"""OfficeHome leave-one-domain-out training for TransDG.

The method is implemented in two stages:

1. Train adaptive prompts and visual enhancement with frozen CLIP encoders.
2. Freeze the resulting prototype bank and train only the shared graph module.

Target-domain samples are evaluated only after source-validation checkpoint
selection and never participate in training, prototype construction, or model
selection.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

import clip
from transclip import CustomCLIP, count_trainable_parameters, l2_normalize


DOMAINS = ("Art", "Clipart", "Product", "RealWorld")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
EPS = 1e-8
REAL_KIND = 0
EXTRAPOLATED_KIND = 1
INTERPOLATED_KIND = 2


@dataclass(frozen=True)
class Sample:
    path: str
    domain: int
    label: int


@dataclass
class LODOSplit:
    train: list[Sample]
    validation: list[Sample]
    target: list[Sample]
    classnames: list[str]
    source_domains: list[str]
    target_domain: str


@dataclass
class PrototypeBank:
    visual: torch.Tensor
    text: torch.Tensor
    reliability_v: torch.Tensor
    reliability_t: torch.Tensor
    kinds: torch.Tensor
    real_visual: torch.Tensor
    real_text: torch.Tensor
    real_reliability_v: torch.Tensor
    real_reliability_t: torch.Tensor
    center_v: torch.Tensor
    center_t: torch.Tensor
    radius_v: torch.Tensor
    radius_t: torch.Tensor

    @property
    def reliability(self) -> torch.Tensor:
        return 0.5 * (self.reliability_v + self.reliability_t)

    @property
    def real_reliability(self) -> torch.Tensor:
        return 0.5 * (self.real_reliability_v + self.real_reliability_t)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu()
            for key, value in vars(self).items()
            if isinstance(value, torch.Tensor)
        }


@dataclass
class FeatureStatistics:
    sum_v: torch.Tensor
    sum_t: torch.Tensor
    sum_squared_norm_v: torch.Tensor
    sum_squared_norm_t: torch.Tensor
    counts: torch.Tensor


class OfficeHomeDataset(Dataset):
    def __init__(self, samples: Sequence[Sample], preprocess) -> None:
        self.samples = list(samples)
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            image_tensor = self.preprocess(image.convert("RGB"))
        return (
            image_tensor,
            torch.tensor(sample.domain, dtype=torch.long),
            torch.tensor(sample.label, dtype=torch.long),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TransDG OfficeHome LODO reproduction")
    parser.add_argument("--dataset-root", default="./data/OfficeHome")
    parser.add_argument("--model-path", default="./ViT-B-32.pt")
    parser.add_argument("--output-root", default="./outputs/officehome")
    parser.add_argument("--lodo", default="all", choices=[*DOMAINS, "all"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[108, 113, 115])
    parser.add_argument("--stage", choices=["both", "stage1", "stage2"], default="both")
    parser.add_argument("--stage1-checkpoint")
    parser.add_argument("--stage1-epochs", type=int, default=15)
    parser.add_argument("--stage2-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-con", type=float, default=0.5)
    parser.add_argument("--lambda-src", type=float, default=1.0)
    parser.add_argument("--lambda-trans", type=float, default=0.5)
    parser.add_argument("--tau-cls", type=float, default=0.07)
    parser.add_argument("--tau-src", type=float, default=0.07)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--eta-ext", type=float, default=1.0)
    parser.add_argument("--kappa-ext", type=float, default=1.0)
    parser.add_argument("--eta-int", type=float, default=1.0)
    parser.add_argument("--mu-int", type=float, default=1.0)
    parser.add_argument("--kappa-int", type=float, default=1.0)
    parser.add_argument("--num-propagation-layers", type=int, default=3)
    parser.add_argument("--pseudo-topk-per-type", type=int, default=3)
    parser.add_argument("--visual-class-chunk-size", type=int, default=1)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def list_classnames(domain_root: Path) -> set[str]:
    if not domain_root.is_dir():
        raise FileNotFoundError(f"Missing OfficeHome domain directory: {domain_root}")
    return {path.name for path in domain_root.iterdir() if path.is_dir()}


def list_images(class_root: Path) -> list[str]:
    if not class_root.is_dir():
        return []
    return sorted(
        str(path)
        for path in class_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_lodo_split(dataset_root: str, target_domain: str, seed: int) -> LODOSplit:
    root = Path(dataset_root)
    source_domains = [domain for domain in DOMAINS if domain != target_domain]
    source_class_sets = [list_classnames(root / domain) for domain in source_domains]
    classnames = sorted(set.intersection(*source_class_sets))
    if not classnames:
        raise RuntimeError("No class directories are shared by all source domains")
    class_to_index = {name: index for index, name in enumerate(classnames)}

    train: list[Sample] = []
    validation: list[Sample] = []
    for domain_index, domain in enumerate(source_domains):
        for classname in classnames:
            paths = list_images(root / domain / classname)
            if len(paths) < 2:
                raise RuntimeError(
                    f"Need at least two images for stratified split: {domain}/{classname}"
                )
            rng = random.Random(f"{seed}:{domain}:{classname}")
            rng.shuffle(paths)
            split_index = max(1, min(len(paths) - 1, int(0.8 * len(paths))))
            label = class_to_index[classname]
            train.extend(Sample(path, domain_index, label) for path in paths[:split_index])
            validation.extend(Sample(path, domain_index, label) for path in paths[split_index:])

    target_index = len(source_domains)
    target = [
        Sample(path, target_index, class_to_index[classname])
        for classname in classnames
        for path in list_images(root / target_domain / classname)
    ]
    if not target:
        raise RuntimeError(f"No target images found for {target_domain}")
    random.Random(seed).shuffle(train)
    return LODOSplit(train, validation, target, classnames, source_domains, target_domain)


def make_loader(
    samples: Sequence[Sample],
    preprocess,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        OfficeHomeDataset(samples, preprocess),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        generator=generator,
    )


@torch.no_grad()
def evaluate_stage1(
    model: CustomCLIP, loader: DataLoader, device: torch.device, use_amp: bool
) -> tuple[float, dict[int, float]]:
    model.eval()
    correct: dict[int, int] = {}
    total: dict[int, int] = {}
    for images, domains, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits, _ = model(images, labels=None)
        predictions = logits.argmax(dim=1).cpu()
        for domain in domains.unique():
            domain_id = int(domain)
            mask = domains == domain
            correct[domain_id] = correct.get(domain_id, 0) + int(
                (predictions[mask] == labels.cpu()[mask]).sum()
            )
            total[domain_id] = total.get(domain_id, 0) + int(mask.sum())
    per_domain = {domain: 100.0 * correct[domain] / total[domain] for domain in total}
    return float(np.mean(list(per_domain.values()))), per_domain


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_clip_backbone(model_path: str, device: torch.device):
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing CLIP checkpoint: {path}. TransDG requires a local ViT-B/32 checkpoint."
        )
    try:
        return clip.load(str(path), device=device)
    except (EOFError, OSError, RuntimeError) as error:
        raise RuntimeError(
            f"Cannot read CLIP checkpoint {path}. Verify that the local ViT-B/32 file is "
            "complete and loadable before starting an experiment."
        ) from error


def save_stage1(path: Path, model: CustomCLIP, epoch: int, validation: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "trainable": model.trainable_state_dict(),
            "epoch": epoch,
            "source_validation_accuracy": validation,
        },
        path,
    )


def load_stage1(path: str | Path, model: CustomCLIP, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("trainable", checkpoint)
    model.load_trainable_state_dict(state)


def train_stage1(
    model: CustomCLIP,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
    num_source_domains: int,
) -> Path:
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.stage1_epochs, eta_min=0.0
    )
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_accuracy = -math.inf
    best_path = output_dir / "stage1_best.pth"
    history = []

    for epoch in range(1, args.stage1_epochs + 1):
        model.train()
        classification_sum = 0.0
        sample_count = 0
        progress = tqdm(train_loader, desc=f"Stage I {epoch}/{args.stage1_epochs}")
        for images, domains, labels in progress:
            images = images.to(device, non_blocking=True)
            domains = domains.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits, _ = model(images, labels)
                classification = F.cross_entropy(logits, labels)
            scaler.scale(classification).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), 1.0)
            scaler.step(optimizer)
            scaler.update()
            batch_size = labels.numel()
            classification_sum += float(classification.detach()) * batch_size
            sample_count += batch_size
            progress.set_postfix(classification=classification_sum / sample_count)

        consistency = full_bank_consistency_update(
            model,
            train_loader,
            optimizer,
            scaler,
            args,
            device,
            num_source_domains,
        )
        scheduler.step()

        mean_validation, per_domain = evaluate_stage1(
            model, validation_loader, device, use_amp
        )
        history.append(
            {
                "epoch": epoch,
                "classification_loss": classification_sum / sample_count,
                "prototype_consistency_loss": consistency,
                "train_objective": (
                    classification_sum / sample_count + args.lambda_con * consistency
                ),
                "source_validation_accuracy": mean_validation,
                "source_validation_by_domain": per_domain,
            }
        )
        if mean_validation > best_accuracy:
            best_accuracy = mean_validation
            save_stage1(best_path, model, epoch, mean_validation)
        save_json(output_dir / "stage1_history.json", {"epochs": history})

    load_stage1(best_path, model, device)
    return best_path


def weighted_center(prototypes: torch.Tensor, reliability: torch.Tensor) -> torch.Tensor:
    weights = reliability / reliability.sum(dim=1, keepdim=True).clamp_min(EPS)
    return (weights.unsqueeze(-1) * prototypes).sum(dim=1)


def _extrapolation_candidate(
    offsets: torch.Tensor,
    reliability: torch.Tensor,
    center: torch.Tensor,
    radius: torch.Tensor,
    support: tuple[int, ...],
    anchor: int,
    eta: float,
    kappa: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    support_weights = {
        index: (0.5 if index == anchor else 0.5 / (len(support) - 1))
        for index in support
    }
    anchor_weight = support_weights[anchor]
    candidate = offsets[anchor] / anchor_weight
    for index in support:
        if index != anchor:
            candidate = candidate - support_weights[index] * offsets[index] / anchor_weight
    base = sum(support_weights[index] * offsets[index] for index in support)
    displacement = candidate - base
    magnitude = displacement.norm() / radius.clamp_min(EPS)
    direction = F.cosine_similarity(
        displacement.unsqueeze(0), base.unsqueeze(0), dim=-1, eps=EPS
    ).squeeze(0)
    directional_deviation = 1.0 - direction
    inherited = sum(
        support_weights[index] * reliability[index] for index in support
    )
    confidence = inherited * torch.exp(-eta * magnitude - kappa * directional_deviation)
    calibrated_offset = confidence * candidate + (1.0 - confidence) * base
    prototype = l2_normalize(center + calibrated_offset)
    return prototype, confidence, calibrated_offset


def _interpolation_candidate(
    offsets: list[torch.Tensor],
    reliability: list[torch.Tensor],
    center: torch.Tensor,
    radius: torch.Tensor,
    pair: tuple[int, int],
    weights: tuple[torch.Tensor, torch.Tensor],
    eta: float,
    mu: float,
    kappa: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a, b = pair
    weight_a, weight_b = weights
    mixed = weight_a * offsets[a] + weight_b * offsets[b]
    balanced = 0.5 * (offsets[a] + offsets[b])
    gamma = mixed.norm() / radius.clamp_min(EPS)
    span = weight_a * weight_b * (offsets[a] - offsets[b]).norm() / radius.clamp_min(EPS)
    cosine = F.cosine_similarity(
        offsets[a].unsqueeze(0), offsets[b].unsqueeze(0), dim=-1, eps=EPS
    ).squeeze(0)
    theta = weight_a * weight_b * (1.0 - cosine)
    inherited = weight_a * reliability[a] + weight_b * reliability[b]
    confidence = inherited * torch.exp(-eta * gamma - mu * span - kappa * theta)
    calibrated_offset = confidence * mixed + (1.0 - confidence) * balanced
    prototype = l2_normalize(center + calibrated_offset)
    return prototype, confidence, calibrated_offset


def expand_prototype_bank(
    real_visual: torch.Tensor,
    real_text: torch.Tensor,
    reliability_v: torch.Tensor,
    reliability_t: torch.Tensor,
    args: argparse.Namespace,
) -> PrototypeBank:
    num_classes, num_domains, _ = real_visual.shape
    center_v = weighted_center(real_visual, reliability_v)
    center_t = weighted_center(real_text, reliability_t)
    real_offsets_v = real_visual - center_v.unsqueeze(1)
    real_offsets_t = real_text - center_t.unsqueeze(1)
    radius_v = real_offsets_v.norm(dim=-1).mean(dim=1).clamp_min(EPS)
    radius_t = real_offsets_t.norm(dim=-1).mean(dim=1).clamp_min(EPS)

    class_visual = []
    class_text = []
    class_rel_v = []
    class_rel_t = []
    expected_kinds: list[int] | None = None

    for class_index in range(num_classes):
        visual = [item for item in real_visual[class_index]]
        text = [item for item in real_text[class_index]]
        rel_v = [item for item in reliability_v[class_index]]
        rel_t = [item for item in reliability_t[class_index]]
        offsets_v = [item for item in real_offsets_v[class_index]]
        offsets_t = [item for item in real_offsets_t[class_index]]
        kinds = [REAL_KIND] * num_domains

        # Appendix D.2: one coupled extrapolation is retained for every subset.
        for subset_size in range(2, num_domains + 1):
            for support in itertools.combinations(range(num_domains), subset_size):
                candidates = []
                for anchor in support:
                    candidate_v = _extrapolation_candidate(
                        real_offsets_v[class_index],
                        reliability_v[class_index],
                        center_v[class_index],
                        radius_v[class_index],
                        support,
                        anchor,
                        args.eta_ext,
                        args.kappa_ext,
                    )
                    candidate_t = _extrapolation_candidate(
                        real_offsets_t[class_index],
                        reliability_t[class_index],
                        center_t[class_index],
                        radius_t[class_index],
                        support,
                        anchor,
                        args.eta_ext,
                        args.kappa_ext,
                    )
                    coupled = 0.5 * (candidate_v[1] + candidate_t[1])
                    candidates.append((coupled, candidate_v, candidate_t))
                _, selected_v, selected_t = max(candidates, key=lambda item: float(item[0]))
                visual.append(selected_v[0])
                text.append(selected_t[0])
                rel_v.append(selected_v[1])
                rel_t.append(selected_t[1])
                offsets_v.append(selected_v[2])
                offsets_t.append(selected_t[2])
                kinds.append(EXTRAPOLATED_KIND)

        # Appendix D.2: enumerate every unordered pair from real + extrapolated supports.
        interpolation_support_count = len(visual)
        for pair in itertools.combinations(range(interpolation_support_count), 2):
            a, b = pair
            coupled_a = 0.5 * (rel_v[a] + rel_t[a])
            coupled_b = 0.5 * (rel_v[b] + rel_t[b])
            denominator = coupled_a + coupled_b + 2.0 * EPS
            weights = ((coupled_a + EPS) / denominator, (coupled_b + EPS) / denominator)
            candidate_v = _interpolation_candidate(
                offsets_v,
                rel_v,
                center_v[class_index],
                radius_v[class_index],
                pair,
                weights,
                args.eta_int,
                args.mu_int,
                args.kappa_int,
            )
            candidate_t = _interpolation_candidate(
                offsets_t,
                rel_t,
                center_t[class_index],
                radius_t[class_index],
                pair,
                weights,
                args.eta_int,
                args.mu_int,
                args.kappa_int,
            )
            visual.append(candidate_v[0])
            text.append(candidate_t[0])
            rel_v.append(candidate_v[1])
            rel_t.append(candidate_t[1])
            kinds.append(INTERPOLATED_KIND)

        if expected_kinds is None:
            expected_kinds = kinds
        elif kinds != expected_kinds:
            raise RuntimeError("Prototype enumeration differs across classes")
        class_visual.append(torch.stack(visual))
        class_text.append(torch.stack(text))
        class_rel_v.append(torch.stack(rel_v))
        class_rel_t.append(torch.stack(rel_t))

    return PrototypeBank(
        visual=torch.stack(class_visual),
        text=torch.stack(class_text),
        reliability_v=torch.stack(class_rel_v),
        reliability_t=torch.stack(class_rel_t),
        kinds=torch.tensor(expected_kinds, device=real_visual.device, dtype=torch.long),
        real_visual=real_visual,
        real_text=real_text,
        real_reliability_v=reliability_v,
        real_reliability_t=reliability_t,
        center_v=center_v,
        center_t=center_t,
        radius_v=radius_v,
        radius_t=radius_t,
    )


@torch.no_grad()
def collect_feature_statistics(
    model: CustomCLIP,
    loader: DataLoader,
    num_classes: int,
    num_domains: int,
    device: torch.device,
    use_amp: bool,
    description: str,
) -> FeatureStatistics:
    feature_dim = 512
    sum_v = torch.zeros(num_classes, num_domains, feature_dim, device=device)
    sum_t = torch.zeros_like(sum_v)
    sum_squared_norm_v = torch.zeros(num_classes, num_domains, device=device)
    sum_squared_norm_t = torch.zeros_like(sum_squared_norm_v)
    counts = torch.zeros(num_classes, num_domains, device=device)
    model.eval()

    for images, domains, labels in tqdm(loader, desc=description):
        images = images.to(device, non_blocking=True)
        domains = domains.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            features = model.extract_features(images, labels)
        visual = l2_normalize(features["visual"].float())
        text = l2_normalize(features["label_text"].float())
        for index in range(labels.numel()):
            class_index = int(labels[index])
            domain_index = int(domains[index])
            sum_v[class_index, domain_index] += visual[index]
            sum_t[class_index, domain_index] += text[index]
            sum_squared_norm_v[class_index, domain_index] += visual[index].square().sum()
            sum_squared_norm_t[class_index, domain_index] += text[index].square().sum()
            counts[class_index, domain_index] += 1

    return FeatureStatistics(
        sum_v=sum_v,
        sum_t=sum_t,
        sum_squared_norm_v=sum_squared_norm_v,
        sum_squared_norm_t=sum_squared_norm_t,
        counts=counts,
    )


def prototype_bank_from_statistics(
    statistics: FeatureStatistics, args: argparse.Namespace
) -> PrototypeBank:
    counts = statistics.counts

    if (counts == 0).any():
        missing = (counts == 0).nonzero().tolist()
        raise RuntimeError(f"Missing source class-domain groups: {missing[:10]}")
    denominator = counts.unsqueeze(-1)
    mean_v = statistics.sum_v / denominator
    mean_t = statistics.sum_t / denominator
    # Variance is computed before prototype normalization (Appendix Eq. A2).
    variance_v = (
        statistics.sum_squared_norm_v / counts - mean_v.square().sum(dim=-1)
    ).clamp_min(0.0)
    variance_t = (
        statistics.sum_squared_norm_t / counts - mean_t.square().sum(dim=-1)
    ).clamp_min(0.0)
    real_visual = l2_normalize(mean_v)
    real_text = l2_normalize(mean_t)
    normalized_count = counts / counts.max(dim=1, keepdim=True).values
    reliability_v = normalized_count * torch.exp(-args.beta * variance_v)
    reliability_t = normalized_count * torch.exp(-args.beta * variance_t)
    return expand_prototype_bank(
        real_visual, real_text, reliability_v, reliability_t, args
    )


@torch.no_grad()
def compute_prototype_bank(
    model: CustomCLIP,
    loader: DataLoader,
    num_classes: int,
    num_domains: int,
    args: argparse.Namespace,
    device: torch.device,
) -> PrototypeBank:
    statistics = collect_feature_statistics(
        model,
        loader,
        num_classes,
        num_domains,
        device,
        args.amp and device.type == "cuda",
        "Construct fixed prototype bank",
    )
    return prototype_bank_from_statistics(statistics, args)


def prototype_alignment_loss(bank: PrototypeBank) -> torch.Tensor:
    """Reliability-weighted paired-anchor consistency from Eq. (13)."""
    reliability = bank.reliability
    similarity = (l2_normalize(bank.visual) * l2_normalize(bank.text)).sum(dim=-1)
    return (reliability * (1.0 - similarity)).sum() / reliability.sum().clamp_min(EPS)


def full_bank_consistency_update(
    model: CustomCLIP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    args: argparse.Namespace,
    device: torch.device,
    num_source_domains: int,
) -> float:
    """Apply the exact full-source gradient of the Stage-I Eq. (13) term.

    Retaining the autograd graph for every source image is prohibitively large.
    We therefore differentiate Eq. (13) with respect to its sufficient
    statistics, then recompute source features and apply the resulting
    vector-Jacobian product. This yields the same gradient while bounding
    memory by one mini-batch.
    """
    use_amp = args.amp and device.type == "cuda"
    statistics = collect_feature_statistics(
        model,
        loader,
        model.num_classes,
        num_source_domains,
        device,
        use_amp,
        "Stage I full-bank statistics",
    )
    differentiable_statistics = FeatureStatistics(
        sum_v=statistics.sum_v.detach().requires_grad_(True),
        sum_t=statistics.sum_t.detach().requires_grad_(True),
        sum_squared_norm_v=statistics.sum_squared_norm_v.detach().requires_grad_(True),
        sum_squared_norm_t=statistics.sum_squared_norm_t.detach().requires_grad_(True),
        counts=statistics.counts.detach(),
    )
    bank = prototype_bank_from_statistics(differentiable_statistics, args)
    alignment = prototype_alignment_loss(bank)
    statistic_tensors = (
        differentiable_statistics.sum_v,
        differentiable_statistics.sum_t,
        differentiable_statistics.sum_squared_norm_v,
        differentiable_statistics.sum_squared_norm_t,
    )
    statistic_gradients = torch.autograd.grad(
        args.lambda_con * alignment, statistic_tensors
    )

    optimizer.zero_grad(set_to_none=True)
    model.train()
    for images, domains, labels in tqdm(loader, desc="Stage I Eq. (13) gradient"):
        images = images.to(device, non_blocking=True)
        domains = domains.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            features = model.extract_features(images, labels)
        visual = features["visual"].float()
        text = features["label_text"].float()
        grad_sum_v = statistic_gradients[0][labels, domains]
        grad_sum_t = statistic_gradients[1][labels, domains]
        grad_squared_v = statistic_gradients[2][labels, domains]
        grad_squared_t = statistic_gradients[3][labels, domains]
        surrogate = (
            (visual * grad_sum_v).sum()
            + (text * grad_sum_t).sum()
            + (visual.square().sum(dim=-1) * grad_squared_v).sum()
            + (text.square().sum(dim=-1) * grad_squared_t).sum()
        )
        scaler.scale(surrogate).backward()

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), 1.0)
    scaler.step(optimizer)
    scaler.update()
    return float(alignment.detach())


class AdaptiveDomainGraph(nn.Module):
    """Shared context-guided propagation module from Eqs. (17)--(19).

    Its parameter count is 1,050,670: four biased 512 -> 512 projections,
    a 3 x 8 anchor-type embedding, and two 11-dimensional context weights.
    The same parameters are reused across all classes and propagation layers.
    """

    def __init__(self, feature_dim: int = 512, type_dim: int = 8) -> None:
        super().__init__()
        self.visual_feature_projection = nn.Linear(feature_dim, feature_dim, bias=True)
        self.text_feature_projection = nn.Linear(feature_dim, feature_dim, bias=True)
        self.visual_message_projection = nn.Linear(feature_dim, feature_dim, bias=True)
        self.text_message_projection = nn.Linear(feature_dim, feature_dim, bias=True)
        self.type_embedding = nn.Embedding(3, type_dim)
        self.context_weights = nn.Parameter(torch.zeros(2, type_dim + 3))
        nn.init.normal_(self.context_weights, std=0.02)

    def _context_logits(
        self,
        query: torch.Tensor,
        anchors: torch.Tensor,
        reliability: torch.Tensor,
        kinds: torch.Tensor,
        receiving_modality: int,
        receiving_center: torch.Tensor,
        receiving_radius: torch.Tensor,
        anchor_center: torch.Tensor,
        anchor_radius: torch.Tensor,
    ) -> torch.Tensor:
        if receiving_modality == 0:
            query_projection = self.visual_feature_projection(query)
            anchor_projection = self.text_feature_projection(anchors)
        else:
            query_projection = self.text_feature_projection(query)
            anchor_projection = self.visual_feature_projection(anchors)
        similarity = (query_projection.unsqueeze(1) * anchor_projection).sum(dim=-1)

        prototype_radius = (anchors - anchor_center).norm(dim=-1) / anchor_radius.clamp_min(EPS)
        instance_radius = (query - receiving_center).norm(dim=-1) / receiving_radius.clamp_min(EPS)
        compatibility = torch.exp(-torch.abs(instance_radius.unsqueeze(1) - prototype_radius))
        type_features = self.type_embedding(kinds)
        context = torch.cat(
            [
                type_features,
                reliability.unsqueeze(-1),
                prototype_radius.unsqueeze(-1),
                compatibility.unsqueeze(-1),
            ],
            dim=-1,
        )
        context_bias = torch.einsum(
            "bak,k->ba", context, self.context_weights[receiving_modality]
        )
        return similarity + context_bias

    def _select_local_anchors(
        self,
        query: torch.Tensor,
        anchors: torch.Tensor,
        reliability: torch.Tensor,
        kinds: torch.Tensor,
        receiving_modality: int,
        receiving_center: torch.Tensor,
        receiving_radius: torch.Tensor,
        anchor_center: torch.Tensor,
        anchor_radius: torch.Tensor,
        pseudo_topk: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = query.shape[0]
        expanded_anchors = anchors.unsqueeze(0).expand(batch_size, -1, -1)
        expanded_reliability = reliability.unsqueeze(0).expand(batch_size, -1)
        expanded_kinds = kinds.unsqueeze(0).expand(batch_size, -1)
        logits = self._context_logits(
            query,
            expanded_anchors,
            expanded_reliability,
            expanded_kinds,
            receiving_modality,
            receiving_center,
            receiving_radius,
            anchor_center,
            anchor_radius,
        )

        selected_indices = []
        real_indices = (kinds == REAL_KIND).nonzero(as_tuple=False).flatten()
        selected_indices.append(real_indices.unsqueeze(0).expand(batch_size, -1))
        for kind in (EXTRAPOLATED_KIND, INTERPOLATED_KIND):
            candidate_indices = (kinds == kind).nonzero(as_tuple=False).flatten()
            k = min(pseudo_topk, candidate_indices.numel())
            local = logits[:, candidate_indices].topk(k=k, dim=1).indices
            selected_indices.append(candidate_indices[local])
        indices = torch.cat(selected_indices, dim=1)
        gather_features = indices.unsqueeze(-1).expand(-1, -1, anchors.shape[-1])
        return (
            torch.gather(expanded_anchors, 1, gather_features),
            torch.gather(expanded_reliability, 1, indices),
            torch.gather(expanded_kinds, 1, indices),
        )

    def _update(
        self,
        query: torch.Tensor,
        bridge_state: torch.Tensor,
        anchors: torch.Tensor,
        reliability: torch.Tensor,
        kinds: torch.Tensor,
        receiving_modality: int,
        receiving_center: torch.Tensor,
        receiving_radius: torch.Tensor,
        anchor_center: torch.Tensor,
        anchor_radius: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prototype_logits = self._context_logits(
            query,
            anchors,
            reliability,
            kinds,
            receiving_modality,
            receiving_center,
            receiving_radius,
            anchor_center,
            anchor_radius,
        )
        if receiving_modality == 0:
            query_projection = self.visual_feature_projection(query)
            bridge_projection = self.text_feature_projection(bridge_state)
            prototype_values = self.visual_message_projection(anchors)
            bridge_value = self.visual_message_projection(bridge_state)
        else:
            query_projection = self.text_feature_projection(query)
            bridge_projection = self.visual_feature_projection(bridge_state)
            prototype_values = self.text_message_projection(anchors)
            bridge_value = self.text_message_projection(bridge_state)

        # Equation (16): the opposite instance is a bridge node in every layer.
        bridge_logit = (query_projection * bridge_projection).sum(dim=-1, keepdim=True)
        logits = torch.cat([prototype_logits, bridge_logit], dim=1)
        attention = F.softmax(logits, dim=1)
        prototype_attention = attention[:, :-1]
        bridge_attention = attention[:, -1:]
        message = (
            (prototype_attention.unsqueeze(-1) * prototype_values).sum(dim=1)
            + bridge_attention * bridge_value
        )
        transfer_reliability = (prototype_attention * reliability).sum(dim=1)
        return l2_normalize(query + message), transfer_reliability

    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        bank: PrototypeBank,
        propagation_layers: int,
        pseudo_topk: int,
        tau_cls: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scores = []
        refined_visual = []
        refined_text = []
        transfer_reliability = []

        for class_index in range(text_features.shape[1]):
            visual_state = (
                visual_features[:, class_index]
                if visual_features.ndim == 3
                else visual_features
            )
            text_state = text_features[:, class_index]
            selected_text = self._select_local_anchors(
                visual_state,
                bank.text[class_index],
                bank.reliability_t[class_index],
                bank.kinds,
                0,
                bank.center_v[class_index],
                bank.radius_v[class_index],
                bank.center_t[class_index],
                bank.radius_t[class_index],
                pseudo_topk,
            )
            selected_visual = self._select_local_anchors(
                text_state,
                bank.visual[class_index],
                bank.reliability_v[class_index],
                bank.kinds,
                1,
                bank.center_t[class_index],
                bank.radius_t[class_index],
                bank.center_v[class_index],
                bank.radius_v[class_index],
                pseudo_topk,
            )

            rel_visual = selected_text[1].mean(dim=1)
            rel_text = selected_visual[1].mean(dim=1)
            for _ in range(propagation_layers):
                previous_visual = visual_state
                previous_text = text_state
                next_visual, rel_visual = self._update(
                    previous_visual,
                    previous_text,
                    *selected_text,
                    0,
                    bank.center_v[class_index],
                    bank.radius_v[class_index],
                    bank.center_t[class_index],
                    bank.radius_t[class_index],
                )
                next_text, rel_text = self._update(
                    previous_text,
                    previous_visual,
                    *selected_visual,
                    1,
                    bank.center_t[class_index],
                    bank.radius_t[class_index],
                    bank.center_v[class_index],
                    bank.radius_v[class_index],
                )
                visual_state = next_visual
                text_state = next_text

            class_score = torch.logsumexp(
                visual_state @ bank.text[class_index].transpose(0, 1) / tau_cls
                + torch.log(bank.reliability_t[class_index].clamp_min(EPS)),
                dim=1,
            )
            scores.append(class_score)
            refined_visual.append(visual_state)
            refined_text.append(text_state)
            transfer_reliability.append(0.5 * (rel_visual + rel_text))

        return (
            torch.stack(scores, dim=1),
            torch.stack(refined_visual, dim=1),
            torch.stack(refined_text, dim=1),
            torch.stack(transfer_reliability, dim=1),
        )


def source_prototype_loss(
    refined_visual: torch.Tensor,
    labels: torch.Tensor,
    domains: torch.Tensor,
    bank: PrototypeBank,
    temperature: float,
) -> torch.Tensor:
    batch_indices = torch.arange(labels.numel(), device=labels.device)
    true_state = refined_visual[batch_indices, labels]
    flattened_prototypes = bank.real_visual.flatten(0, 1)
    # Equation (20) uses the visual reliability r^v for visual prototypes.
    flattened_reliability = bank.real_reliability_v.flatten()
    logits = (
        true_state @ flattened_prototypes.transpose(0, 1) / temperature
        + torch.log(flattened_reliability.clamp_min(EPS))
    )
    target = labels * bank.real_visual.shape[1] + domains
    return F.cross_entropy(logits, target)


def propagated_transfer_loss(
    refined_visual: torch.Tensor,
    refined_text: torch.Tensor,
    labels: torch.Tensor,
    transfer_reliability: torch.Tensor,
) -> torch.Tensor:
    batch_indices = torch.arange(labels.numel(), device=labels.device)
    visual = refined_visual[batch_indices, labels]
    text = refined_text[batch_indices, labels]
    reliability = transfer_reliability[batch_indices, labels]
    distance = 1.0 - (l2_normalize(visual) * l2_normalize(text)).sum(dim=-1)
    return (reliability * distance).sum() / reliability.sum().clamp_min(EPS)


@torch.no_grad()
def evaluate_stage2(
    model: CustomCLIP,
    graph: AdaptiveDomainGraph,
    bank: PrototypeBank,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[float, dict[int, float]]:
    model.eval()
    graph.eval()
    use_amp = args.amp and device.type == "cuda"
    correct: dict[int, int] = {}
    total: dict[int, int] = {}
    for images, domains, labels in loader:
        images = images.to(device, non_blocking=True)
        labels_device = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            features = model.extract_features(images, labels=None)
            scores, _, _, _ = graph(
                features["visual"],
                features["text"],
                bank,
                args.num_propagation_layers,
                args.pseudo_topk_per_type,
                args.tau_cls,
            )
        predictions = scores.argmax(dim=1).cpu()
        for domain in domains.unique():
            domain_id = int(domain)
            mask = domains == domain
            correct[domain_id] = correct.get(domain_id, 0) + int(
                (predictions[mask] == labels[mask]).sum()
            )
            total[domain_id] = total.get(domain_id, 0) + int(mask.sum())
    per_domain = {domain: 100.0 * correct[domain] / total[domain] for domain in total}
    return float(np.mean(list(per_domain.values()))), per_domain


def train_stage2(
    model: CustomCLIP,
    graph: AdaptiveDomainGraph,
    bank: PrototypeBank,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> Path:
    optimizer = torch.optim.AdamW(
        graph.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.stage2_epochs, eta_min=0.0
    )
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_accuracy = -math.inf
    best_path = output_dir / "stage2_best.pth"
    history = []
    model.eval()

    for epoch in range(1, args.stage2_epochs + 1):
        graph.train()
        loss_sum = 0.0
        sample_count = 0
        progress = tqdm(train_loader, desc=f"Stage II {epoch}/{args.stage2_epochs}")
        for images, domains, labels in progress:
            images = images.to(device, non_blocking=True)
            domains = domains.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=use_amp):
                features = model.extract_features(images, labels=None)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                scores, refined_v, refined_t, transfer_reliability = graph(
                    features["visual"],
                    features["text"],
                    bank,
                    args.num_propagation_layers,
                    args.pseudo_topk_per_type,
                    args.tau_cls,
                )
                classification = F.cross_entropy(scores, labels)
                source = source_prototype_loss(
                    refined_v, labels, domains, bank, args.tau_src
                )
                transfer = propagated_transfer_loss(
                    refined_v, refined_t, labels, transfer_reliability
                )
                loss = (
                    classification
                    + args.lambda_src * source
                    + args.lambda_trans * transfer
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(graph.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            batch_size = labels.numel()
            loss_sum += float(loss.detach()) * batch_size
            sample_count += batch_size
            progress.set_postfix(loss=loss_sum / sample_count)
        scheduler.step()

        mean_validation, per_domain = evaluate_stage2(
            model, graph, bank, validation_loader, args, device
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / sample_count,
                "source_validation_accuracy": mean_validation,
                "source_validation_by_domain": per_domain,
            }
        )
        if mean_validation > best_accuracy:
            best_accuracy = mean_validation
            best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "graph": graph.state_dict(),
                    "epoch": epoch,
                    "source_validation_accuracy": mean_validation,
                    "prototype_bank": bank.state_dict(),
                },
                best_path,
            )
        save_json(output_dir / "stage2_history.json", {"epochs": history})

    checkpoint = torch.load(best_path, map_location=device)
    graph.load_state_dict(checkpoint["graph"])
    return best_path


def run_fold(args: argparse.Namespace, target_domain: str, seed: int) -> dict:
    seed_everything(seed)
    split = build_lodo_split(args.dataset_root, target_domain, seed)
    summary = {
        "seed": seed,
        "target_domain": target_domain,
        "source_domains": split.source_domains,
        "classes": len(split.classnames),
        "train_samples": len(split.train),
        "validation_samples": len(split.validation),
        "target_samples": len(split.target),
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, preprocess = load_clip_backbone(args.model_path, device)
    model = CustomCLIP(
        split.classnames,
        clip_model,
        visual_class_chunk_size=args.visual_class_chunk_size,
    ).to(device)
    graph = AdaptiveDomainGraph().to(device)
    stage1_parameters = count_trainable_parameters(model)
    graph_parameters = count_trainable_parameters(graph)
    if stage1_parameters != 12_378_705:
        raise RuntimeError(f"Unexpected Stage-I parameter count: {stage1_parameters}")
    if graph_parameters != 1_050_670:
        raise RuntimeError(f"Unexpected graph parameter count: {graph_parameters}")

    train_loader = make_loader(
        split.train, preprocess, args.batch_size, args.num_workers, True, seed
    )
    prototype_loader = make_loader(
        split.train, preprocess, args.eval_batch_size, args.num_workers, False, seed
    )
    validation_loader = make_loader(
        split.validation, preprocess, args.eval_batch_size, args.num_workers, False, seed
    )
    target_loader = make_loader(
        split.target, preprocess, args.eval_batch_size, args.num_workers, False, seed
    )
    output_dir = Path(args.output_root) / f"seed_{seed}" / target_domain.lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        output_dir / "run_config.json",
        {
            "arguments": vars(args),
            "split": summary,
            "stage1_trainable_parameters": stage1_parameters,
            "graph_trainable_parameters": graph_parameters,
        },
    )

    if args.stage1_checkpoint:
        stage1_path = Path(args.stage1_checkpoint)
        load_stage1(stage1_path, model, device)
    elif args.stage in ("both", "stage1"):
        stage1_path = train_stage1(
            model,
            train_loader,
            validation_loader,
            args,
            device,
            output_dir,
            len(split.source_domains),
        )
    else:
        raise ValueError("--stage stage2 requires --stage1-checkpoint")

    summary.update(
        {
            "stage1_checkpoint": str(stage1_path),
            "stage1_trainable_parameters": stage1_parameters,
            "graph_trainable_parameters": graph_parameters,
        }
    )
    if args.stage == "stage1":
        target_accuracy, _ = evaluate_stage1(
            model, target_loader, device, args.amp and device.type == "cuda"
        )
        summary["target_accuracy_stage1"] = target_accuracy
        save_json(output_dir / "summary.json", summary)
        return summary

    bank = compute_prototype_bank(
        model,
        prototype_loader,
        len(split.classnames),
        len(split.source_domains),
        args,
        device,
    )
    kind_counts = {
        "real": int((bank.kinds == REAL_KIND).sum()),
        "extrapolated": int((bank.kinds == EXTRAPOLATED_KIND).sum()),
        "interpolated": int((bank.kinds == INTERPOLATED_KIND).sum()),
    }
    expected_counts = {"real": 3, "extrapolated": 4, "interpolated": 21}
    if len(split.source_domains) == 3 and kind_counts != expected_counts:
        raise RuntimeError(f"Unexpected prototype enumeration: {kind_counts}")

    stage2_path = train_stage2(
        model,
        graph,
        bank,
        train_loader,
        validation_loader,
        args,
        device,
        output_dir,
    )
    # The held-out target is touched only after both checkpoints are selected.
    target_accuracy, _ = evaluate_stage2(
        model, graph, bank, target_loader, args, device
    )
    summary.update(
        {
            "stage2_checkpoint": str(stage2_path),
            "prototype_count_per_class": kind_counts,
            "target_accuracy": target_accuracy,
        }
    )
    save_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def aggregate_results(runs: list[dict]) -> dict:
    """Report target-wise and overall mean +/- sample standard deviation."""
    completed = [run for run in runs if "target_accuracy" in run]
    if not completed:
        return {}

    by_target = {}
    for target in DOMAINS:
        values = [
            float(run["target_accuracy"])
            for run in completed
            if run["target_domain"] == target
        ]
        if values:
            by_target[target] = {
                "runs": len(values),
                "mean": float(np.mean(values)),
                "sample_standard_deviation": (
                    float(np.std(values, ddof=1)) if len(values) > 1 else None
                ),
            }

    seed_macro_accuracies = []
    for seed in sorted({int(run["seed"]) for run in completed}):
        values = [
            float(run["target_accuracy"])
            for run in completed
            if int(run["seed"]) == seed
        ]
        seed_macro_accuracies.append(float(np.mean(values)))
    return {
        "by_target_domain": by_target,
        "overall_macro": {
            "seed_macro_accuracies": seed_macro_accuracies,
            "mean": float(np.mean(seed_macro_accuracies)),
            "sample_standard_deviation": (
                float(np.std(seed_macro_accuracies, ddof=1))
                if len(seed_macro_accuracies) > 1
                else None
            ),
        },
    }


def main() -> None:
    args = parse_args()
    targets = DOMAINS if args.lodo == "all" else (args.lodo,)
    summaries = [
        run_fold(args, target_domain, seed)
        for seed in args.seeds
        for target_domain in targets
    ]
    if not args.dry_run:
        aggregate = aggregate_results(summaries)
        save_json(
            Path(args.output_root) / "all_results.json",
            {"runs": summaries, "aggregate": aggregate},
        )
        print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
