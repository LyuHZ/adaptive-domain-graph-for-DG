"""Small CPU tests for the paper-aligned TransDG reconstruction."""

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from train import (
    EXTRAPOLATED_KIND,
    INTERPOLATED_KIND,
    REAL_KIND,
    AdaptiveDomainGraph,
    aggregate_results,
    expand_prototype_bank,
    full_bank_consistency_update,
    prototype_alignment_loss,
    source_prototype_loss,
)
from transclip import count_trainable_parameters


def expansion_args():
    return SimpleNamespace(
        eta_ext=1.0,
        kappa_ext=1.0,
        eta_int=1.0,
        mu_int=1.0,
        kappa_int=1.0,
    )


def make_bank(num_classes: int = 3):
    torch.manual_seed(108)
    visual = F.normalize(torch.randn(num_classes, 3, 512), dim=-1)
    text = F.normalize(torch.randn(num_classes, 3, 512), dim=-1)
    reliability_v = torch.rand(num_classes, 3) + 0.2
    reliability_t = torch.rand(num_classes, 3) + 0.2
    return expand_prototype_bank(
        visual, text, reliability_v, reliability_t, expansion_args()
    )


def test_graph_parameter_count_matches_appendix():
    assert count_trainable_parameters(AdaptiveDomainGraph()) == 1_050_670


def test_deterministic_three_domain_candidate_counts():
    bank = make_bank()
    assert bank.visual.shape == (3, 28, 512)
    assert int((bank.kinds == REAL_KIND).sum()) == 3
    assert int((bank.kinds == EXTRAPOLATED_KIND).sum()) == 4
    assert int((bank.kinds == INTERPOLATED_KIND).sum()) == 21


def test_graph_forward_shapes_and_finiteness():
    bank = make_bank()
    graph = AdaptiveDomainGraph()
    visual = F.normalize(torch.randn(2, 3, 512), dim=-1)
    text = F.normalize(torch.randn(2, 3, 512), dim=-1)
    outputs = graph(visual, text, bank, propagation_layers=3, pseudo_topk=3, tau_cls=0.07)
    assert [tuple(output.shape) for output in outputs] == [
        (2, 3),
        (2, 3, 512),
        (2, 3, 512),
        (2, 3),
    ]
    assert all(bool(torch.isfinite(output).all()) for output in outputs)


def test_instance_bridge_changes_visual_state_at_each_layer():
    bank = make_bank(num_classes=2)
    graph = AdaptiveDomainGraph()
    visual = F.normalize(torch.randn(1, 2, 512), dim=-1)
    text_a = F.normalize(torch.randn(1, 2, 512), dim=-1)
    text_b = -text_a
    refined_a = graph(visual, text_a, bank, 1, 3, 0.07)[1]
    refined_b = graph(visual, text_b, bank, 1, 3, 0.07)[1]
    assert not torch.allclose(refined_a, refined_b)


def test_prototype_alignment_matches_equation_13():
    bank = make_bank()
    reliability = 0.5 * (bank.reliability_v + bank.reliability_t)
    similarity = (F.normalize(bank.visual, dim=-1) * F.normalize(bank.text, dim=-1)).sum(-1)
    expected = (reliability * (1.0 - similarity)).sum() / reliability.sum()
    assert torch.allclose(prototype_alignment_loss(bank), expected)


def test_source_loss_uses_visual_reliability_only():
    bank = make_bank(num_classes=2)
    bank.real_reliability_v = torch.tensor([[0.9, 0.7, 0.5], [0.8, 0.6, 0.4]])
    bank.real_reliability_t = torch.full_like(bank.real_reliability_v, 0.01)
    refined = F.normalize(torch.randn(2, 2, 512), dim=-1)
    labels = torch.tensor([0, 1])
    domains = torch.tensor([1, 2])
    actual = source_prototype_loss(refined, labels, domains, bank, 0.07)
    true_states = refined[torch.arange(2), labels]
    logits = (
        true_states @ bank.real_visual.flatten(0, 1).t() / 0.07
        + bank.real_reliability_v.flatten().log()
    )
    expected = F.cross_entropy(logits, labels * 3 + domains)
    assert torch.allclose(actual, expected)


def test_result_aggregation_uses_sample_standard_deviation():
    runs = [
        {"seed": 108, "target_domain": "Art", "target_accuracy": 90.0},
        {"seed": 113, "target_domain": "Art", "target_accuracy": 92.0},
        {"seed": 115, "target_domain": "Art", "target_accuracy": 94.0},
    ]
    aggregate = aggregate_results(runs)
    art = aggregate["by_target_domain"]["Art"]
    assert art["mean"] == 92.0
    assert art["sample_standard_deviation"] == 2.0


def test_full_bank_consistency_gradient_runs_with_bounded_batches():
    class ToyModel(torch.nn.Module):
        num_classes = 2

        def __init__(self):
            super().__init__()
            self.visual_projection = torch.nn.Linear(4, 512)
            self.text_projection = torch.nn.Linear(4, 512)

        def extract_features(self, images, labels):
            visual = F.normalize(self.visual_projection(images), dim=-1)
            label_text = F.normalize(self.text_projection(images), dim=-1)
            return {"visual": visual, "label_text": label_text}

        def trainable_parameters(self):
            return (parameter for parameter in self.parameters() if parameter.requires_grad)

    torch.manual_seed(108)
    images = torch.randn(8, 4)
    domains = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    labels = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
    loader = DataLoader(TensorDataset(images, domains, labels), batch_size=2)
    model = ToyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    args = SimpleNamespace(
        amp=False,
        lambda_con=0.5,
        beta=1.0,
        eta_ext=1.0,
        kappa_ext=1.0,
        eta_int=1.0,
        mu_int=1.0,
        kappa_int=1.0,
    )
    before = model.visual_projection.weight.detach().clone()
    loss = full_bank_consistency_update(
        model, loader, optimizer, scaler, args, torch.device("cpu"), 2
    )
    assert torch.isfinite(torch.tensor(loss))
    assert not torch.equal(before, model.visual_projection.weight)
