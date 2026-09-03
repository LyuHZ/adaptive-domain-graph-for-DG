"""Prompt-induced dual-modal feature model used by TransDG.

The implementation follows Eqs. (1)--(3) in the paper and Appendix D.1.
The CLIP visual and text encoders stay frozen; only the prompt/statistics
projector and visual feature-enhancement modules are optimized in Stage I.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from clip import clip


EPS = 1e-8


def l2_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.nan_to_num(x / x.norm(dim=dim, keepdim=True).clamp_min(EPS))


class FrozenTextEncoder(nn.Module):
    """Expose CLIP's text path for prompt embeddings instead of token IDs."""

    def __init__(self, clip_model: nn.Module):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(
        self, prompt_embeddings: torch.Tensor, tokenized_prompts: torch.Tensor
    ) -> torch.Tensor:
        tokenized_prompts = tokenized_prompts.to(prompt_embeddings.device)
        x = prompt_embeddings + self.positional_embedding.to(
            device=prompt_embeddings.device, dtype=self.dtype
        )
        x = x.permute(1, 0, 2)
        transformer_output = self.transformer(x)
        x = transformer_output[0] if isinstance(transformer_output, tuple) else transformer_output
        x = self.ln_final(x.permute(1, 0, 2)).type(self.dtype)
        eot = tokenized_prompts.argmax(dim=-1)
        return x[torch.arange(x.shape[0], device=x.device), eot] @ self.text_projection


class MultiLayerStatisticsProjector(nn.Module):
    """Map statistics from all 12 ViT layers to one domain and four style tokens.

    For each layer, the token-wise mean and standard deviation are averaged
    before the 768 -> 256 -> 512 projection. This is the explicit form of the
    pooling operation used by the recovered implementation.
    """

    def __init__(
        self,
        visual_width: int = 768,
        embed_dim: int = 512,
        num_layers: int = 12,
        num_style_tokens: int = 4,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.projectors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(visual_width, 256),
                    nn.Linear(256, embed_dim),
                )
                for _ in range(num_layers)
            ]
        )
        self.domain_mixer = nn.Linear(num_layers, 1)
        self.style_mixer = nn.Linear(num_layers, num_style_tokens)

    def forward(self, layer_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # The repository's CLIP returns [layers, batch, tokens, visual_width].
        if layer_states.ndim != 4:
            raise ValueError(f"Expected four-dimensional ViT states, got {layer_states.shape}")
        if layer_states.shape[0] != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} ViT layers, got {layer_states.shape[0]}"
            )

        states = layer_states.float()
        means = states.mean(dim=2)
        stds = states.var(dim=2, unbiased=False).clamp_min(0.0).sqrt()
        statistics = 0.5 * (means + stds)
        projected = torch.stack(
            [projector(statistics[i]) for i, projector in enumerate(self.projectors)],
            dim=1,
        )
        mixed = projected.transpose(1, 2)
        domain_token = self.domain_mixer(mixed).transpose(1, 2)
        style_tokens = self.style_mixer(mixed).transpose(1, 2)
        return domain_token, style_tokens


class DualBranchPromptLearner(nn.Module):
    """Construct the positive class-domain and negative domain prompts."""

    def __init__(
        self,
        classnames: list[str],
        clip_model: nn.Module,
        num_context_tokens: int = 4,
        num_style_tokens: int = 4,
    ) -> None:
        super().__init__()
        self.num_classes = len(classnames)
        self.num_context_tokens = num_context_tokens
        self.num_dynamic_tokens = 1 + num_style_tokens
        self.dtype = clip_model.dtype
        context_dim = clip_model.ln_final.weight.shape[0]

        self.statistics_projector = MultiLayerStatisticsProjector(
            visual_width=clip_model.visual.conv1.out_channels,
            embed_dim=context_dim,
            num_layers=len(clip_model.visual.transformer.resblocks),
            num_style_tokens=num_style_tokens,
        )
        self.positive_context = nn.Parameter(torch.empty(num_context_tokens, context_dim))
        self.negative_context = nn.Parameter(torch.empty(num_context_tokens, context_dim))
        nn.init.normal_(self.positive_context, std=0.02)
        nn.init.normal_(self.negative_context, std=0.02)

        classnames = [name.replace("_", " ") for name in classnames]
        placeholder_count = self.num_dynamic_tokens + num_context_tokens
        placeholder_prefix = " ".join(["X"] * placeholder_count)
        positive_text = [f"{placeholder_prefix} {name}." for name in classnames]
        negative_text = [placeholder_prefix]
        positive_tokens = torch.cat([clip.tokenize(text) for text in positive_text])
        negative_tokens = torch.cat([clip.tokenize(text) for text in negative_text])

        embedding_device = clip_model.token_embedding.weight.device
        positive_tokens = positive_tokens.to(embedding_device)
        negative_tokens = negative_tokens.to(embedding_device)
        with torch.no_grad():
            positive_embedding = clip_model.token_embedding(positive_tokens).type(self.dtype)
            negative_embedding = clip_model.token_embedding(negative_tokens).type(self.dtype)

        prompt_length = 1 + placeholder_count
        self.register_buffer("positive_prefix", positive_embedding[:, :1])
        self.register_buffer("positive_suffix", positive_embedding[:, prompt_length:])
        self.register_buffer("negative_prefix", negative_embedding[:, :1])
        self.register_buffer("negative_suffix", negative_embedding[:, prompt_length:])
        self.register_buffer("positive_tokens", positive_tokens)
        self.register_buffer("negative_tokens", negative_tokens)

    def forward(self, layer_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        domain_token, style_tokens = self.statistics_projector(layer_states)
        batch_size = domain_token.shape[0]
        dynamic_tokens = torch.cat([domain_token, style_tokens], dim=1).to(self.dtype)

        positive_context = self.positive_context.to(self.dtype).view(
            1, 1, self.num_context_tokens, -1
        )
        positive_context = positive_context.expand(batch_size, self.num_classes, -1, -1)
        positive_dynamic = dynamic_tokens.unsqueeze(1).expand(-1, self.num_classes, -1, -1)
        positive = torch.cat(
            [
                self.positive_prefix.unsqueeze(0).expand(batch_size, -1, -1, -1),
                positive_dynamic,
                positive_context,
                self.positive_suffix.unsqueeze(0).expand(batch_size, -1, -1, -1),
            ],
            dim=2,
        )

        negative_context = self.negative_context.to(self.dtype).unsqueeze(0).expand(
            batch_size, -1, -1
        )
        negative = torch.cat(
            [
                self.negative_prefix.expand(batch_size, -1, -1),
                dynamic_tokens,
                negative_context,
                self.negative_suffix.expand(batch_size, -1, -1),
            ],
            dim=1,
        )
        return positive, negative


class VisualResidualUpsampler(nn.Module):
    def __init__(self, embed_dim: int = 512) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, kernel_size=7, stride=3, padding=1, output_padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=7, stride=3, padding=1, output_padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=7, stride=3, padding=1, output_padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=7, stride=3, padding=1, output_padding=2),
        )

    def forward(self, residual: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
        signal = self.layers(residual.unsqueeze(-1).unsqueeze(-1))
        return F.interpolate(signal, size=image_size, mode="bilinear", align_corners=False)


class VisualFeatureEnhancer(nn.Module):
    """Inject a text residual while preserving the input at initialization."""

    def __init__(self, embed_dim: int = 512) -> None:
        super().__init__()
        self.upsampler = VisualResidualUpsampler(embed_dim)
        self.projector = nn.Conv2d(4, 3, kernel_size=1)
        with torch.no_grad():
            self.projector.weight.zero_()
            self.projector.bias.zero_()
            for channel in range(3):
                self.projector.weight[channel, channel, 0, 0] = 1.0

    def forward(self, image: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        signal = self.upsampler(residual.float(), image.shape[-2:])
        return self.projector(torch.cat([image.float(), signal], dim=1))


class CustomCLIP(nn.Module):
    """Frozen CLIP backbone with trainable adaptive prompts and enhancement."""

    def __init__(
        self,
        classnames: list[str],
        clip_model: nn.Module,
        visual_class_chunk_size: int = 1,
    ) -> None:
        super().__init__()
        self.num_classes = len(classnames)
        self.visual_class_chunk_size = max(1, int(visual_class_chunk_size))
        self.dtype = clip_model.dtype
        self.image_encoder = clip_model.visual
        self.text_encoder = FrozenTextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.prompt_learner = DualBranchPromptLearner(classnames, clip_model)
        self.feature_enhancer = VisualFeatureEnhancer(clip_model.text_projection.shape[1])

        for parameter in self.image_encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.text_encoder.parameters():
            parameter.requires_grad_(False)
        self.logit_scale.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.image_encoder.eval()
        self.text_encoder.eval()
        return self

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        prefixes = ("prompt_learner.", "feature_enhancer.")
        return {
            key: value.detach().cpu()
            for key, value in self.state_dict().items()
            if key.startswith(prefixes)
        }

    def load_trainable_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        incompatible = self.load_state_dict(state_dict, strict=False)
        unexpected = [key for key in incompatible.unexpected_keys if not key.startswith("logit_scale")]
        if unexpected:
            raise RuntimeError(f"Unexpected Stage-I keys: {unexpected}")

    def _encode_prompts(
        self, positive: torch.Tensor, negative: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_classes = positive.shape[:2]
        positive_flat = positive.flatten(0, 1)
        positive_tokens = self.prompt_learner.positive_tokens.repeat(batch_size, 1)
        positive_features = self.text_encoder(positive_flat, positive_tokens)
        positive_features = l2_normalize(positive_features).view(batch_size, num_classes, -1)

        negative_tokens = self.prompt_learner.negative_tokens.repeat(batch_size, 1)
        negative_features = l2_normalize(self.text_encoder(negative, negative_tokens))
        return positive_features, negative_features

    def _encode_enhanced_visual(
        self, image: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        enhanced_image = self.feature_enhancer(image, residual)
        visual_features, _ = self.image_encoder(enhanced_image.type(self.dtype))
        return l2_normalize(visual_features)

    def _encode_candidate_visuals(
        self, image: torch.Tensor, residuals: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate every candidate class without using a target label.

        Appendix D.3 requires class-wise inference. Chunking controls memory but
        does not change the candidate-conditioned computation.
        """
        batch_size, num_classes, feature_dim = residuals.shape
        outputs = []
        for start in range(0, num_classes, self.visual_class_chunk_size):
            end = min(start + self.visual_class_chunk_size, num_classes)
            chunk_size = end - start
            chunk_residuals = residuals[:, start:end].reshape(
                batch_size * chunk_size, feature_dim
            )
            chunk_images = image.unsqueeze(1).expand(
                -1, chunk_size, -1, -1, -1
            ).reshape(batch_size * chunk_size, *image.shape[1:])
            chunk_features = self._encode_enhanced_visual(
                chunk_images, chunk_residuals
            ).view(batch_size, chunk_size, -1)
            outputs.append(chunk_features)
        return torch.cat(outputs, dim=1)

    def extract_features(
        self, image: torch.Tensor, labels: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        # The first pass extracts detached multi-layer statistics from frozen CLIP.
        with torch.no_grad():
            _, layer_states = self.image_encoder(image.type(self.dtype))
        positive_prompts, negative_prompts = self.prompt_learner(layer_states)
        text_features, domain_features = self._encode_prompts(
            positive_prompts, negative_prompts
        )
        residuals = l2_normalize(text_features - domain_features.unsqueeze(1))

        if labels is None:
            visual_features = self._encode_candidate_visuals(image, residuals)
        else:
            selected_residual = residuals[
                torch.arange(labels.shape[0], device=labels.device), labels
            ]
            visual_features = self._encode_enhanced_visual(image, selected_residual)
        label_text_features = None
        if labels is not None:
            label_text_features = text_features[
                torch.arange(labels.shape[0], device=labels.device), labels
            ]

        return {
            "visual": visual_features,
            "text": text_features,
            "label_text": label_text_features,
            "domain_text": domain_features,
            "semantic_residual": residuals,
        }

    def forward(
        self, image: torch.Tensor, labels: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        features = self.extract_features(image, labels)
        scale = self.logit_scale.exp().clamp(max=100.0)
        if features["visual"].ndim == 3:
            logits = scale * torch.einsum(
                "bcd,bcd->bc", features["visual"], features["text"]
            )
        else:
            logits = scale * torch.einsum(
                "bd,bcd->bc", features["visual"], features["text"]
            )
        return logits, features


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
