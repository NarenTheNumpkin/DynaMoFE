"""Dynamic Gating Router and DynaMoFE Detector Architecture.

Routes across heterogeneous forensic experts based on transmission degradation signatures:
- Semantic Component Guidance (FCG)
- Spatiotemporal Thumbnail Layout (TALL)
- Frequency-Domain Decomposition (F3Net)
- Spatial Boundary / Texture Baseline (Xception)
"""

from __future__ import annotations

from typing import Any, Sequence
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class DynamicGatingRouter(nn.Module):
    """Predicts instance-level mixture weights across M forensic experts.

    Conditioned on the K-dimensional video degradation signature.
    """

    def __init__(
        self,
        in_dim: int = 16,
        num_experts: int = 4,
        hidden_dim: int = 64,
        temperature: float = 1.0,
        dropout: float = 0.1,
        base_weights: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.num_experts = num_experts
        self.temperature = temperature

        if base_weights is not None:
            base_tensor = torch.tensor(base_weights, dtype=torch.float32)
            base_logits = torch.log(base_tensor + 1e-8)
        else:
            base_logits = torch.zeros(num_experts, dtype=torch.float32)
        self.register_buffer("base_logits", base_logits.view(1, num_experts))

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_experts),
        )

        # Initialize output to zero so initial output matches base_weights exactly
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, deg_features: Tensor) -> Tensor:
        """Compute routing weights w in Delta^{M-1}."""
        delta = self.net(deg_features)
        logits = self.base_logits + delta
        weights = F.softmax(logits / self.temperature, dim=-1)
        return weights


class DynaMoFEDetector(nn.Module):
    """DynaMoFE: Dynamic Mixture of Forensic Experts.

    Combines standardized expert decision margins using dynamically predicted routing weights.
    """

    def __init__(
        self,
        router: DynamicGatingRouter,
        expert_names: Sequence[str] = ("fcg", "tall", "f3net", "xception"),
        means: Sequence[float] | None = None,
        stds: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        self.router = router
        self.expert_names = list(expert_names)
        num_experts = len(self.expert_names)

        if means is None:
            means = [0.0] * num_experts
        if stds is None:
            stds = [1.0] * num_experts

        self.register_buffer("means", torch.tensor(means, dtype=torch.float32).view(1, num_experts))
        self.register_buffer("stds", torch.tensor(stds, dtype=torch.float32).view(1, num_experts))

    def standardize(self, scores: Tensor) -> Tensor:
        """Standardize raw expert margins: (B, M)."""
        return (scores - self.means) / (self.stds + 1e-7)

    def forward(
        self,
        deg_features: Tensor,
        expert_scores: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Aggregate expert scores using degradation-aware dynamic weights.

        Args:
            deg_features: (B, in_dim) degradation vectors.
            expert_scores: (B, num_experts) raw expert decision margins.

        Returns:
            fused_score: (B,) final authenticity decision margin.
            weights: (B, num_experts) dynamic weights assigned to each expert.
        """
        weights = self.router(deg_features)  # (B, M)
        std_scores = self.standardize(expert_scores)  # (B, M)
        fused_score = (weights * std_scores).sum(dim=-1)  # (B,)
        return fused_score, weights
