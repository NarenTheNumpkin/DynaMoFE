"""Dynamic Gating Router and DynaMoFE Detector Architecture (v2 Reliability-Supervised).

Routes across heterogeneous forensic experts based on transmission degradation signatures:
- Semantic Component Guidance (FCG)
- Spatiotemporal Thumbnail Layout (TALL)
- Frequency-Domain Decomposition (F3Net)
- Spatial Boundary / Texture Baseline (Xception)
"""

from __future__ import annotations

import math
from typing import Any, Sequence
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class DynamicGatingRouter(nn.Module):
    """Predicts condition-dependent expert risk and mixture weights across M forensic experts.

    Conditioned on the K-dimensional video degradation signature d:
        \hat{R}_m(d) = E[l(s_m(X), y) | D(X) = d]
        w_{dyn, m}(d) propto exp(log(w_{base, m}) - \hat{R}_m(d) / tau)
    anchored around base synergy priors w_0 in Delta^{M-1}, with optional
    confidence-gated fallback to the static multi-domain prior based on routing entropy:
        alpha(d) = clamp(gamma * (1 - H(w_dyn) / log(M)), 0, 1)
        w(X) = (1 - alpha(X)) * w_base + alpha(X) * w_dyn(X).
    """

    def __init__(
        self,
        in_dim: int = 16,
        num_experts: int = 4,
        hidden_dim: int = 64,
        temperature: float = 1.0,
        dropout: float = 0.1,
        base_weights: Sequence[float] | None = None,
        use_confidence: bool = False,
        confidence_gain: float = 2.0,
        routing_mode: str = "residual",
        delta_scale: float = 0.5,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.num_experts = num_experts
        self.temperature = temperature
        self.use_confidence = use_confidence
        self.confidence_gain = confidence_gain
        self.routing_mode = routing_mode
        self.delta_scale = delta_scale

        if base_weights is not None:
            base_tensor = torch.tensor(base_weights, dtype=torch.float32)
            base_logits = torch.log(base_tensor + 1e-8)
            base_weights_tensor = base_tensor
        else:
            base_weights_tensor = torch.full((num_experts,), 1.0 / num_experts, dtype=torch.float32)
            base_logits = torch.zeros(num_experts, dtype=torch.float32)

        self.register_buffer("base_weights", base_weights_tensor.view(1, num_experts))
        self.register_buffer("base_logits", base_logits.view(1, num_experts))

        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
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

    def predict_risk(self, deg_features: Tensor) -> Tensor:
        """Predict relative condition-dependent risk R_m(d) for each expert: (B, M)."""
        raw_out = self.net(deg_features)
        if self.routing_mode == "residual":
            return -raw_out * self.delta_scale
        return raw_out

    def compute_weights(
        self, deg_features: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute final mixture weights, dynamic weights, and predicted relative risk.

        Returns:
            weights: (B, M) final blended weights.
            w_dyn: (B, M) dynamic weights from predicted inverse risk.
            pred_risk: (B, M) predicted relative risk.
        """
        raw_out = self.net(deg_features)

        if self.routing_mode == "residual":
            delta = raw_out * self.delta_scale
            logits = self.base_logits + delta
            w_dyn = F.softmax(logits, dim=-1)
            pred_risk = -delta
            weights = w_dyn
        else:
            pred_risk = raw_out
            logits = self.base_logits - pred_risk / self.temperature
            w_dyn = F.softmax(logits, dim=-1)

            if self.use_confidence and self.num_experts > 1:
                h_max = math.log(float(self.num_experts))
                h = -(w_dyn * torch.log(w_dyn + 1e-8)).sum(dim=-1, keepdim=True)
                alpha = torch.clamp(self.confidence_gain * (1.0 - h / h_max), 0.0, 1.0)
                weights = (1.0 - alpha) * self.base_weights + alpha * w_dyn
            else:
                weights = w_dyn

        return weights, w_dyn, pred_risk

    def forward(self, deg_features: Tensor) -> Tensor:
        """Compute routing weights w in Delta^{M-1}."""
        weights, _, _ = self.compute_weights(deg_features)
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
        return_risk: bool = False,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        """Aggregate expert scores using degradation-aware dynamic weights.

        Args:
            deg_features: (B, in_dim) degradation vectors.
            expert_scores: (B, num_experts) raw expert decision margins.
            return_risk: whether to return predicted risk tensor.

        Returns:
            fused_score: (B,) final authenticity decision margin.
            weights: (B, num_experts) dynamic weights assigned to each expert.
            pred_risk: (B, num_experts) predicted risk (if return_risk=True).
        """
        weights, _, pred_risk = self.router.compute_weights(deg_features)
        std_scores = self.standardize(expert_scores)
        fused_score = (weights * std_scores).sum(dim=-1)
        if return_risk:
            return fused_score, weights, pred_risk
        return fused_score, weights
