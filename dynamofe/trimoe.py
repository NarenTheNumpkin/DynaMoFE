"""Official TriMoE (Three-Domain Mixture-of-Experts) model implementation.

Directly adapted from official TriMoE repository (Anurag Dutta et al., CVPRW 2024 / IEEE Xplore).
Reference: https://github.com/Anurag-Dutta/TriMoE
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MesoNet(nn.Module):
    """MesoNet-4 architecture for facial deepfake detection."""
    def __init__(self, in_channels: int = 3, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 8, 5, padding=2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 5, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 16, 5, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16, 16),
            nn.LeakyReLU(0.1, True),
            nn.Dropout(0.5),
            nn.Linear(16, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class MobileNetExpert(nn.Module):
    """MobileNetV2 expert backbone."""
    def __init__(self, in_channels: int = 3, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        weights = models.MobileNet_V2_Weights.DEFAULT if (pretrained and in_channels == 3) else None
        base = models.mobilenet_v2(weights=weights)
        if in_channels != 3:
            c = base.features[0][0]
            base.features[0][0] = nn.Conv2d(
                in_channels, c.out_channels, c.kernel_size, c.stride, c.padding, bias=False
            )
        base.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(base.last_channel, num_classes)
        )
        self.net = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SqueezeNetExpert(nn.Module):
    """SqueezeNet 1.1 expert backbone."""
    def __init__(self, in_channels: int = 3, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        weights = models.SqueezeNet1_1_Weights.DEFAULT if (pretrained and in_channels == 3) else None
        base = models.squeezenet1_1(weights=weights)
        if in_channels != 3:
            c = base.features[0]
            base.features[0] = nn.Conv2d(
                in_channels, c.out_channels, c.kernel_size, c.stride
            )
        base.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        base.num_classes = num_classes
        self.net = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResNetExpert(nn.Module):
    """ResNet-18 expert backbone."""
    def __init__(self, in_channels: int = 3, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if (pretrained and in_channels == 3) else None
        base = models.resnet18(weights=weights)
        if in_channels != 3:
            c = base.conv1
            base.conv1 = nn.Conv2d(
                in_channels, c.out_channels, c.kernel_size, c.stride, c.padding, bias=False
            )
        base.fc = nn.Linear(base.fc.in_features, num_classes)
        self.net = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalExpert(nn.Module):
    """Recurrent neural network temporal expert on frame moment statistics."""
    def __init__(self, rnn_type: str = "gru", hidden: int = 128, num_classes: int = 2):
        super().__init__()
        self.bn = nn.BatchNorm1d(4)
        bidir = (rnn_type == "bilstm")
        rnn_h = hidden // 2 if bidir else hidden
        if rnn_type == "gru":
            self.rnn = nn.GRU(4, rnn_h, batch_first=True)
        else:
            self.rnn = nn.LSTM(4, rnn_h, batch_first=True, bidirectional=bidir)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, F, D = x.shape
        x = self.bn(x.reshape(B * F, D)).reshape(B, F, D)
        o, _ = self.rnn(x)
        return self.fc(o[:, -1, :])


class TopKRouter(nn.Module):
    """Two-layer MLP router with top-k sparse softmax gating."""
    def __init__(self, input_dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.top_k = min(top_k, num_experts)
        self.lin1 = nn.Linear(input_dim, 64)
        self.lin2 = nn.Linear(64, num_experts)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        a = self.lin2(F.relu(self.lin1(h)))
        topk_vals, topk_idx = torch.topk(a, self.top_k, dim=-1)
        mask = torch.full_like(a, float("-inf"))
        mask.scatter_(1, topk_idx, topk_vals)
        return torch.softmax(mask, dim=-1)


def build_spatial_experts(num_classes: int = 2, pretrained: bool = True) -> nn.ModuleList:
    return nn.ModuleList([
        MesoNet(3, num_classes),
        MobileNetExpert(3, num_classes, pretrained=pretrained),
        SqueezeNetExpert(3, num_classes, pretrained=pretrained),
        ResNetExpert(3, num_classes, pretrained=pretrained),
    ])


def build_spectral_experts(num_classes: int = 2) -> nn.ModuleList:
    return nn.ModuleList([
        MesoNet(1, num_classes),
        MobileNetExpert(1, num_classes, pretrained=False),
        SqueezeNetExpert(1, num_classes, pretrained=False),
        ResNetExpert(1, num_classes, pretrained=False),
    ])


def build_temporal_experts(num_classes: int = 2) -> nn.ModuleList:
    return nn.ModuleList([
        TemporalExpert("gru", 128, num_classes),
        TemporalExpert("lstm", 128, num_classes),
        TemporalExpert("bilstm", 64, num_classes),
    ])


class TRIMOE(nn.Module):
    """Complete Three-Domain Mixture-of-Experts Architecture."""
    def __init__(self, num_classes: int = 2, top_k: int = 2, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.top_k = top_k
        N_s = 4
        N_t = 3

        self.spatial_experts = build_spatial_experts(num_classes, pretrained=pretrained)
        self.spectral_experts = build_spectral_experts(num_classes)
        self.temporal_experts = build_temporal_experts(num_classes)

        self.spatial_router = TopKRouter(num_classes, N_s, top_k)
        self.spectral_router = TopKRouter(num_classes, N_s, top_k)
        self.temporal_router = TopKRouter(num_classes, N_t, top_k)

        # Initial parameter transfer from spatial to spectral
        self.transfer_spatial_to_spectral()

    def _cnn_pool(self, experts: nn.ModuleList, stream: torch.Tensor) -> List[torch.Tensor]:
        B, F = stream.shape[:2]
        x = stream.reshape(B * F, *stream.shape[2:])
        outputs = []
        for e in experts:
            logits = e(x).reshape(B, F, -1).mean(1)
            outputs.append(logits)
        return outputs

    def forward(
        self,
        spatial: torch.Tensor,
        spectral: torch.Tensor,
        temporal: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Spatial domain
        zs = torch.stack(self._cnn_pool(self.spatial_experts, spatial), 1)  # [B, 4, 2]
        gs = self.spatial_router(zs.mean(1))  # [B, 4]
        z_s = (gs.unsqueeze(-1) * zs).sum(1)  # [B, 2]

        # Spectral domain
        zf = torch.stack(self._cnn_pool(self.spectral_experts, spectral), 1)  # [B, 4, 2]
        gf = self.spectral_router(zf.mean(1))  # [B, 4]
        z_f = (gf.unsqueeze(-1) * zf).sum(1)  # [B, 2]

        # Temporal domain
        zt = torch.stack([e(temporal) for e in self.temporal_experts], 1)  # [B, 3, 2]
        gt = self.temporal_router(zt.mean(1))  # [B, 3]
        z_t = (gt.unsqueeze(-1) * zt).sum(1)  # [B, 2]

        # Discrete decision: Majority voting
        y_s = z_s.argmax(-1)
        y_f = z_f.argmax(-1)
        y_t = z_t.argmax(-1)
        y_hat = (torch.stack([y_s, y_f, y_t], 1).float().mean(1) >= 0.5).long()

        # Continuous probability: average of domain softmax fake probabilities
        p_s = torch.softmax(z_s, dim=-1)[:, 1]
        p_f = torch.softmax(z_f, dim=-1)[:, 1]
        p_t = torch.softmax(z_t, dim=-1)[:, 1]
        p_avg = (p_s + p_f + p_t) / 3.0

        details = {
            "spatial": z_s,
            "spectral": z_f,
            "temporal": z_t,
            "g_spatial": gs,
            "g_spectral": gf,
            "g_temporal": gt,
            "p_spatial": p_s,
            "p_spectral": p_f,
            "p_temporal": p_t,
            "prob": p_avg,
        }
        return y_hat, details

    def transfer_spatial_to_spectral(self) -> None:
        """Channel-collapse parameter transfer from spatial to spectral experts."""
        for sp, spec in zip(self.spatial_experts, self.spectral_experts):
            sp_sd = sp.state_dict()
            spec_sd = spec.state_dict()
            first_k = None
            for k, v in sp_sd.items():
                if v.ndim == 4 and v.shape[1] == 3:
                    first_k = k
                    break
            for k in spec_sd:
                if k == first_k and first_k in sp_sd:
                    spec_sd[k] = sp_sd[k].mean(dim=1, keepdim=True)
                elif k in sp_sd and sp_sd[k].shape == spec_sd[k].shape:
                    spec_sd[k] = sp_sd[k].clone()
            spec.load_state_dict(spec_sd)


class TRIMOELoss(nn.Module):
    """TriMoE total loss: cross-entropy over domain logits plus auxiliary load-balancing loss."""
    def __init__(self, lambda_aux: float = 0.01):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.lambda_aux = lambda_aux

    def _load_balance(self, gates: torch.Tensor) -> torch.Tensor:
        N = gates.shape[1]
        load = gates.mean(dim=0)
        return ((load - 1.0 / N) ** 2).sum()

    def forward(self, logits: Dict[str, torch.Tensor], y: torch.Tensor) -> torch.Tensor:
        L_ce = (
            self.ce(logits["spatial"], y) +
            self.ce(logits["spectral"], y) +
            self.ce(logits["temporal"], y)
        ) / 3.0
        L_aux = (
            self._load_balance(logits["g_spatial"]) +
            self._load_balance(logits["g_spectral"]) +
            self._load_balance(logits["g_temporal"])
        )
        return L_ce + self.lambda_aux * L_aux


def extract_multidomain_inputs(
    frames_rgb_uint8: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute spatial, spectral, and temporal streams on GPU from uint8 RGB frames [F, H, W, 3] or [B, F, H, W, 3]."""
    if frames_rgb_uint8.ndim == 4:
        # Single video: [F, H, W, 3]
        frames_float = frames_rgb_uint8.float() / 255.0
        spatial = frames_float.permute(0, 3, 1, 2)  # [F, 3, H, W]

        grey = spatial.mean(dim=1)  # [F, H, W]
        fft_s = torch.fft.fftshift(torch.fft.fft2(grey), dim=(-2, -1))
        mag = torch.log1p(torch.abs(fft_s))
        m_min = mag.amin(dim=(-2, -1), keepdim=True)
        m_max = mag.amax(dim=(-2, -1), keepdim=True)
        spectral = ((mag - m_min) / (m_max - m_min + 1e-8)).unsqueeze(1)  # [F, 1, H, W]

        arr = spatial.reshape(spatial.shape[0], -1)  # [F, 3*H*W]
        mean = arr.mean(dim=1)
        diff = arr - mean.unsqueeze(1)
        var = (diff ** 2).mean(dim=1)
        std = torch.sqrt(var).clamp(min=1e-8)
        skew = (diff ** 3).mean(dim=1) / (std ** 3)
        kurt = (diff ** 4).mean(dim=1) / (std ** 4)
        temporal = torch.stack([mean, var, skew, kurt], dim=1)  # [F, 4]
        return spatial, spectral, temporal
    elif frames_rgb_uint8.ndim == 5:
        # Batched video: [B, F, H, W, 3]
        B, F, H, W, C = frames_rgb_uint8.shape
        frames_float = frames_rgb_uint8.float() / 255.0
        spatial = frames_float.permute(0, 1, 4, 2, 3)  # [B, F, 3, H, W]

        grey = spatial.mean(dim=2)  # [B, F, H, W]
        fft_s = torch.fft.fftshift(torch.fft.fft2(grey), dim=(-2, -1))
        mag = torch.log1p(torch.abs(fft_s))
        m_min = mag.amin(dim=(-2, -1), keepdim=True)
        m_max = mag.amax(dim=(-2, -1), keepdim=True)
        spectral = ((mag - m_min) / (m_max - m_min + 1e-8)).unsqueeze(2)  # [B, F, 1, H, W]

        arr = spatial.reshape(B, F, -1)
        mean = arr.mean(dim=2)
        diff = arr - mean.unsqueeze(2)
        var = (diff ** 2).mean(dim=2)
        std = torch.sqrt(var).clamp(min=1e-8)
        skew = (diff ** 3).mean(dim=2) / (std ** 3)
        kurt = (diff ** 4).mean(dim=2) / (std ** 4)
        temporal = torch.stack([mean, var, skew, kurt], dim=-1)  # [B, F, 4]
        return spatial, spectral, temporal
    else:
        raise ValueError(f"Expected 4D or 5D tensor, got shape {frames_rgb_uint8.shape}")

