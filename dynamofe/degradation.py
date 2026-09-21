"""Degradation Signature Extractor for DynaMoFE.

Extracts deterministic, physical signal-degradation statistics from video frames
without relying on semantic labels:
1. 2D Spectral energy roll-off (quantization and blur detection)
2. Multi-scale block boundary discontinuity (4-px, 8-px, and 16-px periodic boundary artifacts)
3. Inter-frame temporal dynamics and 2nd-order temporal acceleration
4. Total variation, Laplacian blur variance, and spatial gradient distribution
5. 8x8 block DCT transform energy concentration (AC/DC ratio, HF ratio, zero fraction, entropy)
6. Dynamic range, contrast, pixel clipping, chroma attenuation, and resampling autocorrelation.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F


class DegradationSignatureExtractor:
    """Extracts a K-dimensional degradation signature from video frames."""

    FEATURE_NAMES_16D = [
        "hf_spectral_ratio",       # Ratio of high-frequency to total power
        "mf_spectral_ratio",       # Ratio of mid-frequency to total power
        "spectral_decay_slope",    # Endpoint log-frequency power spectral decay rate across 4 rings
        "blockiness_ratio_h",      # Horizontal 8-pixel periodic block boundary discontinuity
        "blockiness_ratio_v",      # Vertical 8-pixel periodic block boundary discontinuity
        "blockiness_mean",         # Average 8-pixel periodic block boundary discontinuity
        "temporal_diff_mean",      # Mean inter-frame difference (motion magnitude)
        "temporal_diff_std",       # Inter-frame difference standard deviation
        "temporal_diff_max",       # Peak inter-frame change
        "spatial_tv_norm",         # Total variation (spatial edge energy)
        "grad_magnitude_mean",     # Mean spatial gradient
        "grad_magnitude_std",      # Gradient distribution spread
        "luminance_mean",          # Mean luminance
        "luminance_std",           # Contrast / dynamic spread
        "clip_fraction_low",       # Fraction of crushed shadows (<= 3/255)
        "clip_fraction_high",      # Fraction of blown highlights (>= 252/255)
    ]

    FEATURE_NAMES_28D = FEATURE_NAMES_16D + [
        "laplacian_variance",      # High-pass blur / focus indicator
        "edge_density",            # Fraction of sharp edge transitions
        "edge_spread",             # Standard deviation of gradient magnitudes on edges
        "blockiness_4px",          # 4-pixel sub-block boundary discontinuity
        "blockiness_16px",         # 16-pixel macroblock boundary discontinuity
        "dct_ac_dc_ratio",         # 8x8 DCT AC-to-DC power ratio
        "dct_hf_ratio",            # Fraction of AC power in highest spatial frequencies
        "dct_zero_frac",           # Fraction of near-zero quantized high-frequency DCT coefficients
        "dct_entropy",             # Shannon entropy of the 2D DCT coefficient power spectrum
        "chroma_attenuation_ratio",# Ratio of chrominance TV to luminance TV
        "resampling_indicator",    # 2nd-order horizontal difference autocorrelation (periodic resampling)
        "temporal_accel_std",      # Standard deviation of 2nd-order inter-frame acceleration
    ]

    # Default to modern 28D signature
    FEATURE_NAMES = FEATURE_NAMES_28D
    DIMENSION = len(FEATURE_NAMES_28D)  # 28 dimensions

    def __init__(self, size: int = 128, version: str = "v2_28d") -> None:
        self.size = size
        self.version = version
        if version in ("v2_28d", "28", "28d"):
            self.feature_names = self.FEATURE_NAMES_28D
            self.dimension = 28
        elif version in ("v1_16d", "16", "16d"):
            self.feature_names = self.FEATURE_NAMES_16D
            self.dimension = 16
        else:
            raise ValueError(f"Unknown degradation version: {version}. Expected 'v2_28d' or 'v1_16d'.")

    @torch.no_grad()
    def extract_from_numpy(self, frames: np.ndarray) -> np.ndarray:
        """Extract degradation vector from a numpy array of shape (T, H, W, C) in [0, 255]."""
        tensor = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float() / 255.0
        return self.extract_from_tensor(tensor).cpu().numpy()

    @torch.no_grad()
    def extract_from_tensor(self, frames: Tensor) -> Tensor:
        """Extract degradation vector from a float Tensor (T, C, H, W) in [0, 1].

        Returns:
            Tensor of shape (dimension,)
        """
        T, C, H, W = frames.shape
        device = frames.device

        # Resize to standard analysis size for consistent multi-resolution scale
        if (H, W) != (self.size, self.size):
            scaled = F.interpolate(frames, (self.size, self.size), mode="bilinear", align_corners=False)
        else:
            scaled = frames

        # Compute luminance channel: Y = 0.299 R + 0.587 G + 0.114 B
        if C == 3:
            lum = 0.299 * scaled[:, 0:1] + 0.587 * scaled[:, 1:2] + 0.114 * scaled[:, 2:3]
            u_ch = scaled[:, 2:3] - lum  # B - Y
            v_ch = scaled[:, 0:1] - lum  # R - Y
        else:
            lum = scaled[:, 0:1]
            u_ch = torch.zeros_like(lum)
            v_ch = torch.zeros_like(lum)
            
        S = self.size

        # 1. 2D Spectral Energy Roll-off via 2D FFT
        fft_vals = torch.fft.fft2(lum, dim=(-2, -1))
        fft_shift = torch.fft.fftshift(fft_vals, dim=(-2, -1))
        power = (fft_shift.abs() ** 2).mean(dim=(0, 1))  # (S, S) average over frames

        # Create radial frequency grid centered at (S/2, S/2)
        center = S // 2
        y_idx, x_idx = torch.meshgrid(
            torch.arange(S, device=device) - center,
            torch.arange(S, device=device) - center,
            indexing="ij",
        )
        r = torch.sqrt(x_idx.float() ** 2 + y_idx.float() ** 2) / (center * np.sqrt(2))  # normalized to [0, 1]

        total_power = power.sum() + 1e-12
        mf_mask = (r >= 0.25) & (r < 0.6)
        hf_mask = r >= 0.6

        hf_ratio = (power[hf_mask].sum() / total_power).clamp(0.0, 1.0)
        mf_ratio = (power[mf_mask].sum() / total_power).clamp(0.0, 1.0)

        # Estimate decay slope: endpoint spectral slope across 4 frequency rings
        ring_powers = []
        for r_min, r_max in [(0.1, 0.25), (0.25, 0.45), (0.45, 0.65), (0.65, 0.85)]:
            ring_mask = (r >= r_min) & (r < r_max)
            val = power[ring_mask].mean() + 1e-12
            ring_powers.append(torch.log(val))
        ring_powers = torch.stack(ring_powers)
        # Endpoint spectral decay rate across 3 octave intervals
        decay_slope = (ring_powers[-1] - ring_powers[0]) / 3.0

        # 2. Block Boundary Discontinuity (8-pixel periodic boundary indicator)
        lum_2d = lum.squeeze(1)  # (T, S, S)
        diff_h = (lum_2d[:, :, 1:] - lum_2d[:, :, :-1]).abs()  # (T, S, S-1)
        diff_v = (lum_2d[:, 1:, :] - lum_2d[:, :-1, :]).abs()  # (T, S-1, S)

        block_h_idx = torch.arange(7, S - 1, 8, device=device)
        non_block_h_idx = torch.tensor([i for i in range(S - 1) if (i + 1) % 8 != 0], device=device)
        b_h = diff_h[:, :, block_h_idx].mean() / (diff_h[:, :, non_block_h_idx].mean() + 1e-8)

        block_v_idx = torch.arange(7, S - 1, 8, device=device)
        non_block_v_idx = torch.tensor([i for i in range(S - 1) if (i + 1) % 8 != 0], device=device)
        b_v = diff_v[:, block_v_idx, :].mean() / (diff_v[:, non_block_v_idx, :].mean() + 1e-8)
        blockiness_mean = (b_h + b_v) / 2.0

        # 3. Inter-Frame Temporal Difference Statistics
        if T > 1:
            frame_diffs = (lum[1:] - lum[:-1]).abs().mean(dim=(1, 2, 3))
            temp_mean = frame_diffs.mean()
            temp_std = frame_diffs.std() if len(frame_diffs) > 1 else torch.tensor(0.0, device=device)
            temp_max = frame_diffs.max()
            if T > 2:
                second_diffs = (lum[2:] - 2 * lum[1:-1] + lum[:-2]).abs().mean(dim=(1, 2, 3))
                temp_accel = second_diffs.std()
            else:
                temp_accel = torch.tensor(0.0, device=device)
        else:
            temp_mean = torch.tensor(0.0, device=device)
            temp_std = torch.tensor(0.0, device=device)
            temp_max = torch.tensor(0.0, device=device)
            temp_accel = torch.tensor(0.0, device=device)

        # 4. Total Variation & Spatial Gradient
        tv_norm = diff_h.mean() + diff_v.mean()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3) / 8.0
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3) / 8.0
        gx = F.conv2d(lum, sobel_x, padding=1)
        gy = F.conv2d(lum, sobel_y, padding=1)
        g_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-12)
        grad_mean = g_mag.mean()
        grad_std = g_mag.std()

        # 5. Luminance Dynamic Range & Clipping
        lum_mean = lum.mean()
        lum_std = lum.std()
        clip_low = (lum <= (3.0 / 255.0)).float().mean()
        clip_high = (lum >= (252.0 / 255.0)).float().mean()

        features_16 = [
            hf_ratio,
            mf_ratio,
            decay_slope,
            b_h,
            b_v,
            blockiness_mean,
            temp_mean,
            temp_std,
            temp_max,
            tv_norm,
            grad_mean,
            grad_std,
            lum_mean,
            lum_std,
            clip_low,
            clip_high,
        ]

        if self.dimension == 16:
            return torch.stack(features_16).float()

        # --- Extended 12 Features for 28-D Signature ---
        # 6. Spatial Blur & Edge Structure
        laplacian_k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
        lap = F.conv2d(lum, laplacian_k, padding=1)
        lap_var = lap.var(dim=(-2, -1)).mean()

        edge_density = (g_mag > 0.10).float().mean()
        edge_mask = (g_mag > 0.05)
        edge_spread = g_mag[edge_mask].std() if edge_mask.sum() > 10 else torch.tensor(0.0, device=device)

        # 7. Multi-scale Blockiness (4-px and 16-px)
        b4_h_idx = torch.arange(3, S - 1, 4, device=device)
        non_b4_h_idx = torch.tensor([i for i in range(S - 1) if (i + 1) % 4 != 0], device=device)
        b4_h = diff_h[:, :, b4_h_idx].mean() / (diff_h[:, :, non_b4_h_idx].mean() + 1e-8)
        b4_v_idx = torch.arange(3, S - 1, 4, device=device)
        non_b4_v_idx = torch.tensor([i for i in range(S - 1) if (i + 1) % 4 != 0], device=device)
        b4_v = diff_v[:, b4_v_idx, :].mean() / (diff_v[:, non_b4_v_idx, :].mean() + 1e-8)
        b4_mean = (b4_h + b4_v) / 2.0

        b16_h_idx = torch.arange(15, S - 1, 16, device=device)
        non_b16_h_idx = torch.tensor([i for i in range(S - 1) if (i + 1) % 16 != 0], device=device)
        b16_h = diff_h[:, :, b16_h_idx].mean() / (diff_h[:, :, non_b16_h_idx].mean() + 1e-8)
        b16_v_idx = torch.arange(15, S - 1, 16, device=device)
        non_b16_v_idx = torch.tensor([i for i in range(S - 1) if (i + 1) % 16 != 0], device=device)
        b16_v = diff_v[:, b16_v_idx, :].mean() / (diff_v[:, non_b16_v_idx, :].mean() + 1e-8)
        b16_mean = (b16_h + b16_v) / 2.0

        # 8. 8x8 Block DCT Transform Statistics
        lum_blocks = F.unfold(lum, kernel_size=8, stride=8).transpose(1, 2).contiguous()
        lum_blocks = lum_blocks.view(-1, 8, 8)

        dct_mat = torch.zeros((8, 8), device=device)
        for i in range(8):
            for j in range(8):
                scale = 1.0 / np.sqrt(8) if i == 0 else np.sqrt(2.0 / 8)
                dct_mat[i, j] = scale * np.cos((2 * j + 1) * i * np.pi / 16.0)

        dct_coeff = torch.matmul(torch.matmul(dct_mat, lum_blocks), dct_mat.t())
        dct_power = (dct_coeff ** 2).mean(dim=0)

        dc_power = dct_power[0, 0] + 1e-12
        ac_power = dct_power.sum() - dc_power
        dct_ac_dc_ratio = (ac_power / dc_power).clamp(0.0, 100.0)

        u_idx, v_idx = torch.meshgrid(torch.arange(8, device=device), torch.arange(8, device=device), indexing="ij")
        hf_dct_mask = (u_idx + v_idx >= 7)
        hf_dct_power = dct_power[hf_dct_mask].sum()
        dct_hf_ratio = (hf_dct_power / (ac_power + 1e-12)).clamp(0.0, 1.0)

        hf_coeffs = dct_coeff[:, hf_dct_mask].abs()
        dct_zero_frac = (hf_coeffs < 0.01).float().mean()

        p_dct = (dct_power / (dct_power.sum() + 1e-12)).clamp(min=1e-12)
        dct_entropy = -(p_dct * p_dct.log()).sum()

        # 9. Chroma Attenuation & Resampling Indicator
        tv_u = (u_ch[:, :, :, 1:] - u_ch[:, :, :, :-1]).abs().mean() + (u_ch[:, :, 1:, :] - u_ch[:, :, :-1, :]).abs().mean()
        tv_v = (v_ch[:, :, :, 1:] - v_ch[:, :, :, :-1]).abs().mean() + (v_ch[:, :, 1:, :] - v_ch[:, :, :-1, :]).abs().mean()
        chroma_ratio = (tv_u + tv_v) / (2.0 * tv_norm + 1e-7)

        autocorr_h = (diff_h[:, :, 2:] * diff_h[:, :, :-2]).mean() / (diff_h[:, :, 2:].var() + 1e-8)
        resampling_indicator = autocorr_h.clamp(-1.0, 1.0)

        features_28 = features_16 + [
            lap_var,
            edge_density,
            edge_spread,
            b4_mean,
            b16_mean,
            dct_ac_dc_ratio,
            dct_hf_ratio,
            dct_zero_frac,
            dct_entropy,
            chroma_ratio,
            resampling_indicator,
            temp_accel,
        ]

        return torch.stack(features_28).float()
