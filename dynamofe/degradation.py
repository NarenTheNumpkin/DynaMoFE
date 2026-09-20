"""Degradation Signature Extractor for DynaMoFE.

Extracts deterministic, physical signal-degradation statistics from video frames
without relying on semantic labels:
1. 2D Spectral energy roll-off (quantization and blur detection)
2. Block boundary discontinuity (8-pixel periodic boundary artifact indicator associated with block-based compression)
3. Inter-frame temporal difference statistics (mean, standard deviation, and peak)
4. Total variation and spatial gradient energy
5. Dynamic range, contrast, and pixel clipping statistics
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F


class DegradationSignatureExtractor:
    """Extracts a K-dimensional degradation signature from video frames."""

    FEATURE_NAMES = [
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

    DIMENSION = len(FEATURE_NAMES)  # 16 dimensions

    def __init__(self, size: int = 128) -> None:
        self.size = size

    @torch.no_grad()
    def extract_from_numpy(self, frames: np.ndarray) -> np.ndarray:
        """Extract degradation vector from a numpy array of shape (T, H, W, C) in [0, 255]."""
        tensor = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float() / 255.0
        return self.extract_from_tensor(tensor).cpu().numpy()

    @torch.no_grad()
    def extract_from_tensor(self, frames: Tensor) -> Tensor:
        """Extract degradation vector from a float Tensor (T, C, H, W) in [0, 1].

        Returns:
            Tensor of shape (DIMENSION,)
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
        else:
            lum = scaled[:, 0:1]
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

        # 2. Block Boundary Discontinuity (8-pixel periodic boundary indicator capturing block-transform boundary artifacts)
        grid = 8
        boundary_cols = torch.arange(grid - 1, S - 1, grid, device=device)
        internal_cols = torch.arange(grid // 2 - 1, S - 1, grid, device=device)

        diff_h = torch.abs(lum[:, :, :, 1:] - lum[:, :, :, :-1])  # (T, 1, S, S-1)
        diff_v = torch.abs(lum[:, :, 1:, :] - lum[:, :, :-1, :])  # (T, 1, S-1, S)

        b_jump_h = diff_h[:, :, :, boundary_cols].mean()
        i_jump_h = diff_h[:, :, :, internal_cols].mean() + 1e-7
        blockiness_h = (b_jump_h / i_jump_h).clamp(0.0, 10.0)

        b_jump_v = diff_v[:, :, boundary_cols, :].mean()
        i_jump_v = diff_v[:, :, internal_cols, :].mean() + 1e-7
        blockiness_v = (b_jump_v / i_jump_v).clamp(0.0, 10.0)
        blockiness_mean = 0.5 * (blockiness_h + blockiness_v)

        # 3. Inter-Frame Temporal Difference Statistics (mean, standard deviation, peak)
        if T > 1:
            frame_diffs = torch.abs(scaled[1:] - scaled[:-1]).mean(dim=(1, 2, 3))  # (T-1,)
            temp_mean = frame_diffs.mean()
            temp_std = frame_diffs.std() if len(frame_diffs) > 1 else torch.tensor(0.0, device=device)
            temp_max = frame_diffs.max()
        else:
            temp_mean = torch.tensor(0.0, device=device)
            temp_std = torch.tensor(0.0, device=device)
            temp_max = torch.tensor(0.0, device=device)

        # 4. Total Variation & Spatial Gradient
        tv_norm = (diff_h.mean() + diff_v.mean())
        grad_mag = torch.sqrt(diff_h[:, :, :-1, :] ** 2 + diff_v[:, :, :, :-1] ** 2 + 1e-12)
        grad_mean = grad_mag.mean()
        grad_std = grad_mag.std()

        # 5. Luminance Dynamic Range & Clipping
        lum_mean = lum.mean()
        lum_std = lum.std()
        clip_low = (lum <= (3.0 / 255.0)).float().mean()
        clip_high = (lum >= (252.0 / 255.0)).float().mean()

        features = torch.stack([
            hf_ratio,
            mf_ratio,
            decay_slope,
            blockiness_h,
            blockiness_v,
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
        ]).float()

        return features
