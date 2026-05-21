from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from data_loading import load_prior_id_features, load_prior_mt_features, load_prior_seismic_features
from lithosphere_model import LayerGrid
from parameterization import ParameterModelBase


def _normalize_channels(stack: np.ndarray) -> np.ndarray:
    means = np.nanmean(stack, axis=(1, 2), keepdims=True)
    filled = np.where(np.isfinite(stack), stack, means)
    stds = np.std(filled, axis=(1, 2), keepdims=True)
    stds = np.where(stds > 1e-6, stds, 1.0)
    return (filled - means) / stds


def build_prior_feature_pack(
    data_dir: Path,
    grid: LayerGrid,
    stride: int,
) -> tuple[torch.Tensor, list[str]]:
    arrays, channel_names = load_prior_id_features(data_dir, stride)
    seis_arrays, seis_names = load_prior_seismic_features(data_dir, grid)
    arrays.extend(seis_arrays)
    channel_names.extend(seis_names)
    mt_arrays, mt_names = load_prior_mt_features(data_dir, grid)
    arrays.extend(mt_arrays)
    channel_names.extend(mt_names)
    stack = _normalize_channels(np.stack(arrays, axis=0))
    return torch.as_tensor(stack[None, ...], dtype=torch.float64), channel_names


class MultiSourcePriorNet(nn.Module):
    def __init__(self, in_channels: int, theta_dim: int, width: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=2, ceil_mode=True),
            nn.Conv2d(width, 2 * width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(2 * width, 2 * width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * width, 4 * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * width, theta_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class PriorParameterModel(ParameterModelBase):
    def __init__(self, ny: int, nx: int, in_channels: int, width: int = 32, **kwargs: Any) -> None:
        super().__init__(ny, nx, **kwargs)
        self.uses_k_property_mode = self.k_mode in ("dct", "full")
        self.uses_a_property_mode = self.a_mode in ("dct", "full")
        self.theta_dim = self.n_bottom_field
        if self.uses_k_property_mode:
            self.theta_dim += self.n_k_field
        if self.uses_a_property_mode:
            self.theta_dim += self.n_a_field
        self.prior_net = MultiSourcePriorNet(in_channels, self.theta_dim, width=width)

    def forward(self, features: torch.Tensor | None = None) -> dict[str, torch.Tensor | None]:
        if features is None:
            raise ValueError("PriorParameterModel requires feature tensor input.")
        theta_raw = self.prior_net(features)[0]
        cursor = 0
        bottom_raw = theta_raw[cursor : cursor + self.n_bottom_field]
        cursor += self.n_bottom_field
        K_raw = None
        A_raw = None

        if self.uses_k_property_mode:
            K_raw = theta_raw[cursor : cursor + self.n_k_field]
            cursor += self.n_k_field
        if self.uses_a_property_mode:
            A_raw = theta_raw[cursor : cursor + self.n_a_field]
        return {
            "bottom_coeff": self._scaled_coeff(self.bottom_mode, bottom_raw, self.alpha_scale, "bottom"),
            "K_coeff": self._scaled_coeff(self.k_mode, K_raw, self.k_property_scale, "K"),
            "A_coeff": self._scaled_coeff(self.a_mode, A_raw, self.a_property_scale, "A"),
        }
