from __future__ import annotations

from typing import Any

import torch
from torch import nn

from parameterization import ParameterModelBase


class DirectParameterModel(ParameterModelBase):
    def __init__(self, ny: int, nx: int, **kwargs: Any) -> None:
        super().__init__(ny, nx, **kwargs)

        self.raw_bottom_field = self._make_raw_field_parameter(self.bottom_mode, self.n_bottom_field, (ny, nx))
        self.raw_K_field = self._make_raw_field_parameter(self.k_mode, self.n_k_params)
        self.raw_A_field = self._make_raw_field_parameter(self.a_mode, self.n_a_params)

    def _make_raw_field_parameter(self, mode: str, n_params: int, full_shape: tuple[int, ...] | None = None) -> nn.Parameter | None:
        if mode == "fixed":
            return None
        shape = full_shape if mode == "full" and full_shape is not None else (n_params,)
        return nn.Parameter(torch.zeros(shape, dtype=torch.float64))

    def forward(self, features: torch.Tensor | None = None) -> dict[str, Any]:
        return {
            "bottom_coeff": self._scaled_coeff(self.bottom_mode, self.raw_bottom_field, self.alpha_scale, "bottom"),
            "K_coeff": self._scaled_coeff(self.k_mode, self.raw_K_field, self.k_property_scale, "K"),
            "A_coeff": self._scaled_coeff(self.a_mode, self.raw_A_field, self.a_property_scale, "A"),
        }
