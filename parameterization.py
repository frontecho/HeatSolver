from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy.fft import idct
from torch import nn

from lithosphere_model import THERMAL_LAYER_NAMES

PARAMETER_MODES = ("fixed", "dct", "full")


class DCTField:
    def __init__(self, ny: int, nx: int, x: int = 5, y: int = 5):
        self.ny = ny
        self.nx = nx
        self.x = min(x, nx)
        self.y = min(y, ny)
        if self.x < 1 or self.y < 1:
            raise ValueError(f"Invalid DCT shape for grid {ny}x{nx}: x={x}, y={y}")
        self.n_params = self.x * self.y

        basis_y = np.zeros((ny, self.y), dtype=np.float64)
        basis_x = np.zeros((nx, self.x), dtype=np.float64)
        eye_y = np.eye(ny, dtype=np.float64)
        eye_x = np.eye(nx, dtype=np.float64)
        for i in range(self.y):
            basis_y[:, i] = idct(eye_y[:, i], norm="ortho")
        for j in range(self.x):
            basis_x[:, j] = idct(eye_x[:, j], norm="ortho")

        self._basis_y_np = basis_y
        self._basis_x_np = basis_x

    def decode_torch(self, coeff: torch.Tensor, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        coeff = coeff.to(dtype=dtype, device=device).reshape(self.y, self.x)
        basis_y = torch.as_tensor(self._basis_y_np, dtype=dtype, device=device)
        basis_x = torch.as_tensor(self._basis_x_np, dtype=dtype, device=device)
        return basis_y @ coeff @ basis_x.transpose(0, 1)


class LayeredDCTField:
    def __init__(self, n_layers: int, ny: int, nx: int, x: int, y: int):
        self.n_layers = n_layers
        self.dct = DCTField(ny, nx, x, y)
        self.x = self.dct.x
        self.y = self.dct.y
        self.n_params_per_layer = self.dct.n_params
        self.n_params = n_layers * self.n_params_per_layer

    def decode_torch(self, coeffs: torch.Tensor, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        coeffs = coeffs.to(dtype=dtype, device=device).reshape(self.n_layers, self.n_params_per_layer)
        return torch.stack([self.dct.decode_torch(coeffs[i], dtype=dtype, device=device) for i in range(self.n_layers)])


class ParameterModelBase(nn.Module):
    def __init__(
        self,
        ny: int,
        nx: int,
        *,
        bottom_x: int = 5,
        bottom_y: int = 5,
        k_property_x: int = 5,
        k_property_y: int = 5,
        a_property_x: int = 5,
        a_property_y: int = 5,
        k_property_scale: float = 0.7,
        a_property_scale: float = 0.7,
        alpha_scale: float = 300.0,
        bottom_mode: str = "dct",
        k_mode: str = "fixed",
        a_mode: str = "fixed",
    ) -> None:
        super().__init__()
        self.ny = ny
        self.nx = nx
        self.bottom_mode = bottom_mode.lower()
        self.k_mode = k_mode.lower()
        self.a_mode = a_mode.lower()
        self.k_property_scale = k_property_scale
        self.a_property_scale = a_property_scale
        self.alpha_scale = alpha_scale

        self.register_buffer("_zero", torch.zeros(0, dtype=torch.float64))
        self.n_property_layers = len(THERMAL_LAYER_NAMES)
        self.bottom_dct = LayeredDCTField(1, ny, nx, bottom_x, bottom_y) if self.bottom_mode == "dct" else None
        self.k_dct = LayeredDCTField(self.n_property_layers, ny, nx, k_property_x, k_property_y) if self.k_mode == "dct" else None
        self.a_dct = LayeredDCTField(self.n_property_layers, ny, nx, a_property_x, a_property_y) if self.a_mode == "dct" else None
        self.n_bottom_field = self._field_param_count(self.bottom_mode, self.bottom_dct, 1, "bottom")
        self.n_k_field = self._field_param_count(self.k_mode, self.k_dct, self.n_property_layers, "K")
        self.n_a_field = self._field_param_count(self.a_mode, self.a_dct, self.n_property_layers, "A")
        self.n_k_mean = self.n_property_layers if self.k_mode != "fixed" else 0
        self.n_a_mean = self.n_property_layers if self.a_mode != "fixed" else 0
        self.n_k_params = self.n_k_mean + self.n_k_field
        self.n_a_params = self.n_a_mean + self.n_a_field

    def _field_param_count(self, mode: str, dct: LayeredDCTField | None, n_layers: int, name: str) -> int:
        if mode == "fixed":
            return 0
        if mode == "full":
            return n_layers * self.ny * self.nx
        if mode == "dct":
            if dct is None:
                raise ValueError(f"DCT {name} mode is missing its DCT field.")
            return dct.n_params
        raise ValueError(f"Unsupported {name} mode: {mode}")

    def _scaled_coeff(self, mode: str, raw: torch.Tensor | None, scale: float, name: str) -> torch.Tensor:
        if mode in ("dct", "full"):
            if raw is None:
                raise ValueError(f"{name} {mode} mode is missing coefficients.")
            return scale * torch.tanh(raw)
        if raw is None:
            return torch.zeros(0, dtype=self._zero.dtype, device=self._zero.device)
        return torch.zeros(0, dtype=raw.dtype, device=raw.device)

    def _decode_coeff(
        self,
        mode: str,
        dct: LayeredDCTField | None,
        coeff: torch.Tensor,
        fixed_shape: tuple[int, ...],
        full_shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if mode == "fixed":
            return torch.zeros(fixed_shape, dtype=dtype, device=device)
        coeff = coeff.to(dtype=dtype, device=device)
        if mode == "full":
            return coeff.reshape(full_shape)
        if dct is None:
            raise ValueError("DCT mode is missing its DCT field.")
        return dct.decode_torch(coeff, dtype=dtype, device=device)

    def _decode_property_log_xy(
        self,
        mode: str,
        dct: LayeredDCTField | None,
        coeff: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        zero_shape = (self.n_property_layers, self.ny, self.nx)
        if mode == "fixed":
            return torch.zeros(zero_shape, dtype=dtype, device=device)
        coeff = coeff.to(dtype=dtype, device=device)
        mean = coeff[: self.n_property_layers]
        field_coeff = coeff[self.n_property_layers :]
        field_xy = self._decode_coeff(
            mode,
            dct,
            field_coeff,
            zero_shape,
            zero_shape,
            dtype=dtype,
            device=device,
        )
        return mean.reshape(self.n_property_layers, 1, 1) + field_xy

    def apply_log_multiplier_torch(
        self,
        solver: Any,
        reference: np.ndarray,
        mode: str,
        *,
        log_xy: torch.Tensor | None = None,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mode = mode.lower()
        missing = [name for name in THERMAL_LAYER_NAMES if name not in solver.layer_masks]
        if mode != "fixed" and missing:
            raise ValueError(f"Solver is missing layer masks required for parameterization: {missing}")
        field = torch.as_tensor(reference, dtype=dtype, device=device).clone()
        zero_xy = torch.zeros((len(THERMAL_LAYER_NAMES), solver.ny, solver.nx), dtype=dtype, device=device)
        if mode == "fixed":
            return field, zero_xy
        if log_xy is None:
            raise ValueError(f"{mode} property mode requires per-layer 2-D log multipliers.")
        log_xy = log_xy.to(dtype=dtype, device=device)
        multiplier = torch.zeros_like(field)
        for i, name in enumerate(THERMAL_LAYER_NAMES):
            weight = torch.as_tensor(solver.layer_masks[name], dtype=dtype, device=device)
            if bool(torch.any(weight > 0.0)):
                scale = torch.exp(log_xy[i][:, :, None])
                multiplier = multiplier + weight * scale
        return field * multiplier, log_xy

    def decode_configured_parameter_fields(
        self,
        solver: Any,
        coeffs: dict[str, Any],
        *,
        T_ref: float,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        T_bot_delta_layers = self._decode_coeff(self.bottom_mode, self.bottom_dct, coeffs["bottom_coeff"], (1, self.ny, self.nx), (1, self.ny, self.nx), dtype=dtype, device=device)
        K_log_xy = self._decode_property_log_xy(self.k_mode, self.k_dct, coeffs["K_coeff"], dtype=dtype, device=device)
        A_log_xy = self._decode_property_log_xy(self.a_mode, self.a_dct, coeffs["A_coeff"], dtype=dtype, device=device)
        T_bot_delta = T_bot_delta_layers[0]
        T_bot = torch.full((solver.ny, solver.nx), float(T_ref), dtype=dtype, device=device) if self.bottom_mode == "fixed" else float(T_ref) + T_bot_delta

        K0, _ = self.apply_log_multiplier_torch(
            solver,
            solver.K0_ref,
            self.k_mode,
            log_xy=K_log_xy,
            dtype=dtype,
            device=device,
        )
        A, _ = self.apply_log_multiplier_torch(
            solver,
            solver.A_ref,
            self.a_mode,
            log_xy=A_log_xy,
            dtype=dtype,
            device=device,
        )
        extras = {
            "T_bot_delta": T_bot_delta,
            "K_log_xy": K_log_xy,
            "A_log_xy": A_log_xy,
        }
        return T_bot, K0, A, extras
