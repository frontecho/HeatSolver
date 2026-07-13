from __future__ import annotations

"""Low-dimensional direct parameterization for geothermal inversion."""

from typing import Any

import numpy as np
from scipy.fft import idct

from lithosphere_model import THERMAL_LAYER_NAMES

PARAMETER_MODES = ("fixed", "dct", "full")


class DCTBasis:
    def __init__(self, ny: int, nx: int, modes_y: int, modes_x: int) -> None:
        self.ny = ny
        self.nx = nx
        self.modes_y = min(modes_y, ny)
        self.modes_x = min(modes_x, nx)
        if self.modes_y < 1 or self.modes_x < 1:
            raise ValueError("DCT mode counts must be positive")
        self.size = self.modes_y * self.modes_x
        self.basis_y = idct(
            np.eye(ny, dtype=np.float64)[:, : self.modes_y],
            axis=0,
            norm="ortho",
        )
        self.basis_x = idct(
            np.eye(nx, dtype=np.float64)[:, : self.modes_x],
            axis=0,
            norm="ortho",
        )

    def decode(self, coefficients: np.ndarray) -> np.ndarray:
        matrix = np.asarray(coefficients).reshape(self.modes_y, self.modes_x)
        return self.basis_y @ matrix @ self.basis_x.T

    def decode_layers(self, coefficients: np.ndarray) -> np.ndarray:
        matrices = np.asarray(coefficients).reshape(-1, self.modes_y, self.modes_x)
        return np.einsum(
            "ay,lyx,bx->lab",
            self.basis_y,
            matrices,
            self.basis_x,
            optimize=True,
        )

    def adjoint(self, field: np.ndarray) -> np.ndarray:
        return (self.basis_y.T @ field @ self.basis_x).ravel()

    def adjoint_layers(self, fields: np.ndarray) -> np.ndarray:
        return np.einsum(
            "ay,lab,bx->lyx",
            self.basis_y,
            fields,
            self.basis_x,
            optimize=True,
        ).ravel()


class DirectParameterization:
    def __init__(
        self,
        ny: int,
        nx: int,
        *,
        bottom_mode: str = "dct",
        k_mode: str = "fixed",
        a_mode: str = "fixed",
        bottom_dct_x: int = 3,
        bottom_dct_y: int = 3,
        k_dct_x: int = 5,
        k_dct_y: int = 5,
        a_dct_x: int = 5,
        a_dct_y: int = 5,
        bottom_bound_C: float = 800.0,
        k_log_bound: float = 0.7,
        a_log_bound: float = 0.7,
    ) -> None:
        self.ny = ny
        self.nx = nx
        self.n_layers = len(THERMAL_LAYER_NAMES)
        self.bottom_mode = bottom_mode
        self.k_mode = k_mode
        self.a_mode = a_mode
        for mode in (bottom_mode, k_mode, a_mode):
            if mode not in PARAMETER_MODES:
                raise ValueError(f"unknown parameter mode: {mode}")

        self.bottom_dct = DCTBasis(ny, nx, bottom_dct_y, bottom_dct_x) if bottom_mode == "dct" else None
        self.k_dct = DCTBasis(ny, nx, k_dct_y, k_dct_x) if k_mode == "dct" else None
        self.a_dct = DCTBasis(ny, nx, a_dct_y, a_dct_x) if a_mode == "dct" else None
        self.bottom_bound_C = bottom_bound_C
        self.k_log_bound = k_log_bound
        self.a_log_bound = a_log_bound

        offset = 0
        self.bottom_slice, offset = self._allocate(offset, self._bottom_size())
        self.k_slice, offset = self._allocate(offset, self._property_size(k_mode, self.k_dct))
        self.a_slice, offset = self._allocate(offset, self._property_size(a_mode, self.a_dct))
        self.size = offset

    @staticmethod
    def _allocate(offset: int, size: int) -> tuple[slice, int]:
        return slice(offset, offset + size), offset + size

    def _bottom_size(self) -> int:
        if self.bottom_mode == "fixed":
            return 0
        if self.bottom_mode == "full":
            return self.ny * self.nx
        return self.bottom_dct.size

    def _property_size(self, mode: str, basis: DCTBasis | None) -> int:
        if mode == "fixed":
            return 0
        field_size = self.ny * self.nx if mode == "full" else basis.size
        return self.n_layers * (1 + field_size)

    def initial(self) -> np.ndarray:
        return np.zeros(self.size, dtype=np.float64)

    def bounds(self) -> list[tuple[float, float]]:
        result: list[tuple[float, float]] = []
        result.extend([(-self.bottom_bound_C, self.bottom_bound_C)] * (self.bottom_slice.stop - self.bottom_slice.start))
        result.extend([(-self.k_log_bound, self.k_log_bound)] * (self.k_slice.stop - self.k_slice.start))
        result.extend([(-self.a_log_bound, self.a_log_bound)] * (self.a_slice.stop - self.a_slice.start))
        return result

    def _decode_bottom(self, coefficients: np.ndarray) -> np.ndarray:
        if self.bottom_mode == "fixed":
            return np.zeros((self.ny, self.nx), dtype=np.float64)
        if self.bottom_mode == "full":
            return coefficients.reshape(self.ny, self.nx)
        return self.bottom_dct.decode(coefficients)

    def _decode_property(
        self,
        mode: str,
        basis: DCTBasis | None,
        coefficients: np.ndarray,
    ) -> np.ndarray:
        if mode == "fixed":
            return np.zeros((self.n_layers, self.ny, self.nx), dtype=np.float64)
        means = coefficients[: self.n_layers, None, None]
        field_coefficients = coefficients[self.n_layers :]
        if mode == "full":
            fields = field_coefficients.reshape(self.n_layers, self.ny, self.nx)
        else:
            fields = basis.decode_layers(field_coefficients)
        return means + fields

    def _apply_property(
        self,
        solver: Any,
        reference: np.ndarray,
        mode: str,
        log_xy: np.ndarray,
    ) -> np.ndarray:
        if mode == "fixed":
            return reference.copy()
        missing = [name for name in THERMAL_LAYER_NAMES if name not in solver.layer_masks]
        if missing:
            raise ValueError(f"missing layer masks: {missing}")
        multiplier = np.zeros_like(reference)
        for index, name in enumerate(THERMAL_LAYER_NAMES):
            multiplier += solver.layer_masks[name] * np.exp(log_xy[index, :, :, None])
        return reference * multiplier

    def decode(self, solver: Any, parameters: np.ndarray, *, T_ref: float) -> dict[str, np.ndarray]:
        parameters = np.asarray(parameters, dtype=np.float64)
        if parameters.shape != (self.size,):
            raise ValueError(f"parameters must have shape {(self.size,)}")
        bottom_coeff = parameters[self.bottom_slice]
        k_coeff = parameters[self.k_slice]
        a_coeff = parameters[self.a_slice]
        T_bot_delta = self._decode_bottom(bottom_coeff)
        K_log_xy = self._decode_property(self.k_mode, self.k_dct, k_coeff)
        A_log_xy = self._decode_property(self.a_mode, self.a_dct, a_coeff)
        T_bot = np.full((self.ny, self.nx), T_ref) + T_bot_delta
        return {
            "T_bot": T_bot,
            "K0": self._apply_property(solver, solver.K0_ref, self.k_mode, K_log_xy),
            "A": self._apply_property(solver, solver.A_ref, self.a_mode, A_log_xy),
            "T_bot_delta": T_bot_delta,
            "K_log_xy": K_log_xy,
            "A_log_xy": A_log_xy,
            "bottom_coeff": bottom_coeff,
            "K_coeff": k_coeff,
            "A_coeff": a_coeff,
        }

    def _property_pullback(
        self,
        solver: Any,
        reference: np.ndarray,
        field_gradient: np.ndarray,
        mode: str,
        basis: DCTBasis | None,
        log_xy: np.ndarray,
    ) -> np.ndarray:
        if mode == "fixed":
            return np.zeros(0, dtype=np.float64)
        grad_log = np.empty_like(log_xy)
        for index, name in enumerate(THERMAL_LAYER_NAMES):
            derivative = (
                reference
                * solver.layer_masks[name]
                * np.exp(log_xy[index, :, :, None])
            )
            grad_log[index] = np.sum(field_gradient * derivative, axis=2)
        mean_gradient = np.sum(grad_log, axis=(1, 2))
        if mode == "full":
            field_coeff_gradient = grad_log.ravel()
        else:
            field_coeff_gradient = basis.adjoint_layers(grad_log)
        return np.concatenate([mean_gradient, field_coeff_gradient])

    def pullback(
        self,
        solver: Any,
        state: dict[str, np.ndarray],
        grad_T_bot: np.ndarray,
        grad_K0: np.ndarray,
        grad_A: np.ndarray,
    ) -> np.ndarray:
        gradient = np.zeros(self.size, dtype=np.float64)
        if self.bottom_mode == "full":
            gradient[self.bottom_slice] = grad_T_bot.ravel()
        elif self.bottom_mode == "dct":
            gradient[self.bottom_slice] = self.bottom_dct.adjoint(grad_T_bot)
        gradient[self.k_slice] = self._property_pullback(
            solver,
            solver.K0_ref,
            grad_K0,
            self.k_mode,
            self.k_dct,
            state["K_log_xy"],
        )
        gradient[self.a_slice] = self._property_pullback(
            solver,
            solver.A_ref,
            grad_A,
            self.a_mode,
            self.a_dct,
            state["A_log_xy"],
        )
        return gradient

    def config(self) -> dict[str, int | float | str]:
        return {
            "ny": self.ny,
            "nx": self.nx,
            "bottom_mode": self.bottom_mode,
            "k_mode": self.k_mode,
            "a_mode": self.a_mode,
            "bottom_dct_x": 0 if self.bottom_dct is None else self.bottom_dct.modes_x,
            "bottom_dct_y": 0 if self.bottom_dct is None else self.bottom_dct.modes_y,
            "k_dct_x": 0 if self.k_dct is None else self.k_dct.modes_x,
            "k_dct_y": 0 if self.k_dct is None else self.k_dct.modes_y,
            "a_dct_x": 0 if self.a_dct is None else self.a_dct.modes_x,
            "a_dct_y": 0 if self.a_dct is None else self.a_dct.modes_y,
            "bottom_bound_C": self.bottom_bound_C,
            "k_log_bound": self.k_log_bound,
            "a_log_bound": self.a_log_bound,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DirectParameterization:
        return cls(**config)
