"""Discrete cosine spatial basis and its transpose operators."""
from __future__ import annotations

import numpy as np
from scipy.fft import idct


class DCTBasis:
    def __init__(self, ny: int, nx: int, modes_y: int, modes_x: int) -> None:
        self.ny = ny
        self.nx = nx
        self.modes_y = min(modes_y, ny)
        self.modes_x = min(modes_x, nx)
        if self.modes_y < 1 or self.modes_x < 1:
            raise ValueError("DCT mode counts must be positive")
        self.size = self.modes_y * self.modes_x
        self.scale = np.sqrt(ny * nx)
        self.basis_y = idct(
            np.eye(ny, self.modes_y, dtype=np.float64),
            axis=0,
            norm="ortho",
        )
        self.basis_x = idct(
            np.eye(nx, self.modes_x, dtype=np.float64),
            axis=0,
            norm="ortho",
        )
        spatial_envelope = self.scale * np.multiply.outer(
            np.sum(np.abs(self.basis_y), axis=1),
            np.sum(np.abs(self.basis_x), axis=1),
        ) - 1.0
        self.spatial_bound_scale = (
            0.0 if self.size == 1 else 1.0 / np.max(spatial_envelope)
        )

    def decode(self, coefficients: np.ndarray) -> np.ndarray:
        matrix = np.asarray(coefficients).reshape(self.modes_y, self.modes_x)
        return self.scale * (self.basis_y @ matrix @ self.basis_x.T)

    def decode_layers(self, coefficients: np.ndarray) -> np.ndarray:
        matrices = np.asarray(coefficients).reshape(-1, self.modes_y, self.modes_x)
        return self.scale * np.einsum(
            "ay,lyx,bx->lab",
            self.basis_y,
            matrices,
            self.basis_x,
            optimize=True,
        )

    def adjoint(self, field: np.ndarray) -> np.ndarray:
        return (self.scale * self.basis_y.T @ field @ self.basis_x).ravel()

    def adjoint_layers(self, fields: np.ndarray) -> np.ndarray:
        return self.scale * np.einsum(
            "ay,lab,bx->lyx",
            self.basis_y,
            fields,
            self.basis_x,
            optimize=True,
        ).ravel()
