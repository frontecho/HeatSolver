from __future__ import annotations

import numpy as np

from heat_solver import HeatSolver
from experiments.model import THERMAL_LAYER_NAMES


def make_layer_masks(ny: int, nx: int, nz: int) -> dict[str, np.ndarray]:
    masks = {name: np.zeros((ny, nx, nz), dtype=bool) for name in THERMAL_LAYER_NAMES}
    splits = np.array_split(np.arange(nz), len(THERMAL_LAYER_NAMES))
    for name, idx in zip(THERMAL_LAYER_NAMES, splits):
        masks[name][:, :, idx] = True
    return masks


def make_solver(
    ny: int = 3,
    nx: int = 3,
    nz: int = 5,
    *,
    bottom_temp: float = 600.0,
    surface_temp: float = 10.0,
    K0: float = 2.5,
    c: float = 0.0,
    A: float = 0.0,
    max_picard: int = 4,
    picard_tol: float = 1e-10,
    cg_rtol: float = 1e-10,
) -> tuple[HeatSolver, np.ndarray]:
    x = np.broadcast_to(np.arange(nx, dtype=float)[None, :] * 1_000.0, (ny, nx)).copy()
    y = np.broadcast_to(np.arange(ny, dtype=float)[:, None] * 1_000.0, (ny, nx)).copy()
    z_nodes_km = np.linspace(0.0, 4.0, nz)
    surface = np.full((ny, nx), surface_temp, dtype=float)
    K0_ref = np.full((ny, nx, nz), K0, dtype=float)
    c_ref = np.full((ny, nx, nz), c, dtype=float)
    A_ref = np.full((ny, nx, nz), A, dtype=float)
    solver = HeatSolver(
        x=x,
        y=y,
        z_nodes_km=z_nodes_km,
        surface_temp=surface,
        K0_ref=K0_ref,
        c_ref=c_ref,
        A_ref=A_ref,
        layer_masks=make_layer_masks(ny, nx, nz),
        max_picard=max_picard,
        picard_tol=picard_tol,
        cg_rtol=cg_rtol,
    )
    return solver, np.full((ny, nx), bottom_temp, dtype=float)
