from __future__ import annotations

"""Clean baseline 3-D steady-state heat conduction solver.

Model
-----
Solve
    -div( k(x, T) grad(T) ) = A(x)
with
    T = T_surf on the top
    T = T_bot  on the bottom
    dT/dn = 0  on the four side boundaries
"""

import time
from typing import Optional

import numpy as np
import torch
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import pyamg
from torch import nn


# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------

class HeatSolver(nn.Module):
    @staticmethod
    def _conductivity_from_temperature(K0: np.ndarray, c: np.ndarray, T: np.ndarray) -> np.ndarray:
        return K0 / np.maximum(1.0 + c * T, 0.1)

    @staticmethod
    def _harmonic_mean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return 2.0 * a * b / np.maximum(a + b, 1e-30)

    @staticmethod
    def _require_finite(name: str, arr: np.ndarray) -> None:
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains NaN or inf values.")

    def __init__(
        self,
        *,
        x: np.ndarray,
        y: np.ndarray,
        z_nodes_km: np.ndarray,
        surface_temp: np.ndarray,
        K0_ref: np.ndarray,
        c_ref: np.ndarray,
        A_ref: np.ndarray,
        layer_masks: dict[str, np.ndarray] | None = None,
        max_picard: int = 20,
        picard_tol: float = 1e-5,
        cg_rtol: float = 1e-8,
        amg_rebuild_interval: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.z_nodes_km = np.asarray(z_nodes_km, dtype=float)
        self.surface_temp = np.asarray(surface_temp, dtype=float)
        self.K0_ref = np.asarray(K0_ref, dtype=float)
        self.c_ref = np.asarray(c_ref, dtype=float)
        self.A_ref = np.asarray(A_ref, dtype=float)
        self.layer_masks = {} if layer_masks is None else {name: np.asarray(mask, dtype=float) for name, mask in layer_masks.items()}

        if self.surface_temp.ndim != 2:
            raise ValueError("surface_temp must be a 2-D array.")
        self.ny, self.nx = self.surface_temp.shape
        self.nz = self.z_nodes_km.size
        if self.nz < 3:
            raise ValueError("z_nodes_km must contain at least 3 nodes.")
        if self.x.shape != (self.ny, self.nx) or self.y.shape != (self.ny, self.nx):
            raise ValueError("x, y, and surface_temp must have matching 2-D shapes.")
        field_shape = (self.ny, self.nx, self.nz)
        for name, arr in [("K0_ref", self.K0_ref), ("c_ref", self.c_ref), ("A_ref", self.A_ref)]:
            if arr.shape != field_shape:
                raise ValueError(f"{name} must have shape {field_shape}, got {arr.shape}.")
        for name, mask in self.layer_masks.items():
            if mask.shape != field_shape:
                raise ValueError(f"layer mask {name!r} must have shape {field_shape}, got {mask.shape}.")

        dz_vals = np.diff(self.z_nodes_km)
        if not np.all(np.isfinite(self.z_nodes_km)) or not np.all(dz_vals > 0):
            raise ValueError("z_nodes_km must be finite and strictly increasing.")
        if not np.allclose(dz_vals, dz_vals[0]):
            raise ValueError("HeatSolver requires uniform vertical spacing.")
        dx_vals = np.diff(np.sort(np.unique(self.x[np.isfinite(self.x)])))
        dy_vals = np.diff(np.sort(np.unique(self.y[np.isfinite(self.y)])))
        if dx_vals.size == 0 or dy_vals.size == 0:
            raise ValueError("x and y must contain at least two unique finite coordinates.")

        self.dz = float(dz_vals[0] * 1e3)
        self.z_max = float(self.z_nodes_km[-1] * 1e3)
        self.dx = float(np.median(dx_vals))
        self.dy = float(np.median(dy_vals))
        self.max_picard = max_picard
        self.picard_tol = picard_tol
        self.cg_rtol = cg_rtol
        self.amg_rebuild_interval = amg_rebuild_interval

        self._require_finite("x", self.x)
        self._require_finite("y", self.y)
        self._require_finite("surface_temp", self.surface_temp)
        self._require_finite("K0_ref", self.K0_ref)
        self._require_finite("c_ref", self.c_ref)
        self._require_finite("A_ref", self.A_ref)

        self._built = True
        self._solve_calls = 0
        self._last_T: Optional[np.ndarray] = None
        self.last_solve_info: Optional[dict[str, float | int | bool]] = None

    def _solve_sparse_system(
        self,
        A_mat: sp.csr_matrix,
        rhs: np.ndarray,
        x0: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, int]:
        ml = pyamg.smoothed_aggregation_solver(A_mat)
        num_iters = 0

        def _cb(_: np.ndarray) -> None:
            nonlocal num_iters
            num_iters += 1

        sol, info = spla.cg(
            A_mat,
            rhs,
            x0=x0,
            M=ml.aspreconditioner(),
            rtol=self.cg_rtol,
            maxiter=500,
            callback=_cb,
        )
        if info != 0:
            raise RuntimeError(f"CG did not converge, info={info}")
        if num_iters == 0:
            num_iters = 1
        return sol, num_iters

    def clear_warm_start(self) -> None:
        self._last_T = None
        self.last_solve_info = None

    # ------------------------------------------------------------------
    # Matrix assembly
    # ------------------------------------------------------------------
    def _build_matrix_and_rhs(
        self,
        K0: np.ndarray,
        A_source: np.ndarray,
        T_iter: np.ndarray,
        T_bot: np.ndarray,
    ) -> tuple[sp.csr_matrix, np.ndarray]:
        ny, nx, nz = self.ny, self.nx, self.nz
        nz2 = nz - 2
        T_surf = self.surface_temp

        k_node = self._conductivity_from_temperature(K0, self.c_ref, T_iter)
        kfz = self._harmonic_mean(k_node[:, :, :-1], k_node[:, :, 1:])
        kfx = self._harmonic_mean(k_node[:-1, :, :], k_node[1:, :, :])
        kfy = self._harmonic_mean(k_node[:, :-1, :], k_node[:, 1:, :])

        def flat(i: np.ndarray, j: np.ndarray, k2: np.ndarray) -> np.ndarray:
            return i * (nx * nz2) + j * nz2 + k2

        ii = np.arange(ny)[:, None, None] * np.ones((1, nx, nz2), dtype=int)
        jj = np.ones((ny, 1, nz2), dtype=int) * np.arange(nx)[None, :, None]
        k2 = np.ones((ny, nx, 1), dtype=int) * np.arange(nz2)[None, None, :]
        kk = k2 + 1
        row = flat(ii, jj, k2).ravel()

        rows, cols, vals = [], [], []

        # z-
        cf_zm = kfz[:, :, :-1][:, :, :nz2] / (self.dz * self.dz)
        mask = (kk > 1).ravel()
        rows.append(row[mask])
        cols.append(flat(ii.ravel()[mask], jj.ravel()[mask], k2.ravel()[mask] - 1))
        vals.append(-cf_zm.ravel()[mask])

        # z+
        cf_zp = kfz[:, :, 1:][:, :, :nz2] / (self.dz * self.dz)
        mask = (kk < nz - 2).ravel()
        rows.append(row[mask])
        cols.append(flat(ii.ravel()[mask], jj.ravel()[mask], k2.ravel()[mask] + 1))
        vals.append(-cf_zp.ravel()[mask])

        # row axis (i) follows y, so use dy here
        cf_xm = np.zeros((ny, nx, nz2), dtype=float)
        cf_xm[1:, :, :] = kfx[:, :, 1:-1] / (self.dy * self.dy)
        mask = (ii > 0).ravel()
        rows.append(row[mask])
        cols.append(flat(ii.ravel()[mask] - 1, jj.ravel()[mask], k2.ravel()[mask]))
        vals.append(-cf_xm.ravel()[mask])

        # row axis (i) follows y, so use dy here
        cf_xp = np.zeros((ny, nx, nz2), dtype=float)
        cf_xp[:-1, :, :] = kfx[:, :, 1:-1] / (self.dy * self.dy)
        mask = (ii < ny - 1).ravel()
        rows.append(row[mask])
        cols.append(flat(ii.ravel()[mask] + 1, jj.ravel()[mask], k2.ravel()[mask]))
        vals.append(-cf_xp.ravel()[mask])

        # column axis (j) follows x, so use dx here
        cf_ym = np.zeros((ny, nx, nz2), dtype=float)
        cf_ym[:, 1:, :] = kfy[:, :, 1:-1] / (self.dx * self.dx)
        mask = (jj > 0).ravel()
        rows.append(row[mask])
        cols.append(flat(ii.ravel()[mask], jj.ravel()[mask] - 1, k2.ravel()[mask]))
        vals.append(-cf_ym.ravel()[mask])

        # column axis (j) follows x, so use dx here
        cf_yp = np.zeros((ny, nx, nz2), dtype=float)
        cf_yp[:, :-1, :] = kfy[:, :, 1:-1] / (self.dx * self.dx)
        mask = (jj < nx - 1).ravel()
        rows.append(row[mask])
        cols.append(flat(ii.ravel()[mask], jj.ravel()[mask] + 1, k2.ravel()[mask]))
        vals.append(-cf_yp.ravel()[mask])

        diag = (cf_zm + cf_zp + cf_xm + cf_xp + cf_ym + cf_yp).ravel()
        rows.append(row)
        cols.append(row)
        vals.append(diag)

        A_mat = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(ny * nx * nz2, ny * nx * nz2),
        )

        rhs = A_source[:, :, 1:-1].reshape(-1).copy()
        coeff_surf = kfz[:, :, 0] / (self.dz * self.dz)
        coeff_bot = kfz[:, :, -1] / (self.dz * self.dz)

        idx_surf = np.arange(ny * nx) * nz2
        idx_bot = np.arange(ny * nx) * nz2 + (nz2 - 1)
        rhs[idx_surf] += (coeff_surf * T_surf).reshape(-1)
        rhs[idx_bot] += (coeff_bot * T_bot).reshape(-1)

        return A_mat, rhs

    def build_frozen_system(
        self,
        T: np.ndarray,
        T_bot: np.ndarray,
        K0: Optional[np.ndarray] = None,
        A_source: Optional[np.ndarray] = None,
    ) -> tuple[sp.csr_matrix, np.ndarray]:
        if not self._built:
            raise RuntimeError("HeatSolver reference fields were not initialized.")
        K0_use = self.K0_ref if K0 is None else K0
        A_use = self.A_ref if A_source is None else A_source
        self._require_finite("T", T)
        self._require_finite("T_bot", T_bot)
        self._require_finite("K0", K0_use)
        self._require_finite("A_source", A_use)
        return self._build_matrix_and_rhs(K0_use, A_use, T, T_bot)

    # ------------------------------------------------------------------
    # Forward solve
    # ------------------------------------------------------------------
    def solve(
        self,
        T_bot: np.ndarray,
        K0: Optional[np.ndarray] = None,
        A_source: Optional[np.ndarray] = None,
        T_init: Optional[np.ndarray] = None,
        reuse_last: bool = True,
        verbose: bool = False,
    ) -> np.ndarray:
        if not self._built:
            raise RuntimeError("HeatSolver reference fields were not initialized.")

        K0_use = self.K0_ref if K0 is None else K0
        A_use = self.A_ref if A_source is None else A_source
        T_surf = self.surface_temp
        n_unknowns = self.ny * self.nx * (self.nz - 2)
        self._solve_calls += 1
        solve_id = self._solve_calls

        self._require_finite("T_bot", T_bot)
        self._require_finite("T_surf", T_surf)
        self._require_finite("K0", K0_use)
        self._require_finite("A_source", A_use)

        if T_init is not None:
            self._require_finite("T_init", T_init)
            T = np.array(T_init, dtype=float, copy=True)
        elif reuse_last and self._last_T is not None and self._last_T.shape == (self.ny, self.nx, self.nz):
            T = self._last_T.copy()
        else:
            alpha = np.linspace(0.0, 1.0, self.nz)
            T = T_surf[:, :, None] * (1.0 - alpha) + T_bot[:, :, None] * alpha
        T[:, :, 0] = T_surf
        T[:, :, -1] = T_bot
        solve_t0 = time.time()
        converged = False
        ml = None
        cg_iters_total = 0

        if verbose:
            print(
                f"Solve {solve_id} start: "
                f"grid={self.ny}x{self.nx}x{self.nz}, unknowns={n_unknowns}, max_picard={self.max_picard}"
            )
            print(
                f"Solve {solve_id} boundary temperatures: "
                f"Tsurf={np.nanmin(T_surf):.1f}-{np.nanmax(T_surf):.1f} C, "
                f"Tbot={np.nanmin(T_bot):.1f}-{np.nanmax(T_bot):.1f} C"
            )

        for it in range(self.max_picard):
            A_mat, rhs = self._build_matrix_and_rhs(K0_use, A_use, T, T_bot)
            rebuild_amg = (
                ml is None
                or (
                    self.amg_rebuild_interval is not None
                    and self.amg_rebuild_interval > 0
                    and it % self.amg_rebuild_interval == 0
                )
            )
            if rebuild_amg:
                ml = pyamg.smoothed_aggregation_solver(A_mat)
            cg_iters = 0

            def _cb(_: np.ndarray) -> None:
                nonlocal cg_iters
                cg_iters += 1

            u_int, info = spla.cg(
                A_mat,
                rhs,
                x0=T[:, :, 1:-1].reshape(-1),
                M=ml.aspreconditioner(),
                rtol=self.cg_rtol,
                maxiter=500,
                callback=_cb,
            )
            if info != 0:
                raise RuntimeError(f"CG did not converge, info={info}")
            if cg_iters == 0:
                cg_iters = 1
            cg_iters_total += cg_iters

            T_new = np.concatenate(
                [T_surf[:, :, None], u_int.reshape(self.ny, self.nx, self.nz - 2), T_bot[:, :, None]],
                axis=2,
            )
            rel = np.linalg.norm(T_new - T) / (np.linalg.norm(T_new) + 1e-30)
            T = T_new
            if rel < self.picard_tol:
                converged = True
                break

        self._last_T = T.copy()
        self.last_solve_info = {
            "solve_id": solve_id,
            "picard_iters": it + 1,
            "cg_iters_total": cg_iters_total,
            "converged": converged,
            "final_rel": float(rel),
            "total_time_s": float(time.time() - solve_t0),
        }

        if verbose:
            status = "converged" if converged else "reached max_picard"
            if self.amg_rebuild_interval is None:
                amg_msg = "reused AMG after first Picard step"
            elif self.amg_rebuild_interval > 0:
                amg_msg = f"rebuilt AMG every {self.amg_rebuild_interval} Picard step(s)"
            else:
                amg_msg = "reused AMG after first Picard step"
            print(
                f"Solve {solve_id} done: "
                f"{status}, iterations={it + 1}, rel={rel:.3e}, "
                f"cg_iters={cg_iters_total}, total_time={time.time() - solve_t0:.2f}s, "
                f"final_T={T.min():.1f}-{T.max():.1f} C, "
                f"{amg_msg}"
            )

        return T

    # ------------------------------------------------------------------
    # Heat flow
    # ------------------------------------------------------------------
    @staticmethod
    def _surface_temperature_gradient_np(T: np.ndarray, dz: float) -> np.ndarray:
        return (-3.0 * T[:, :, 0] + 4.0 * T[:, :, 1] - T[:, :, 2]) / (2.0 * dz)

    def surface_heat_flow_upward_np(self, T: np.ndarray, K0: Optional[np.ndarray] = None) -> np.ndarray:
        """Return upward surface heat flow in mW/m^2."""
        K0_use = self.K0_ref if K0 is None else K0
        self._require_finite("T", T)
        self._require_finite("K0", K0_use)
        k_top_nodes = self._conductivity_from_temperature(K0_use[:, :, :2], self.c_ref[:, :, :2], T[:, :, :2])
        k_top_face = self._harmonic_mean(k_top_nodes[:, :, 0], k_top_nodes[:, :, 1])
        return k_top_face * self._surface_temperature_gradient_np(T, self.dz) * 1e3

    def forward(
        self,
        T_bot: torch.Tensor,
        K0: Optional[torch.Tensor] = None,
        A_source: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._built:
            raise RuntimeError("HeatSolver reference fields were not initialized.")
        K0_use = K0
        A_use = A_source
        if K0_use is None:
            K0_use = torch.as_tensor(self.K0_ref, dtype=T_bot.dtype, device=T_bot.device)
        else:
            K0_use = K0_use.to(device=T_bot.device, dtype=T_bot.dtype)
        if A_use is None:
            A_use = torch.as_tensor(self.A_ref, dtype=T_bot.dtype, device=T_bot.device)
        else:
            A_use = A_use.to(device=T_bot.device, dtype=T_bot.dtype)
        return _HeatSolveFunction.apply(T_bot, K0_use, A_use, self)


class _HeatSolveFunction(torch.autograd.Function):
    @staticmethod
    def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy().astype(np.float64, copy=False)

    @staticmethod
    def _harmonic_mean_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return 2.0 * a * b / torch.clamp(a + b, min=1e-30)

    @staticmethod
    def _conductivity_from_temperature_np(K0: np.ndarray, c: np.ndarray, T: np.ndarray) -> np.ndarray:
        return K0 / np.maximum(1.0 + c * T, 0.1)

    @staticmethod
    def _conductivity_from_temperature_torch(
        K0: torch.Tensor,
        c: torch.Tensor,
        T: torch.Tensor,
    ) -> torch.Tensor:
        return K0 / torch.clamp(1.0 + c * T, min=0.1)

    @staticmethod
    def _surface_temperature_gradient_torch(T: torch.Tensor, dz: float) -> torch.Tensor:
        return (-3.0 * T[:, :, 0] + 4.0 * T[:, :, 1] - T[:, :, 2]) / (2.0 * dz)

    @staticmethod
    def _surface_heat_flow_upward_torch(
        solver: HeatSolver,
        T: torch.Tensor,
        K0: torch.Tensor,
        c_ref: torch.Tensor,
    ) -> torch.Tensor:
        k_top_nodes = _HeatSolveFunction._conductivity_from_temperature_torch(K0[:, :, :2], c_ref[:, :, :2], T[:, :, :2])
        k_top_face = _HeatSolveFunction._harmonic_mean_torch(k_top_nodes[:, :, 0], k_top_nodes[:, :, 1])
        return k_top_face * _HeatSolveFunction._surface_temperature_gradient_torch(T, solver.dz) * 1e3

    @staticmethod
    def _frozen_residual_torch(
        solver: HeatSolver,
        T: torch.Tensor,
        T_bot: torch.Tensor,
        K0: torch.Tensor,
        A_source: torch.Tensor,
        c_ref: torch.Tensor,
        T_surf: torch.Tensor,
    ) -> torch.Tensor:
        dz2 = solver.dz * solver.dz
        dx2 = solver.dx * solver.dx
        dy2 = solver.dy * solver.dy

        T_full = torch.cat([T_surf[:, :, None], T[:, :, 1:-1], T_bot[:, :, None]], dim=2)
        T_int = T_full[:, :, 1:-1]
        k_node = _HeatSolveFunction._conductivity_from_temperature_torch(K0, c_ref, T_full)
        kfz = _HeatSolveFunction._harmonic_mean_torch(k_node[:, :, :-1], k_node[:, :, 1:])
        kfx = _HeatSolveFunction._harmonic_mean_torch(k_node[:-1, :, :], k_node[1:, :, :])
        kfy = _HeatSolveFunction._harmonic_mean_torch(k_node[:, :-1, :], k_node[:, 1:, :])

        cf_zm = kfz[:, :, :-1] / dz2
        cf_zp = kfz[:, :, 1:] / dz2
        residual = (cf_zm + cf_zp) * T_int
        residual[:, :, 1:] = residual[:, :, 1:] - cf_zm[:, :, 1:] * T_int[:, :, :-1]
        residual[:, :, :-1] = residual[:, :, :-1] - cf_zp[:, :, :-1] * T_int[:, :, 1:]

        coeff_row = kfx[:, :, 1:-1] / dy2
        residual[:-1, :, :] = residual[:-1, :, :] + coeff_row * (T_int[:-1, :, :] - T_int[1:, :, :])
        residual[1:, :, :] = residual[1:, :, :] + coeff_row * (T_int[1:, :, :] - T_int[:-1, :, :])

        coeff_col = kfy[:, :, 1:-1] / dx2
        residual[:, :-1, :] = residual[:, :-1, :] + coeff_col * (T_int[:, :-1, :] - T_int[:, 1:, :])
        residual[:, 1:, :] = residual[:, 1:, :] + coeff_col * (T_int[:, 1:, :] - T_int[:, :-1, :])

        residual = residual - A_source[:, :, 1:-1]
        residual[:, :, 0] = residual[:, :, 0] - (kfz[:, :, 0] / dz2) * T_surf
        residual[:, :, -1] = residual[:, :, -1] - (kfz[:, :, -1] / dz2) * T_bot
        return residual

    @staticmethod
    def _corrected_adjoint_rhs(
        solver: HeatSolver,
        T: np.ndarray,
        K0: np.ndarray,
        c_ref: np.ndarray,
        grad_T_full: np.ndarray,
        lambda0: np.ndarray,
    ) -> np.ndarray:
        k_field = _HeatSolveFunction._conductivity_from_temperature_np(K0, c_ref, T)
        inv_1cT = 1.0 / np.maximum(1.0 + c_ref * T, 0.1)
        dk_dT = -k_field * c_ref * inv_1cT

        v_full = np.zeros_like(T)
        v_full[:, :, 1:-1] = lambda0
        MTv = np.zeros_like(T)

        a = k_field[:, :, :-1]
        b = k_field[:, :, 1:]
        ab2 = np.maximum((a + b) ** 2, 1e-60)
        u_diff = (T[:, :, :-1] - T[:, :, 1:]) / (solver.dz ** 2)
        v_diff = v_full[:, :, :-1] - v_full[:, :, 1:]
        MTv[:, :, :-1] += 2.0 * b ** 2 / ab2 * dk_dT[:, :, :-1] * u_diff * v_diff
        MTv[:, :, 1:] += 2.0 * a ** 2 / ab2 * dk_dT[:, :, 1:] * u_diff * v_diff

        a = k_field[:-1, :, :]
        b = k_field[1:, :, :]
        ab2 = np.maximum((a + b) ** 2, 1e-60)
        u_diff = (T[:-1, :, :] - T[1:, :, :]) / (solver.dy ** 2)
        v_diff = v_full[:-1, :, :] - v_full[1:, :, :]
        MTv[:-1, :, :] += 2.0 * b ** 2 / ab2 * dk_dT[:-1, :, :] * u_diff * v_diff
        MTv[1:, :, :] += 2.0 * a ** 2 / ab2 * dk_dT[1:, :, :] * u_diff * v_diff

        a = k_field[:, :-1, :]
        b = k_field[:, 1:, :]
        ab2 = np.maximum((a + b) ** 2, 1e-60)
        u_diff = (T[:, :-1, :] - T[:, 1:, :]) / (solver.dx ** 2)
        v_diff = v_full[:, :-1, :] - v_full[:, 1:, :]
        MTv[:, :-1, :] += 2.0 * b ** 2 / ab2 * dk_dT[:, :-1, :] * u_diff * v_diff
        MTv[:, 1:, :] += 2.0 * a ** 2 / ab2 * dk_dT[:, 1:, :] * u_diff * v_diff

        return grad_T_full[:, :, 1:-1].reshape(-1) - MTv[:, :, 1:-1].reshape(-1)

    @staticmethod
    def _direct_output_grads(
        solver: HeatSolver,
        T_saved: torch.Tensor,
        K0_saved: torch.Tensor,
        grad_T: Optional[torch.Tensor],
        grad_q: Optional[torch.Tensor],
        needs_K0: bool,
        work_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = T_saved.device
        c_ref_t = torch.as_tensor(solver.c_ref, dtype=work_dtype, device=device)
        with torch.enable_grad():
            T_var = T_saved.detach().to(dtype=work_dtype).requires_grad_(True)
            K0_var = K0_saved.detach().to(dtype=work_dtype).requires_grad_(needs_K0)

            output_scalar = torch.zeros((), dtype=work_dtype, device=device)
            if grad_T is not None:
                output_scalar = output_scalar + (grad_T.to(dtype=work_dtype) * T_var).sum()
            if grad_q is not None:
                q_var = _HeatSolveFunction._surface_heat_flow_upward_torch(solver, T_var, K0_var, c_ref_t)
                output_scalar = output_scalar + (grad_q.to(dtype=work_dtype) * q_var).sum()

            inputs: list[torch.Tensor] = [T_var]
            if needs_K0:
                inputs.append(K0_var)
            grads = torch.autograd.grad(output_scalar, inputs, allow_unused=True)

        return grads[0], grads[1] if needs_K0 else None

    @staticmethod
    def _solve_adjoint_state(
        solver: HeatSolver,
        T_np: np.ndarray,
        T_bot_np: np.ndarray,
        K0_np: np.ndarray,
        A_np: np.ndarray,
        grad_T_full_np: np.ndarray,
    ) -> np.ndarray:
        adj_rhs = grad_T_full_np[:, :, 1:-1].reshape(-1)
        A_mat, _ = solver.build_frozen_system(T_np, T_bot_np, K0=K0_np, A_source=A_np)
        A_t = A_mat.transpose().tocsr()
        lambda0_np, _ = solver._solve_sparse_system(A_t, adj_rhs)
        corrected_rhs = _HeatSolveFunction._corrected_adjoint_rhs(
            solver,
            T_np,
            K0_np,
            np.asarray(solver.c_ref, dtype=np.float64),
            grad_T_full_np,
            lambda0_np.reshape(solver.ny, solver.nx, solver.nz - 2),
        )
        lambda_np, _ = solver._solve_sparse_system(A_t, corrected_rhs, x0=lambda0_np)
        return lambda_np.reshape(solver.ny, solver.nx, solver.nz - 2)

    @staticmethod
    def _residual_input_grads(
        solver: HeatSolver,
        T_saved: torch.Tensor,
        T_bot_saved: torch.Tensor,
        K0_saved: torch.Tensor,
        A_saved: torch.Tensor,
        lambda_t: torch.Tensor,
        needs_T_bot: bool,
        needs_K0: bool,
        needs_A: bool,
        work_dtype: torch.dtype,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not (needs_T_bot or needs_K0 or needs_A):
            return None, None, None

        device = T_saved.device
        T_bot_var = T_bot_saved.detach().to(dtype=work_dtype).requires_grad_(needs_T_bot)
        K0_var = K0_saved.detach().to(dtype=work_dtype).requires_grad_(needs_K0)
        A_var = A_saved.detach().to(dtype=work_dtype).requires_grad_(needs_A)
        c_ref_t = torch.as_tensor(solver.c_ref, dtype=work_dtype, device=device)
        T_surf_t = torch.as_tensor(solver.surface_temp, dtype=work_dtype, device=device)

        with torch.enable_grad():
            residual = _HeatSolveFunction._frozen_residual_torch(
                solver,
                T_saved.detach().to(dtype=work_dtype),
                T_bot_var,
                K0_var,
                A_var,
                c_ref_t,
                T_surf_t,
            )
            correction_scalar = (lambda_t * residual).sum()

            inputs: list[torch.Tensor] = []
            if needs_T_bot:
                inputs.append(T_bot_var)
            if needs_K0:
                inputs.append(K0_var)
            if needs_A:
                inputs.append(A_var)
            grads = torch.autograd.grad(correction_scalar, inputs, allow_unused=True)

        grad_T_bot = grad_K0 = grad_A = None
        idx = 0
        if needs_T_bot:
            grad_T_bot = grads[idx]
            idx += 1
        if needs_K0:
            grad_K0 = grads[idx]
            idx += 1
        if needs_A:
            grad_A = grads[idx]
        return grad_T_bot, grad_K0, grad_A

    @staticmethod
    def _combine_input_grads(
        solver: HeatSolver,
        T_bot_saved: torch.Tensor,
        K0_saved: torch.Tensor,
        A_saved: torch.Tensor,
        grad_T: Optional[torch.Tensor],
        grad_T_full_np: np.ndarray,
        direct_grad_K0: Optional[torch.Tensor],
        correction_grads: tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]],
        needs_T_bot: bool,
        needs_K0: bool,
        needs_A: bool,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        device = T_bot_saved.device
        corr_T_bot, corr_K0, corr_A = correction_grads
        grad_T_bot_out = grad_K0_out = grad_A_out = None

        if needs_T_bot:
            grad_T_bot_np = np.zeros((solver.ny, solver.nx), dtype=np.float64)
            if grad_T is not None:
                grad_T_bot_np = grad_T_bot_np + grad_T_full_np[:, :, -1]
            grad_T_bot_out = torch.as_tensor(grad_T_bot_np, dtype=T_bot_saved.dtype, device=device)
            if corr_T_bot is not None:
                grad_T_bot_out = grad_T_bot_out - corr_T_bot.to(dtype=T_bot_saved.dtype)

        if needs_K0:
            grad_K0_out = direct_grad_K0
            if corr_K0 is not None:
                grad_K0_out = grad_K0_out - corr_K0 if grad_K0_out is not None else -corr_K0
            grad_K0_out = grad_K0_out.to(dtype=K0_saved.dtype)

        if needs_A:
            grad_A_out = torch.zeros_like(A_saved)
            if corr_A is not None:
                grad_A_out = grad_A_out - corr_A.to(dtype=torch.float64)
            grad_A_out = grad_A_out.to(dtype=A_saved.dtype)

        return grad_T_bot_out, grad_K0_out, grad_A_out

    @staticmethod
    def forward(
        ctx,
        T_bot: torch.Tensor,
        K0: torch.Tensor,
        A_source: torch.Tensor,
        solver: HeatSolver,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        T_bot_np = _HeatSolveFunction._to_numpy(T_bot)
        K0_np = _HeatSolveFunction._to_numpy(K0)
        A_np = _HeatSolveFunction._to_numpy(A_source)

        T_np = solver.solve(T_bot_np, K0=K0_np, A_source=A_np, verbose=False)
        q_np = solver.surface_heat_flow_upward_np(T_np, K0=K0_np)

        T_out = torch.as_tensor(T_np, dtype=T_bot.dtype, device=T_bot.device)
        q_out = torch.as_tensor(q_np, dtype=T_bot.dtype, device=T_bot.device)

        ctx.solver = solver
        ctx.save_for_backward(T_bot.detach(), K0.detach(), A_source.detach(), T_out.detach())
        return T_out, q_out

    @staticmethod
    def backward(
        ctx,
        grad_T: Optional[torch.Tensor],
        grad_q: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], None]:
        solver: HeatSolver = ctx.solver
        T_bot_saved, K0_saved, A_saved, T_saved = ctx.saved_tensors
        needs_T_bot, needs_K0, needs_A, _ = ctx.needs_input_grad
        if grad_T is None and grad_q is None:
            return None, None, None, None

        work_dtype = torch.float64
        T_np = _HeatSolveFunction._to_numpy(T_saved.to(dtype=work_dtype))
        T_bot_np = _HeatSolveFunction._to_numpy(T_bot_saved.to(dtype=work_dtype))
        K0_np = _HeatSolveFunction._to_numpy(K0_saved.to(dtype=work_dtype))
        A_np = _HeatSolveFunction._to_numpy(A_saved.to(dtype=work_dtype))

        direct_grad_T, direct_grad_K0 = _HeatSolveFunction._direct_output_grads(
            solver,
            T_saved,
            K0_saved,
            grad_T,
            grad_q,
            needs_K0,
            work_dtype,
        )
        grad_T_full_np = _HeatSolveFunction._to_numpy(direct_grad_T)
        lambda_np = _HeatSolveFunction._solve_adjoint_state(solver, T_np, T_bot_np, K0_np, A_np, grad_T_full_np)
        lambda_t = torch.as_tensor(lambda_np, dtype=work_dtype, device=T_saved.device)
        correction_grads = _HeatSolveFunction._residual_input_grads(
            solver,
            T_saved,
            T_bot_saved,
            K0_saved,
            A_saved,
            lambda_t,
            needs_T_bot,
            needs_K0,
            needs_A,
            work_dtype,
        )
        grad_T_bot_out, grad_K0_out, grad_A_out = _HeatSolveFunction._combine_input_grads(
            solver,
            T_bot_saved,
            K0_saved,
            A_saved,
            grad_T,
            grad_T_full_np,
            direct_grad_K0,
            correction_grads,
            needs_T_bot,
            needs_K0,
            needs_A,
        )
        return grad_T_bot_out, grad_K0_out, grad_A_out, None
