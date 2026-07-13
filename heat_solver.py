"""Finite-volume solver for 3-D steady-state heat conduction."""

from __future__ import annotations

import time

import numpy as np
import pyamg
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class HeatSolver:
    """Solve ``-div(k(T) grad(T)) = A`` with vertical Dirichlet boundaries."""

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
        picard_tol: float = 1e-6,
        cg_rtol: float = 1e-9,
    ) -> None:
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.z_nodes_km = np.asarray(z_nodes_km, dtype=np.float64)
        self.surface_temp = np.asarray(surface_temp, dtype=np.float64)
        self.K0_ref = np.asarray(K0_ref, dtype=np.float64)
        self.c_ref = np.asarray(c_ref, dtype=np.float64)
        self.A_ref = np.asarray(A_ref, dtype=np.float64)
        self.layer_masks = {
            name: np.asarray(mask, dtype=np.float64)
            for name, mask in (layer_masks or {}).items()
        }

        if self.surface_temp.ndim != 2:
            raise ValueError("surface_temp must be a 2-D array")
        self.ny, self.nx = self.surface_temp.shape
        self.nz = self.z_nodes_km.size
        if self.nz < 3:
            raise ValueError("z_nodes_km must contain at least three nodes")
        if self.x.shape != (self.ny, self.nx) or self.y.shape != (self.ny, self.nx):
            raise ValueError("x, y, and surface_temp must have matching shapes")

        self.field_shape = (self.ny, self.nx, self.nz)
        for name, field in (
            ("K0_ref", self.K0_ref),
            ("c_ref", self.c_ref),
            ("A_ref", self.A_ref),
        ):
            if field.shape != self.field_shape:
                raise ValueError(f"{name} must have shape {self.field_shape}")
        for name, mask in self.layer_masks.items():
            if mask.shape != self.field_shape:
                raise ValueError(f"layer mask {name!r} must have shape {self.field_shape}")

        dz = np.diff(self.z_nodes_km)
        if not np.all(dz > 0.0) or not np.allclose(dz, dz[0]):
            raise ValueError("z_nodes_km must be uniformly increasing")
        if not np.allclose(self.x, self.x[0, :]) or not np.allclose(
            self.y,
            self.y[:, :1],
        ):
            raise ValueError("x and y must define a rectilinear grid")
        x_unique = self.x[0]
        y_unique = self.y[:, 0]
        if self.nx < 2 or self.ny < 2:
            raise ValueError("x and y must each contain at least two coordinates")
        dx = np.diff(x_unique)
        dy = np.diff(y_unique)
        if not np.allclose(dx, dx[0]) or not np.allclose(dy, dy[0]):
            raise ValueError("x and y must be uniformly spaced")

        for name, field in (
            ("x", self.x),
            ("y", self.y),
            ("surface_temp", self.surface_temp),
            ("K0_ref", self.K0_ref),
            ("c_ref", self.c_ref),
            ("A_ref", self.A_ref),
        ):
            if not np.all(np.isfinite(field)):
                raise ValueError(f"{name} contains non-finite values")

        self.dz = float(dz[0] * 1e3)
        self.dx = float(dx[0])
        self.dy = float(dy[0])
        self.max_picard = max_picard
        self.picard_tol = picard_tol
        self.cg_rtol = cg_rtol

        self._prepare_matrix_topology()
        self._last_T: np.ndarray | None = None
        self._preconditioner = None
        self.last_solve_info: dict[str, float | int] | None = None
        self._solve_calls = 0

    @staticmethod
    def _conductivity_terms(
        K0: np.ndarray,
        c: np.ndarray,
        T: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raw = 1.0 + c * T
        denominator = np.maximum(raw, 0.1)
        conductivity = K0 / denominator
        dk_dT = np.where(raw > 0.1, -K0 * c / denominator**2, 0.0)
        return conductivity, dk_dT, 1.0 / denominator

    @staticmethod
    def _conductivity_from_temperature(
        K0: np.ndarray,
        c: np.ndarray,
        T: np.ndarray,
    ) -> np.ndarray:
        return K0 / np.maximum(1.0 + c * T, 0.1)

    @staticmethod
    def _harmonic_mean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return 2.0 * a * b / np.maximum(a + b, 1e-30)

    def _prepare_matrix_topology(self) -> None:
        ny, nx, nz2 = self.ny, self.nx, self.nz - 2
        ii = np.broadcast_to(np.arange(ny)[:, None, None], (ny, nx, nz2))
        jj = np.broadcast_to(np.arange(nx)[None, :, None], (ny, nx, nz2))
        kk = np.broadcast_to(np.arange(nz2)[None, None, :], (ny, nx, nz2))

        def flat(i: np.ndarray, j: np.ndarray, k: np.ndarray) -> np.ndarray:
            return i * (nx * nz2) + j * nz2 + k

        row = flat(ii, jj, kk).ravel()
        self._zm = (kk > 0).ravel()
        self._zp = (kk < nz2 - 1).ravel()
        self._ym = (ii > 0).ravel()
        self._yp = (ii < ny - 1).ravel()
        self._xm = (jj > 0).ravel()
        self._xp = (jj < nx - 1).ravel()

        rows = [row[mask] for mask in (self._zm, self._zp, self._ym, self._yp, self._xm, self._xp)]
        cols = [
            flat(ii.ravel()[self._zm], jj.ravel()[self._zm], kk.ravel()[self._zm] - 1),
            flat(ii.ravel()[self._zp], jj.ravel()[self._zp], kk.ravel()[self._zp] + 1),
            flat(ii.ravel()[self._ym] - 1, jj.ravel()[self._ym], kk.ravel()[self._ym]),
            flat(ii.ravel()[self._yp] + 1, jj.ravel()[self._yp], kk.ravel()[self._yp]),
            flat(ii.ravel()[self._xm], jj.ravel()[self._xm] - 1, kk.ravel()[self._xm]),
            flat(ii.ravel()[self._xp], jj.ravel()[self._xp] + 1, kk.ravel()[self._xp]),
        ]
        self._matrix_rows = np.concatenate([*rows, row])
        self._matrix_cols = np.concatenate([*cols, row])
        self._surface_rows = np.arange(ny * nx) * nz2
        self._bottom_rows = self._surface_rows + nz2 - 1
        self._n_unknowns = ny * nx * nz2

    def _build_matrix_and_rhs(
        self,
        K0: np.ndarray,
        A_source: np.ndarray,
        T: np.ndarray,
        T_bot: np.ndarray,
    ) -> tuple[sp.csr_matrix, np.ndarray]:
        k = self._conductivity_from_temperature(K0, self.c_ref, T)
        face_z = self._harmonic_mean(k[:, :, :-1], k[:, :, 1:])
        face_y = self._harmonic_mean(k[:-1, :, :], k[1:, :, :])
        face_x = self._harmonic_mean(k[:, :-1, :], k[:, 1:, :])
        nz2 = self.nz - 2

        czm = face_z[:, :, :-1] / self.dz**2
        czp = face_z[:, :, 1:] / self.dz**2
        cym = np.zeros((self.ny, self.nx, nz2), dtype=np.float64)
        cyp = np.zeros_like(cym)
        cxm = np.zeros_like(cym)
        cxp = np.zeros_like(cym)
        cym[1:] = face_y[:, :, 1:-1] / self.dy**2
        cyp[:-1] = face_y[:, :, 1:-1] / self.dy**2
        cxm[:, 1:] = face_x[:, :, 1:-1] / self.dx**2
        cxp[:, :-1] = face_x[:, :, 1:-1] / self.dx**2
        diagonal = czm + czp + cym + cyp + cxm + cxp

        values = np.concatenate(
            [
                -czm.ravel()[self._zm],
                -czp.ravel()[self._zp],
                -cym.ravel()[self._ym],
                -cyp.ravel()[self._yp],
                -cxm.ravel()[self._xm],
                -cxp.ravel()[self._xp],
                diagonal.ravel(),
            ]
        )
        matrix = sp.csr_matrix(
            (values, (self._matrix_rows, self._matrix_cols)),
            shape=(self._n_unknowns, self._n_unknowns),
        )

        rhs = A_source[:, :, 1:-1].ravel().copy()
        rhs[self._surface_rows] += (czm[:, :, 0] * self.surface_temp).ravel()
        rhs[self._bottom_rows] += (czp[:, :, -1] * T_bot).ravel()
        return matrix, rhs

    def build_frozen_system(
        self,
        T: np.ndarray,
        T_bot: np.ndarray,
        K0: np.ndarray | None = None,
        A_source: np.ndarray | None = None,
    ) -> tuple[sp.csr_matrix, np.ndarray]:
        return self._build_matrix_and_rhs(
            self.K0_ref if K0 is None else K0,
            self.A_ref if A_source is None else A_source,
            T,
            T_bot,
        )

    def clear_warm_start(self) -> None:
        self._last_T = None
        self._preconditioner = None
        self.last_solve_info = None

    def solve(
        self,
        T_bot: np.ndarray,
        K0: np.ndarray | None = None,
        A_source: np.ndarray | None = None,
        *,
        T_init: np.ndarray | None = None,
        reuse_last: bool = True,
    ) -> np.ndarray:
        T_bot = np.asarray(T_bot, dtype=np.float64)
        K0 = self.K0_ref if K0 is None else np.asarray(K0, dtype=np.float64)
        A_source = self.A_ref if A_source is None else np.asarray(A_source, dtype=np.float64)
        if T_bot.shape != (self.ny, self.nx):
            raise ValueError(f"T_bot must have shape {(self.ny, self.nx)}")
        if K0.shape != self.field_shape or A_source.shape != self.field_shape:
            raise ValueError(f"K0 and A_source must have shape {self.field_shape}")

        if T_init is not None:
            T = np.asarray(T_init, dtype=np.float64).copy()
        elif reuse_last and self._last_T is not None:
            T = self._last_T.copy()
        else:
            fraction = np.linspace(0.0, 1.0, self.nz)
            T = self.surface_temp[:, :, None] * (1.0 - fraction) + T_bot[:, :, None] * fraction
        if T.shape != self.field_shape:
            raise ValueError(f"T_init must have shape {self.field_shape}")
        T[:, :, 0] = self.surface_temp
        T[:, :, -1] = T_bot

        self._solve_calls += 1
        start = time.perf_counter()
        cg_iterations = 0
        preconditioner = None
        for picard_iteration in range(1, self.max_picard + 1):
            matrix, rhs = self._build_matrix_and_rhs(K0, A_source, T, T_bot)
            if preconditioner is None:
                preconditioner = pyamg.smoothed_aggregation_solver(matrix).aspreconditioner()
            iteration_count = 0

            def count_iteration(_: np.ndarray) -> None:
                nonlocal iteration_count
                iteration_count += 1

            interior, info = spla.cg(
                matrix,
                rhs,
                x0=T[:, :, 1:-1].ravel(),
                M=preconditioner,
                rtol=self.cg_rtol,
                maxiter=500,
                callback=count_iteration,
            )
            if info != 0:
                raise RuntimeError(f"CG did not converge: info={info}")
            cg_iterations += max(iteration_count, 1)
            T_new = np.concatenate(
                [
                    self.surface_temp[:, :, None],
                    interior.reshape(self.ny, self.nx, self.nz - 2),
                    T_bot[:, :, None],
                ],
                axis=2,
            )
            relative_change = np.linalg.norm(T_new - T) / np.linalg.norm(T_new)
            T = T_new
            if relative_change < self.picard_tol:
                break
        else:
            raise RuntimeError(
                f"Picard iteration did not converge after {self.max_picard} iterations"
            )

        self._last_T = T.copy()
        self._preconditioner = preconditioner
        self.last_solve_info = {
            "solve_id": self._solve_calls,
            "picard_iterations": picard_iteration,
            "cg_iterations": cg_iterations,
            "relative_change": float(relative_change),
            "elapsed_s": time.perf_counter() - start,
        }
        return T

    def surface_heat_flow(self, T: np.ndarray, K0: np.ndarray | None = None) -> np.ndarray:
        """Return upward surface heat flow in mW/m2."""
        K0 = self.K0_ref if K0 is None else K0
        k = self._conductivity_from_temperature(
            K0[:, :, :2],
            self.c_ref[:, :, :2],
            T[:, :, :2],
        )
        face_k = self._harmonic_mean(k[:, :, 0], k[:, :, 1])
        return 1e3 * face_k * (T[:, :, 1] - T[:, :, 0]) / self.dz

    @staticmethod
    def _add_face_vjp(
        T: np.ndarray,
        k: np.ndarray,
        dk_dT: np.ndarray,
        dk_dK0: np.ndarray | None,
        v: np.ndarray,
        grad_T: np.ndarray,
        grad_K0: np.ndarray | None,
        axis: int,
        spacing: float,
    ) -> None:
        left = [slice(None)] * 3
        right = [slice(None)] * 3
        left[axis] = slice(None, -1)
        right[axis] = slice(1, None)
        left_index = tuple(left)
        right_index = tuple(right)

        ka = k[left_index]
        kb = k[right_index]
        denominator = np.maximum((ka + kb) ** 2, 1e-60)
        harmonic = 2.0 * ka * kb / np.maximum(ka + kb, 1e-30)
        delta_v = v[left_index] - v[right_index]
        delta_T = T[left_index] - T[right_index]

        direct = delta_v * harmonic / spacing**2
        grad_T[left_index] += direct
        grad_T[right_index] -= direct

        common = delta_v * delta_T / spacing**2
        grad_ka = common * (2.0 * kb**2 / denominator)
        grad_kb = common * (2.0 * ka**2 / denominator)
        grad_T[left_index] += grad_ka * dk_dT[left_index]
        grad_T[right_index] += grad_kb * dk_dT[right_index]
        if grad_K0 is not None:
            grad_K0[left_index] += grad_ka * dk_dK0[left_index]
            grad_K0[right_index] += grad_kb * dk_dK0[right_index]

    def _residual_vjp(
        self,
        T: np.ndarray,
        vector: np.ndarray,
        k: np.ndarray,
        dk_dT: np.ndarray,
        dk_dK0: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Return derivatives of ``vector.T @ residual`` with respect to T and K0."""
        v = np.zeros_like(T)
        v[:, :, 1:-1] = vector.reshape(self.ny, self.nx, self.nz - 2)
        grad_T = np.zeros_like(T)
        grad_K0 = None if dk_dK0 is None else np.zeros_like(T)
        self._add_face_vjp(T, k, dk_dT, dk_dK0, v, grad_T, grad_K0, 2, self.dz)
        self._add_face_vjp(T, k, dk_dT, dk_dK0, v, grad_T, grad_K0, 0, self.dy)
        self._add_face_vjp(T, k, dk_dT, dk_dK0, v, grad_T, grad_K0, 1, self.dx)
        return grad_T, grad_K0

    def _output_vjp(
        self,
        T: np.ndarray,
        K0: np.ndarray,
        grad_T: np.ndarray | None,
        grad_q: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        direct_T = np.zeros_like(T) if grad_T is None else np.asarray(grad_T, dtype=np.float64).copy()
        direct_K0 = np.zeros_like(K0)
        if grad_q is None:
            return direct_T, direct_K0

        weight = np.asarray(grad_q, dtype=np.float64)
        k, dk_dT, dk_dK0 = self._conductivity_terms(
            K0[:, :, :2],
            self.c_ref[:, :, :2],
            T[:, :, :2],
        )
        ka = k[:, :, 0]
        kb = k[:, :, 1]
        denominator = np.maximum((ka + kb) ** 2, 1e-60)
        harmonic = 2.0 * ka * kb / np.maximum(ka + kb, 1e-30)
        gradient_weight = 1e3 * weight * harmonic / self.dz
        direct_T[:, :, 0] -= gradient_weight
        direct_T[:, :, 1] += gradient_weight

        harmonic_weight = 1e3 * weight * (T[:, :, 1] - T[:, :, 0]) / self.dz
        grad_ka = harmonic_weight * (2.0 * kb**2 / denominator)
        grad_kb = harmonic_weight * (2.0 * ka**2 / denominator)
        direct_T[:, :, 0] += grad_ka * dk_dT[:, :, 0]
        direct_T[:, :, 1] += grad_kb * dk_dT[:, :, 1]
        direct_K0[:, :, 0] += grad_ka * dk_dK0[:, :, 0]
        direct_K0[:, :, 1] += grad_kb * dk_dK0[:, :, 1]
        return direct_T, direct_K0

    def adjoint_gradient(
        self,
        T: np.ndarray,
        K0: np.ndarray,
        *,
        grad_T: np.ndarray | None = None,
        grad_q: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Differentiate an output-weighted scalar with one nonlinear adjoint solve."""
        direct_T, direct_K0 = self._output_vjp(T, K0, grad_T, grad_q)
        rhs = direct_T[:, :, 1:-1].ravel()
        k, dk_dT, dk_dK0 = self._conductivity_terms(K0, self.c_ref, T)
        adjoint_iterations = 0
        if np.any(rhs):
            def transpose_matvec(vector: np.ndarray) -> np.ndarray:
                residual_grad_T, _ = self._residual_vjp(T, vector, k, dk_dT)
                return residual_grad_T[:, :, 1:-1].ravel()

            operator = spla.LinearOperator(
                (self._n_unknowns, self._n_unknowns),
                matvec=transpose_matvec,
                dtype=np.float64,
            )

            def count_iteration(_: float) -> None:
                nonlocal adjoint_iterations
                adjoint_iterations += 1

            adjoint, info = spla.gmres(
                operator,
                rhs,
                M=self._preconditioner,
                rtol=self.cg_rtol,
                maxiter=500,
                callback=count_iteration,
                callback_type="legacy",
            )
            if info != 0:
                raise RuntimeError(f"GMRES adjoint did not converge: info={info}")
        else:
            adjoint = np.zeros(self._n_unknowns, dtype=np.float64)

        residual_grad_T, residual_grad_K0 = self._residual_vjp(
            T,
            adjoint,
            k,
            dk_dT,
            dk_dK0,
        )
        grad_T_bot = direct_T[:, :, -1] - residual_grad_T[:, :, -1]
        grad_K0 = direct_K0 - residual_grad_K0
        grad_A = np.zeros_like(T)
        grad_A[:, :, 1:-1] = adjoint.reshape(self.ny, self.nx, self.nz - 2)
        if self.last_solve_info is not None:
            self.last_solve_info["adjoint_iterations"] = adjoint_iterations
        return grad_T_bot, grad_K0, grad_A
