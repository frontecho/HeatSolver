"""Nonlinear conduction and implicit adjoint/tangent solves for a supplied conductivity law."""
from __future__ import annotations

import time
import numpy as np
import scipy.sparse as sp
from scipy.linalg.lapack import dgttrf, dgttrs
from scipy.sparse.linalg import gmres, LinearOperator
from heat_solver import HeatSolver


class NonlinearHeatSolver:
    """Picard forward solve and exact implicit derivatives for a supplied K(T) law.

    The law's evaluate(T) method returns conductivity and its temperature
    derivative on the solver grid. Geometry and linear tolerances come from
    linear_solver; no region, depth interval or material model is prescribed.
    """
    def __init__(self, linear_solver: HeatSolver, law, tolerance_C=2e-5, residual_tolerance=2e-10, max_iterations=45):
        if np.any(linear_solver.c_ref):
            raise ValueError('The inner solver must use frozen conductivity')
        self.linear_solver, self.law = linear_solver, law
        self.tolerance_C, self.residual_tolerance = tolerance_C, residual_tolerance
        self.max_iterations = max_iterations

    def solve(self, A, bottom, top, initial=None):
        start = time.perf_counter()
        s = self.linear_solver
        s.surface_temp = np.asarray(top).copy()
        if initial is None:
            z = s.z_nodes_km
            f = (z-z[0])/(z[-1]-z[0])
            T = top[..., None]*(1-f)+bottom[..., None]*f
        else:
            T = np.array(initial, copy=True)
        T[..., 0], T[..., -1] = top, bottom
        history = []
        for j in range(self.max_iterations):
            K, _ = self.law.evaluate(T)
            updated = s.solve(bottom, K0=K, A_source=A, T_init=T, reuse_last=False)
            change = float(np.max(abs(updated-T)))
            history.append(change)
            T = updated
            if change <= self.tolerance_C:
                K, derivative = self.law.evaluate(T)
                matrix, rhs = s._build_matrix_and_rhs(K, A, T, bottom)
                residual = float(np.linalg.norm(matrix@T[..., 1:-1].ravel()-rhs)/max(np.linalg.norm(rhs), 1e-30))
                if residual <= self.residual_tolerance:
                    self.info = dict(picard_iterations=j+1, max_updates_C=history,
                        nonlinear_relative_residual=residual, elapsed_seconds=time.perf_counter()-start,
                        linear_solver='full 3D column-preconditioned CG', relaxation=1.)
                    return T, K, derivative
        raise RuntimeError(f'Picard failed after {self.max_iterations} iterations, last update {change:g} C')

    def jacobian(self, T, K, derivative):
        s = self.linear_solver
        shape = T.shape
        diag = np.zeros(shape)
        minus = [np.zeros(shape) for _ in range(3)]
        plus = [np.zeros(shape) for _ in range(3)]
        for axis, spacing, weight in [(0, s.dy, s.horizontal_face_y),
                (1, s.dx, s.horizontal_face_x), (2, s.dz, s.horizontal_area_factor)]:
            left, right = [slice(None)]*3, [slice(None)]*3
            left[axis], right[axis] = slice(None, -1), slice(1, None)
            a, b = tuple(left), tuple(right)
            ka, kb = K[a], K[b]
            factor = weight/spacing**2
            face = 2*ka*kb/(ka+kb)
            dt = T[a]-T[b]
            dfa = factor*(face+dt*2*kb**2/(ka+kb)**2*derivative[a])
            dfb = factor*(-face+dt*2*ka**2/(ka+kb)**2*derivative[b])
            diag[a] += dfa
            diag[b] -= dfb
            plus[axis][a] = dfb
            minus[axis][b] = -dfa
        interior = lambda a: a[..., 1:-1].ravel()
        values = np.concatenate([interior(minus[2])[s._zm], interior(plus[2])[s._zp],
            interior(minus[0])[s._ym], interior(plus[0])[s._yp],
            interior(minus[1])[s._xm], interior(plus[1])[s._xp], interior(diag)])
        J = sp.csr_matrix((values, (s._matrix_rows, s._matrix_cols)), shape=(s._n_unknowns, s._n_unknowns))
        top, bottom = minus[2][..., 1].copy(), plus[2][..., -2].copy()
        return J, top, bottom

    @staticmethod
    def factorize_jacobian(J, transpose=False):
        matrix = J.T.tocsr() if transpose else J
        dl, dd, du, du2, pivot, info = dgttrf(matrix.diagonal(-1), matrix.diagonal(), matrix.diagonal(1))
        if info != 0:
            raise RuntimeError(f'Nonlinear column factorization failed: {info}')
        def apply(x):
            result, info = dgttrs(dl, dd, du, du2, pivot, np.asarray(x)[:, None])
            if info != 0:
                raise RuntimeError(f'Column preconditioner failed: {info}')
            return result[:, 0]
        preconditioner = LinearOperator(matrix.shape, matvec=apply)
        def solve(rhs):
            if not np.any(rhs):
                return np.zeros_like(rhs)
            value, code = gmres(matrix, rhs, M=preconditioner, rtol=2e-10, atol=0,
                                 restart=35, maxiter=50)
            if code != 0:
                raise RuntimeError(f'Nonlinear Jacobian solve failed: {code}')
            return value
        return solve

    def adjoint(self, T, K, derivative, grad_T):
        J, top, bottom = self.jacobian(T, K, derivative)
        solve = self.factorize_jacobian(J, transpose=True)
        lam = solve(grad_T[..., 1:-1].ravel()).reshape(T.shape[0], T.shape[1], T.shape[2]-2)
        grad_A = np.zeros_like(T)
        grad_A[..., 1:-1] = lam*self.linear_solver.horizontal_area_factor
        return grad_A, grad_T[..., 0]-top*lam[..., 0], grad_T[..., -1]-bottom*lam[..., -1]

    def tangents(self, T, K, derivative, components, basis):
        J, top, bottom = self.jacobian(T, K, derivative)
        solve = self.factorize_jacobian(J)
        n = len(components)+len(basis)+1
        for j in range(n):
            field = np.zeros_like(T)
            rhs = np.zeros_like(T[..., 1:-1])
            if j < len(components):
                rhs += components[j, ..., 1:-1]*self.linear_solver.horizontal_area_factor
            elif j < n-1:
                field[..., -1] = basis[j-len(components)]
                rhs[..., -1] -= bottom*field[..., -1]
            else:
                field[..., 0] = 1.
                rhs[..., 0] -= top
            field[..., 1:-1] = solve(rhs.ravel()).reshape(rhs.shape)
            yield field
