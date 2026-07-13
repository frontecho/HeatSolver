from __future__ import annotations

import unittest

import numpy as np

from heat_solver import HeatSolver
from tests.helpers import make_solver


class HeatSolverTest(unittest.TestCase):
    def test_linear_solution_and_surface_heat_flow(self) -> None:
        solver, T_bot = make_solver(
            ny=3,
            nx=3,
            nz=5,
            surface_temp=10.0,
            bottom_temp=410.0,
            K0=2.5,
        )
        T = solver.solve(T_bot)
        q = solver.surface_heat_flow(T)

        np.testing.assert_allclose(T[1, 1], np.linspace(10.0, 410.0, 5), atol=1e-7)
        np.testing.assert_allclose(q, 250.0, atol=1e-7)
        self.assertEqual(solver.last_solve_info["solve_id"], 1)

    def test_constant_source_matches_quadratic_solution(self) -> None:
        solver, T_bot = make_solver(
            ny=2,
            nx=2,
            nz=7,
            surface_temp=10.0,
            bottom_temp=410.0,
            K0=2.5,
            A=1e-6,
        )
        T = solver.solve(T_bot)
        depth = solver.z_nodes_km[-1] * 1e3
        z = solver.z_nodes_km * 1e3
        top_gradient = 400.0 / depth + 1e-6 * depth / (2.0 * 2.5)
        expected = 10.0 + top_gradient * z - 1e-6 * z**2 / (2.0 * 2.5)

        np.testing.assert_allclose(T[0, 0], expected, atol=1e-7)
        np.testing.assert_allclose(solver.surface_heat_flow(T), 2.5 * top_gradient * 1e3, atol=1e-7)

    def test_nonlinear_picard_residual(self) -> None:
        solver, T_bot = make_solver(
            ny=2,
            nx=2,
            nz=6,
            bottom_temp=520.0,
            K0=3.0,
            c=5e-4,
            A=1e-6,
            max_picard=50,
            picard_tol=1e-11,
            cg_rtol=1e-12,
        )
        T = solver.solve(T_bot)
        matrix, rhs = solver.build_frozen_system(T, T_bot)
        residual = matrix @ T[:, :, 1:-1].ravel() - rhs

        self.assertLess(np.linalg.norm(residual) / np.linalg.norm(rhs), 1e-10)
        self.assertGreater(solver.last_solve_info["picard_iterations"], 1)

    def test_conductivity_clamp_has_zero_temperature_derivative(self) -> None:
        k, dk_dT, dk_dK0 = HeatSolver._conductivity_terms(
            np.array([2.0]),
            np.array([1.0]),
            np.array([-2.0]),
        )
        np.testing.assert_allclose(k, 20.0)
        np.testing.assert_allclose(dk_dT, 0.0)
        np.testing.assert_allclose(dk_dK0, 10.0)

    def test_residual_vjp_matches_directional_difference(self) -> None:
        solver, T_bot = make_solver(
            ny=2,
            nx=3,
            nz=5,
            bottom_temp=470.0,
            c=4e-4,
            A=8e-7,
            max_picard=50,
            picard_tol=1e-12,
            cg_rtol=1e-12,
        )
        T_bot = T_bot + np.array([[0.0, 3.0, -2.0], [4.0, -1.0, 2.0]])
        K0 = solver.K0_ref * np.linspace(0.95, 1.05, solver.nz)
        T = solver.solve(T_bot, K0=K0, reuse_last=False)
        vector = np.linspace(-0.4, 0.7, solver._n_unknowns)
        direction_T = np.linspace(-0.3, 0.5, T.size).reshape(T.shape)
        direction_T[:, :, 0] = 0.0
        direction_K0 = np.linspace(-0.2, 0.1, K0.size).reshape(K0.shape)
        grad_T, grad_K0 = solver._residual_vjp(T, K0, vector)

        def weighted_residual(T_value: np.ndarray, K0_value: np.ndarray) -> float:
            matrix, rhs = solver.build_frozen_system(
                T_value,
                T_value[:, :, -1],
                K0=K0_value,
            )
            return float(vector @ (matrix @ T_value[:, :, 1:-1].ravel() - rhs))

        eps = 1e-5
        finite_difference = (
            weighted_residual(T + eps * direction_T, K0 + eps * direction_K0)
            - weighted_residual(T - eps * direction_T, K0 - eps * direction_K0)
        ) / (2.0 * eps)
        vjp = float(np.sum(grad_T * direction_T) + np.sum(grad_K0 * direction_K0))
        np.testing.assert_allclose(vjp, finite_difference, rtol=2e-7, atol=1e-10)

    def test_nonlinear_adjoint_matches_finite_difference(self) -> None:
        solver, T_bot = make_solver(
            ny=2,
            nx=2,
            nz=5,
            bottom_temp=460.0,
            c=5e-4,
            A=1e-6,
            max_picard=60,
            picard_tol=1e-12,
            cg_rtol=1e-12,
        )
        T_bot = T_bot + np.array([[0.0, 4.0], [-3.0, 2.0]])
        K0 = solver.K0_ref * 1.03
        A = solver.A_ref * 0.9
        weight_T = np.linspace(-0.5, 0.7, np.prod(solver.field_shape)).reshape(solver.field_shape) / np.prod(solver.field_shape)
        weight_q = np.array([[0.3, -0.2], [0.1, 0.4]]) * 0.01 / 4.0

        T = solver.solve(T_bot, K0=K0, A_source=A, reuse_last=False)
        grad_T_bot, grad_K0, grad_A = solver.adjoint_gradient(
            T,
            K0,
            grad_T=weight_T,
            grad_q=weight_q,
        )

        def objective(T_bot_value: np.ndarray, K0_value: np.ndarray, A_value: np.ndarray) -> float:
            T_value = solver.solve(
                T_bot_value,
                K0=K0_value,
                A_source=A_value,
                reuse_last=False,
            )
            return float(np.sum(weight_T * T_value) + np.sum(weight_q * solver.surface_heat_flow(T_value, K0_value)))

        checks = [
            ("T_bot", (0, 1), 1e-3, grad_T_bot[0, 1]),
            ("K0", (1, 0, 2), 1e-5, grad_K0[1, 0, 2]),
            ("A", (0, 1, 2), 1e-10, grad_A[0, 1, 2]),
        ]
        for name, index, eps, adjoint in checks:
            plus = [T_bot.copy(), K0.copy(), A.copy()]
            minus = [T_bot.copy(), K0.copy(), A.copy()]
            position = {"T_bot": 0, "K0": 1, "A": 2}[name]
            plus[position][index] += eps
            minus[position][index] -= eps
            finite_difference = (objective(*plus) - objective(*minus)) / (2.0 * eps)
            np.testing.assert_allclose(adjoint, finite_difference, rtol=2e-5, atol=2e-7)


if __name__ == "__main__":
    unittest.main()
