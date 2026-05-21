from __future__ import annotations

import unittest

import numpy as np
import torch

from heat_solver import HeatSolver
from tests.helpers import make_layer_masks, make_solver


class HeatSolverValidationTest(unittest.TestCase):
    def base_kwargs(self) -> dict:
        solver, _ = make_solver(ny=3, nx=3, nz=5)
        return {
            "x": solver.x.copy(),
            "y": solver.y.copy(),
            "z_nodes_km": solver.z_nodes_km.copy(),
            "surface_temp": solver.surface_temp.copy(),
            "K0_ref": solver.K0_ref.copy(),
            "c_ref": solver.c_ref.copy(),
            "A_ref": solver.A_ref.copy(),
            "layer_masks": {name: mask.copy() for name, mask in solver.layer_masks.items()},
        }

    def test_rejects_bad_shapes_and_nonfinite_values(self) -> None:
        kwargs = self.base_kwargs()
        bad = dict(kwargs)
        bad["surface_temp"] = np.zeros((3, 3, 1))
        with self.assertRaises(ValueError):
            HeatSolver(**bad)

        bad = dict(kwargs)
        bad["x"] = np.zeros((2, 3))
        with self.assertRaises(ValueError):
            HeatSolver(**bad)

        bad = dict(kwargs)
        bad["K0_ref"] = np.zeros((3, 3, 4))
        with self.assertRaises(ValueError):
            HeatSolver(**bad)

        bad = dict(kwargs)
        bad["K0_ref"] = kwargs["K0_ref"].copy()
        bad["K0_ref"][0, 0, 0] = np.inf
        with self.assertRaises(ValueError):
            HeatSolver(**bad)

    def test_rejects_bad_vertical_and_horizontal_coordinates(self) -> None:
        kwargs = self.base_kwargs()
        bad = dict(kwargs)
        bad["z_nodes_km"] = np.array([0.0, 1.0])
        with self.assertRaises(ValueError):
            HeatSolver(**bad)

        bad = dict(kwargs)
        bad["z_nodes_km"] = np.array([0.0, 1.0, 3.0, 4.0, 5.0])
        with self.assertRaises(ValueError):
            HeatSolver(**bad)

        bad = dict(kwargs)
        bad["x"] = np.zeros_like(kwargs["x"])
        with self.assertRaises(ValueError):
            HeatSolver(**bad)

    def test_rejects_bad_layer_mask_shape(self) -> None:
        kwargs = self.base_kwargs()
        bad = dict(kwargs)
        bad["layer_masks"] = {"sediment": np.zeros((2, 2, 2), dtype=bool)}
        with self.assertRaises(ValueError):
            HeatSolver(**bad)


class HeatSolverUtilitiesTest(unittest.TestCase):
    def test_conductivity_clamps_small_denominator(self) -> None:
        K0 = np.array([2.0])
        c = np.array([1.0])
        T = np.array([-2.0])
        out = HeatSolver._conductivity_from_temperature(K0, c, T)
        self.assertEqual(float(out[0]), 20.0)

    def test_harmonic_mean_handles_tiny_denominator(self) -> None:
        out = HeatSolver._harmonic_mean(np.array([1.0, 0.0]), np.array([3.0, 0.0]))
        self.assertAlmostEqual(float(out[0]), 1.5)
        self.assertEqual(float(out[1]), 0.0)

    def test_surface_gradient_matches_linear_profile(self) -> None:
        T = np.zeros((2, 2, 5), dtype=float)
        for k in range(5):
            T[:, :, k] = 10.0 + 2.0 * k
        grad = HeatSolver._surface_temperature_gradient_np(T, dz=1000.0)
        np.testing.assert_allclose(grad, np.full((2, 2), 2.0 / 1000.0))


class HeatSolverSolveTest(unittest.TestCase):
    def test_no_source_solution_is_linear_and_heat_flow_is_analytical(self) -> None:
        solver, T_bot = make_solver(ny=3, nx=3, nz=5, surface_temp=10.0, bottom_temp=410.0, K0=2.5, A=0.0)
        T = solver.solve(T_bot)
        q = solver.surface_heat_flow_upward_np(T)

        expected = np.linspace(10.0, 410.0, solver.nz)
        np.testing.assert_allclose(T[1, 1, :], expected, atol=1e-5)
        np.testing.assert_allclose(q, np.full((3, 3), 2.5 * (400.0 / 4000.0) * 1000.0), atol=1e-5)
        self.assertTrue(solver.last_solve_info["converged"])
        self.assertEqual(solver.last_solve_info["solve_id"], 1)

        solver.solve(T_bot)
        self.assertEqual(solver.last_solve_info["solve_id"], 2)
        solver.clear_warm_start()
        self.assertIsNone(solver._last_T)
        self.assertIsNone(solver.last_solve_info)

    def test_build_frozen_system_shapes(self) -> None:
        solver, T_bot = make_solver(ny=2, nx=3, nz=5)
        T_init = solver.solve(T_bot)
        mat, rhs = solver.build_frozen_system(T_init, T_bot)
        self.assertEqual(mat.shape, (2 * 3 * 3, 2 * 3 * 3))
        self.assertEqual(rhs.shape, (2 * 3 * 3,))

    def test_constant_source_solution_matches_quadratic_profile_and_heat_flow(self) -> None:
        solver, T_bot = make_solver(ny=2, nx=2, nz=7, surface_temp=10.0, bottom_temp=410.0, K0=2.5, A=1e-6)
        T = solver.solve(T_bot)
        q = solver.surface_heat_flow_upward_np(T)

        z = solver.z_nodes_km * 1000.0
        depth = float(z[-1])
        source = 1e-6
        conductivity = 2.5
        gradient_top = (400.0 / depth) + source * depth / (2.0 * conductivity)
        expected = 10.0 + gradient_top * z - source * z * z / (2.0 * conductivity)

        np.testing.assert_allclose(T[0, 0, :], expected, atol=1e-5)
        np.testing.assert_allclose(q, np.full((2, 2), conductivity * gradient_top * 1000.0), atol=1e-5)

    def test_temperature_dependent_conductivity_picard_residual_is_small(self) -> None:
        solver, T_bot = make_solver(ny=2, nx=2, nz=6, bottom_temp=520.0, K0=3.0, c=5e-4, A=1e-6, max_picard=20, picard_tol=1e-8)
        T = solver.solve(T_bot)
        mat, rhs = solver.build_frozen_system(T, T_bot)
        interior = T[:, :, 1:-1].reshape(-1)
        residual = mat @ interior - rhs

        self.assertTrue(solver.last_solve_info["converged"])
        self.assertGreater(solver.last_solve_info["picard_iters"], 1)
        self.assertLess(np.linalg.norm(residual) / (np.linalg.norm(rhs) + 1e-30), 1e-7)
        self.assertTrue(np.all(np.isfinite(T)))

    def test_custom_fields_temperature_initialization_and_warm_start_paths(self) -> None:
        solver, T_bot = make_solver(ny=2, nx=2, nz=5, bottom_temp=430.0, A=0.0, max_picard=1)
        reference_T = solver.solve(T_bot, reuse_last=False)
        custom_A = np.full_like(solver.A_ref, 1e-6)
        custom_T = solver.solve(T_bot, A_source=custom_A, reuse_last=False)
        self.assertGreater(float(custom_T[0, 0, 2]), float(reference_T[0, 0, 2]))

        q_ref = solver.surface_heat_flow_upward_np(reference_T)
        q_custom_k = solver.surface_heat_flow_upward_np(reference_T, K0=2.0 * solver.K0_ref)
        np.testing.assert_allclose(q_custom_k, 2.0 * q_ref, atol=1e-8)

        captured: list[np.ndarray] = []
        original_build = solver._build_matrix_and_rhs

        def capture_initial_iter(K0: np.ndarray, A_source: np.ndarray, T_iter: np.ndarray, T_bot_in: np.ndarray):
            captured.append(T_iter.copy())
            return original_build(K0, A_source, T_iter, T_bot_in)

        solver._build_matrix_and_rhs = capture_initial_iter  # type: ignore[method-assign]
        T_init = np.full((solver.ny, solver.nx, solver.nz), 123.0)
        solver.solve(T_bot, T_init=T_init, reuse_last=True)
        np.testing.assert_allclose(captured[-1][:, :, 1:-1], 123.0)

        solver._last_T = np.full((solver.ny, solver.nx, solver.nz), 321.0)
        solver.solve(T_bot, reuse_last=True)
        np.testing.assert_allclose(captured[-1][:, :, 1:-1], 321.0)

        solver.solve(T_bot, reuse_last=False)
        self.assertFalse(np.allclose(captured[-1][:, :, 1:-1], 321.0))


class HeatSolverAutogradTest(unittest.TestCase):
    def test_forward_backpropagates_finite_bottom_gradient(self) -> None:
        solver, T_bot_np = make_solver(ny=2, nx=2, nz=4, bottom_temp=400.0, max_picard=3)
        T_bot = torch.tensor(T_bot_np, dtype=torch.float64, requires_grad=True)
        T, q = solver(T_bot)
        loss = T.mean() + 0.01 * q.mean()
        loss.backward()

        self.assertEqual(T.shape, (2, 2, 4))
        self.assertEqual(q.shape, (2, 2))
        self.assertIsNotNone(T_bot.grad)
        self.assertTrue(torch.all(torch.isfinite(T_bot.grad)))

    def test_pde_gradients_match_finite_difference(self) -> None:
        solver, T_bot_np = make_solver(ny=2, nx=2, nz=4, bottom_temp=420.0, A=1e-6, max_picard=3)
        T_bot = torch.tensor(T_bot_np + np.array([[0.0, 4.0], [-3.0, 2.0]]), dtype=torch.float64, requires_grad=True)
        K0 = torch.tensor(solver.K0_ref * 1.03, dtype=torch.float64, requires_grad=True)
        A = torch.tensor(solver.A_ref * 0.9, dtype=torch.float64, requires_grad=True)
        w_T = torch.linspace(-0.5, 0.7, solver.ny * solver.nx * solver.nz, dtype=torch.float64).reshape(solver.ny, solver.nx, solver.nz)
        w_q = torch.tensor([[0.3, -0.2], [0.1, 0.4]], dtype=torch.float64)

        def loss_fn(T_bot_in: torch.Tensor, K0_in: torch.Tensor, A_in: torch.Tensor) -> torch.Tensor:
            solver.clear_warm_start()
            T, q = solver(T_bot_in, K0=K0_in, A_source=A_in)
            return (T * w_T).mean() + 0.01 * (q * w_q).mean()

        loss = loss_fn(T_bot, K0, A)
        loss.backward()

        def finite_difference(which: str, index: tuple[int, ...], eps: float) -> float:
            T_bot_p = T_bot.detach().clone()
            T_bot_m = T_bot.detach().clone()
            K0_p = K0.detach().clone()
            K0_m = K0.detach().clone()
            A_p = A.detach().clone()
            A_m = A.detach().clone()
            if which == "T_bot":
                T_bot_p[index] += eps
                T_bot_m[index] -= eps
            elif which == "K0":
                K0_p[index] += eps
                K0_m[index] -= eps
            else:
                A_p[index] += eps
                A_m[index] -= eps
            with torch.no_grad():
                fp = float(loss_fn(T_bot_p, K0_p, A_p).cpu())
                fm = float(loss_fn(T_bot_m, K0_m, A_m).cpu())
            return (fp - fm) / (2.0 * eps)

        checks = [
            ("T_bot", (0, 1), T_bot.grad, 1e-3),
            ("K0", (0, 0, 1), K0.grad, 1e-5),
            ("A_source", (1, 0, 1), A.grad, 1e-9),
        ]
        for name, index, grad, eps in checks:
            self.assertIsNotNone(grad)
            autodiff = float(grad[index])
            fd = finite_difference(name, index, eps)
            self.assertAlmostEqual(autodiff, fd, delta=5e-3 * max(1.0, abs(autodiff), abs(fd)))

    def test_constructor_accepts_complete_layer_masks(self) -> None:
        solver, _ = make_solver(ny=2, nx=2, nz=5)
        self.assertEqual(set(solver.layer_masks), set(make_layer_masks(2, 2, 5)))

    def test_constructor_preserves_float_layer_weights(self) -> None:
        kwargs = HeatSolverValidationTest().base_kwargs()
        kwargs["layer_masks"] = {name: mask.astype(float) * 0.5 for name, mask in kwargs["layer_masks"].items()}
        solver = HeatSolver(**kwargs)
        self.assertTrue(np.issubdtype(solver.layer_masks["sediment"].dtype, np.floating))
        self.assertEqual(float(np.max(solver.layer_masks["sediment"])), 0.5)


if __name__ == "__main__":
    unittest.main()
