from __future__ import annotations

import unittest

import numpy as np
import torch

from direct_parameter import DirectParameterModel
from inversion import (
    _build_optimizer,
    _tensor_to_numpy,
    run_inversion,
)
from tests.helpers import make_solver


class InversionHelperTest(unittest.TestCase):
    def test_tensor_to_numpy_handles_tensor_and_missing_values(self) -> None:
        arr = _tensor_to_numpy(torch.tensor([1.0, 2.0]))
        np.testing.assert_allclose(arr, np.array([1.0, 2.0]))
        self.assertEqual(_tensor_to_numpy(None).shape, (0,))

    def test_build_optimizer_and_reject_invalid_name(self) -> None:
        param = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
        self.assertIsInstance(_build_optimizer("adamw", [param], lr=0.1, weight_decay=0.0), torch.optim.AdamW)
        self.assertIsInstance(_build_optimizer("lbfgs", [param], lr=0.1, weight_decay=0.0), torch.optim.LBFGS)
        with self.assertRaises(ValueError):
            _build_optimizer("sgd", [param], lr=0.1, weight_decay=0.0)


class RunInversionTest(unittest.TestCase):
    def test_no_valid_observations_raise(self) -> None:
        solver, _ = make_solver(ny=2, nx=2, nz=4)
        model = DirectParameterModel(2, 2, bottom_mode="fixed", k_mode="fixed", a_mode="fixed")
        with self.assertRaises(ValueError):
            run_inversion(solver, np.full((2, 2), np.nan), model, verbose=False)

    def test_fixed_model_evaluates_without_trainable_parameters(self) -> None:
        solver, T_bot = make_solver(ny=2, nx=2, nz=4, bottom_temp=500.0)
        baseline_T = solver.solve(T_bot)
        q_obs = solver.surface_heat_flow_upward_np(baseline_T)
        solver.clear_warm_start()
        model = DirectParameterModel(2, 2, bottom_mode="fixed", k_mode="fixed", a_mode="fixed")
        out = run_inversion(solver, q_obs, model, T_ref=500.0, steps=1, optimizer_name="adamw", verbose=False)

        self.assertGreaterEqual(len(out["history"]), 1)
        self.assertTrue(np.isfinite(out["final_loss"]))
        self.assertAlmostEqual(out["history"][-1]["data_misfit"], 0.5 * out["history"][-1]["rmse"] ** 2)
        self.assertEqual(out["T"].shape, (2, 2, 4))
        self.assertEqual(out["q_pred"].shape, (2, 2))
        self.assertNotIn("parameterization", out)

    def test_one_step_adamw_direct_dct_bottom(self) -> None:
        solver, T_bot = make_solver(ny=2, nx=2, nz=4, bottom_temp=500.0)
        baseline_T = solver.solve(T_bot)
        q_obs = solver.surface_heat_flow_upward_np(baseline_T)
        solver.clear_warm_start()
        model = DirectParameterModel(2, 2, bottom_x=1, bottom_y=1, bottom_mode="dct", k_mode="fixed", a_mode="fixed")
        out = run_inversion(
            solver,
            q_obs,
            model,
            T_ref=500.0,
            steps=1,
            lr=0.01,
            optimizer_name="adamw",
            history_every=1,
            diagnose_gradients=True,
            verbose=False,
        )

        self.assertGreaterEqual(len(out["history"]), 2)
        self.assertTrue(np.isfinite(out["final_rmse"]))
        self.assertEqual(out["alpha_bot"].shape, (1,))
        self.assertEqual(out["valid_mask"].shape, (2, 2))
        self.assertFalse(out["early_stopped"])
        self.assertIn("grad_data_bottom", out["history"][-1])

    def test_early_stop_restores_best_model(self) -> None:
        solver, T_bot = make_solver(ny=2, nx=2, nz=4, bottom_temp=500.0)
        baseline_T = solver.solve(T_bot)
        q_obs = solver.surface_heat_flow_upward_np(baseline_T)
        solver.clear_warm_start()
        model = DirectParameterModel(2, 2, bottom_x=1, bottom_y=1, bottom_mode="dct", k_mode="fixed", a_mode="fixed")
        out = run_inversion(
            solver,
            q_obs,
            model,
            T_ref=500.0,
            steps=4,
            lr=0.01,
            optimizer_name="adamw",
            history_every=1,
            early_stop_patience=1,
            early_stop_min_delta=1e9,
            verbose=False,
        )

        self.assertTrue(out["early_stopped"])
        self.assertEqual(out["early_stop_step"], 1)
        self.assertEqual(out["early_stop_best_step"], 0)
        self.assertEqual(len(out["history"]), 2)
        self.assertAlmostEqual(out["final_loss"], out["history"][0]["loss"])


if __name__ == "__main__":
    unittest.main()
