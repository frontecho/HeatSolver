from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from inversion import (
    CHECKPOINT_FORMAT_VERSION,
    _coefficient_regularization,
    evaluate_objective,
    load_checkpoint,
    load_initial_checkpoint,
    run_inversion,
    save_checkpoint,
    verify_input_hashes,
)
from parameterization import DirectParameterization
from tests.helpers import make_solver


class DirectObjectiveTest(unittest.TestCase):
    def test_regularization_matches_field_energy_across_modes(self) -> None:
        dct = DirectParameterization(
            2,
            3,
            bottom_mode="dct",
            k_mode="fixed",
            a_mode="fixed",
            bottom_dct_x=1,
            bottom_dct_y=1,
        )
        full = DirectParameterization(
            2,
            3,
            bottom_mode="full",
            k_mode="fixed",
            a_mode="fixed",
        )
        weights = {"bottom": 1.0, "K": 0.0, "A": 0.0}
        dct_value, _, _ = _coefficient_regularization(
            np.array([2.0]),
            dct,
            weights,
        )
        full_value, _, _ = _coefficient_regularization(
            np.full(full.size, 2.0),
            full,
            weights,
        )
        self.assertEqual(dct_value, full_value)

    def test_parameter_gradient_matches_directional_difference(self) -> None:
        solver, truth_bottom = make_solver(
            ny=2,
            nx=2,
            nz=5,
            bottom_temp=480.0,
            c=5e-4,
            A=1e-6,
            max_picard=60,
            picard_tol=1e-12,
            cg_rtol=1e-12,
        )
        truth_temperature = solver.solve(truth_bottom, reuse_last=False)
        q_obs = solver.surface_heat_flow(truth_temperature)
        model = DirectParameterization(
            2,
            2,
            bottom_mode="dct",
            k_mode="dct",
            a_mode="dct",
            bottom_dct_x=1,
            bottom_dct_y=1,
            k_dct_x=1,
            k_dct_y=1,
            a_dct_x=1,
            a_dct_y=1,
        )
        rng = np.random.default_rng(7)
        parameters = rng.normal(0.0, 0.02, model.size)
        direction = rng.normal(size=model.size)
        _, gradient, _ = evaluate_objective(
            solver,
            q_obs,
            model,
            parameters,
            T_ref=450.0,
            lam_bottom=0.03,
            lam_k=0.04,
            lam_a=0.05,
        )

        def objective(x: np.ndarray) -> float:
            solver.clear_warm_start()
            value, _, _ = evaluate_objective(
                solver,
                q_obs,
                model,
                x,
                T_ref=450.0,
                lam_bottom=0.03,
                lam_k=0.04,
                lam_a=0.05,
            )
            return value

        eps = 2e-6
        finite_difference = (
            objective(parameters + eps * direction)
            - objective(parameters - eps * direction)
        ) / (2.0 * eps)
        np.testing.assert_allclose(gradient @ direction, finite_difference, rtol=3e-5, atol=2e-6)

    def test_lbfgsb_reduces_synthetic_misfit(self) -> None:
        solver, truth_bottom = make_solver(
            ny=2,
            nx=2,
            nz=5,
            bottom_temp=540.0,
            max_picard=10,
            picard_tol=1e-12,
            cg_rtol=1e-12,
        )
        q_obs = solver.surface_heat_flow(solver.solve(truth_bottom, reuse_last=False))
        solver.clear_warm_start()
        model = DirectParameterization(
            2,
            2,
            bottom_mode="dct",
            k_mode="fixed",
            a_mode="fixed",
            bottom_dct_x=1,
            bottom_dct_y=1,
            bottom_bound_C=200.0,
        )
        result = run_inversion(
            solver,
            q_obs,
            model,
            T_ref=500.0,
            maxiter=12,
            verbose=False,
        )

        self.assertLess(result["metrics"]["data_misfit"], result["history"][0]["data_misfit"] * 1e-6)
        np.testing.assert_allclose(result["T_bot"], truth_bottom, atol=1e-4)
        self.assertLessEqual(result["pde_evaluations"], result["optimization"]["function_evaluations"] + 1)

    def test_fixed_model_is_evaluated_without_optimizer(self) -> None:
        solver, T_bot = make_solver(ny=2, nx=2, nz=4, bottom_temp=500.0)
        q_obs = solver.surface_heat_flow(solver.solve(T_bot))
        model = DirectParameterization(
            2,
            2,
            bottom_mode="fixed",
            k_mode="fixed",
            a_mode="fixed",
        )
        result = run_inversion(solver, q_obs, model, T_ref=500.0, verbose=False)

        self.assertEqual(result["parameters"].size, 0)
        self.assertEqual(result["optimization"]["message"], "fixed model")
        self.assertAlmostEqual(result["final_loss"], 0.0, places=12)


class CheckpointTest(unittest.TestCase):
    def test_input_hash_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            verify_input_hashes({"grid.csv": "old"}, {"grid.csv": "new"})

    def test_npz_checkpoint_round_trip(self) -> None:
        parameters = np.array([1.0, -2.0, 3.5])
        metadata = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "data": {"t_ref": 1225.0},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.npz"
            save_checkpoint(path, parameters, metadata)
            loaded_parameters, loaded_metadata = load_checkpoint(path)

        np.testing.assert_array_equal(loaded_parameters, parameters)
        self.assertEqual(loaded_metadata, metadata)

    def test_checkpoint_rejects_old_format(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.npz"
            save_checkpoint(path, np.zeros(1), {"format_version": 1})
            with self.assertRaises(ValueError):
                load_checkpoint(path)

    def test_initial_checkpoint_requires_exact_parameterization(self) -> None:
        data = {"stride": 1}
        first = DirectParameterization(
            2,
            2,
            bottom_mode="dct",
            k_mode="fixed",
            a_mode="fixed",
            bottom_dct_x=2,
            bottom_dct_y=2,
        )
        second = DirectParameterization(
            2,
            2,
            bottom_mode="full",
            k_mode="fixed",
            a_mode="fixed",
        )
        metadata = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "data": data,
            "parameterization": first.config(),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.npz"
            save_checkpoint(path, first.initial(), metadata)
            with self.assertRaises(ValueError):
                load_initial_checkpoint(path, data, second.config())


if __name__ == "__main__":
    unittest.main()
