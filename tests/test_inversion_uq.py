from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from inversion import CHECKPOINT_FORMAT_VERSION, load_checkpoint, save_checkpoint
from inversion_uq import _finalize_moments, _update_moments, run_uq, split_observations
from parameterization import DirectParameterization
from tests.helpers import make_solver


class UQUtilitiesTest(unittest.TestCase):
    def test_split_is_deterministic_and_disjoint(self) -> None:
        q_obs = np.arange(20, dtype=float).reshape(4, 5)
        q_obs[0, 0] = np.nan
        first = split_observations(q_obs, 0.25, seed=4)
        second = split_observations(q_obs, 0.25, seed=4)

        for left, right in zip(first, second, strict=True):
            np.testing.assert_array_equal(left, right)
        _, train_mask, holdout_mask = first
        self.assertFalse(np.any(train_mask & holdout_mask))
        self.assertEqual(np.count_nonzero(train_mask) + np.count_nonzero(holdout_mask), 19)

    def test_online_moments_match_numpy(self) -> None:
        samples = [
            np.array([1.0, 4.0]),
            np.array([2.0, 6.0]),
            np.array([6.0, 8.0]),
        ]
        moments: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for count, sample in enumerate(samples, 1):
            _update_moments(moments, {"field": sample}, count)
        result = _finalize_moments(moments, len(samples))

        np.testing.assert_allclose(result["field_mean"], np.mean(samples, axis=0))
        np.testing.assert_allclose(result["field_std"], np.std(samples, axis=0, ddof=1))


class UQRunTest(unittest.TestCase):
    def test_ensemble_starts_independently_of_fitted_template(self) -> None:
        solver, T_bot = make_solver(ny=2, nx=2, nz=4, bottom_temp=500.0)
        q_obs = solver.surface_heat_flow(solver.solve(T_bot))
        model = DirectParameterization(
            2,
            2,
            bottom_mode="dct",
            k_mode="fixed",
            a_mode="fixed",
            bottom_dct_x=1,
            bottom_dct_y=1,
        )
        metadata = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "data": {
                "data_dir": "unused",
                "stride": 1,
                "dz_km": 1.0,
                "z_max_km": 3.0,
                "t_ref": 500.0,
                "layer_transition_xy_km": 0.0,
                "layer_transition_z_km": 0.0,
                "input_sha256": {},
            },
            "parameterization": model.config(),
            "objective": {
                "lam_bottom": 0.0,
                "lam_k": 0.0,
                "lam_a": 0.0,
            },
        }
        grid = SimpleNamespace(shape=(2, 2))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "checkpoint.npz"
            output_dir = root / "uq"
            save_checkpoint(checkpoint_path, np.array([20.0]), metadata)
            with patch("inversion_uq.build_problem", return_value=(grid, solver, q_obs, {})):
                summaries = run_uq(
                    checkpoint_path,
                    output_dir,
                    members=2,
                    holdout_fraction=0.25,
                    seed=10,
                    initial_noise_fraction=0.0,
                    maxiter=0,
                    verbose=False,
                )
            with np.load(output_dir / "aggregate_fields.npz") as fields:
                q_std = fields["q_pred_std"]
            member_parameters, member_metadata = load_checkpoint(
                output_dir / "members" / "member_000" / "checkpoint.npz"
            )

            self.assertTrue((output_dir / "manifest.json").exists())
            self.assertTrue((output_dir / "member_summaries.csv").exists())
            self.assertEqual(len(summaries), 2)
            np.testing.assert_allclose(q_std, 0.0)
            np.testing.assert_allclose(member_parameters, 0.0)
            self.assertIsNone(member_metadata["run"]["initial_checkpoint"])
            self.assertEqual(
                member_metadata["run"]["template_checkpoint"],
                str(checkpoint_path.resolve()),
            )
            self.assertTrue(all(row["holdout_rmse_mW_m2"] < 1e-10 for row in summaries))


if __name__ == "__main__":
    unittest.main()
