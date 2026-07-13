from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from evaluation import evaluate_checkpoint
from inversion import save_checkpoint
from parameterization import DirectParameterization
from tests.helpers import make_solver


class EvaluationTest(unittest.TestCase):
    def test_checkpoint_t_ref_is_used_to_rebuild_fixed_bottom(self) -> None:
        solver, _ = make_solver(ny=2, nx=2, nz=4)
        model = DirectParameterization(
            2,
            2,
            bottom_mode="fixed",
            k_mode="fixed",
            a_mode="fixed",
        )
        metadata = {
            "format_version": 1,
            "data": {
                "data_dir": "unused",
                "stride": 1,
                "dz_km": 1.0,
                "z_max_km": 3.0,
                "t_ref": 1225.0,
                "layer_transition_xy_km": 0.0,
                "layer_transition_z_km": 0.0,
            },
            "parameterization": model.config(),
            "objective": {},
        }
        grid = SimpleNamespace(shape=(2, 2))
        q_obs = np.zeros((2, 2))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint.npz"
            save_checkpoint(checkpoint_path, model.initial(), metadata)
            with patch("evaluation.build_problem", return_value=(grid, solver, q_obs)):
                _, _, fields, _, loaded_metadata = evaluate_checkpoint(checkpoint_path)

        np.testing.assert_allclose(fields["T_bot"], 1225.0)
        self.assertEqual(loaded_metadata, metadata)


if __name__ == "__main__":
    unittest.main()
