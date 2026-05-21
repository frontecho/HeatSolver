from __future__ import annotations

import unittest

import numpy as np
import torch

from direct_parameter import DirectParameterModel
from lithosphere_model import THERMAL_LAYER_NAMES
from tests.helpers import make_solver


DECODED_OUTPUT_KEYS = {
    "T_bot_delta",
    "K_log_xy",
    "A_log_xy",
}


class DirectParameterModelTest(unittest.TestCase):
    def assert_decoded_keys_absent(self, out: dict[str, object]) -> None:
        for key in DECODED_OUTPUT_KEYS:
            self.assertNotIn(key, out)

    def test_forward_shapes_for_fixed_dct_and_full_modes(self) -> None:
        fixed = DirectParameterModel(2, 3, bottom_mode="fixed", k_mode="fixed", a_mode="fixed")
        fixed_out = fixed()
        self.assertEqual(fixed_out["bottom_coeff"].numel(), 0)
        self.assertEqual(fixed.bottom_mode, "fixed")
        self.assert_decoded_keys_absent(fixed_out)

        dct = DirectParameterModel(2, 3, bottom_x=2, bottom_y=2, bottom_mode="dct", k_mode="dct", a_mode="dct", alpha_scale=200.0)
        dct.raw_bottom_field.data.fill_(1.0)
        dct_out = dct()
        solver, _ = make_solver(ny=2, nx=3)
        _, _, _, decoded = dct.decode_configured_parameter_fields(solver, dct_out, T_ref=1300.0, dtype=torch.float64, device=torch.device("cpu"))
        self.assertEqual(decoded["K_log_xy"].shape, (len(THERMAL_LAYER_NAMES), 2, 3))
        self.assertEqual(dct.bottom_dct.n_params, 4)
        self.assertEqual(dct.alpha_scale, 200.0)
        self.assertGreater(float(torch.max(torch.abs(decoded["T_bot_delta"])).detach()), 1.0)
        self.assertGreater(dct_out["K_coeff"].numel(), 0)
        self.assert_decoded_keys_absent(dct_out)

        full = DirectParameterModel(2, 3, bottom_mode="full", k_mode="full", a_mode="full", alpha_scale=150.0)
        full.raw_bottom_field.data.fill_(1.0)
        full_out = full()
        solver, _ = make_solver(ny=2, nx=3)
        _, _, _, decoded = full.decode_configured_parameter_fields(solver, full_out, T_ref=1300.0, dtype=torch.float64, device=torch.device("cpu"))
        self.assertEqual(decoded["T_bot_delta"].shape, (2, 3))
        self.assertAlmostEqual(float(torch.max(decoded["T_bot_delta"]).detach()), 150.0 * np.tanh(1.0))
        self.assertEqual(decoded["K_log_xy"].shape, (len(THERMAL_LAYER_NAMES), 2, 3))
        self.assertEqual(full_out["K_coeff"].numel(), len(THERMAL_LAYER_NAMES) * 2 * 3)
        self.assertEqual(full_out["A_coeff"].numel(), len(THERMAL_LAYER_NAMES) * 2 * 3)
        self.assertEqual(full.raw_bottom_field.numel(), 6)
        self.assert_decoded_keys_absent(full_out)

    def test_full_property_modes_do_not_require_nz(self) -> None:
        model = DirectParameterModel(2, 2, k_mode="full", a_mode="full")
        out = model()
        self.assertEqual(out["K_coeff"].numel(), len(THERMAL_LAYER_NAMES) * 2 * 2)
        self.assertEqual(out["A_coeff"].numel(), len(THERMAL_LAYER_NAMES) * 2 * 2)


if __name__ == "__main__":
    unittest.main()
