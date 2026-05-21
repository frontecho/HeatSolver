from __future__ import annotations

import unittest

import numpy as np
import torch

from data_loading import nearest_point_grid
from lithosphere_model import THERMAL_LAYER_NAMES
from prior_network import MultiSourcePriorNet, PriorParameterModel, _normalize_channels
from tests.helpers import make_solver


DECODED_OUTPUT_KEYS = {
    "theta_raw",
    "T_bot_delta",
    "K_log_xy",
    "A_log_xy",
}


class PriorNetworkUtilityTest(unittest.TestCase):
    def test_normalize_channels_handles_nan_and_constant_channel(self) -> None:
        stack = np.array([
            [[1.0, np.nan], [3.0, 5.0]],
            [[2.0, 2.0], [2.0, 2.0]],
        ])
        out = _normalize_channels(stack)
        self.assertTrue(np.all(np.isfinite(out)))
        np.testing.assert_allclose(out[1], np.zeros((2, 2)))
        self.assertAlmostEqual(float(np.mean(out[0])), 0.0)

    def test_nearest_point_grid_handles_empty_and_valid_sources(self) -> None:
        lon = np.array([[0.0, 1.0], [0.0, 1.0]])
        lat = np.array([[0.0, 0.0], [1.0, 1.0]])
        empty = nearest_point_grid(lon, lat, np.array([np.nan]), np.array([0.0]), np.array([1.0]))
        self.assertTrue(np.all(np.isnan(empty)))
        out = nearest_point_grid(lon, lat, np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([10.0, 20.0]))
        self.assertEqual(out[0, 0], 10.0)
        self.assertEqual(out[1, 1], 20.0)


class MultiSourcePriorNetTest(unittest.TestCase):
    def test_output_shape_and_finite_values(self) -> None:
        torch.manual_seed(0)
        net = MultiSourcePriorNet(in_channels=3, theta_dim=7, width=4, dropout=0.0).eval()
        x = torch.randn(2, 3, 5, 4)
        out = net(x)
        self.assertEqual(out.shape, (2, 7))
        self.assertTrue(torch.all(torch.isfinite(out)))


class PriorParameterModelTest(unittest.TestCase):
    def assert_decoded_keys_absent(self, out: dict[str, object]) -> None:
        for key in DECODED_OUTPUT_KEYS:
            self.assertNotIn(key, out)

    def test_requires_features(self) -> None:
        model = PriorParameterModel(3, 4, in_channels=2, bottom_x=2, bottom_y=2, width=4)
        with self.assertRaises(ValueError):
            model()

    def test_fixed_property_output_when_property_modes_are_fixed(self) -> None:
        torch.manual_seed(1)
        model = PriorParameterModel(3, 4, in_channels=2, bottom_x=2, bottom_y=2, width=4).eval()
        out = model(torch.randn(1, 2, 3, 4))
        self.assertEqual(out["bottom_coeff"].shape, (4,))
        self.assertEqual(out["K_coeff"].numel(), 0)
        self.assertEqual(out["A_coeff"].numel(), 0)
        self.assert_decoded_keys_absent(out)

    def test_full_mode_output_shapes(self) -> None:
        torch.manual_seed(6)
        model = PriorParameterModel(
            3,
            4,
            in_channels=2,
            width=4,
            bottom_mode="full",
            k_mode="full",
            a_mode="full",
        ).eval()
        out = model(torch.randn(1, 2, 3, 4))
        solver, _ = make_solver(ny=3, nx=4, nz=5)
        _, _, _, decoded = model.decode_configured_parameter_fields(solver, out, T_ref=1300.0, dtype=torch.float64, device=torch.device("cpu"))
        expected_property_params = len(THERMAL_LAYER_NAMES) + len(THERMAL_LAYER_NAMES) * 3 * 4
        expected_theta_dim = 3 * 4 + 2 * expected_property_params

        self.assertEqual(decoded["T_bot_delta"].shape, (3, 4))
        self.assertEqual(decoded["K_log_xy"].shape, (len(THERMAL_LAYER_NAMES), 3, 4))
        self.assertEqual(decoded["A_log_xy"].shape, (len(THERMAL_LAYER_NAMES), 3, 4))
        self.assertEqual(model.theta_dim, expected_theta_dim)
        self.assertEqual(out["bottom_coeff"].numel(), 3 * 4)
        self.assertEqual(out["K_coeff"].numel(), expected_property_params)
        self.assertEqual(out["A_coeff"].numel(), expected_property_params)
        self.assert_decoded_keys_absent(out)

    def test_dct_property_output_shapes(self) -> None:
        torch.manual_seed(2)
        model = PriorParameterModel(
            3,
            4,
            in_channels=2,
            bottom_x=2,
            bottom_y=2,
            k_property_x=2,
            k_property_y=2,
            a_property_x=2,
            a_property_y=2,
            width=4,
            k_mode="dct",
            a_mode="dct",
        ).eval()
        out = model(torch.randn(1, 2, 3, 4))
        solver, _ = make_solver(ny=3, nx=4, nz=5)
        _, _, _, decoded = model.decode_configured_parameter_fields(solver, out, T_ref=1300.0, dtype=torch.float64, device=torch.device("cpu"))
        expected_property_params = len(THERMAL_LAYER_NAMES) + len(THERMAL_LAYER_NAMES) * 4
        expected_theta_dim = 4 + 2 * expected_property_params
        self.assertEqual(decoded["T_bot_delta"].shape, (3, 4))
        self.assertEqual(decoded["K_log_xy"].shape, (len(THERMAL_LAYER_NAMES), 3, 4))
        self.assertEqual(decoded["A_log_xy"].shape, (len(THERMAL_LAYER_NAMES), 3, 4))
        self.assertEqual(model.theta_dim, expected_theta_dim)
        self.assertEqual(out["K_coeff"].numel(), expected_property_params)
        self.assertEqual(out["A_coeff"].numel(), expected_property_params)
        self.assert_decoded_keys_absent(out)

    def test_k_dct_a_fixed_output_shapes(self) -> None:
        torch.manual_seed(4)
        model = PriorParameterModel(
            3,
            4,
            in_channels=2,
            bottom_x=2,
            bottom_y=2,
            k_property_x=2,
            k_property_y=2,
            width=4,
            k_mode="dct",
            a_mode="fixed",
        ).eval()
        out = model(torch.randn(1, 2, 3, 4))
        solver, _ = make_solver(ny=3, nx=4, nz=5)
        _, _, _, decoded = model.decode_configured_parameter_fields(solver, out, T_ref=1300.0, dtype=torch.float64, device=torch.device("cpu"))
        expected_property_params = len(THERMAL_LAYER_NAMES) + len(THERMAL_LAYER_NAMES) * 4
        expected_theta_dim = 4 + expected_property_params

        self.assertEqual(decoded["K_log_xy"].shape, (len(THERMAL_LAYER_NAMES), 3, 4))
        self.assertEqual(decoded["A_log_xy"].shape, (len(THERMAL_LAYER_NAMES), 3, 4))
        self.assertTrue(torch.all(decoded["A_log_xy"] == 0.0))
        self.assertEqual(model.theta_dim, expected_theta_dim)
        self.assertEqual(out["K_coeff"].numel(), expected_property_params)
        self.assertEqual(out["A_coeff"].numel(), 0)
        self.assert_decoded_keys_absent(out)

    def test_k_fixed_a_dct_output_shapes(self) -> None:
        torch.manual_seed(5)
        model = PriorParameterModel(
            3,
            4,
            in_channels=2,
            bottom_x=2,
            bottom_y=2,
            a_property_x=2,
            a_property_y=2,
            width=4,
            k_mode="fixed",
            a_mode="dct",
        ).eval()
        out = model(torch.randn(1, 2, 3, 4))
        solver, _ = make_solver(ny=3, nx=4, nz=5)
        _, _, _, decoded = model.decode_configured_parameter_fields(solver, out, T_ref=1300.0, dtype=torch.float64, device=torch.device("cpu"))
        expected_property_params = len(THERMAL_LAYER_NAMES) + len(THERMAL_LAYER_NAMES) * 4
        expected_theta_dim = 4 + expected_property_params

        self.assertEqual(decoded["K_log_xy"].shape, (len(THERMAL_LAYER_NAMES), 3, 4))
        self.assertTrue(torch.all(decoded["K_log_xy"] == 0.0))
        self.assertEqual(decoded["A_log_xy"].shape, (len(THERMAL_LAYER_NAMES), 3, 4))
        self.assertEqual(model.theta_dim, expected_theta_dim)
        self.assertEqual(out["K_coeff"].numel(), 0)
        self.assertEqual(out["A_coeff"].numel(), expected_property_params)
        self.assert_decoded_keys_absent(out)

    def test_checkpoint_relevant_property_parameter_counts(self) -> None:
        model = PriorParameterModel(
            10,
            10,
            in_channels=2,
            bottom_x=3,
            bottom_y=3,
            k_property_x=6,
            k_property_y=6,
            a_property_x=8,
            a_property_y=8,
            width=32,
            k_mode="dct",
            a_mode="dct",
        )
        self.assertEqual(model.n_k_params, 185)
        self.assertEqual(model.n_a_params, 325)
        self.assertEqual(model.theta_dim, model.n_bottom_field + 185 + 325)
        self.assertEqual(model.prior_net.head[-1].weight.shape, (model.theta_dim, 128))


if __name__ == "__main__":
    unittest.main()
