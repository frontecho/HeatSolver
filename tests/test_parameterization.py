from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from direct_parameter import DirectParameterModel
from lithosphere_model import THERMAL_LAYER_NAMES
from parameterization import DCTField, LayeredDCTField
from tests.helpers import make_layer_masks, make_solver


class DCTFieldTest(unittest.TestCase):
    def test_clamps_modes_and_rejects_invalid_dimensions(self) -> None:
        field = DCTField(3, 4, x=8, y=9)
        self.assertEqual(field.y, 3)
        self.assertEqual(field.x, 4)
        self.assertEqual(field.n_params, 12)
        with self.assertRaises(ValueError):
            DCTField(0, 4)

    def test_torch_decode_shape_and_zero_coefficients(self) -> None:
        field = DCTField(4, 3, x=2, y=2)
        alpha = torch.arange(field.n_params, dtype=torch.float64)
        actual = field.decode_torch(alpha, dtype=torch.float64, device=torch.device("cpu"))
        self.assertEqual(actual.shape, (4, 3))
        torch.testing.assert_close(field.decode_torch(torch.zeros(field.n_params), dtype=torch.float64, device=torch.device("cpu")), torch.zeros((4, 3), dtype=torch.float64))


class LayeredDCTFieldTest(unittest.TestCase):
    def test_layered_field_shape_and_counts(self) -> None:
        field = LayeredDCTField(5, 3, 4, x=3, y=2)
        coeffs = torch.arange(field.n_params, dtype=torch.float64)
        out = field.decode_torch(coeffs, dtype=torch.float64, device=torch.device("cpu"))
        self.assertEqual(field.n_params_per_layer, 6)
        self.assertEqual(field.n_params, 30)
        self.assertEqual(out.shape, (5, 3, 4))


class ApplyLogMultiplierTest(unittest.TestCase):
    def make_dummy_solver(self) -> SimpleNamespace:
        ny, nx, nz = 2, 2, 5
        return SimpleNamespace(
            ny=ny,
            nx=nx,
            K0_ref=np.full((ny, nx, nz), 2.0),
            A_ref=np.full((ny, nx, nz), 1e-6),
            layer_masks=make_layer_masks(ny, nx, nz),
        )

    def make_dummy_model(self) -> DirectParameterModel:
        return DirectParameterModel(2, 2, bottom_mode="fixed", k_mode="dct", a_mode="fixed")

    def test_fixed_mode_returns_reference_and_zero_summary(self) -> None:
        solver = self.make_dummy_solver()
        ref = np.full((2, 2, 5), 2.0)
        field, summary = self.make_dummy_model().apply_log_multiplier_torch(
            solver,
            ref,
            "fixed",
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        np.testing.assert_allclose(field.numpy(), ref)
        self.assertEqual(summary.shape, (len(THERMAL_LAYER_NAMES), 2, 2))
        self.assertTrue(torch.all(summary == 0.0))

    def test_dct_mode_requires_complete_masks_and_applies_layer_scaling(self) -> None:
        solver = self.make_dummy_solver()
        bad_solver = SimpleNamespace(ny=2, nx=2, layer_masks={})
        log_xy = torch.zeros((len(THERMAL_LAYER_NAMES), 2, 2), dtype=torch.float64)
        with self.assertRaises(ValueError):
            self.make_dummy_model().apply_log_multiplier_torch(
                bad_solver,
                np.ones((2, 2, 5)),
                "dct",
                log_xy=log_xy,
                dtype=torch.float64,
                device=torch.device("cpu"),
            )

        log_xy[0, :, :] = np.log(2.0)
        field, summary = self.make_dummy_model().apply_log_multiplier_torch(
            solver,
            np.ones((2, 2, 5)),
            "dct",
            log_xy=log_xy,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        sediment_mask = torch.as_tensor(solver.layer_masks["sediment"])
        self.assertTrue(torch.allclose(field[sediment_mask], torch.full_like(field[sediment_mask], 2.0)))
        self.assertEqual(summary.shape, log_xy.shape)

    def test_dct_mode_blends_float_layer_weights(self) -> None:
        solver = self.make_dummy_solver()
        soft = {name: mask.astype(float) * 0.0 for name, mask in solver.layer_masks.items()}
        soft["sediment"][:, :, :] = 0.25
        soft["upper_crust"][:, :, :] = 0.75
        solver.layer_masks = soft
        log_xy = torch.zeros((len(THERMAL_LAYER_NAMES), 2, 2), dtype=torch.float64)
        log_xy[0, :, :] = np.log(2.0)
        log_xy[1, :, :] = np.log(4.0)

        field, _ = self.make_dummy_model().apply_log_multiplier_torch(
            solver,
            np.ones((2, 2, 5)),
            "dct",
            log_xy=log_xy,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )

        np.testing.assert_allclose(field.numpy(), np.full((2, 2, 5), 0.25 * 2.0 + 0.75 * 4.0))

    def test_full_mode_applies_layer_2d_scaling(self) -> None:
        solver = self.make_dummy_solver()
        log_xy = torch.zeros((len(THERMAL_LAYER_NAMES), 2, 2), dtype=torch.float64)
        log_xy[0, :, :] = np.log(3.0)
        field, summary = self.make_dummy_model().apply_log_multiplier_torch(
            solver,
            np.ones((2, 2, 5)),
            "full",
            log_xy=log_xy,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        sediment_mask = torch.as_tensor(solver.layer_masks["sediment"])
        self.assertTrue(torch.allclose(field[sediment_mask], torch.full_like(field[sediment_mask], 3.0)))
        self.assertEqual(summary.shape, log_xy.shape)


class DecodeConfiguredParameterFieldsTest(unittest.TestCase):
    def test_decodes_fixed_bottom_and_properties(self) -> None:
        solver, _ = make_solver(ny=2, nx=2, nz=5)
        model = DirectParameterModel(2, 2, bottom_mode="fixed", k_mode="fixed", a_mode="fixed")
        coeffs = model()
        T_bot, K0, A, extras = model.decode_configured_parameter_fields(
            solver,
            coeffs,
            T_ref=1300.0,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        np.testing.assert_allclose(T_bot.numpy(), np.full((2, 2), 1300.0))
        np.testing.assert_allclose(K0.numpy(), solver.K0_ref)
        np.testing.assert_allclose(A.numpy(), solver.A_ref)
        self.assertEqual(extras["T_bot_delta"].shape, (2, 2))

    def test_decodes_nonfixed_modes(self) -> None:
        solver, _ = make_solver(ny=2, nx=2, nz=5)
        model = DirectParameterModel(
            2,
            2,
            bottom_mode="full",
            k_mode="dct",
            a_mode="full",
            k_property_x=1,
            k_property_y=1,
        )
        coeffs = model()
        coeffs["bottom_coeff"] = torch.ones((2, 2), dtype=torch.float64)
        coeffs["K_coeff"] = torch.zeros_like(coeffs["K_coeff"])
        coeffs["A_coeff"] = torch.zeros_like(coeffs["A_coeff"])
        T_bot, K0, A, extras = model.decode_configured_parameter_fields(
            solver,
            coeffs,
            T_ref=1000.0,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        np.testing.assert_allclose(T_bot.numpy(), np.full((2, 2), 1001.0))
        self.assertEqual(K0.shape, (2, 2, 5))
        self.assertEqual(A.shape, (2, 2, 5))
        self.assertEqual(extras["T_bot_delta"].shape, (2, 2))
        self.assertEqual(extras["K_log_xy"].shape, (len(THERMAL_LAYER_NAMES), 2, 2))
        self.assertEqual(extras["A_log_xy"].shape, (len(THERMAL_LAYER_NAMES), 2, 2))

    def test_decodes_property_mean_plus_field_terms(self) -> None:
        solver, _ = make_solver(ny=2, nx=2, nz=5)
        model = DirectParameterModel(
            2,
            2,
            bottom_mode="fixed",
            k_mode="dct",
            a_mode="full",
            k_property_x=1,
            k_property_y=1,
        )
        k_mean = torch.linspace(0.1, 0.5, len(THERMAL_LAYER_NAMES), dtype=torch.float64)
        k_field_coeff = torch.arange(model.n_k_field, dtype=torch.float64) / 10.0
        a_mean = torch.linspace(-0.5, -0.1, len(THERMAL_LAYER_NAMES), dtype=torch.float64)
        a_field = torch.arange(model.n_a_field, dtype=torch.float64).reshape(len(THERMAL_LAYER_NAMES), 2, 2) / 20.0
        coeffs = {
            "bottom_coeff": torch.zeros(0, dtype=torch.float64),
            "K_coeff": torch.cat([k_mean, k_field_coeff]),
            "A_coeff": torch.cat([a_mean, a_field.reshape(-1)]),
        }
        _, _, _, extras = model.decode_configured_parameter_fields(
            solver,
            coeffs,
            T_ref=1300.0,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        expected_k_field = model.k_dct.decode_torch(k_field_coeff, dtype=torch.float64, device=torch.device("cpu"))
        expected_a_field = a_field

        torch.testing.assert_close(extras["K_log_xy"], k_mean[:, None, None] + expected_k_field)
        torch.testing.assert_close(extras["A_log_xy"], a_mean[:, None, None] + expected_a_field)


if __name__ == "__main__":
    unittest.main()
