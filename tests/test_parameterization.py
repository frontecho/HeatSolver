from __future__ import annotations

import unittest

import numpy as np

from lithosphere_model import THERMAL_LAYER_NAMES
from parameterization import DCTBasis, DirectParameterization
from tests.helpers import make_solver


class DCTBasisTest(unittest.TestCase):
    def test_decode_and_adjoint_satisfy_dot_product_identity(self) -> None:
        basis = DCTBasis(5, 4, modes_y=3, modes_x=2)
        coefficients = np.linspace(-0.4, 0.7, basis.size)
        field = np.linspace(-0.2, 0.5, 20).reshape(5, 4)
        np.testing.assert_allclose(
            np.sum(basis.decode(coefficients) * field),
            coefficients @ basis.adjoint(field),
            rtol=1e-14,
            atol=1e-14,
        )


class DirectParameterizationTest(unittest.TestCase):
    def test_fixed_parameterization_returns_reference_fields(self) -> None:
        solver, _ = make_solver(ny=2, nx=3, nz=5, A=1e-6)
        model = DirectParameterization(
            2,
            3,
            bottom_mode="fixed",
            k_mode="fixed",
            a_mode="fixed",
        )
        state = model.decode(solver, model.initial(), T_ref=1250.0)

        self.assertEqual(model.size, 0)
        np.testing.assert_allclose(state["T_bot"], 1250.0)
        np.testing.assert_allclose(state["K0"], solver.K0_ref)
        np.testing.assert_allclose(state["A"], solver.A_ref)

    def test_parameter_counts_and_bounds(self) -> None:
        model = DirectParameterization(
            3,
            4,
            bottom_mode="dct",
            k_mode="dct",
            a_mode="full",
            bottom_dct_x=2,
            bottom_dct_y=3,
            k_dct_x=2,
            k_dct_y=2,
            bottom_bound_C=500.0,
            k_log_bound=0.4,
            a_log_bound=0.2,
        )
        expected = 6 + (5 + 5 * 4) + (5 + 5 * 12)
        self.assertEqual(model.size, expected)
        self.assertEqual(model.bounds()[0], (-500.0, 500.0))
        self.assertEqual(model.bounds()[model.k_slice.start], (-0.4, 0.4))
        self.assertEqual(model.bounds()[model.a_slice.start], (-0.2, 0.2))

    def test_soft_layer_masks_mix_log_multipliers(self) -> None:
        solver, _ = make_solver(ny=2, nx=2, nz=5, K0=2.0)
        solver.layer_masks = {
            name: np.zeros(solver.field_shape, dtype=np.float64)
            for name in THERMAL_LAYER_NAMES
        }
        solver.layer_masks["sediment"][:] = 0.25
        solver.layer_masks["mantle"][:] = 0.75
        model = DirectParameterization(
            2,
            2,
            bottom_mode="fixed",
            k_mode="full",
            a_mode="fixed",
            k_log_bound=2.0,
        )
        parameters = model.initial()
        parameters[model.k_slice.start] = np.log(2.0)
        parameters[model.k_slice.start + 4] = np.log(4.0)
        state = model.decode(solver, parameters, T_ref=1300.0)
        np.testing.assert_allclose(state["K0"], 2.0 * (0.25 * 2.0 + 0.75 * 4.0))

    def test_pullback_matches_directional_difference(self) -> None:
        solver, _ = make_solver(ny=2, nx=3, nz=5, K0=2.5, A=1e-6)
        model = DirectParameterization(
            2,
            3,
            bottom_mode="dct",
            k_mode="dct",
            a_mode="full",
            bottom_dct_x=2,
            bottom_dct_y=2,
            k_dct_x=2,
            k_dct_y=1,
            bottom_bound_C=1000.0,
            k_log_bound=1.0,
            a_log_bound=1.0,
        )
        rng = np.random.default_rng(3)
        parameters = rng.normal(0.0, 0.03, model.size)
        direction = rng.normal(size=model.size)
        grad_T_bot = rng.normal(size=(2, 3))
        grad_K0 = rng.normal(size=solver.field_shape)
        grad_A = rng.normal(size=solver.field_shape)
        state = model.decode(solver, parameters, T_ref=1250.0)
        gradient = model.pullback(solver, state, grad_T_bot, grad_K0, grad_A)

        def scalar(x: np.ndarray) -> float:
            decoded = model.decode(solver, x, T_ref=1250.0)
            return float(
                np.sum(grad_T_bot * decoded["T_bot"])
                + np.sum(grad_K0 * decoded["K0"])
                + np.sum(grad_A * decoded["A"])
            )

        eps = 1e-6
        finite_difference = (
            scalar(parameters + eps * direction)
            - scalar(parameters - eps * direction)
        ) / (2.0 * eps)
        np.testing.assert_allclose(gradient @ direction, finite_difference, rtol=2e-7, atol=2e-7)


if __name__ == "__main__":
    unittest.main()
