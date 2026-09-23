import unittest
import numpy as np
from spatial_basis import DCTBasis


class DCTBasisTest(unittest.TestCase):
    def test_dc_coefficient_is_grid_independent_field_mean(self) -> None:
        for shape in ((3, 4), (7, 2)):
            basis = DCTBasis(*shape, modes_y=2, modes_x=2)
            coefficients = np.zeros(basis.size)
            coefficients[0] = 100.0
            np.testing.assert_allclose(basis.decode(coefficients), 100.0)

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
