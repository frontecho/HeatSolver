import numpy as np

from uncertainty import Tangent, depth_statistics, field_response, property_jacobians
from test.test_uncertainty import small_problem


def test_joint_T_K_A_response_includes_reference_and_temperature_derivatives(tmp_path):
    problem = small_problem(tmp_path)
    coefficients = np.linspace(-.1, .1, len(problem.p.names))
    problem.evaluate(coefficients)
    tangent = Tangent(problem)
    direction = np.random.default_rng(114).normal(size=len(coefficients))*.1
    offsets = {key: np.full_like(problem.p.offset[key], .02) for key in ['A', 'K']}
    dT, _ = tangent.solve(*tangent.properties(coefficients=direction, offsets=offsets))
    derivatives = field_response(tangent, direction, dT, offsets)
    original = {key: value.copy() for key, value in problem.p.offset.items()}
    states = []
    step = 1e-4
    for sign in [1., -1.]:
        problem.p.offset = {key: original[key]+sign*step*offsets[key] for key in original}
        problem.p.set(coefficients+sign*step*direction)
        T = tangent.T+sign*step*dT
        states.append(dict(T=T, K=problem.p.evaluate(T)[0], A=problem.p.A*1e6))
    for field in ['T', 'K', 'A']:
        np.testing.assert_allclose((states[0][field]-states[1][field])/(2*step), derivatives[field], rtol=2e-6, atol=1e-8)


def test_rowwise_property_jacobians_preserve_units_and_mode_order(tmp_path):
    problem = small_problem(tmp_path)
    problem.evaluate(np.zeros(len(problem.p.names)))
    tangent = Tangent(problem)
    n = len(problem.p.names)
    temperature = []
    expected = []
    for direction in np.eye(n):
        dT, _ = tangent.solve(*tangent.properties(coefficients=direction))
        temperature.append(dT)
        expected.append(field_response(tangent, direction, dT))
    for row in range(tangent.T.shape[0]):
        matrices = property_jacobians(tangent, np.array(temperature), row)
        for field in ['T', 'K', 'A']:
            reference = np.array([values[field][row].ravel() for values in expected]).T
            np.testing.assert_allclose(matrices[field], reference, rtol=1e-12, atol=1e-12)


def test_depth_statistics_compare_local_errors_with_spatial_mean_using_area_weights():
    mean = np.array([[[10., 20.]], [[30., 40.]]])
    variance = np.array([[[1., 4.]], [[9., 16.]]])
    area = np.array([[1.], [3.]])
    values = depth_statistics(mean, variance, area)
    np.testing.assert_allclose(values['mean'], [25., 35.])
    np.testing.assert_allclose(values['point_sd_rms'], np.sqrt([7., 13.]))
    np.testing.assert_allclose(values['point_sd_mean'], [2.5, 3.5])
    np.testing.assert_allclose(values['point_sd_median'], [3., 4.])
    np.testing.assert_allclose(values['point_sd_p05'], [1., 2.])
    np.testing.assert_allclose(values['point_sd_p95'], [3., 4.])
    # Averaging uncertainties is different from preserving spatial covariance.
    opposite_errors = np.array([1., -1.])
    assert abs(np.array([.25, .75])@opposite_errors) == .5
