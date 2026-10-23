import json

import h5py
import numpy as np
import pandas as pd
import pytest
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

from data_io import write_json
import uncertainty as input_uncertainty
from uncertainty import validate_cross_covariance
from uncertainty import (conditional_gp_draws, depth_kernel,
    heat_production_bootstrap, reference_logit_direction)
from uncertainty import KEYS, Tangent, covariances, linearize, observation_errors
from test.test_uncertainty import small_problem


def test_boundary_and_reference_tangents_include_boundary_conductivity(tmp_path):
    problem = small_problem(tmp_path)
    c = np.linspace(-.05, .08, len(problem.p.names))
    problem.evaluate(c)
    tangent = Tangent(problem)
    rng = np.random.default_rng(77)
    offsets = {key: rng.normal(size=problem.p.masks.shape)*.02 for key in ['A', 'K']}
    boundary = np.zeros_like(tangent.T)
    boundary[..., 0] = 1.+np.linspace(0, 2, boundary.shape[1])
    boundary[..., -1] = 30.+np.linspace(-5, 5, boundary.shape[0])[:, None]
    dT, obs = tangent.solve(*tangent.properties(offsets=offsets), boundary)
    original = {key: problem.p.offset[key].copy() for key in offsets}
    top, bottom = problem.b['surface_T'].copy(), problem.b['T_bot'].copy()
    temperatures, predictions = [], []
    step = 1e-3
    for sign in [1, -1]:
        for key in offsets:
            problem.p.offset[key] = original[key]+sign*step*offsets[key]
        problem.b['surface_T'] = top+sign*step*boundary[..., 0]
        problem.b['T_bot'] = bottom+sign*step*boundary[..., -1]
        problem.evaluate(c)
        temperatures.append(problem.last['T'].copy())
        predictions.append(np.concatenate([problem.last[key] for key in KEYS]))
    np.testing.assert_allclose(dT, (temperatures[0]-temperatures[1])/(2*step), rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(obs, (predictions[0]-predictions[1])/(2*step), rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(dT[..., [0, -1]], boundary[..., [0, -1]])


def test_effective_external_response_matches_paired_complete_nonlinear_refits(tmp_path):
    problem = small_problem(tmp_path)
    c = np.zeros(len(problem.p.names))
    problem.evaluate(c)
    for key in KEYS:
        problem.data[key]['observed'] = problem.last[key]
    G = np.empty((len(c), problem.last['T'].size))
    J = linearize(problem, c, G)
    W, noise = observation_errors(problem)
    H, _, _ = covariances(J, W, noise, problem.regularization/problem.p.scales**2)
    response = cho_solve(cho_factor(H), J.T@W)
    tangent = Tangent(problem)
    offsets = dict(A=np.full(problem.p.masks.shape, .02), K=np.full(problem.p.masks.shape, -.015))
    boundary = np.zeros_like(tangent.T)
    boundary[..., -1] = 20.
    effective, _, dc, direct = tangent.external(response, offsets, boundary)
    direct_obs = tangent.solve(*tangent.properties(offsets=offsets), boundary)[1]
    np.testing.assert_allclose(effective.ravel(), direct.ravel()+dc@G, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(H@dc, -J.T@W@direct_obs, atol=1e-10)
    original = {key: problem.p.offset[key].copy() for key in offsets}
    bottom = problem.b['T_bot'].copy()
    fits, fields = [], []
    step = .01
    for sign in [1, -1]:
        for key in offsets:
            problem.p.offset[key] = original[key]+sign*step*offsets[key]
        problem.b['T_bot'] = bottom+sign*step*boundary[..., -1]
        fit = minimize(problem.evaluate, c.copy(), jac=True, method='L-BFGS-B',
            options=dict(maxiter=150, gtol=1e-9, ftol=1e-15, maxls=40))
        assert abs(fit.jac).max() < 2e-6
        problem.evaluate(fit.x)
        fits.append(fit.x)
        fields.append(problem.last['T'].copy())
    np.testing.assert_allclose(dc, (fits[0]-fits[1])/(2*step), rtol=.002, atol=5e-6)
    np.testing.assert_allclose(effective, (fields[0]-fields[1])/(2*step), rtol=.002, atol=2e-4)


def test_separable_gp_simulator_matches_exact_dense_conditional_covariance():
    query = np.array([[0., 0.], [40., 0.], [80., 20.]])
    depths = np.array([0., .5, 1.])
    train = np.array([[10., 0.], [50., 10.], [60., 0.], [20., 10.]])
    a, b = np.array([0., .1, .4, .3]), np.array([.2, .5, .9, .8])
    C = np.exp(-cdist(train, train)/150)*depth_kernel(a[:, None], b[:, None], a[None], b[None])+np.eye(4)/3
    samples = conditional_gp_draws(query, depths, train, a, b, C, 40000, np.random.default_rng(18))
    cross = (np.exp(-cdist(query, train)/150)[:, None, :]*depth_kernel(
        depths[None, :, None], depths[None, :, None], a[None, None], b[None, None])).reshape(-1, len(a))
    prior = np.kron(np.exp(-cdist(query, query)/150), np.exp(-abs(depths[:, None]-depths[None])/.5))
    expected = prior-cross@np.linalg.solve(C, cross.T)
    actual = np.cov(samples.reshape(-1, samples.shape[-1]))
    np.testing.assert_allclose(actual, expected, rtol=.08, atol=.009)


def test_publication_resampling_keeps_shared_curve_covariance():
    table = pd.DataFrame(dict(study=np.repeat(['a', 'b', 'c', 'd'], 4),
        sio2=np.tile([49., 50., 50., 51.], 4), rock=['MET']*16,
        A=np.repeat([.2, 1., 2., 5.], 4)))
    curves = heat_production_bootstrap(table, np.array([49.5, 50.5]), 200, np.random.default_rng(5))
    np.testing.assert_array_equal(curves[:, 0], curves[:, 2])
    np.testing.assert_array_equal(curves[:, 1], curves[:, 3])
    assert np.std(curves[:, 0]) > .2


def test_reference_clipping_and_joint_covariance_constraints(tmp_path):
    problem = small_problem(tmp_path)
    key = 'K'
    ref = problem.b['K_reference']
    lo, hi = problem.p.bounds[key].T
    ref[0, 0, 0, 0] = lo[0]
    derivative = np.ones_like(ref)*.01
    direction = reference_logit_direction(problem, key, derivative)
    assert direction[0, 0, 0, 0] == 0.
    expected = (hi[1]-lo[1])/((ref[1]-lo[1])*(hi[1]-ref[1]))*.01
    np.testing.assert_allclose(direction[1], expected)
    noise = np.array([[2., .4], [.4, 1.]])
    cross = np.array([[.4], [.1]])
    assert validate_cross_covariance(noise, cross) < 1.
    with pytest.raises(ValueError, match='positive semidefinite'):
        validate_cross_covariance(noise, cross*10)




@pytest.mark.parametrize('correlation', [0., .2])
def test_external_exports_propagate_joint_covariance_and_regional_means(tmp_path, monkeypatch, correlation):
    problem = small_problem(tmp_path)
    c = np.zeros(len(problem.p.names))
    problem.evaluate(c)
    fit = tmp_path/'fit'
    fit.mkdir()
    layout = dict(A=[None, (2, 2), None, None, None], K=[(1, 1), None, None, None, (1, 1)])
    training = dict(modes=layout, constant_modes=dict(A=[True]*5, K=[True]*5), regularization=dict(A=4., K=4.))
    write_json(fit/'training_config.json', training)
    np.savez(fit/'result.npz', names=problem.p.names, coefficients=c, T=problem.last['T'], K=problem.last['K'], A=problem.last['A'])
    linear = tmp_path/'linear'
    input_uncertainty_output = tmp_path/'propagated'
    from uncertainty import analyze
    analyze(problem, c, linear)
    inputs = tmp_path/'inputs'
    inputs.mkdir()
    count = 8
    reference_count = 3+count
    settings = dict(gp_draws=count, boundary_scenario_sd_C={}, A_scale_scenario_sd_percent={})
    write_json(inputs/'summary.json', dict(settings=settings))
    with h5py.File(inputs/'calibration.h5', 'w') as h:
        for dataset in ['A/modes', 'A/tail_modes', 'deep_K/modes']:
            h[dataset] = np.zeros((2, 1))
        h['GP/log_K_draws'] = np.zeros((count, 2))
    rng = np.random.default_rng(643)
    sources = ['A_curve', 'A_curve_tail', 'deep_K', *(['shallow_GP']*count), 'boundary_scenario', 'A_scale_scenario']
    names = ['A_curve_1', 'A_curve_tail_1', 'deep_K_1', *[f'shallow_GP_{i+1}' for i in range(count)], 'surface_offset_per_C', 'sediment_A_per_percent']
    offsets = [dict(A=rng.normal(size=problem.p.masks.shape)*.005, K=rng.normal(size=problem.p.masks.shape)*.004) for _ in sources]
    boundary = np.zeros_like(problem.last['T'])
    boundary[..., 0] = 1.

    def modes(p, h, boundary_sd, A_scale_sd, profiles, forward):
        for i, source in enumerate(sources):
            yield source, names[i], offsets[i] if i != reference_count else {}, boundary if i == reference_count else None, 2. if i == reference_count else 1.

    monkeypatch.setattr(input_uncertainty, 'input_modes', modes)
    W, noise = observation_errors(problem)
    cross = correlation*np.linalg.cholesky(noise)[:, :reference_count]
    cross_file = None
    if correlation:
        cross_file = tmp_path/'cross.npz'
        np.savez(cross_file, mode_names=names[:reference_count], cross_covariance=cross)
    input_uncertainty.propagate_inputs(tmp_path, linear, inputs, input_uncertainty_output, cross_file)
    G = np.empty((len(c), problem.last['T'].size))
    J = linearize(problem, c, G)
    H, _, _ = covariances(J, W, noise, problem.regularization/problem.p.scales**2)
    B = cho_solve(cho_factor(H), J.T@W)
    noise_response = (G.T@B@cross).T.reshape(reference_count, *problem.last['T'].shape)
    with h5py.File(input_uncertainty_output/'responses.h5') as h:
        effective = np.array([h['temperature/'+name][:] for name in names[:reference_count]])
    expected_cross = 2*np.sum(effective*noise_response, axis=0)
    with h5py.File(linear/'fields.h5') as old, h5py.File(input_uncertainty_output/'fields.h5') as h:
        expected = old['temperature_sd/observation_noise'][:]**2+np.sum(effective**2, axis=0)+expected_cross
        np.testing.assert_allclose(h['temperature_variance/observation_reference_cross'][:], expected_cross, atol=2e-10)
        np.testing.assert_allclose(h['temperature_variance/observation_plus_reference'][:], expected, atol=2e-10)
    weights = problem.b['core']/problem.b['horizontal_scale']**2
    weights /= weights.sum()
    region_effective = np.einsum('yx,iyxz->iz', weights, effective)
    region_noise = np.einsum('yx,iyxz->iz', weights, noise_response)
    profiles = pd.read_csv(input_uncertainty_output/'profiles.csv')
    actual = profiles[profiles.source.eq('observation_reference_cross')].variance_C2.to_numpy()
    np.testing.assert_allclose(actual, 2*np.sum(region_effective*region_noise, axis=0), atol=2e-10)
