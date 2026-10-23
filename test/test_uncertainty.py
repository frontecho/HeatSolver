import json

import h5py
import numpy as np
import pandas as pd
import pytest
from scipy.linalg import cholesky

from config import A_BOUNDS, BETA, K_BOUNDS, REGULARIZATION
from data_io import load_model
from observations import load as load_observations
from objective import Problem
from uncertainty import (analyze, covariances, linearize, observation_errors,
                         standard_deviation)
from test.test_workflow import synthetic_bundle


def small_problem(tmp_path):
    folder = tmp_path/'problem'
    synthetic_bundle(folder)
    layout = dict(A=[None, (2, 2), None, None, None], K=[(1, 1), None, None, None, (1, 1)])
    constants = dict(A=[True]*5, K=[True]*5)
    problem = Problem(load_model(folder), load_observations(folder), layout, constants,
                      a_bounds=A_BOUNDS, k_bounds=K_BOUNDS, beta=BETA,
                      regularization=REGULARIZATION)
    problem.solver.tolerance, problem.solver.residual_tolerance = 1e-9, 1e-12
    return problem


def test_tangent_matches_complete_nonlinear_pde_and_observation_derivatives(tmp_path):
    problem = small_problem(tmp_path)
    c = np.linspace(-.1, .15, len(problem.p.names))
    G = np.empty((len(c), int(np.prod(problem.b['masks'].shape[1:]))))
    objective, gradient = problem.evaluate(c)
    J = linearize(problem, c, G)
    from uncertainty import Tangent
    tangent = Tangent(problem)
    for direction, dT in zip(np.eye(len(c)), G):
        rhs = problem.solver.tangent_rhs(tangent.T, tangent.K, *tangent.properties(coefficients=direction))
        residual = tangent.matrix@dT.reshape(tangent.T.shape)[..., 1:-1].ravel()-rhs
        assert np.linalg.norm(residual)/np.linalg.norm(rhs) < 3e-10
    # The transpose observation derivatives must agree with the established adjoint.
    W, _ = observation_errors(problem)
    errors = np.concatenate([problem.last[key]-problem.data[key].observed.to_numpy()
                             for key in ['heatflow', 'gradient', 'curie']])
    expected_gradient = J.T@W@errors+problem.regularization*c/problem.p.scales**2
    np.testing.assert_allclose(gradient, expected_gradient, rtol=2e-8, atol=1e-7)
    assert np.isfinite(objective)
    step = 3e-4
    for i in range(len(c)):
        observations, temperature = [], []
        for sign in [1, -1]:
            perturbed = c.copy()
            perturbed[i] += sign*step
            problem.evaluate(perturbed)
            observations.append(np.concatenate([problem.last[key] for key in ['heatflow', 'gradient', 'curie']]))
            temperature.append(problem.last['T'].copy())
        np.testing.assert_allclose(J[:, i], (observations[0]-observations[1])/(2*step), rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(G[i], ((temperature[0]-temperature[1])/(2*step)).ravel(), rtol=2e-5, atol=3e-5)
    np.testing.assert_array_equal(G.reshape(len(c), *temperature[0].shape)[..., [0, -1]], 0.)


def test_propagated_covariance_matches_repeated_correlated_linear_fits():
    rng = np.random.default_rng(871)
    J = rng.normal(size=(4, 2))
    noise = np.array([[1., .5, 0., 0.], [.5, 2., 0., 0.], [0., 0., 1., -.3], [0., 0., -.3, 1.]])
    W = np.diag([1., .2, 1., 2.])
    prior = np.array([2., 3.])
    H, propagated, gaussian = covariances(J, W, noise, prior)
    # Independently solve augmented weighted least-squares problems for 40,000 noise realizations.
    errors = cholesky(noise, lower=True)@rng.normal(size=(4, 40000))
    whitening = cholesky(W)
    design = np.vstack([whitening@J, np.diag(np.sqrt(prior))])
    target = np.vstack([whitening@errors, np.zeros((2, errors.shape[1]))])
    fitted = np.linalg.lstsq(design, target, rcond=None)[0]
    np.testing.assert_allclose(np.cov(fitted), propagated, rtol=.025, atol=.002)
    np.testing.assert_allclose(H@gaussian, np.eye(2), atol=1e-13)
    assert np.trace(propagated) < np.trace(gaussian)


def test_noise_covariance_preserves_curie_correlation_and_separates_gradient_weights(tmp_path):
    problem = small_problem(tmp_path)
    g = problem.data['gradient']
    g.loc[:, ['X', 'Y']] = g.loc[0, ['X', 'Y']].to_numpy()
    g['sigma'] = 5*np.sqrt(len(g))
    problem.weights['gradient'] = 1/g.sigma.to_numpy()**2
    covariance = 2*np.eye(5)+np.ones((5, 5))
    problem.b['curie_precision'] = np.linalg.inv(covariance)
    W, noise = observation_errors(problem)
    start = len(problem.data['heatflow'])
    np.testing.assert_allclose(np.diag(noise)[start:start+len(g)], 25.)
    np.testing.assert_allclose(np.diag(W)[start:start+len(g)], 1/(25*len(g)))
    np.testing.assert_allclose(noise[-5:, -5:], covariance, atol=1e-14)
    g['noise_sigma'] = 3.
    _, noise = observation_errors(problem)
    np.testing.assert_allclose(np.diag(noise)[start:start+len(g)], 9.)


def test_analysis_exports_fixed_boundary_fields_and_propagates_region_means(tmp_path):
    problem = small_problem(tmp_path)
    c = np.zeros(len(problem.p.names))
    output = tmp_path/'uncertainty'
    analyze(problem, c, output)
    assert not (output/'temperature_jacobian.npy').exists()
    with h5py.File(output/'fields.h5') as h:
        for name in ['observation_noise', 'conditional_gaussian']:
            sd = h['temperature_sd/'+name][:]
            np.testing.assert_array_equal(sd[..., [0, -1]], 0.)
            assert sd[..., 1:-1].max() > 0.
    matrices = np.load(output/'parameters.npz')
    assert np.linalg.eigvalsh(matrices['covariance_noise']).min() > 0.
    assert np.linalg.eigvalsh(matrices['covariance_gaussian']).min() > 0.
    G = np.empty((len(c), problem.last['T'].size))
    linearize(problem, c, G)
    weights = problem.b['core']/problem.b['horizontal_scale']**2
    weights /= weights.sum()
    mean_derivative = np.einsum('yx,iyxz->zi', weights, G.reshape(len(c), *problem.last['T'].shape))
    profiles = pd.read_csv(output/'profiles.csv')
    expected = standard_deviation(mean_derivative, matrices['covariance_noise'])
    np.testing.assert_allclose(profiles.observation_noise_sd_C, expected, atol=1e-10)
    with pytest.raises(FileExistsError):
        analyze(problem, c, output)
    with pytest.raises(ValueError, match='inside their bounds'):
        analyze(problem, np.full(len(c), 4.), tmp_path/'bounded')


def test_complete_workflow_writes_main_results_and_preserves_variance_sum(tmp_path, monkeypatch):
    import config
    import uncertainty
    from config import CRS, SOURCES
    from data_io import write_json
    from pyproj import Transformer
    import si.regions

    problem = small_problem(tmp_path)
    coefficients = np.zeros(len(problem.p.names))
    problem.predict(coefficients)
    fit = tmp_path/'fit'
    fit.mkdir()
    np.savez(fit/'result.npz', coefficients=coefficients, T=problem.last['T'])
    write_json(fit/'training_config.json', dict(
        modes=dict(A=[None, (2, 2), None, None, None], K=[(1, 1), None, None, None, (1, 1)]),
        constant_modes=dict(A=[True]*5, K=[True]*5), regularization=dict(A=4., K=4.)))
    bundle = dict(np.load(tmp_path/'problem/bundle.npz'))
    bundle['longitude'], bundle['latitude'] = Transformer.from_crs(CRS, 4326, always_xy=True).transform(bundle['x'], bundle['y'])
    np.savez(tmp_path/'problem/bundle.npz', **bundle)
    weights = bundle['core']/bundle['core'].sum()
    monkeypatch.setattr(si.regions, 'region_weights', lambda b, path: dict(ENCC=weights, WNCC=weights))
    for key, column in [('heatflow', 'ids'), ('gradient', 'id')]:
        table = problem.data[key].copy()
        table[column] = [f'r{i}' for i in range(1, len(table)+1)]
        table.to_csv(tmp_path/f'problem/{key}.csv', index=False)
    raw = tmp_path/'raw_heatflow.json'
    write_json(raw, [dict(a=f'r{i}', o=30., h=10., j=3.) for i in range(1, 8)])
    monkeypatch.setitem(SOURCES, 'heatflow', raw)
    csrm = tmp_path/'csrm'
    csrm.mkdir()
    for lon, lat in [(112., 36.), (114., 36.), (112., 39.), (114., 39.)]:
        columns = np.zeros((2, 7))
        columns[:, 0], columns[:, 6] = [0., 60.], [60., 64.]
        np.savetxt(csrm/f'CSRM1.0_{lon}_{lat}_SiO2.info', columns)
    monkeypatch.setitem(SOURCES, 'csrm', csrm)
    references = tmp_path/'references'
    pd.DataFrame(dict(sio2_wt_percent=[60., 64.])).to_csv(references/'heat_production_relation.csv', index=False)
    pd.DataFrame([dict(study=study, sio2=silica, rock='MET', A=.2+.01*silica+.1*i)
                  for i, study in enumerate(['a', 'b', 'c']) for silica in [58., 60., 62., 64., 66.]]).to_csv(
                      references/'georoc_ncc.csv', index=False)
    pd.DataFrame([dict(reference=ref, sample=sample, layer=layer, group='rock', log_K0=np.log(2.5+.1*i+.05*sample))
                  for layer in ['upper_crust', 'middle_crust', 'lower_crust', 'mantle']
                  for i, ref in enumerate(['a', 'b']) for sample in [0, 1]]).to_csv(
                      references/'deep_conductivity_samples.csv', index=False)
    pd.DataFrame(dict(X=np.arange(8)*5., Y=np.arange(8)*3., depth_top_m=np.arange(8)*100.,
                      depth_bottom_m=np.arange(8)*100.+200., glim=['ss']*8, site_id=np.arange(8),
                      record_id=np.arange(8), k_W_mK=[2., 2.5, 2.1, 2.4, 2.2, 2.6, 2.3, 2.7])).to_csv(
                          references/'shallow_conductivity_samples.csv', index=False)
    monkeypatch.setattr(config, 'CALIBRATION_SETTINGS', dict(
        config.CALIBRATION_SETTINGS, bootstrap_replicates=20, gp_draws=4, calibration_tail_draws=2))

    summary = uncertainty.run(tmp_path)

    output = tmp_path/'uncertainty'
    assert not (tmp_path/'si').exists()
    assert not (output/'fields').exists()
    with h5py.File(output/'fields.h5') as h:
        for field in ['T', 'K', 'A']:
            total = sum(h[f'variance/{source}/{field}'][:] for source in summary['source_sets']['typical_reference'])
            np.testing.assert_allclose(h[f'std/typical_reference/{field}'][:]**2, total, atol=1e-10)
            assert np.isfinite(total).all() and total.max() > 0
        np.testing.assert_allclose(h['mean/T'][:], problem.last['T'], rtol=1e-9, atol=1e-7)
    statistics = pd.read_csv(output/'depth_statistics.csv')
    assert set(statistics.field) == {'T', 'K', 'A'}
    checks = pd.read_csv(output/'nonlinear_checks.csv')
    assert len(checks) == 12 and np.isfinite(checks.select_dtypes('number')).all().all()
    assert (output/'linear/parameters.npz').is_file()
    assert (output/'inputs/calibration.h5').is_file()
    assert (output/'external/responses.h5').is_file()
    with pytest.raises(FileExistsError):
        uncertainty.run(tmp_path)
