import h5py
import numpy as np
import pandas as pd
import pytest
from config import A_BOUNDS, BETA, CONSTANT_MODES, K_BOUNDS, MODES, REGULARIZATION
from data_io import load_model
from observations import load as load_observations
from objective import Problem
from solver import Solver


def test_sample_heat_production_uses_elemental_mass_fractions():
    from references import sample_heat_production
    # 1 wt% elemental K, 1 ppm Th and 1 ppm U, independently.
    k2o_per_percent_K = (2*39.0983+15.999)/(2*39.0983)
    calculated = sample_heat_production([k2o_per_percent_K, 0, 0], [0, 1, 0], [0, 0, 1])
    np.testing.assert_allclose(calculated, [.091449, .070686, .265383], rtol=1e-14)
    np.testing.assert_allclose(sample_heat_production(1, 0, 0), .07591650643342152, rtol=1e-13)


class Constant:
    def __init__(self, k):
        self.k = k

    def evaluate(self, T):
        return np.full(T.shape, self.k), np.zeros(T.shape)


def test_layered_column_matches_analytic_solution():
    z = np.linspace(0, 40, 41)
    x, y = np.meshgrid(np.arange(4)*2e4, np.arange(3)*2e4)
    solver = Solver(x, y, z, np.ones(x.shape), np.ones(x.shape, bool), Constant(2.5), 1e-12, 1e-12)
    A = np.full((3, 4, 41), 1e-6)
    T, _, _ = solver.solve(A, np.full(x.shape, 10.), np.full(x.shape, 600.))
    zm = z*1e3
    exact = 10+(590+1e-6*zm[-1]**2/5)*zm/zm[-1]-1e-6*zm**2/5
    np.testing.assert_allclose(T, np.broadcast_to(exact, T.shape), atol=1e-8)


def synthetic_bundle(folder, nx=8):
    folder.mkdir(parents=True)
    rng = np.random.default_rng(1)
    ny, z = 6, np.arange(0, 60.1, 2.)
    x, y = np.meshgrid(np.arange(nx)*2e4, np.arange(ny)*2e4)
    interfaces = np.stack([1+rng.random(x.shape), 12+rng.random(x.shape), 22+rng.random(x.shape),
                           35+3*rng.random(x.shape)], axis=-1)
    above = np.clip((interfaces[..., None]-z)/2+.5, 0, 1)
    masks = np.moveaxis(np.concatenate([above[..., :1, :], np.diff(above, axis=-2), 1-above[..., -1:, :]], axis=-2), -2, 0)
    shape = (5, ny, nx, len(z))
    A = np.array([1.5, 1.6, .8, .4, .03])[:, None, None, None]*(1+.2*rng.random(shape))
    K = np.array([2., 2.9, 2.4, 2.1, 3.8])[:, None, None, None]*(1+.05*rng.random(shape))
    active = np.ones(x.shape, bool)
    active[0, 0] = False
    np.savez(folder/'bundle.npz', x=x, y=y, z_km=z, surface_T=10+rng.random(x.shape), T_bot=1100+50*rng.random(x.shape),
             horizontal_scale=1+.01*rng.random(x.shape), domain_mask=active, core=active, masks=masks,
             curie_precision=.05*np.eye(5))
    (folder.parent/'references').mkdir()
    with h5py.File(folder.parent/'references/references.h5', 'w') as h:
        h['fields/K_reference'], h['fields/A_reference'] = K, A
    pick = lambda n: (2e4+rng.random(n)*1e5, 2e4+rng.random(n)*6e4)
    X, Y = pick(7)
    pd.DataFrame(dict(X=X, Y=Y, observed=60+rng.random(7)*20, sigma=10.)).to_csv(folder/'heatflow.csv', index=False)
    X, Y = pick(6)
    pd.DataFrame(dict(X=X, Y=Y, top_km=.3, bottom_km=3.7, observed=25+rng.random(6)*10,
                      sigma=5.)).to_csv(folder/'gradient.csv', index=False)
    X, Y = pick(5)
    pd.DataFrame(dict(X=X, Y=Y, observed=20+rng.random(5)*5)).to_csv(folder/'curie.csv', index=False)


def test_objective_gradient_matches_finite_differences(tmp_path, monkeypatch):
    monkeypatch.setitem(REGULARIZATION, 'A', 64.)
    bundle = tmp_path/'problem'
    synthetic_bundle(bundle)
    problem = Problem(load_model(bundle), load_observations(bundle),
                      layout=MODES, constant_modes=CONSTANT_MODES,
                      a_bounds=A_BOUNDS, k_bounds=K_BOUNDS, beta=BETA,
                      regularization=REGULARIZATION)
    problem.solver.tolerance, problem.solver.residual_tolerance = 1e-9, 1e-12
    rng = np.random.default_rng(2)
    c = .3*rng.standard_normal(len(problem.p.names))
    _, gradient = problem.evaluate(c)
    for _ in range(3):
        direction = rng.standard_normal(len(c))
        h = 1e-5
        fd = (problem.evaluate(c+h*direction)[0]-problem.evaluate(c-h*direction)[0])/(2*h)
        np.testing.assert_allclose(gradient@direction, fd, rtol=2e-6)


@pytest.mark.parametrize('key', ['heatflow', 'gradient', 'curie'])
def test_all_observations_affect_objective_and_metrics(tmp_path, key):
    bundle = tmp_path/'problem'
    synthetic_bundle(bundle)
    problem = Problem(load_model(bundle), load_observations(bundle),
                      layout=MODES, constant_modes=CONSTANT_MODES,
                      a_bounds=A_BOUNDS, k_bounds=K_BOUNDS, beta=BETA,
                      regularization=REGULARIZATION)
    coefficients = np.zeros(len(problem.p.names))
    loss, gradient = problem.evaluate(coefficients)
    metrics = problem.metrics()
    for name, table in problem.data.items():
        assert metrics[name]['n'] == len(table)
    problem.data[key].loc[0, 'observed'] += 1e3
    changed_loss, changed_gradient = problem.evaluate(coefficients)
    assert changed_loss > loss
    assert not np.allclose(changed_gradient, gradient)
    assert problem.metrics()[key]['rmse'] > metrics[key]['rmse']


def test_fit_starts_170_parameters_from_zero(tmp_path, monkeypatch):
    import json
    import fit as fitting
    bundle = tmp_path/'problem'
    synthetic_bundle(bundle, nx=9)
    assert fitting.FIT_OPTIONS['maxiter'] == 60
    assert fitting.REGULARIZATION == dict(A=4., K=4.)
    monkeypatch.setattr(fitting, 'FIT_OPTIONS', dict(fitting.FIT_OPTIONS, maxiter=1))
    fit = fitting.fit

    def fit_from_zero(problem, output, start, options):
        np.testing.assert_array_equal(start, np.zeros(170))
        fit(problem, output, start, options)

    monkeypatch.setattr(fitting, 'fit', fit_from_zero)
    fitting.run(tmp_path)
    assert {path.name for path in (tmp_path/'fit').iterdir()} == {
        'result.npz', 'summary.json', 'training_config.json', 'heatflow.csv', 'gradient.csv', 'curie.csv'}
    with np.load(tmp_path/'fit/result.npz') as result:
        names = result['names'].tolist()
        assert len(names) == 170
        assert sum(name.startswith('A_') for name in names) == 99
        assert sum(name.startswith('K_') for name in names) == 71
        assert not any(name.startswith('A_') and name.endswith('_y0_x0') for name in names)
    configuration = json.loads((tmp_path/'fit/training_config.json').read_text())
    assert configuration['parameters'] == 170
    assert configuration['regularization'] == dict(A=4., K=4.)
    problem = Problem(load_model(bundle), load_observations(bundle),
                      layout=MODES, constant_modes=CONSTANT_MODES,
                      a_bounds=A_BOUNDS, k_bounds=K_BOUNDS, beta=BETA,
                      regularization=REGULARIZATION)
    with np.load(tmp_path/'fit/result.npz') as result:
        problem.p.set(result['coefficients'])
        np.testing.assert_allclose(problem.p.A, result['A'], rtol=0, atol=1e-18)
        np.testing.assert_allclose(problem.p.evaluate(result['T'])[0], result['K'], rtol=1e-12)
        np.testing.assert_array_equal(result['T'][..., 0], problem.b['surface_T'])
        np.testing.assert_array_equal(result['T'][..., -1], problem.b['T_bot'])
    with pytest.raises(FileExistsError):
        fitting.run(tmp_path)


def test_custom_layer_and_constant_modes_have_correct_gradients(tmp_path):
    bundle = tmp_path/'problem'
    synthetic_bundle(bundle)
    layout = dict(A=[(1, 1), (2, 3), (1, 1), None, (2, 2)],
                  K=[(1, 1), (2, 2), None, (1, 1), (2, 1)])
    constants = dict(A=[False, True, True, False, True], K=[False, False, False, True, True])
    problem = Problem(load_model(bundle), load_observations(bundle), layout=layout, constant_modes=constants,
                      a_bounds=A_BOUNDS, k_bounds=K_BOUNDS, beta=BETA,
                      regularization=REGULARIZATION)
    problem.solver.tolerance, problem.solver.residual_tolerance = 1e-9, 1e-12
    assert 'A_sediment_y0_x0' not in problem.p.names
    assert 'A_middle_crust_y0_x0' in problem.p.names
    assert 'A_mantle_y1_x1' in problem.p.names
    assert 'K_lower_crust_y0_x0' in problem.p.names
    assert not any(name.startswith('K_sediment_') for name in problem.p.names)
    assert not any(name.startswith('A_lower_crust_') for name in problem.p.names)
    rng = np.random.default_rng(7)
    c = .2*rng.standard_normal(len(problem.p.names))
    _, gradient = problem.evaluate(c)
    for _ in range(3):
        direction = rng.standard_normal(len(c))
        # Avoid amplifying the forward linear-solve error in this low-misfit case.
        h = 1e-3
        fd = (problem.evaluate(c+h*direction)[0]-problem.evaluate(c-h*direction)[0])/(2*h)
        np.testing.assert_allclose(gradient@direction, fd, rtol=2e-6, atol=1e-7)


def test_forward_model_needs_no_observations(tmp_path):
    from model import Model

    bundle = tmp_path/'problem'
    synthetic_bundle(bundle)
    inverse = Problem(load_model(bundle), load_observations(bundle),
                      layout=MODES, constant_modes=CONSTANT_MODES,
                      a_bounds=A_BOUNDS, k_bounds=K_BOUNDS, beta=BETA,
                      regularization=REGULARIZATION)
    coefficients = np.zeros(len(inverse.p.names))
    inverse.evaluate(coefficients)
    b = load_model(bundle)
    b.pop('curie_precision')
    forward = Model(b,
                      layout=MODES, constant_modes=CONSTANT_MODES,
                      a_bounds=A_BOUNDS, k_bounds=K_BOUNDS, beta=BETA)
    state = forward.forward(coefficients)
    for key in ['T', 'K', 'A', 'q']:
        np.testing.assert_array_equal(state[key], inverse.last[key])


def test_attribution_uses_saved_property_fields(tmp_path, monkeypatch):
    import si.attribution as attribution
    bundle = tmp_path/'problem'
    synthetic_bundle(bundle)
    problem = Problem(load_model(bundle), load_observations(bundle),
                      layout=MODES, constant_modes=CONSTANT_MODES,
                      a_bounds=A_BOUNDS, k_bounds=K_BOUNDS, beta=BETA,
                      regularization=REGULARIZATION)
    A = problem.p.values['A'].copy()
    K = problem.p.values['K'].copy()
    A[1] += .1
    problem.p.values = dict(A=A, K=K)
    expected = problem.solver.solve(np.sum(problem.p.masks*A, axis=0)*1e-6,
                                    problem.b['surface_T'], problem.b['T_bot'])[0]
    fitted = tmp_path/'fit'
    fitted.mkdir()
    # Read the saved layer fields independently of the coefficients.
    np.savez(fitted/'result.npz', layer_A=A, layer_K=K, coefficients=np.zeros(len(problem.p.names)))
    weight = problem.b['core']/problem.b['core'].sum()
    monkeypatch.setattr(attribution, 'region_weights', lambda b, region_path: dict(all=weight))
    attribution.run(tmp_path)
    result = pd.read_csv(tmp_path/'si/attribution.csv')
    np.testing.assert_allclose(result.joint, np.einsum('yx,yxz->z', weight, expected), rtol=0, atol=1e-6)
    assert np.max(np.abs(result.joint-result.reference)) > .1


def test_reference_fields_load_with_updated_prescribed_boundary(tmp_path):
    bundle = tmp_path/'problem'
    synthetic_bundle(bundle)
    original = Problem(load_model(bundle), load_observations(bundle),
                      layout=MODES, constant_modes=CONSTANT_MODES,
                      a_bounds=A_BOUNDS, k_bounds=K_BOUNDS, beta=BETA,
                      regularization=REGULARIZATION)
    data = dict(np.load(bundle/'bundle.npz'))
    assert 'A_reference' not in data and 'K_reference' not in data
    data['surface_T'] += 1
    np.savez(bundle/'bundle.npz', **data)
    problem = Problem(load_model(bundle), load_observations(bundle),
                      layout=MODES, constant_modes=CONSTANT_MODES,
                      a_bounds=A_BOUNDS, k_bounds=K_BOUNDS, beta=BETA,
                      regularization=REGULARIZATION)
    np.testing.assert_array_equal(problem.b['A_reference'], original.b['A_reference'])
    np.testing.assert_array_equal(problem.b['K_reference'], original.b['K_reference'])
    problem.evaluate(np.zeros(len(problem.p.names)))
    np.testing.assert_array_equal(problem.last['T'][..., 0], data['surface_T'])
    np.testing.assert_array_equal(problem.last['T'][..., -1], data['T_bot'])


def test_synthetic_noise_and_paired_ablation_fits(tmp_path, monkeypatch):
    import json
    from si import synthetic
    monkeypatch.setitem(synthetic.OPTIONS, 'maxiter', 1)
    synthetic.run(tmp_path)
    out = tmp_path/'si/synthetic'
    configuration = json.loads((out/'configuration.json').read_text())
    assert configuration['parallel_workers'] == 4
    assert configuration['modes'] != configuration['truth_modes']
    truth = np.load(out/'truth.npz')
    truncated = np.load(out/'truncated_truth.npz')
    assert len(truth['coefficients']) == 340
    assert np.count_nonzero(truncated['coefficients']) == 27
    for key, count in [('A', 180), ('K', 160)]:
        selected = np.array([name.startswith(key+'_') for name in truth['names']])
        assert selected.sum() == count
        omitted = selected & (truncated['coefficients'] == 0)
        fraction = np.sum(truth['coefficients'][omitted]**2)/np.sum(truth['coefficients'][selected]**2)
        assert fraction > .1
    assert np.sqrt(np.mean((truth['A']-truncated['A'])**2))*1e6 > .01
    assert np.sqrt(np.mean((truth['Kref']-truncated['Kref'])**2)) > .01
    tables = {key: pd.read_csv(out/'problem'/f'{key}.csv') for key in ['heatflow', 'gradient', 'curie']}
    assert configuration['curie_relative_sigma'] == .075
    assert configuration['curie_error_model'] == 'independent_gaussian'
    assert configuration['complete_rectangular_mode_support']
    assert configuration['layout_selected_using_truth']
    np.testing.assert_array_equal(tables['heatflow'].noise_sigma, 5.)
    np.testing.assert_array_equal(tables['gradient'].noise_sigma, 2.5)
    g, c = tables['gradient'], tables['curie']
    cells = np.floor((g[['X', 'Y']].to_numpy()+1e4)/2e4).astype(int)
    counts = g.groupby([cells[:, 0], cells[:, 1]])['observed'].transform('size')
    np.testing.assert_allclose(g.sigma, 2.5*np.sqrt(counts))
    np.testing.assert_allclose(c.sigma, .075*c.truth)
    covariance = np.load(out/'noise_covariance.npz')['curie']
    expected = np.diag(c.sigma.to_numpy()**2)
    np.testing.assert_allclose(covariance, expected, atol=1e-12)
    noise = np.random.default_rng(configuration['seeds']['noise'])
    for key in ['gradient', 'heatflow', 'curie']:
        table = tables[key]
        np.testing.assert_allclose(table.noise, noise.normal(0, table.noise_sigma.to_numpy()), atol=1e-12)
    precision = np.load(out/'problem/bundle.npz')['curie_precision']
    np.testing.assert_allclose(precision@covariance, np.eye(len(c)), atol=1e-12)
    assert np.linalg.eigvalsh(covariance).min() > 0
    for case in synthetic.CASES:
        summary = json.loads((out/case/'summary.json').read_text())
        settings = json.loads((out/case/'training_config.json').read_text())
        result = np.load(out/case/'result.npz')
        assert settings['parameters'] == len(result['coefficients']) == 27
        assert sum(name.startswith('A_') for name in result['names']) == 14
        assert sum(name.startswith('K_') for name in result['names']) == 13
        assert settings['complete_rectangular_mode_support']
        assert settings['layout_selected_using_truth']
        assert set(result['names']) == set(truncated['names'][truncated['coefficients'] != 0])
        assert settings['modes'] == configuration['modes']
        assert summary['iterations'] == 1
        assert settings['options'] == configuration['options']
        for key, table in tables.items():
            observed = pd.read_csv(out/case/f'{key}.csv').observed
            expected = table.truth+(0 if case == 'joint_noise_free' else table.noise)
            np.testing.assert_allclose(observed, expected, atol=1e-12)
        if case == 'no_curie':
            assert summary['losses']['curie'] == 0
        else:
            curie = pd.read_csv(out/case/'curie.csv')
            independent_loss = .5*np.sum(((curie.predicted-curie.observed)/curie.sigma)**2)
            np.testing.assert_allclose(summary['losses']['curie'], independent_loss, rtol=1e-12)
        if case == 'no_heatflow':
            assert summary['losses']['heatflow'] == 0

    # Check the adjoint with the complete low-order inverse basis.
    problem = synthetic.inverse_problem(out/'problem')
    problem.solver.tolerance, problem.solver.residual_tolerance = 1e-9, 1e-12
    coefficients, direction = np.full(27, .03), np.linspace(-.6, .7, 27)
    _, gradient = problem.evaluate(coefficients)
    # Keep the derivative check within the same piecewise-linear Curie-depth intervals.
    step = 1e-4
    upper = problem.evaluate(coefficients+step*direction)[0]
    upper_segments = np.argmax(problem.ops['curie']@problem.last['T'].reshape(-1, 201) >= 580., axis=1)
    lower = problem.evaluate(coefficients-step*direction)[0]
    lower_segments = np.argmax(problem.ops['curie']@problem.last['T'].reshape(-1, 201) >= 580., axis=1)
    np.testing.assert_array_equal(upper_segments, lower_segments)
    np.testing.assert_allclose(gradient@direction, (upper-lower)/(2*step), rtol=2e-5, atol=1e-3)


def test_ncc_geojson_controls_domain_and_keeps_summary_regions_separate(tmp_path, monkeypatch):
    import json
    import geometry
    from si import regions as regional

    def feature(name, west, east):
        return dict(type='Feature', properties=dict(name=name), geometry=dict(type='Polygon',
            coordinates=[[[west, 0], [east, 0], [east, 1], [west, 1], [west, 0]]]))

    path = tmp_path/'regions.geojson'
    path.write_text(json.dumps(dict(type='FeatureCollection', features=[
        feature('WNCC', 0, 1), feature('TNCO', 1, 2), feature('ENCC', 2, 3), feature('NCC', 0, 3)])),
        encoding='utf-8')
    monkeypatch.setattr(regional, 'WINDOWS', {})
    lon, lat = np.array([[.5, 1.5, 2.5, 3.5]]), np.full((1, 4), .5)
    core = geometry.inside(geometry.regions(path), lon, lat)
    np.testing.assert_array_equal(core, [[True, True, True, False]])
    weights = regional.region_weights(dict(longitude=lon, latitude=lat, core=core,
                                          horizontal_scale=np.ones_like(lon)), path)
    assert set(weights) == {'WNCC', 'TNCO', 'ENCC'}
    for i, name in enumerate(['WNCC', 'TNCO', 'ENCC']):
        np.testing.assert_array_equal(weights[name], np.eye(4)[i:i+1])


def test_deep_conductivity_exports_filtered_samples_and_weights(tmp_path):
    import references
    columns = ['Sample ID', 'Primary Reference (Bibtexkey)', 'Petrographic term', 'Temperature (K)',
        'Pressure (Pa)', 'Saturating Fluid', 'Value bulk thermal conductivity (W/(m·K))',
        'Value total porosity (%)', 'Value effective porosity (%)']
    rows = [[rock, 'study_2000', rock, temperature, 1e5, 'water', 3., 1., 1.]
            for rock in ['granite', 'gneiss', 'gabbro', 'amphibolite', 'peridotite', 'serpentinite']
            for temperature in [293.15, 373.15]]
    source = tmp_path/'p3.txt'
    pd.DataFrame(rows, columns=columns).to_csv(source, sep='\t', index=False, encoding='latin1')
    output = tmp_path/'references'
    output.mkdir()
    values = references.deep_conductivity(output, source)
    records = pd.read_csv(output/'deep_conductivity_records.csv')
    samples = pd.read_csv(output/'deep_conductivity_samples.csv')
    parameters = pd.read_csv(output/'deep_conductivity_parameters.csv')
    assert not records['sample'].eq('serpentinite').any()
    for i, layer in enumerate(['upper_crust', 'middle_crust', 'lower_crust', 'mantle']):
        part = samples[samples.layer == layer]
        np.testing.assert_allclose(part.weight.sum(), 1., atol=1e-14)
        np.testing.assert_allclose(part.weight@part.log_K0, values[i], atol=1e-14)
    np.testing.assert_allclose(parameters.log_K0, values, atol=1e-14)
