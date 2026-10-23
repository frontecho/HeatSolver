import numpy as np

from si.basal import boundary, case_c_slice, interpolate_boundary, moho_flux


def test_moho_flux_matches_constant_conductivity_linear_geotherm():
    z = np.array([0., 1., 3., 5.])
    b = dict(z_km=z, thermal_interfaces_km=np.array([[[0., 1., 2., 2.5]]]))
    temperature = (12. + 10.*z)[None, None, :]
    conductivity = np.full_like(temperature, 3.)
    np.testing.assert_allclose(moho_flux(z, b['thermal_interfaces_km'][..., 3], temperature, conductivity), [[30.]])


def test_source_interpolation_preserves_linear_field_and_marks_extension():
    nodes = np.array([[100., 33., 1000.], [101., 33., 1100.],
                      [100., 34., 1200.], [101., 34., 1300.]])
    b = dict(longitude=np.array([[100.25, 100.75, 102.]]),
             latitude=np.array([[33.25, 33.75, 34.]]),
             x=np.zeros((1, 3)), y=np.zeros((1, 3)))
    T, extended = interpolate_boundary(nodes, b['longitude'], b['latitude'])
    np.testing.assert_allclose(T, [[1075., 1225., 1300.]])
    np.testing.assert_array_equal(extended, [[False, False, True]])


def test_case_c_uses_complete_columns_and_extends_only_below_source_base():
    data = np.zeros((25, 2, 5))
    data[:, :, 0] = [108., 116.]
    data[:, :, 1] = 37.
    data[:, 0, 2] = np.linspace(-150000, 0, 25)
    data[:, 1, 2] = np.linspace(-80000, 0, 25)
    data[:, 0, 3] = 10+(1370-10)*(-data[:, 0, 2]/150000)
    data[:, 1, 3] = 10+(1328-10)*(-data[:, 1, 2]/80000)
    nodes, extended = case_c_slice(data.reshape(-1, 5))
    np.testing.assert_allclose(nodes[:, 2], [10+1360*2/3, 1328+.5*(100-80)])
    np.testing.assert_array_equal(extended, [False, True])


def test_modified_boundaries_preserve_area_mean():
    rng = np.random.default_rng(55)
    b = dict(T_bot=1000.+rng.random((6, 8))*400., core=np.ones((6, 8), bool),
             domain_mask=np.ones((6, 8), bool), horizontal_scale=1.+rng.random((6, 8))*.1)
    b['core'][0] = False
    weight = b['core']/b['horizontal_scale']**2
    weight /= weight.sum()
    for case in ['original', 'smoothed', 'uniform']:
        values = boundary(b, case)
        np.testing.assert_allclose(np.sum(weight*values), np.sum(weight*b['T_bot']), atol=1e-10)
    np.testing.assert_allclose(np.ptp(boundary(b, 'uniform')), 0.)
    assert np.ptp(boundary(b, 'smoothed')) < np.ptp(b['T_bot'])


def test_complete_basal_workflow_uses_common_cases_and_training_settings(tmp_path, monkeypatch):
    import json
    import pandas as pd
    import pytest
    from pyproj import Transformer
    from config import CRS
    from data_io import write_json
    from fit import fit
    import si.basal as basal
    from test.test_uncertainty import small_problem

    problem = small_problem(tmp_path)
    bundle = dict(np.load(tmp_path/'problem/bundle.npz'))
    bundle['longitude'], bundle['latitude'] = Transformer.from_crs(CRS, 4326, always_xy=True).transform(bundle['x'], bundle['y'])
    bundle['thermal_interfaces_km'] = np.broadcast_to([1., 12., 22., 35.], (*bundle['x'].shape, 4))
    np.savez(tmp_path/'problem/bundle.npz', **bundle)
    weights = bundle['core']/bundle['core'].sum()
    monkeypatch.setattr(basal, 'region_weights', lambda b, path: {
        name: weights for name in ['WNCC', 'ENCC', 'N Shanxi', 'Ordos interior']})
    options = dict(maxiter=1, maxfun=10)
    fit(problem, tmp_path/'fit', np.zeros(len(problem.p.names)), options)
    training = dict(parameters=len(problem.p.names),
        modes=dict(A=[None, (2, 2), None, None, None], K=[(1, 1), None, None, None, (1, 1)]),
        constant_modes=dict(A=[True]*5, K=[True]*5), regularization=dict(A=4., K=4.), options=options)
    write_json(tmp_path/'fit/training_config.json', training)
    training = json.loads((tmp_path/'fit/training_config.json').read_text())
    nodes = np.array([[112., 36.], [114., 36.], [112., 39.], [114., 39.]])
    columns = np.zeros((25, 4, 5))
    columns[:, :, :2] = nodes
    columns[:, :, 2] = np.linspace(-150000., 0., 25)[:, None]
    columns[:, :, 3] = 10.-columns[:, :, 2]*.011
    sun = tmp_path/'sun.txt'
    np.savetxt(sun, columns.reshape(-1, 5))
    zhang = tmp_path/'zhang.xlsx'
    pd.DataFrame(np.column_stack([nodes, np.full(4, 100.), np.full(4, 1150.)]),
                 columns=['lon', 'lat', 'depth', 'T']).to_excel(zhang, sheet_name='Temperature', index=False)
    yan = tmp_path/'yan.txt'
    np.savetxt(yan, np.column_stack([nodes, np.full(4, 100.), np.zeros((4, 2)), np.full(4, 1473.15)]),
               header='lon lat depth unused unused T_K')
    monkeypatch.setattr(basal, 'FILES', dict(sun2022_c=sun, zhang2025=zhang, yan2019=yan))

    basal.run(tmp_path)

    output = tmp_path/'si/basal'
    assert {p.name for p in output.iterdir() if p.is_dir()} == set(basal.CASES)
    assert (output/'original/result.npz').read_bytes() == (tmp_path/'fit/result.npz').read_bytes()
    boundaries = np.load(output/'boundaries.npz')
    np.testing.assert_allclose(boundaries['sun2022_c'], 1110.)
    np.testing.assert_allclose(boundaries['zhang2025'], 1150.)
    np.testing.assert_allclose(boundaries['yan2019'], 1200.)
    for case in basal.CASES:
        assert json.loads((output/case/'training_config.json').read_text()) == training
        np.testing.assert_array_equal(np.load(output/case/'T_bot.npy'), boundaries[case])
        saved = np.load(output/case/'result.npz')
        np.testing.assert_allclose(saved['T'][..., -1], boundaries[case])
    metrics = pd.read_csv(output/'metrics.csv')
    assert metrics.case.tolist() == basal.CASES
    assert np.isfinite(metrics.select_dtypes('number')).all().all()
    assert set(pd.read_csv(output/'depth_profiles.csv').case) == set(basal.CASES)
    assert len(pd.read_csv(output/'layer_properties.csv')) == 60
    with pytest.raises(FileExistsError):
        basal.run(tmp_path)
