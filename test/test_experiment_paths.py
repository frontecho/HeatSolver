import numpy as np
import pandas as pd
import pytest

import prepare as preparation


def test_prepare_writes_problem_inside_experiment_root(tmp_path, monkeypatch):
    x, y = np.meshgrid([0., 1.], [0., 1.])
    geometry = dict(x=x, y=y, domain_mask=np.ones(x.shape, bool))
    boundary = dict(surface_T=np.full(x.shape, 10.), T_bot=np.full(x.shape, 1000.))
    table = pd.DataFrame(dict(observed=[42.]))
    monkeypatch.setattr(preparation.geometry, 'build', lambda *args: geometry.copy())
    monkeypatch.setattr(preparation.boundary, 'build', lambda *args: boundary)
    monkeypatch.setattr(preparation.observations, 'prepare',
                        lambda *args: (table, table, table, np.eye(1)))

    results = tmp_path/'run01'
    preparation.prepare(results)

    assert {path.name for path in results.iterdir()} == {'problem'}
    assert {path.name for path in (results/'problem').iterdir()} == {
        'bundle.npz', 'heatflow.csv', 'gradient.csv', 'curie.csv'}
    with np.load(results/'problem/bundle.npz') as saved:
        np.testing.assert_array_equal(saved['surface_T'], boundary['surface_T'])
        np.testing.assert_array_equal(saved['T_bot'], boundary['T_bot'])
    for name in ['heatflow', 'gradient', 'curie']:
        pd.testing.assert_frame_equal(pd.read_csv(results/f'problem/{name}.csv'), table)
    with pytest.raises(FileExistsError):
        preparation.prepare(results)
