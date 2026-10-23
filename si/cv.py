"""Spatial cross-validation: heat flow and gradients in three subsets of 150 km blocks, each withheld in turn.

Training excludes sites within 40 km of withheld ones; Curie depths and all other inputs are unchanged. Three
parameterizations are compared: the default one, additional A constant modes,
and more upper- and middle-crust A modes. Baselines are the unadjusted reference, the training mean and
inverse-distance interpolation of the 32 nearest training sites.

usage: python -m si.cv RESULTS   (writes RESULTS/si/cv)
"""
import runtime  # Set BLAS thread counts before importing NumPy.
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from config import A_BOUNDS, BETA, CONSTANT_MODES, FIT_OPTIONS, K_BOUNDS, MODES, REGULARIZATION, SOURCES
from fit import fit
from data_io import load_model
from observations import load as load_observations
from objective import Problem
from si.regions import region_weights

# Mode layout, L-BFGS memory, constant-mode switches.
MODELS = dict(selected=(MODES, CONSTANT_MODES),
              with_A_constants=(MODES, dict(CONSTANT_MODES, A=[True, True, True, True, False])),
              more_modes=(dict(MODES, A=[(3, 5), (9, 17), (5, 9), (3, 5), None]), CONSTANT_MODES))
KEYS = ['heatflow', 'gradient']


def folds(data):
    """Subset of each site by 150 km block colouring, and the training sites of each subset."""
    xy = {k: data[k][['X', 'Y']].to_numpy() for k in KEYS}
    origin = np.minimum(*(v.min(axis=0) for v in xy.values()))
    fold = {k: (np.floor((v-origin)/150e3).astype(int)@[1, 2]) % 3 for k, v in xy.items()}
    train = {}
    for i in range(3):
        tree = cKDTree(np.concatenate([xy[k][fold[k] == i] for k in KEYS]))
        train |= {(k, i): (fold[k] != i) & (tree.query(xy[k])[0] > 40e3) for k in KEYS}
    return fold, train


def refit(results, name, i):
    layout, constant_modes = MODELS[name]
    problem = Problem(load_model(results/'problem'), load_observations(results/'problem'), layout, constant_modes,
                      a_bounds=A_BOUNDS, k_bounds=K_BOUNDS, beta=BETA,
                      regularization=REGULARIZATION)
    _, train = folds(problem.data)
    for k in KEYS:
        problem.weights[k] = problem.weights[k]*train[k, i]
    fit(problem, results/'si/cv'/f'{name}_{i}', np.zeros(len(problem.p.names)),
        FIT_OPTIONS)


def run(results):
    out = results/'si/cv'
    jobs = [(name, i) for name in MODELS for i in range(3)]
    with ProcessPoolExecutor(3) as pool:
        list(pool.map(refit, [results]*len(jobs), *zip(*jobs)))

    problem = Problem(load_model(results/'problem'), load_observations(results/'problem'),
                      layout=MODES, constant_modes=CONSTANT_MODES,
                      a_bounds=A_BOUNDS, k_bounds=K_BOUNDS, beta=BETA,
                      regularization=REGULARIZATION)
    problem.evaluate(np.zeros(len(problem.p.names)))
    fold, train = folds(problem.data)
    rows = []
    for k in KEYS:
        d = problem.data[k]
        y, xy = d.observed.to_numpy(), d[['X', 'Y']].to_numpy()
        # Gradients: each 20 km cell has total weight one, as in the fit.
        cells = np.floor((xy+1e4)/2e4).astype(int)
        weight = 1/pd.Series(range(len(d))).groupby([cells[:, 0], cells[:, 1]]).transform('size').to_numpy() if k == 'gradient' else np.ones(len(d))
        t = d[['X', 'Y', 'observed']].assign(fold=fold[k], weight=weight, reference=problem.last[k])
        t['mean'], t['idw'] = np.nan, np.nan
        for name in MODELS:
            t[name] = np.nan
        for i in range(3):
            test, tr = fold[k] == i, train[k, i]
            t.loc[test, 'mean'] = np.average(y[tr], weights=weight[tr])
            distance, index = cKDTree(xy[tr]).query(xy[test], k=32)
            w = distance**-2
            t.loc[test, 'idw'] = np.sum(w*y[tr][index], axis=1)/w.sum(axis=1)
            for name in MODELS:
                t.loc[test, name] = pd.read_csv(out/f'{name}_{i}'/f'{k}.csv').predicted.to_numpy()[test]
        t.to_csv(out/f'{k}.csv', index=False)
        rows += [dict(observable=k, model=m, rmse=np.sqrt(np.average((t[m]-y)**2, weights=weight)))
                 for m in ['reference', 'mean', 'idw', *MODELS]]
    metrics = pd.DataFrame(rows).pivot(index='model', columns='observable', values='rmse')
    metrics.to_csv(out/'metrics.csv')

    # Withheld heat flow averaged with Gaussian kernels (truncated at 2 sigma) over the sites of one subset.
    t, rows = pd.read_csv(out/'heatflow.csv'), []
    for sigma in [0, 50, 100, 150]:
        pairs = {m: [] for m in ['selected', 'idw']}
        for i in range(3):
            f = t[t.fold == i]
            distance = cdist(f[['X', 'Y']], f[['X', 'Y']])/1e3
            W = np.eye(len(f)) if sigma == 0 else np.exp(-.5*(distance/sigma)**2)*(distance <= 2*sigma)
            W /= W.sum(axis=1, keepdims=True)
            keep = (1/np.sum(W**2, axis=1) >= 3) & ((W > 0).sum(axis=1) >= 5) if sigma else np.ones(len(f), bool)
            for m in pairs:
                pairs[m].append(np.c_[W@f.observed, W@f[m]][keep])
        for m, v in pairs.items():
            v = np.concatenate(v)
            rows.append(dict(sigma_km=sigma, model=m, n=len(v), rmse=np.sqrt(np.mean((v[:, 1]-v[:, 0])**2)),
                             r=np.corrcoef(v.T)[0, 1]))
    pd.DataFrame(rows).to_csv(out/'averaging.csv', index=False)

    # East minus west column heat production of each crustal layer in the fitted models.
    weights, b, rows = region_weights(problem.b, SOURCES['regions']), problem.b, []
    dz = np.r_[.5, np.ones(len(b['z_km'])-2), .5]*(b['z_km'][1]-b['z_km'][0])
    for name, path in [('reference', None), ('selected_all', results/'fit')]+[(f'{m}_{i}', out/f'{m}_{i}') for m, i in jobs]:
        A = problem.p.values['A'] if path is None else np.load(path/'result.npz')['layer_A']
        Q = np.sum(b['masks']*A*dz, axis=-1)
        rows.append(dict(model=name, **{layer: float(np.sum((weights['ENCC']-weights['WNCC'])*Q[i]))
                                        for i, layer in enumerate(['sediment', 'upper', 'middle', 'lower'])}))
    pd.DataFrame(rows).to_csv(out/'layer_heat_production.csv', index=False)


if __name__ == '__main__':
    run(Path(sys.argv[1]))
