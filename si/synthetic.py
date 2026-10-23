"""Low-order recovery of a more complex, multiscale synthetic A/K model.

usage: python -m si.synthetic RESULTS   (writes RESULTS/si/synthetic)
"""
import runtime  # Set BLAS thread counts before importing NumPy.
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from config import A_BOUNDS, BETA, K_BOUNDS
from fit import fit
from data_io import load_model
from observations import load as load_observations
from objective import Problem

SEED = 917170
HEATFLOW_SIGMA = 5.
GRADIENT_SIGMA = 2.5
CURIE_RELATIVE_SIGMA = .075
CONSTANT_MODES = dict(A=[False]*5, K=[True]*5)
REGULARIZATION = dict(A=4., K=4.)
OPTIONS = dict(maxiter=150, maxfun=500, maxls=25, maxcor=40, ftol=1e-10, gtol=1e-4)
TRUTH_MODES = dict(A=[(5, 9), (6, 11), (5, 9), (4, 7), None],
                   K=[(6, 11), (5, 9), (4, 7), (3, 5), (2, 3)])
FIT_MODES = dict(A=[(2, 2), (2, 3), (2, 3), (1, 2), None],
                 K=[(2, 3), (1, 2), (2, 1), (1, 2), (1, 1)])
# RMS of the random logit-field component in each layer, before adding broad anomalies.
TRUTH_RANDOM_RMS = dict(A=.25, K=.20)
# Broad anomalies in the truth only; the inverse basis uses complete low-order rectangles.
PERTURBATIONS = {'A_sediment_y1_x1': .28, 'A_upper_crust_y0_x1': -.35, 'A_upper_crust_y1_x2': .18,
                 'A_middle_crust_y1_x0': .35, 'A_middle_crust_y1_x2': -.25, 'A_lower_crust_y0_x1': .28,
                 'K_sediment_y0_x0': .12, 'K_sediment_y1_x1': -.30, 'K_upper_crust_y0_x1': .18,
                 'K_middle_crust_y1_x0': -.20, 'K_lower_crust_y0_x1': .17, 'K_mantle_y0_x0': .08}
CASES = ['joint', 'no_curie', 'no_heatflow', 'joint_noise_free']


def bundle(folder):
    """Smooth interfaces, uniform layer references, random observation sites and noise."""
    y, x = np.meshgrid(np.arange(11)*2e4, np.arange(19)*2e4, indexing='ij')
    z = np.arange(0, 100.01, .5)
    u, v = x/x.max(), y/y.max()
    sediment, moho = 4+2*np.sin(np.pi*u)*np.sin(np.pi*v), 38+3*np.cos(np.pi*u)
    interfaces = np.stack([sediment, sediment+.32*(moho-sediment), sediment+.75*(moho-sediment), moho], axis=-1)
    above = np.clip((interfaces[..., None]-z)/.5+.5, 0, 1)
    masks = np.moveaxis(np.concatenate([above[..., :1, :], np.diff(above, axis=-2), 1-above[..., -1:, :]], axis=-2), -2, 0)
    layer = lambda values: np.broadcast_to(np.array(values)[:, None, None, None], masks.shape)
    np.savez(folder/'bundle.npz', x=x, y=y, z_km=z, masks=masks, core=np.ones(x.shape, bool), domain_mask=np.ones(x.shape, bool),
             horizontal_scale=np.ones(x.shape), surface_T=12+2*np.cos(np.pi*v),
             T_bot=1200+80*np.cos(np.pi*u)*np.cos(np.pi*v), curie_precision=np.eye(80))
    (folder.parent/'references').mkdir()
    with h5py.File(folder.parent/'references/references.h5', 'w') as h:
        h['fields/K_reference'] = layer([2.4, 2.9, 2.4, 2.15, 3.8])
        h['fields/A_reference'] = layer([1.2, 1.4, .75, .25, .03])
    rng = np.random.default_rng(SEED)
    sites = lambda n: dict(Y=rng.uniform(1e4, y.max()-1e4, n), X=rng.uniform(1e4, x.max()-1e4, n))
    g, q, c = sites(160), sites(120), sites(80)
    top = rng.uniform(.1, 1., 160)
    g.update(top_km=top, bottom_km=top+rng.uniform(.5, 1.5, 160))
    for key, table, sigma in [('gradient', g, GRADIENT_SIGMA), ('heatflow', q, HEATFLOW_SIGMA), ('curie', c, 1.)]:
        d = pd.DataFrame(dict(table, observed=0., sigma=sigma, noise_sigma=sigma))
        if key == 'gradient':
            cells = np.floor((d[['X', 'Y']].to_numpy()+1e4)/2e4).astype(int)
            d['sigma'] *= np.sqrt(d.groupby([cells[:, 0], cells[:, 1]])['observed'].transform('size'))
        d.to_csv(folder/f'{key}.csv', index=False)


def inverse_problem(folder):
    """Use every mode in the selected low-order rectangular layout."""
    return Problem(load_model(folder), load_observations(folder), FIT_MODES, CONSTANT_MODES,
                      a_bounds=A_BOUNDS, k_bounds=K_BOUNDS, beta=BETA, regularization=REGULARIZATION)


def fit_case(results, case, options):
    """Fit one case from zero using the shared truth and saved noise realization."""
    out = results/'si/synthetic'
    problem = inverse_problem(out/'problem')
    for d in problem.data.values():
        d['observed'] = d.truth+(0 if case == 'joint_noise_free' else d.noise)
    if case == 'no_heatflow':
        problem.weights['heatflow'] *= 0
    if case == 'no_curie':
        problem.b['curie_precision'] *= 0
    fit(problem, out/case, np.zeros(len(problem.p.names)), options)
    configuration = dict(modes=FIT_MODES, constant_modes=CONSTANT_MODES, regularization=REGULARIZATION,
                         complete_rectangular_mode_support=True, layout_selected_using_truth=True,
                         case=case, parameters=len(problem.p.names), options=options)
    (out/case/'training_config.json').write_text(json.dumps(configuration, indent=2)+'\n', encoding='utf-8')


def run(results):
    out = results/'si/synthetic'
    (out/'problem').mkdir(parents=True)
    bundle(out/'problem')
    problem = Problem(load_model(out/'problem'), load_observations(out/'problem'), TRUTH_MODES, CONSTANT_MODES,
                      a_bounds=A_BOUNDS, k_bounds=K_BOUNDS, beta=BETA, regularization=REGULARIZATION)
    p = problem.p
    rng = np.random.default_rng(SEED+1)
    truth = np.zeros(len(p.names))
    for key, layer, sl, _ in p.blocks:
        ny, nx = TRUTH_MODES[key][layer]
        frequencies = np.array([(y, x) for y in range(ny) for x in range(nx)
                                if (y, x) != (0, 0) or CONSTANT_MODES[key][layer]])
        spectrum = 1/(1+(frequencies[:, 0]/2)**2+(frequencies[:, 1]/4)**2)
        coefficients = rng.standard_normal(len(frequencies))*spectrum
        truth[sl] = coefficients*TRUTH_RANDOM_RMS[key]/np.linalg.norm(coefficients)
    for name, value in PERTURBATIONS.items():
        truth[p.names.index(name)] += value
    fitted_names = inverse_problem(out/'problem').p.names
    represented = np.array([name in fitted_names for name in p.names])
    states = {}
    for name, c in [('reference', np.zeros(len(p.names))), ('truth', truth),
                    ('truncated_truth', truth*represented)]:
        problem.last = None
        problem.evaluate(c)
        s = problem.last
        states[name] = dict(T=s['T'], A=s['A'], Kref=1/np.sum(p.masks/p.values['K'], axis=0),
                            heatflow=s['heatflow'], curie=s['curie'], gradient=s['gradient'])
        np.savez_compressed(out/f'{name}.npz', coefficients=c, names=p.names,
                            layer_A=p.values['A'], layer_K=p.values['K'], **states[name])
    noise = np.random.default_rng(SEED+2)
    for key in ['gradient', 'heatflow']:
        problem.data[key]['truth'] = states['truth'][key]
        problem.data[key]['noise'] = noise.normal(0, problem.data[key].noise_sigma.to_numpy())
    d = problem.data['curie']
    d['truth'] = states['truth']['curie']
    d['sigma'] = d['noise_sigma'] = CURIE_RELATIVE_SIGMA*d.truth
    covariance = np.diag(d.sigma.to_numpy()**2)
    d['noise'] = noise.normal(0, d.sigma.to_numpy())
    geometry = dict(np.load(out/'problem/bundle.npz'))
    geometry['curie_precision'] = np.diag(1/d.sigma.to_numpy()**2)
    np.savez(out/'problem/bundle.npz', **geometry)
    np.savez(out/'noise_covariance.npz', curie=covariance)
    for key, d in problem.data.items():
        d['observed'] = d.truth+d.noise
        d.to_csv(out/'problem'/f'{key}.csv', index=False)
    configuration = dict(modes=FIT_MODES, constant_modes=CONSTANT_MODES, regularization=REGULARIZATION,
                         options=OPTIONS, complete_rectangular_mode_support=True, layout_selected_using_truth=True,
                         truth_modes=TRUTH_MODES,
                         truth_random_logit_rms=TRUTH_RANDOM_RMS,
                         truth_random_spectrum='1/(1+(y/2)^2+(x/4)^2)',
                         truth_broad_anomalies=PERTURBATIONS,
                         seeds=dict(sites=SEED, truth=SEED+1, noise=SEED+2), cases=CASES,
                         heatflow_sigma_mW_m2=HEATFLOW_SIGMA, gradient_sigma_C_km=GRADIENT_SIGMA, gradient_cell_weighting=True,
                         curie_relative_sigma=CURIE_RELATIVE_SIGMA, curie_error_model='independent_gaussian',
                         parallel_workers=len(CASES))
    (out/'configuration.json').write_text(json.dumps(configuration, indent=2)+'\n', encoding='utf-8')
    with ProcessPoolExecutor(len(CASES)) as pool:
        list(pool.map(fit_case, [results]*len(CASES), CASES, [OPTIONS]*len(CASES)))

    # RMSE against the input model: volume-weighted T, crustal A/K20, and heat flow and Curie depth at all sites.
    dz = np.r_[.25, np.full(len(problem.b['z_km'])-2, .5), .25]
    volume = np.broadcast_to(dz, states['truth']['T'].shape)
    crust = volume*p.masks[:4].sum(axis=0)
    t, rows = states['truth'], []
    for case in CASES:
        r = np.load(out/case/'result.npz')
        rms = lambda a, b, w=None: float(np.sqrt(np.average((a-b)**2, weights=w)))
        rows.append(dict(case=case, T_C=rms(r['T'], t['T'], volume), A_crust_uW_m3=rms(r['A']*1e6, t['A']*1e6, crust),
                         K_crust_W_mK=rms(1/np.sum(p.masks/r['layer_K'], axis=0), t['Kref'], crust),
                         heatflow_mW_m2=rms(pd.read_csv(out/case/'heatflow.csv').predicted, t['heatflow']),
                         curie_km=rms(pd.read_csv(out/case/'curie.csv').predicted, t['curie'])))
    pd.DataFrame(rows).to_csv(out/'metrics.csv', index=False)


if __name__ == '__main__':
    run(Path(sys.argv[1]))
