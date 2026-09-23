"""Run an explicitly configured NCC paper experiment in a fresh output directory."""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
from scipy.optimize import minimize
from project import ROOT, write_json, digest
from experiments.model import InverseProblem
from experiments.analyze import export


def load_config(path):
    path = Path(path)
    config = json.loads(path.read_text(encoding='utf-8'))
    if config['action'] not in ['fit', 'reference', 'attribution', 'synthetic', 'baselines']:
        raise ValueError('Unknown experiment action')
    return config


def problem(config):
    return InverseProblem(ROOT/config['bundle'], variant=config.get('variant', 'AK170_budget'),
        regularization=config.get('regularization'), fold=config.get('fold', -1),
        split=ROOT/config['split'] if 'split' in config else None,
        qfactor=config.get('qfactor', 1.), boundary=config.get('boundary', 'original'))


def initialize(obj, config):
    initial = np.asarray(config.get('initial_coefficients', np.zeros(obj.n)), dtype=float)
    if initial.shape != (obj.n,) or not np.isfinite(initial).all():
        raise ValueError('Initial coefficients do not match the configured model')
    if obj.fold >= 0 and np.any(initial):
        raise ValueError('Spatial validation must start from the untrained reference')
    if 'initial_temperature' in config:
        source = config['initial_temperature']
        path = ROOT/source['path']
        if digest(path) != source['sha256']:
            raise ValueError('Initial temperature checksum mismatch')
        with np.load(path) as state:
            temperature = state['T'].copy()
        if source['adjust_boundary']:
            temperature += (obj.b['T_bot']-obj.original_bottom)[...,None]*(obj.b['z_km']/100)
        if temperature.shape != obj.b['A'].shape:
            raise ValueError('Initial temperature has the wrong grid shape')
        obj.last = {'T': temperature}
    return initial


def run(config_path, output):
    config = load_config(config_path)
    output = Path(output).resolve()
    if config['action'] == 'baselines':
        from experiments.analyze import baselines
        return baselines(ROOT/config['bundle'], ROOT/config['split'], output)
    if config['action'] == 'synthetic':
        from experiments import synthetic
        synthetic.SEED = config['seed']
        return synthetic.main(config['steps'], output=output)
    obj = problem(config)
    initial = initialize(obj, config)
    if config['action'] == 'attribution':
        return attribution(obj, output, config)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(config, coefficient_names=obj.properties.names,
                    core_sha256=digest(ROOT/'heat_solver.py'), runner_sha256=digest(__file__),
                    source_sha256={name: digest(ROOT/name) for name in [
                        'heat_solver.py', 'nonlinear_heat_solver.py', 'observations.py',
                        'spatial_basis.py', 'heat_flux.py', 'fit_metrics.py',
                        'experiments/model.py',
                        'experiments/analyze.py']})
    write_json(output/'manifest.json', manifest)
    history = []
    start = time.perf_counter()

    def objective(c):
        value, gradient = obj.evaluate(c)
        history.append(dict(evaluation=len(history)+1, objective=value, seconds=time.perf_counter()-start))
        return value, gradient

    def checkpoint(c):
        np.savez_compressed(output/'checkpoint.npz', coefficients=c, T=obj.last['T'])
        write_json(output/'progress.json', history[-1])

    if config['action'] == 'reference':
        obj.evaluate(initial)
        fit = SimpleNamespace(x=initial, nit=0, nfev=1, success=True, message='Untrained bounded reference')
    else:
        fit = minimize(objective, initial, jac=True, method='L-BFGS-B',
                       bounds=[(-4, 4)]*obj.n, options=config['options'], callback=checkpoint)
        obj.evaluate(fit.x)
    export(obj, output, fit)
    write_json(output/'fit_history.json', history)


def attribution(obj, output, config):
    """Four forward solutions with independently reference/fitted A and K."""
    output.mkdir(parents=True, exist_ok=False)
    p = obj.properties
    p.set(np.zeros(p.n))
    reference = {key: value.copy() for key, value in p.values.items()}
    p.set(obj.initial_coefficients)
    fitted = {key: value.copy() for key, value in p.values.items()}
    states = {}
    for name, ia, ik in [('reference', 0, 0), ('A_only', 1, 0), ('K_only', 0, 1), ('joint', 1, 1)]:
        p.values['A'] = (fitted if ia else reference)['A'].copy()
        p.values['K'] = (fitted if ik else reference)['K'].copy()
        A = np.sum(p.masks*p.values['A'], axis=0)*1e-6
        T, K, _ = obj.solver.solve(A, obj.b['T_bot'], obj.b['surface_T'])
        states[name] = T
        np.savez_compressed(output/(name+'.npz'), T=T, K=K, A=A)
    contribution_a = .5*(states['A_only']-states['reference']+states['joint']-states['K_only'])
    contribution_k = .5*(states['K_only']-states['reference']+states['joint']-states['A_only'])
    np.testing.assert_allclose(contribution_a+contribution_k, states['joint']-states['reference'], atol=1e-12)
    np.savez_compressed(output/'temperature_contributions.npz', A=contribution_a, K=contribution_k)
    write_json(output/'manifest.json', dict(config, source_sha256={name: digest(ROOT/name) for name in [
        'heat_solver.py', 'nonlinear_heat_solver.py', 'observations.py', 'spatial_basis.py',
        'heat_flux.py', 'experiments/model.py',
        'experiments/run.py']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', type=Path, help='New directory; existing results are never overwritten')
    parser.add_argument('--check', action='store_true', help='Load and validate inputs without solving or fitting')
    args = parser.parse_args()
    config = load_config(args.config)
    if args.check:
        if config['action'] == 'synthetic':
            from experiments.synthetic import Problem
            obj = Problem()
        else:
            obj = problem(config)
            initialize(obj, config)
        print(json.dumps(dict(parameters=obj.n, heatflow=len(obj.heatflow),
                              gradients=len(obj.records), magnetic_depths=len(obj.curie))))
    else:
        if args.output is None:
            parser.error('--output is required unless --check is supplied')
        run(args.config, args.output)


if __name__ == '__main__':
    main()
