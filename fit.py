"""Fit the property coefficients by L-BFGS-B.

usage: python -m fit RESULTS
"""
import runtime  # Set BLAS thread counts before importing NumPy.
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from config import A_BOUNDS, BETA, CONSTANT_MODES, FIT_OPTIONS, K_BOUNDS, MODES, REGULARIZATION
from data_io import load_model
from observations import load as load_observations
from objective import Problem


def fit(problem, output, start, options):
    """Fit from coefficients start; observations with zero weight are predicted but not fitted."""
    output.mkdir(parents=True)
    history = []

    def callback(_):
        history.append(problem.last['losses'].copy())

    result = minimize(problem.evaluate, start, jac=True, method='L-BFGS-B',
                      bounds=[(-4, 4)]*len(start), options=options, callback=callback)
    problem.evaluate(result.x)
    s = problem.last
    np.savez_compressed(output/'result.npz', T=s['T'], K=s['K'], A=s['A'], q=s['q'], coefficients=result.x,
                        names=np.array(problem.p.names), layer_A=problem.p.values['A'], layer_K=problem.p.values['K'])
    misfit = problem.metrics()
    for key in ['heatflow', 'gradient', 'curie']:
        table = problem.data[key].assign(predicted=s[key])
        table.to_csv(output/f'{key}.csv', index=False)
    (output/'summary.json').write_text(json.dumps(dict(
        evaluation_region='all_domain_observations', success=bool(result.success), message=str(result.message), iterations=int(result.nit),
        evaluations=int(result.nfev), gradient_inf=float(abs(result.jac).max()), objective=float(result.fun),
        losses={k: float(v) for k, v in s['losses'].items()}, misfit=misfit, history=history), indent=2)+'\n')


def run(results):
    """Fit the configured model from zero coefficients."""
    bundle, output = results/'problem', results/'fit'
    problem = Problem(load_model(bundle), load_observations(bundle),
                      layout=MODES, constant_modes=CONSTANT_MODES,
                      a_bounds=A_BOUNDS, k_bounds=K_BOUNDS, beta=BETA,
                      regularization=REGULARIZATION)
    fit(problem, output, np.zeros(len(problem.p.names)), FIT_OPTIONS)
    training_config = dict(parameters=len(problem.p.names), modes=MODES, constant_modes=CONSTANT_MODES,
                           regularization=REGULARIZATION, options=FIT_OPTIONS)
    (output/'training_config.json').write_text(json.dumps(training_config, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fit from zero coefficients.')
    parser.add_argument('results', type=Path)
    cli = parser.parse_args()
    run(cli.results)
