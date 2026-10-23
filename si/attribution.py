"""Temperature changes from inverted heat production and conductivity: forward models with reference or inverted A and K.

usage: python -m si.attribution RESULTS   (writes RESULTS/si/attribution.csv)
"""
import runtime  # Set BLAS thread counts before importing NumPy.
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from config import A_BOUNDS, BETA, CONSTANT_MODES, K_BOUNDS, MODES, SOURCES
from data_io import load_model
from model import Model
from si.regions import region_weights


def run(results):
    problem = Model(load_model(results/'problem'),
                      layout=MODES, constant_modes=CONSTANT_MODES,
                      a_bounds=A_BOUNDS, k_bounds=K_BOUNDS, beta=BETA)
    p, b = problem.p, problem.b
    p.set(np.zeros(len(p.names)))
    reference = dict(p.values)
    with np.load(results/'fit/result.npz') as fitted:
        inverted = dict(A=fitted['layer_A'], K=fitted['layer_K'])
    weights, profiles = region_weights(b, SOURCES['regions']), {}
    for name, A, K in [('reference', reference, reference), ('A_only', inverted, reference),
                       ('K_only', reference, inverted), ('joint', inverted, inverted)]:
        p.values = dict(A=A['A'], K=K['K'])
        T = problem.solver.solve(np.sum(p.masks*p.values['A'], axis=0)*1e-6, b['surface_T'], b['T_bot'])[0]
        profiles[name] = np.concatenate([np.einsum('yx,yxz->z', w, T) for w in weights.values()])
    t = pd.DataFrame(dict(region=np.repeat(list(weights), len(b['z_km'])), depth_km=np.tile(b['z_km'], len(weights)), **profiles))
    # Symmetric two-factor split: each contribution averages both orders of introducing the change.
    t['A_contribution'] = .5*(t.A_only-t.reference+t.joint-t.K_only)
    t['K_contribution'] = .5*(t.K_only-t.reference+t.joint-t.A_only)
    (results/'si').mkdir(exist_ok=True)
    t.to_csv(results/'si/attribution.csv', index=False)


if __name__ == '__main__':
    run(Path(sys.argv[1]))
