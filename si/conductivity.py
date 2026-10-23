"""Conductivity measurements compared with reference and inverted K at 20 C, harmonic means over each interval.

Records are those used to construct the shallow reference field.

usage: python -m si.conductivity RESULTS   (writes RESULTS/si/conductivity.csv)
"""
import runtime  # Set BLAS thread counts before importing NumPy.
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator
from config import A_BOUNDS, BETA, CONSTANT_MODES, CRS, K_BOUNDS, MODES
from data_io import load_model
from model import Model

def run(results):
    problem = Model(load_model(results/'problem'),
                      layout=MODES, constant_modes=CONSTANT_MODES,
                      a_bounds=A_BOUNDS, k_bounds=K_BOUNDS, beta=BETA)
    p, b = problem.p, problem.b
    d = pd.read_csv(results/'references/shallow_conductivity_samples.csv', low_memory=False)
    site = d.longitude.round(6).astype(str)+' '+d.latitude.round(6).astype(str)
    t = pd.DataFrame(dict(site=site, longitude=d.longitude, latitude=d.latitude,
                          top_km=d.depth_top_m/1000, bottom_km=d.depth_bottom_m/1000, observed=d.k_W_mK))
    x, y = Transformer.from_crs(4326, CRS, always_xy=True).transform(t.longitude.to_numpy(), t.latitude.to_numpy())
    axes = b['y'][:, 0], b['x'][0]
    inside = RegularGridInterpolator(axes, b['core'].astype(float), bounds_error=False, fill_value=0)(np.c_[y, x]) >= 1-1e-10
    t, x, y = t[inside].reset_index(drop=True), x[inside], y[inside]
    nodes, w = np.polynomial.legendre.leggauss(12)
    depth = t.top_km.to_numpy()[:, None]+(t.bottom_km-t.top_km).to_numpy()[:, None]*(nodes+1)/2
    points = np.stack([np.broadcast_to(y[:, None], depth.shape), np.broadcast_to(x[:, None], depth.shape), depth], axis=-1)
    for name, c in [('reference', np.zeros(len(p.names))), ('inverted', np.load(results/'fit/result.npz')['coefficients'])]:
        p.set(c)
        K20 = RegularGridInterpolator((*axes, b['z_km']), 1/np.sum(p.masks/p.values['K'], axis=0))(points)
        t[name] = 1/(np.sum(w/2/K20, axis=1))
    (results/'si').mkdir(exist_ok=True)
    t.to_csv(results/'si/conductivity.csv', index=False)


if __name__ == '__main__':
    run(Path(sys.argv[1]))
