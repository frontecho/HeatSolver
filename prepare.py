"""Prepare geometry, boundary temperatures and observations from fixed inputs.

usage: python -m prepare RESULTS
"""
import runtime  # Set BLAS thread counts before importing NumPy.
import argparse
from pathlib import Path
import numpy as np
import boundary, geometry, observations
from config import BOX, CRS, SOURCES, X, Y, Z


def prepare(results):
    output = results/'problem'
    b = geometry.build(X, Y, Z, BOX, CRS, SOURCES['regions'], SOURCES['csrm'], SOURCES['crust1'])
    b.update(boundary.build(b, SOURCES['surface_T'], SOURCES['bottom_T'], X, Y, CRS))
    axes = b['y'][:, 0], b['x'][0]
    heatflow, gradient, curie, b['curie_precision'] = observations.prepare(
        axes, b['domain_mask'], SOURCES['heatflow'], SOURCES['curie'], SOURCES['countries'], SOURCES['regions'], BOX, CRS)
    output.mkdir(parents=True)
    np.savez_compressed(output/'bundle.npz', **b)
    for key, table in [('heatflow', heatflow), ('gradient', gradient), ('curie', curie)]:
        table.to_csv(output/f'{key}.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    cli = parser.parse_args()
    prepare(cli.results)
