"""Observation preparation, declared errors and table loading."""
import json
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist
from pyproj import Transformer
from geometry import in_box, inside, regions
from solver import sampler


def prepare(full_axes, active, heatflow_source, curie_source, country_path, region_path, box, crs):
    """Heat-flow cell medians, interval gradients and Curie depths whose stencils lie in the domain."""
    forward = Transformer.from_crs(4326, crs, always_xy=True)
    ncc = regions(region_path)
    countries = json.loads(country_path.read_text())
    china = next(feature['geometry']['coordinates'] for feature in countries['features'] if feature['properties']['ADMIN'] == 'China')
    areas = [abs(np.sum(np.asarray(p[0])[:, 0]*np.roll(np.asarray(p[0])[:, 1], -1)
        - np.asarray(p[0])[:, 1]*np.roll(np.asarray(p[0])[:, 0], -1))) for p in china]
    mainland = [china[int(np.argmax(areas))]]
    eligible = lambda lon, lat: in_box(lon, lat, box) & inside(ncc, lon, lat) & inside(mainland, lon, lat)
    raw = pd.DataFrame(json.loads(heatflow_source.read_text(encoding='utf-8-sig')))
    raw['lon'], raw['lat'], raw['q'] = (pd.to_numeric(raw[k]) for k in 'bco')
    selected = raw.p.isin(['A', 'B']) & eligible(raw.lon.to_numpy(), raw.lat.to_numpy())
    rows = raw[selected].copy()
    rows['X'], rows['Y'] = forward.transform(rows.lon.to_numpy(), rows.lat.to_numpy())
    within = lambda d: sampler(full_axes, d[['Y', 'X']].to_numpy())@active.ravel() >= 1-1e-10
    rows['ix'], rows['iy'] = (np.floor((rows[k]+1e4)/2e4).astype(int) for k in ['X', 'Y'])

    # One median per 20 km cell; sigma is the scaled median absolute deviation, at least 10 mW/m2.
    heatflow = rows.groupby(['ix', 'iy']).apply(lambda g: pd.Series(dict(
        X=g.X.median(), Y=g.Y.median(), observed=g.q.median(),
        sigma=max(10., 1.4826*np.median(abs(g.q-g.q.median()))), records=len(g), ids=';'.join(g.a.astype(str)))),
        include_groups=False).reset_index(drop=True)
    heatflow = heatflow[within(heatflow)].reset_index(drop=True)

    # Reported intervals in metres, upper 5 km; each 20 km cell has total weight one, error at least 5 C/km.
    depths = rows.g.astype(str).str.extract(r'^\s*(\d+(?:\.\d+)?)\s*[-–—~～－]\s*(\d+(?:\.\d+)?)\s*$').astype(float)/1000
    gradient = rows.assign(top_km=depths[0], bottom_km=depths[1], observed=pd.to_numeric(rows.h, errors='coerce'),
                           reported=pd.to_numeric(rows.i, errors='coerce').fillna(0))
    gradient = gradient[(gradient.observed > 0) & (gradient.bottom_km > gradient.top_km) & (gradient.bottom_km <= 5)]
    gradient = gradient[within(gradient)].copy()
    weight = 1/gradient.groupby(['ix', 'iy'])['observed'].transform('size')
    gradient['sigma'] = np.maximum(5., gradient.reported)/np.sqrt(weight)
    gradient = gradient[['a', 'X', 'Y', 'top_km', 'bottom_km', 'observed', 'sigma']].rename(columns=dict(a='id'))

    lon, lat, depth = np.loadtxt(curie_source).T
    keep = eligible(lon, lat)
    x, y = forward.transform(lon[keep], lat[keep])
    curie = pd.DataFrame(dict(X=x, Y=y, observed=depth[keep]))
    curie = curie[within(curie)].reset_index(drop=True)
    # Declared error model: 15% depth errors, Gaussian correlation over 150 km with a 25% nugget.
    xy, error = curie[['X', 'Y']].to_numpy()/1e3, .15*curie.observed.to_numpy()
    correlation = .75*np.exp(-.5*(cdist(xy, xy)/150)**2)+.25*np.eye(len(xy))
    precision = cho_solve(cho_factor(error[:, None]*correlation*error[None]), np.eye(len(xy)))
    return heatflow, gradient.reset_index(drop=True), curie, (precision+precision.T)/2


def load(bundle):
    return {key: pd.read_csv(bundle/f'{key}.csv') for key in ['heatflow', 'gradient', 'curie']}


