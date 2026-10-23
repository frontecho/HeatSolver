"""KML-derived regional weights and shared SI longitude/latitude windows."""
import json
import numpy as np
from shapely import contains_xy
from shapely.geometry import shape

WINDOWS = {'N Shanxi': (111, 114, 38.5, 40.5), 'Ordos interior': (107, 109, 36, 38),
           'Shandong': (117.5, 120.5, 35.5, 38), 'SE Ordos': (109, 111, 35, 37),
           'NE Ordos Block': (110, 112, 39, 41), 'North China Basin': (115, 117, 35, 37)}
FULL = {'SE Ordos': 'Southeastern Ordos', 'N Shanxi': 'Northern Shanxi', 'NE Ordos Block': 'Northeastern Ordos'}


def inside(lon, lat, window):
    w, e, s, n = window
    return (lon >= w) & (lon <= e) & (lat >= s) & (lat <= n)


def region_weights(b, region_path):
    lon, lat, core = b['longitude'], b['latitude'], b['core']
    masks = {f['properties']['name']: contains_xy(shape(f['geometry']), lon, lat)
             for f in json.loads(region_path.read_text())['features']
             if f['properties']['name'] in ['WNCC', 'TNCO', 'ENCC']}
    masks |= {name: inside(lon, lat, window) for name, window in WINDOWS.items()
              if name in ['N Shanxi', 'Ordos interior']}
    area = 400/b['horizontal_scale']**2
    return {name: area*(m & core)/np.sum(area*(m & core)) for name, m in masks.items()}
