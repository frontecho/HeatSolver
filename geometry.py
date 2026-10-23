"""Model grid, NCC extent and five-layer geometry from fixed inputs."""
import json
import re
import numpy as np
from matplotlib.path import Path as Polygon
from pyproj import Proj, Transformer
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree


def regions(path):
    """Complete NCC polygon exported to the fixed regional GeoJSON."""
    features = json.loads(path.read_text(encoding='utf-8'))['features']
    ncc = next(f['geometry'] for f in features if f['properties']['name'] == 'NCC')
    return [ncc['coordinates']]


def inside(polygons, lon, lat):
    points = np.column_stack([np.ravel(lon), np.ravel(lat)])
    result = np.zeros(len(points), bool)
    for rings in polygons:
        within = Polygon(np.asarray(rings[0])).contains_points(points, radius=1e-10)
        for hole in rings[1:]:
            within &= ~Polygon(np.asarray(hole)).contains_points(points, radius=-1e-10)
        result |= within
    return result.reshape(np.shape(lon))


def in_box(lon, lat, box):
    w, e, s, n = box
    return (lon >= w) & (lon <= e) & (lat >= s) & (lat <= n)


def composition_profiles(folder):
    """CSRM sediment-base/Moho depths and SiO2 profiles, ordered by coordinates."""
    return sorted([(float(m[1]), float(m[2]), np.loadtxt(path)[:, [0, 6]]) for path in folder.iterdir()
                   if (m := re.fullmatch(r'CSRM1\.0_([\d.]+)_([\d.]+)_SiO2.info', path.name))], key=lambda p: p[:2])


def idw4(points, query):
    distance, index = cKDTree(points).query(query, k=4)
    weights = 1/np.maximum(distance, 1.)**2
    return index, weights/weights.sum(axis=1, keepdims=True)


def build(xaxis, yaxis, z, box, crs, region_path, csrm, crust1):
    forward = Transformer.from_crs(4326, crs, always_xy=True)
    inverse = Transformer.from_crs(crs, 4326, always_xy=True)
    polygons = regions(region_path)
    xfull, yfull = np.meshgrid(xaxis, yaxis)
    lonfull, latfull = inverse.transform(xfull, yfull)
    domain = in_box(lonfull, latfull, box)
    iy, ix = np.where(domain)
    crop = slice(iy.min(), iy.max()+1), slice(ix.min(), ix.max()+1)
    x, y, lon, lat = xfull[crop], yfull[crop], lonfull[crop], latfull[crop]
    factors = Proj(crs).get_factors(lon, lat)
    b = dict(x=x, y=y, longitude=lon, latitude=lat, z_km=z, domain_mask=domain[crop],
             core=domain[crop] & inside(polygons, lon, lat),
             horizontal_scale=np.sqrt(factors.meridional_scale*factors.parallel_scale))

    # CSRM-1.0 composition profiles: the first and last samples give the sediment base and Moho.
    profiles = composition_profiles(csrm)
    index, weights = idw4(np.column_stack(forward.transform([p[0] for p in profiles], [p[1] for p in profiles])),
                          np.column_stack([x.ravel(), y.ravel()]))
    sediment_base = np.sum(np.array([p[2][0, 0] for p in profiles])[index]*weights, axis=1).reshape(x.shape)
    moho = np.sum(np.array([p[2][-1, 0] for p in profiles])[index]*weights, axis=1).reshape(x.shape)
    # CRUST1.0 thickness fractions of the upper, middle and lower crystalline crust.
    bounds = np.loadtxt(crust1).reshape(180, 360, 9)
    thickness = -np.diff(bounds[..., 5:9], axis=-1)
    fractions = RegularGridInterpolator((np.arange(-89.5, 90), np.arange(-179.5, 180)),
                                        (thickness/thickness.sum(axis=-1, keepdims=True))[::-1])(
                                            np.column_stack([lat.ravel(), lon.ravel()])).reshape(*x.shape, 3)
    span = moho-sediment_base
    interfaces = np.stack([sediment_base, sediment_base+span*fractions[..., 0],
                           sediment_base+span*fractions[..., :2].sum(axis=-1), moho], axis=-1)
    above = np.clip((interfaces[..., None]-z)/(z[1]-z[0])+.5, 0, 1)
    b['masks'] = np.moveaxis(np.concatenate([above[..., :1, :], np.diff(above, axis=-2), 1-above[..., -1:, :]], axis=-2), -2, 0)
    b['thermal_interfaces_km'] = interfaces

    return b
