"""Prescribed surface and 100-km temperatures on the supplied geometry."""
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree


def surface_temperature(source, xaxis, yaxis, crs):
    """Area average of the LSTC land-surface temperature, nearest-filled where the raster has no data."""
    shape = (len(yaxis), len(xaxis))
    with rasterio.open(source) as src:
        raw = src.read(1)
        raw[~np.isfinite(raw) | (raw == src.nodata)] = np.nan
        T = np.full(shape, np.nan)
        reproject(raw, T, src_transform=src.transform, src_crs=src.crs, src_nodata=np.nan,
                  dst_transform=from_origin(xaxis[0]-1e4, yaxis[-1]+1e4, 2e4, 2e4), dst_crs=crs, dst_nodata=np.nan,
                  resampling=Resampling.average)
    T = T[::-1].ravel()
    xy = np.column_stack([a.ravel() for a in np.meshgrid(xaxis, yaxis)])
    known = np.isfinite(T)
    T[~known] = T[known][cKDTree(xy[known]).query(xy[~known])[1]]
    return T.reshape(shape)


def build(b, surface_source, bottom_source, xaxis, yaxis, crs):
    crop = np.ix_(np.searchsorted(yaxis, b['y'][:, 0]), np.searchsorted(xaxis, b['x'][0]))
    surface = surface_temperature(surface_source, xaxis, yaxis, crs)[crop]
    sun = np.loadtxt(bottom_source).reshape(25, -1, 5)[0]
    bottom = LinearNDInterpolator(sun[:, :2], sun[:, 3])(b['longitude'], b['latitude'])
    return dict(surface_T=surface, T_bot=bottom)
