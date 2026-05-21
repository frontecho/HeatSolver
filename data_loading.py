from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

GRID_FILENAME = "grid_1km.csv"
LAYER_FILENAME = "grid_1km_layer_synthesis.csv"
STRATUM_FILLED_FILENAME = "grid_1km_stratum_filled.csv"
THERMAL_PROPERTIES_FILENAME = "thermal_properties.json"
THERMAL_GRID_FILENAME = "grid_1km_thermal.csv"
HEATFLOW_FILENAME = "sample_heatflow_processed.csv"

GEOPHYS_2D_SPECS = [
    ("grid_1km_gebco.csv", ["DEM"]),
    ("grid_1km_egm2008.csv", ["geoid", "vgg"]),
    ("grid_1km_wgm2012.csv", ["Bouguer", "Freeair", "Isostatic"]),
    ("grid_1km_mag_anomaly.csv", ["WDMAM2"]),
]
SEISMIC_MODEL_FILENAME = "CSRM1.0.csv"
MT_RESISTIVITY_FILENAME = "NCC.MT.resistivity.csv"
SEISMIC_DEPTH_BINS_KM = [(0.0, 20.0), (20.0, 60.0), (60.0, 120.0)]
MT_DEPTH_BINS_KM = [(0.0, 5.0), (5.0, 20.0), (20.0, 60.0)]


def build_regular_index(grid_csv: Path | str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gdf = pd.read_csv(grid_csv, encoding="utf-8-sig")
    x_unique = np.sort(gdf["X"].unique())
    y_unique = np.sort(gdf["Y"].unique())
    xi = pd.Series(np.arange(len(x_unique)), index=x_unique)
    yi = pd.Series(np.arange(len(y_unique)), index=y_unique)
    ci = xi[gdf["X"].values].values
    ri = yi[gdf["Y"].values].values
    return gdf, x_unique, y_unique, ri, ci


def fill_regular_grid(values: np.ndarray, ny: int, nx: int, ri: np.ndarray, ci: np.ndarray) -> np.ndarray:
    arr = np.full((ny, nx), np.nan, dtype=float)
    arr[ri, ci] = values.astype(float)
    return arr


def _block_reduce_nanmean(arr: np.ndarray, stride: int) -> np.ndarray:
    ny, nx = arr.shape
    out = np.full(((ny + stride - 1) // stride, (nx + stride - 1) // stride), np.nan, dtype=float)
    for i0 in range(0, ny, stride):
        for j0 in range(0, nx, stride):
            block = arr[i0 : i0 + stride, j0 : j0 + stride]
            if np.isfinite(block).any():
                out[i0 // stride, j0 // stride] = float(np.nanmean(block))
    return out


def load_surface_temperature_grid(grid_csv: Path | str, thermal_csv: Path | str, stride: int) -> np.ndarray:
    gdf, x_unique, y_unique, ri, ci = build_regular_index(grid_csv)
    tdf = pd.read_csv(thermal_csv, encoding="utf-8-sig", usecols=["id", "Surface_Temperature"])
    merged = gdf[["id"]].merge(tdf, on="id", how="left")
    full = fill_regular_grid(merged["Surface_Temperature"].to_numpy(dtype=float), len(y_unique), len(x_unique), ri, ci)
    return _block_reduce_nanmean(full, stride)


def load_prior_id_features(data_dir: Path, stride: int) -> tuple[list[np.ndarray], list[str]]:
    gdf, x_unique, y_unique, ri, ci = build_regular_index(data_dir / GRID_FILENAME)
    ny_full, nx_full = len(y_unique), len(x_unique)
    base = gdf[["id"]].copy()
    arrays: list[np.ndarray] = []
    channel_names: list[str] = []
    for filename, columns in GEOPHYS_2D_SPECS:
        fdf = pd.read_csv(data_dir / filename, encoding="utf-8-sig", usecols=["id", *columns])
        merged = base.merge(fdf, on="id", how="left")
        for col in columns:
            full = fill_regular_grid(merged[col].values, ny_full, nx_full, ri, ci)
            arrays.append(full[::stride, ::stride])
            channel_names.append(col)
    return arrays, channel_names


def nearest_point_grid(
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    points_lon: np.ndarray,
    points_lat: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    mask = np.isfinite(points_lon) & np.isfinite(points_lat) & np.isfinite(values)
    if not np.any(mask):
        return np.full_like(lon_grid, np.nan, dtype=float)
    pts = np.column_stack([points_lon[mask], points_lat[mask]])
    tree = cKDTree(pts)
    query = np.column_stack([lon_grid.reshape(-1), lat_grid.reshape(-1)])
    _, idx = tree.query(query, k=1)
    return values[mask][idx].reshape(lon_grid.shape)


def load_prior_seismic_features(data_dir: Path, grid: Any) -> tuple[list[np.ndarray], list[str]]:
    path = data_dir / SEISMIC_MODEL_FILENAME
    if not path.exists():
        return [], []
    df = pd.read_csv(path, usecols=["lon", "lat", "depth", "vs", "vp"])
    df = df.dropna(subset=["lon", "lat", "depth"])
    features: list[np.ndarray] = []
    names: list[str] = []
    for value_col in ["vs", "vp"]:
        sub = df.dropna(subset=[value_col])
        if sub.empty:
            continue
        for z0, z1 in SEISMIC_DEPTH_BINS_KM:
            sel = sub[(sub["depth"] >= z0) & (sub["depth"] < z1)]
            if sel.empty:
                continue
            agg = sel.groupby(["lon", "lat"], as_index=False)[value_col].mean()
            features.append(
                nearest_point_grid(
                    grid.lon,
                    grid.lat,
                    agg["lon"].to_numpy(dtype=float),
                    agg["lat"].to_numpy(dtype=float),
                    agg[value_col].to_numpy(dtype=float),
                )
            )
            names.append(f"{value_col}_{z0:.0f}_{z1:.0f}km")
    return features, names


def load_prior_mt_features(data_dir: Path, grid: Any) -> tuple[list[np.ndarray], list[str]]:
    path = data_dir / MT_RESISTIVITY_FILENAME
    if not path.exists():
        return [], []
    df = pd.read_csv(path, usecols=["lon", "lat", "depth", "resistivity"])
    features: list[np.ndarray] = []
    names: list[str] = []
    for z0, z1 in MT_DEPTH_BINS_KM:
        sel = df[(df["depth"] >= z0) & (df["depth"] < z1)].copy()
        if sel.empty:
            continue
        sel["log10_resistivity"] = np.log10(np.clip(sel["resistivity"].values.astype(float), 1e-6, None))
        agg = sel.groupby(["lon", "lat"], as_index=False)["log10_resistivity"].mean()
        features.append(
            nearest_point_grid(
                grid.lon,
                grid.lat,
                agg["lon"].to_numpy(dtype=float),
                agg["lat"].to_numpy(dtype=float),
                agg["log10_resistivity"].to_numpy(dtype=float),
            )
        )
        names.append(f"mt_log10rho_{z0:.0f}_{z1:.0f}km")
    return features, names


def load_heat_flow_observations(grid_csv: Path, heatflow_csv: Path, stride: int) -> np.ndarray:
    gdf, x_unique, y_unique, ri, ci = build_regular_index(grid_csv)
    hdf = pd.read_csv(heatflow_csv, encoding="utf-8-sig", usecols=["X", "Y", "Heat_Flow"])
    x = hdf["X"].to_numpy(dtype=float)
    y = hdf["Y"].to_numpy(dtype=float)
    heat_flow = hdf["Heat_Flow"].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(heat_flow)

    ny = (len(y_unique) + stride - 1) // stride
    nx = (len(x_unique) + stride - 1) // stride
    out = np.full((ny, nx), np.nan, dtype=float)
    if not np.any(valid):
        return out

    grid_points = np.column_stack([gdf["X"].to_numpy(dtype=float), gdf["Y"].to_numpy(dtype=float)])
    obs_points = np.column_stack([x[valid], y[valid]])
    _, nearest = cKDTree(grid_points).query(obs_points, k=1)
    obs_ri = ri[nearest] // stride
    obs_ci = ci[nearest] // stride
    values = heat_flow[valid]

    sums = np.zeros((ny, nx), dtype=float)
    counts = np.zeros((ny, nx), dtype=float)
    np.add.at(sums, (obs_ri, obs_ci), values)
    np.add.at(counts, (obs_ri, obs_ci), 1.0)
    observed = counts > 0.0
    out[observed] = sums[observed] / counts[observed]
    return out
