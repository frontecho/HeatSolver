"""Load the grids and observations used by direct thermal inversion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

GRID_FILENAME = "grid_1km.csv"
LAYER_FILENAME = "grid_1km_layer.csv"
STRATUM_FILLED_FILENAME = "grid_1km_stratum_filled.csv"
THERMAL_PROPERTIES_FILENAME = "thermal_properties.json"
THERMAL_GRID_FILENAME = "grid_1km_thermal.csv"
HEATFLOW_FILENAME = "sample_heatflow_processed.csv"


def build_regular_index(
    grid_csv: Path | str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grid = pd.read_csv(grid_csv, encoding="utf-8-sig")
    x_unique = np.sort(grid["X"].unique())
    y_unique = np.sort(grid["Y"].unique())
    columns = np.searchsorted(x_unique, grid["X"].to_numpy())
    rows = np.searchsorted(y_unique, grid["Y"].to_numpy())
    return grid, x_unique, y_unique, rows, columns


def fill_regular_grid(
    values: np.ndarray,
    ny: int,
    nx: int,
    rows: np.ndarray,
    columns: np.ndarray,
) -> np.ndarray:
    field = np.full((ny, nx), np.nan, dtype=np.float64)
    field[rows, columns] = np.asarray(values, dtype=np.float64)
    return field


def block_reduce_nanmean(field: np.ndarray, stride: int) -> np.ndarray:
    ny, nx = field.shape
    out_ny = ny // stride
    out_nx = nx // stride
    field = field[: out_ny * stride, : out_nx * stride]
    blocks = field.reshape(out_ny, stride, out_nx, stride)
    finite = np.isfinite(blocks)
    sums = np.where(finite, blocks, 0.0).sum(axis=(1, 3))
    counts = finite.sum(axis=(1, 3))
    return np.divide(
        sums,
        counts,
        out=np.full((out_ny, out_nx), np.nan),
        where=counts > 0,
    )


def load_surface_temperature_grid(
    grid_csv: Path | str,
    thermal_csv: Path | str,
    stride: int,
) -> np.ndarray:
    grid, x_unique, y_unique, rows, columns = build_regular_index(grid_csv)
    thermal = pd.read_csv(
        thermal_csv,
        encoding="utf-8-sig",
        usecols=["id", "Surface_Temperature"],
    )
    merged = grid[["id"]].merge(thermal, on="id", how="left")
    full = fill_regular_grid(
        merged["Surface_Temperature"].to_numpy(),
        y_unique.size,
        x_unique.size,
        rows,
        columns,
    )
    return block_reduce_nanmean(full, stride)


def load_heat_flow_observations(
    grid_csv: Path,
    heatflow_csv: Path,
    stride: int,
) -> np.ndarray:
    grid, x_unique, y_unique, rows, columns = build_regular_index(grid_csv)
    observations = pd.read_csv(
        heatflow_csv,
        encoding="utf-8-sig",
        usecols=["X", "Y", "Heat_Flow"],
    ).dropna()
    output = np.full(
        (
            y_unique.size // stride,
            x_unique.size // stride,
        ),
        np.nan,
    )
    grid_points = grid[["X", "Y"]].to_numpy(dtype=np.float64)
    observation_points = observations[["X", "Y"]].to_numpy(dtype=np.float64)
    nearest = cKDTree(grid_points).query(observation_points, k=1)[1]
    observation_rows = rows[nearest]
    observation_columns = columns[nearest]
    inside = (
        (observation_rows < output.shape[0] * stride)
        & (observation_columns < output.shape[1] * stride)
    )
    observation_rows = observation_rows[inside] // stride
    observation_columns = observation_columns[inside] // stride
    values = observations["Heat_Flow"].to_numpy(dtype=np.float64)[inside]
    sums = np.zeros_like(output)
    counts = np.zeros_like(output)
    np.add.at(sums, (observation_rows, observation_columns), values)
    np.add.at(counts, (observation_rows, observation_columns), 1.0)
    np.divide(sums, counts, out=output, where=counts > 0)
    return output
