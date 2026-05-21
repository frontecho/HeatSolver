"""Ordos Basin lithosphere thermal model.

Loads 1 km grid data, resamples to coarser stride, and provides
thermal property queries (A, K) as a function of grid position and depth.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

THERMAL_LAYER_NAMES = ["sediment", "upper_crust", "middle_crust", "lower_crust", "mantle"]
SEDIMENT_STRATUM_NAMES = [
    "xinshengjie",
    "baiexi",
    "zhuluoxi",
    "sandiexi",
    "erdiexi",
    "shitanxi",
    "aotaoxi",
    "hanwuxi",
]


@dataclass(frozen=True)
class ThermalReferenceFields:
    z_nodes_km: np.ndarray
    K0_ref: np.ndarray
    c_ref: np.ndarray
    A_ref: np.ndarray
    layer_masks: dict[str, np.ndarray]


def _require_finite(name: str, arr: np.ndarray) -> None:
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or inf values.")


def _block_reduce_nanmean(arr: np.ndarray, stride: int) -> np.ndarray:
    ny, nx = arr.shape
    out = np.full(((ny + stride - 1) // stride, (nx + stride - 1) // stride), np.nan, dtype=float)
    for i0 in range(0, ny, stride):
        for j0 in range(0, nx, stride):
            block = arr[i0 : i0 + stride, j0 : j0 + stride]
            if np.isfinite(block).any():
                out[i0 // stride, j0 // stride] = float(np.nanmean(block))
    return out


def _block_reduce_1d_nanmean(arr: np.ndarray, stride: int) -> np.ndarray:
    n = arr.shape[0]
    out = np.full((n + stride - 1) // stride, np.nan, dtype=float)
    for i0 in range(0, n, stride):
        block = arr[i0 : i0 + stride]
        if np.isfinite(block).any():
            out[i0 // stride] = float(np.nanmean(block))
    return out


class LayerGrid:
    """Resampled 2-D grid of crustal layer boundaries.

    Layer structure (top to bottom):
        sediment    0 → sed_base  (Total_Sediment from layer_csv)
        upper_crust sed_base → uc_base
        middle_crust uc_base → mc_base
        lower_crust  mc_base → moho
        mantle       moho → z_max

    Total_Sediment, Moho, thk_uc, and thk_mc come from layer_csv.
    thk_uc and thk_mc define the uc/mc interface within the crystalline
    crust; they are scaled so uc+mc <= moho-sed.

    Attributes (all 2-D arrays, shape (ny, nx)):
        x, y            projected coordinates (m)
        lon, lat        geographic coordinates (degrees)
        sed_thick       sediment thickness (km)
        uc_thick        upper-crust thickness (km), below sediment base
        mc_thick        middle-crust thickness (km)
        lc_thick        lower-crust thickness (km)
        moho            Moho depth (km)
    """

    def __init__(
        self,
        grid_csv: Path | str,
        layer_csv: Path | str,
        stride: int = 10,
        stratum_csv: Path | str | None = None,
    ) -> None:
        grid_csv   = Path(grid_csv)
        layer_csv  = Path(layer_csv)

        gdf  = pd.read_csv(grid_csv,   encoding="utf-8-sig")
        ldf  = pd.read_csv(layer_csv,  encoding="utf-8-sig")

        df = gdf.merge(ldf[["id", "Total_Sediment", "Moho", "thk_uc", "thk_mc"]], on="id", how="left")

        if stratum_csv is not None:
            sdf = pd.read_csv(Path(stratum_csv), encoding="utf-8-sig")[["id", *SEDIMENT_STRATUM_NAMES]]
            df = df.merge(sdf, on="id", how="left")


        # --- build regular 2-D grid via integer indices ---
        x_unique = np.sort(df["X"].unique())
        y_unique = np.sort(df["Y"].unique())
        nx_full  = len(x_unique)
        ny_full  = len(y_unique)

        xi = pd.Series(np.arange(nx_full), index=x_unique)
        yi = pd.Series(np.arange(ny_full), index=y_unique)
        ci = xi[df["X"].values].values
        ri = yi[df["Y"].values].values

        def _fill(col, dtype=float):
            arr = np.full((ny_full, nx_full), np.nan, dtype=dtype)
            arr[ri, ci] = df[col].values.astype(dtype)
            return arr

        lon_g  = _fill("lon")
        lat_g  = _fill("lat")
        x_g    = _fill("X")
        y_g    = _fill("Y")

        if stratum_csv is None:
            stratum_raw: dict[str, np.ndarray] = {}
            sed_raw = df["Total_Sediment"].values.astype(float)  # km
        else:
            stratum_raw = {name: df[name].values.astype(float) / 1000.0 for name in SEDIMENT_STRATUM_NAMES}
            sed_raw = np.sum([stratum_raw[name] for name in SEDIMENT_STRATUM_NAMES], axis=0)
        moh_raw = df["Moho"].values.astype(float)                    # km
        cryst   = np.maximum(moh_raw - sed_raw, 0.0)  # crystalline crust thickness

        # scale thk_uc / thk_mc so their sum fits within crystalline crust
        uc_raw  = df["thk_uc"].values.astype(float)
        mc_raw  = df["thk_mc"].values.astype(float)
        uc_mc   = uc_raw + mc_raw
        # avoid division by zero where uc_mc == 0
        scale   = np.where(uc_mc > 0, cryst / uc_mc, 0.0)
        # cap scale at 1 so we never expand beyond crystalline crust
        scale   = np.minimum(scale, 1.0)
        uc_scaled = uc_raw * scale
        mc_scaled = mc_raw * scale
        lc_scaled = np.maximum(cryst - uc_scaled - mc_scaled, 0.0)

        def _fill_arr(arr):
            g = np.full((ny_full, nx_full), np.nan)
            g[ri, ci] = arr
            return g

        sed_g  = _fill_arr(sed_raw)
        stratum_g = {name: _fill_arr(values) for name, values in stratum_raw.items()}
        uc_g   = _fill_arr(uc_scaled)
        mc_g   = _fill_arr(mc_scaled)
        lc_g   = _fill_arr(lc_scaled)
        moho_g = _fill_arr(moh_raw)


        # --- aggregate to coarse grid ---
        s = stride
        self.lon          = _block_reduce_nanmean(lon_g, s)
        self.lat          = _block_reduce_nanmean(lat_g, s)
        x_coarse          = _block_reduce_1d_nanmean(x_unique.astype(float), s)
        y_coarse          = _block_reduce_1d_nanmean(y_unique.astype(float), s)
        self.x            = np.broadcast_to(x_coarse[None, :], (y_coarse.size, x_coarse.size)).copy()
        self.y            = np.broadcast_to(y_coarse[:, None], (y_coarse.size, x_coarse.size)).copy()
        self.sed_thick    = _block_reduce_nanmean(sed_g, s)
        self.stratum_thick = {name: _block_reduce_nanmean(stratum_g[name], s) for name in SEDIMENT_STRATUM_NAMES} if stratum_csv is not None else None
        self.uc_thick     = _block_reduce_nanmean(uc_g, s)
        self.mc_thick     = _block_reduce_nanmean(mc_g, s)
        self.lc_thick     = _block_reduce_nanmean(lc_g, s)
        self.moho         = _block_reduce_nanmean(moho_g, s)
        self.shape        = self.x.shape

    # convenience: depth of each interface at (i, j)
    def boundaries(self, i: int, j: int) -> dict[str, float]:
        sed  = float(self.sed_thick[i, j])
        uc   = float(self.uc_thick[i, j])
        mc   = float(self.mc_thick[i, j])
        moh  = float(self.moho[i, j])
        return {
            "sed_base": sed,
            "uc_base":  sed + uc,
            "mc_base":  sed + uc + mc,
            "lc_base":  moh,
            "moho":     moh,
        }


def _horizontal_sigma_cells(grid: LayerGrid, transition_km: float) -> tuple[float, float]:
    x_unique = np.unique(grid.x[0, :])
    y_unique = np.unique(grid.y[:, 0])
    if x_unique.size < 2 or y_unique.size < 2:
        raise ValueError("Horizontal layer transition requires at least two grid nodes along x and y.")
    dx_km = float(np.median(np.diff(x_unique)) / 1000.0)
    dy_km = float(np.median(np.diff(y_unique)) / 1000.0)
    if dx_km <= 0.0 or dy_km <= 0.0:
        raise ValueError("Grid spacing must be positive for horizontal layer transition.")
    return transition_km / dy_km, transition_km / dx_km


def _smooth_reference_fields(
    K0: np.ndarray,
    c: np.ndarray,
    A: np.ndarray,
    layer_masks: dict[str, np.ndarray],
    sigma_y: float,
    sigma_x: float,
    sigma_z: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    field_sigma = (sigma_y, sigma_x, sigma_z)
    K0 = gaussian_filter(K0, sigma=field_sigma, mode="nearest")
    c = gaussian_filter(c, sigma=field_sigma, mode="nearest")
    A = gaussian_filter(A, sigma=field_sigma, mode="nearest")

    mask_stack = np.stack([layer_masks[name].astype(float) for name in THERMAL_LAYER_NAMES], axis=0)
    mask_stack = gaussian_filter(mask_stack, sigma=(0.0, sigma_y, sigma_x, sigma_z), mode="nearest")
    mask_stack = np.clip(mask_stack, 0.0, None)
    denom = np.sum(mask_stack, axis=0, keepdims=True)
    mask_stack = mask_stack / np.where(denom > 0.0, denom, 1.0)
    smoothed_masks = {name: mask_stack[i] for i, name in enumerate(THERMAL_LAYER_NAMES)}
    return K0, c, A, smoothed_masks


def _stratum_reference_fields(
    grid: LayerGrid,
    z_nodes_km: np.ndarray,
    stratum_parameters: dict[str, tuple[float, float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stratum_thick = getattr(grid, "stratum_thick", None)
    if stratum_thick is None:
        raise ValueError("grid has no stratum thickness data.")
    ny, nx = grid.shape
    nz = z_nodes_km.size
    z3 = z_nodes_km[np.newaxis, np.newaxis, :]
    K0 = np.zeros((ny, nx, nz), dtype=float)
    c = np.zeros((ny, nx, nz), dtype=float)
    A = np.zeros((ny, nx, nz), dtype=float)
    top = np.zeros((ny, nx), dtype=float)
    assigned = np.zeros((ny, nx, nz), dtype=bool)
    last_values = stratum_parameters[SEDIMENT_STRATUM_NAMES[-1]]
    for name in SEDIMENT_STRATUM_NAMES:
        bottom = top + stratum_thick[name]
        mask = (z3 >= top[:, :, np.newaxis]) & (z3 < bottom[:, :, np.newaxis])
        A_value, K0_value, c_value = stratum_parameters[name]
        K0[mask] = K0_value
        c[mask] = c_value
        A[mask] = A_value
        assigned |= mask
        top = bottom
        last_values = (A_value, K0_value, c_value)
    A_last, K0_last, c_last = last_values
    K0[~assigned] = K0_last
    c[~assigned] = c_last
    A[~assigned] = A_last
    return K0, c, A, assigned


def load_layer_parameters(path: Path | str) -> tuple[float, float, dict[str, tuple[float | None, float, float]]]:
    with open(Path(path), encoding="utf-8") as f:
        payload = json.load(f)
    explicit_layers = payload["layers"]
    upper_crust_cfg = explicit_layers["upper_crust"]
    a0 = float(upper_crust_cfg["A0"])
    decay_depth = float(upper_crust_cfg["D"])

    layers: dict[str, tuple[float | None, float, float]] = {}
    for layer in THERMAL_LAYER_NAMES:
        cfg = explicit_layers[layer]
        A = None if layer == "upper_crust" else float(cfg["A"])
        K0 = float(cfg["K0"])
        c = float(cfg["c"])
        layers[layer] = (A, K0, c)
    return a0, decay_depth, layers

def load_stratum_parameters(path: Path | str) -> dict[str, tuple[float, float, float]]:
    with open(Path(path), encoding="utf-8") as f:
        payload = json.load(f)
    source = payload["sedments"]
    parameters: dict[str, tuple[float, float, float]] = {}
    for name in SEDIMENT_STRATUM_NAMES:
        default = source[name]["default"]
        parameters[name] = (float(default["A"]), float(default["K0"]), float(default["c"]))
    return parameters

def build_reference_fields(
    grid: LayerGrid,
    z_nodes_km: np.ndarray,
    thermal_properties_path: Path | str,
    layer_transition_z_km: float = 0.0,
    layer_transition_xy_km: float = 0.0,
) -> ThermalReferenceFields:
    z_nodes_km = np.asarray(z_nodes_km, dtype=float)
    if z_nodes_km.ndim != 1 or z_nodes_km.size < 3:
        raise ValueError("z_nodes_km must be a 1-D array with at least 3 nodes.")
    if not np.all(np.isfinite(z_nodes_km)) or not np.all(np.diff(z_nodes_km) > 0):
        raise ValueError("z_nodes_km must be finite and strictly increasing.")
    if not np.isfinite(layer_transition_z_km) or layer_transition_z_km < 0.0:
        raise ValueError("layer_transition_z_km must be finite and non-negative.")
    if not np.isfinite(layer_transition_xy_km) or layer_transition_xy_km < 0.0:
        raise ValueError("layer_transition_xy_km must be finite and non-negative.")

    _require_finite("grid.sed_thick", grid.sed_thick)
    _require_finite("grid.uc_thick", grid.uc_thick)
    _require_finite("grid.mc_thick", grid.mc_thick)
    _require_finite("grid.moho", grid.moho)
    grid_stratum_thick = getattr(grid, "stratum_thick", None)
    if grid_stratum_thick is not None:
        for name, values in grid_stratum_thick.items():
            _require_finite(f"grid.stratum_thick[{name}]", values)

    a0, decay_depth, layers = load_layer_parameters(thermal_properties_path)
    has_stratum = grid_stratum_thick is not None
    stratum_params = load_stratum_parameters(thermal_properties_path) if has_stratum else {}

    ny, nx = grid.shape
    nz = z_nodes_km.size
    sed_base = grid.sed_thick
    uc_base = grid.sed_thick + grid.uc_thick
    mc_base = grid.sed_thick + grid.uc_thick + grid.mc_thick
    moho = grid.moho

    z3 = z_nodes_km[np.newaxis, np.newaxis, :]
    sed3 = sed_base[:, :, np.newaxis]
    uc3 = uc_base[:, :, np.newaxis]
    mc3 = mc_base[:, :, np.newaxis]
    moho3 = moho[:, :, np.newaxis]

    in_sed = z3 < sed3
    in_uc = (~in_sed) & (z3 < uc3)
    in_mc = (~in_sed) & (~in_uc) & (z3 < mc3)
    in_lc = (~in_sed) & (~in_uc) & (~in_mc) & (z3 < moho3)
    in_mn = ~(in_sed | in_uc | in_mc | in_lc)

    A_uc = a0 * np.exp(-(z3 - sed3) / decay_depth) * np.ones_like(z3)
    if has_stratum:
        sed_K0, sed_c, sed_A, sed_mask = _stratum_reference_fields(grid, z_nodes_km, stratum_params)
    else:
        sed_K0 = np.full((ny, nx, nz), layers["sediment"][1], dtype=float)
        sed_c = np.full((ny, nx, nz), layers["sediment"][2], dtype=float)
        sed_A = np.full((ny, nx, nz), layers["sediment"][0], dtype=float)
        sed_mask = in_sed

    K0 = np.empty((ny, nx, nz), dtype=float)
    c = np.empty((ny, nx, nz), dtype=float)
    A = np.empty((ny, nx, nz), dtype=float)

    K0[sed_mask] = sed_K0[sed_mask]
    K0[in_uc] = layers["upper_crust"][1]
    K0[in_mc] = layers["middle_crust"][1]
    K0[in_lc] = layers["lower_crust"][1]
    K0[in_mn] = layers["mantle"][1]

    c[sed_mask] = sed_c[sed_mask]
    c[in_uc] = layers["upper_crust"][2]
    c[in_mc] = layers["middle_crust"][2]
    c[in_lc] = layers["lower_crust"][2]
    c[in_mn] = layers["mantle"][2]

    A[sed_mask] = sed_A[sed_mask]
    A[in_uc] = A_uc[in_uc]
    A[in_mc] = layers["middle_crust"][0]
    A[in_lc] = layers["lower_crust"][0]
    A[in_mn] = layers["mantle"][0]
    layer_masks = {
        "sediment": sed_mask,
        "upper_crust": in_uc,
        "middle_crust": in_mc,
        "lower_crust": in_lc,
        "mantle": in_mn,
    }

    if layer_transition_z_km > 0.0 or layer_transition_xy_km > 0.0:
        sigma_y, sigma_x = (0.0, 0.0)
        if layer_transition_xy_km > 0.0:
            sigma_y, sigma_x = _horizontal_sigma_cells(grid, layer_transition_xy_km)
        dz_km = float(np.median(np.diff(z_nodes_km)))
        sigma_z = layer_transition_z_km / dz_km if layer_transition_z_km > 0.0 else 0.0
        K0, c, A, layer_masks = _smooth_reference_fields(K0, c, A, layer_masks, sigma_y, sigma_x, sigma_z)

    A *= 1e-6

    return ThermalReferenceFields(
        z_nodes_km=z_nodes_km,
        K0_ref=K0,
        c_ref=c,
        A_ref=A,
        layer_masks=layer_masks,
    )

