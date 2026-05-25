from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from data_loading import (
    GRID_FILENAME,
    LAYER_FILENAME,
    STRATUM_FILLED_FILENAME,
    THERMAL_GRID_FILENAME,
    THERMAL_PROPERTIES_FILENAME,
    build_regular_index,
    load_surface_temperature_grid,
)
from direct_parameter import DirectParameterModel
from heat_solver import HeatSolver
from lithosphere_model import LayerGrid, build_reference_fields
from prior_network import PriorParameterModel, build_prior_feature_pack


REFERENCE_TEMPERATURE_FILENAME = "grid_1km_temperature_0_200km.csv"
TEMPERATURE_COLUMN_RE = re.compile(r"^Temperature_(\d+(?:\.\d+)?)km_C$")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _find_upward(start: Path, filename: str) -> Path | None:
    for parent in (start, *start.parents):
        candidate = parent / filename
        if candidate.exists():
            return candidate
    return None


def _resolve_checkpoint(output_dir: Path, checkpoint: Path | None) -> Path:
    if checkpoint is not None:
        path = checkpoint if checkpoint.is_absolute() else output_dir / checkpoint
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    direct = output_dir / "checkpoint.pt"
    if direct.exists():
        return direct

    matches = sorted(output_dir.rglob("checkpoint.pt"))
    if not matches:
        raise FileNotFoundError(f"No checkpoint.pt found in {output_dir}")
    if len(matches) > 1:
        joined = "\n".join(str(path) for path in matches[:20])
        raise ValueError(
            "Multiple checkpoint.pt files were found. Pass --checkpoint explicitly.\n"
            f"Found:\n{joined}"
        )
    return matches[0]


def _first(sources: list[dict[str, Any]], names: tuple[str, ...], default: Any = None) -> Any:
    for source in sources:
        for name in names:
            value = source.get(name)
            if value is not None:
                return value
    return default


def _run_context(checkpoint: dict[str, Any], checkpoint_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = _load_json(checkpoint_path.parent / "summary.json")
    member_summary = _load_json(checkpoint_path.parent / "member_summary.json")
    manifest_path = _find_upward(checkpoint_path.parent, "manifest.json")
    manifest = _load_json(manifest_path) if manifest_path is not None else {}

    run_options: dict[str, Any] = {}
    run_options.update(manifest.get("options") or {})
    run_options.update(summary.get("run") or {})
    run_options.update({key: value for key, value in member_summary.items() if key.startswith("config_")})

    model_config: dict[str, Any] = {}
    model_config.update(summary.get("model_config") or {})
    model_config.update(checkpoint.get("model_config") or {})
    return run_options, model_config


def _depth_label(depth_km: float) -> str:
    if math.isclose(depth_km, round(depth_km), abs_tol=1e-6):
        return f"{int(round(depth_km)):03d}km"
    text = f"{depth_km:.3f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"{text}km"


def _base_grid_frame(grid: LayerGrid) -> pd.DataFrame:
    rows, cols = np.indices(grid.shape)
    return pd.DataFrame(
        {
            "row": rows.reshape(-1),
            "col": cols.reshape(-1),
            "lon": grid.lon.reshape(-1),
            "lat": grid.lat.reshape(-1),
            "X": grid.x.reshape(-1),
            "Y": grid.y.reshape(-1),
        }
    )


def _wide_field_frame(grid: LayerGrid, field: np.ndarray, z_nodes_km: np.ndarray, prefix: str, unit: str) -> pd.DataFrame:
    values = field.reshape(-1, field.shape[-1])
    columns = [f"{prefix}_{_depth_label(float(depth))}_{unit}" for depth in z_nodes_km]
    return pd.concat([_base_grid_frame(grid), pd.DataFrame(values, columns=columns)], axis=1)


def _block_reduce_nanmean_3d(arr: np.ndarray, stride: int) -> np.ndarray:
    ny, nx, nz = arr.shape
    out = np.full(((ny + stride - 1) // stride, (nx + stride - 1) // stride, nz), np.nan, dtype=float)
    for i0 in range(0, ny, stride):
        for j0 in range(0, nx, stride):
            block = arr[i0 : i0 + stride, j0 : j0 + stride, :]
            finite = np.isfinite(block)
            counts = finite.sum(axis=(0, 1))
            sums = np.where(finite, block, 0.0).sum(axis=(0, 1))
            valid = counts > 0
            out[i0 // stride, j0 // stride, valid] = sums[valid] / counts[valid]
    return out


def _temperature_columns(path: Path) -> tuple[list[str], np.ndarray]:
    header = pd.read_csv(path, encoding="utf-8-sig", nrows=0)
    pairs: list[tuple[float, str]] = []
    for column in header.columns:
        match = TEMPERATURE_COLUMN_RE.match(column)
        if match:
            pairs.append((float(match.group(1)), column))
    if not pairs:
        raise ValueError(f"No Temperature_XXXkm_C columns found in {path}")
    pairs.sort(key=lambda item: item[0])
    return [column for _, column in pairs], np.array([depth for depth, _ in pairs], dtype=float)


def _load_reference_temperature(
    reference_csv: Path,
    data_dir: Path,
    stride: int,
    target_depths_km: np.ndarray,
    target_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    columns, ref_depths = _temperature_columns(reference_csv)
    grid_df, x_unique, y_unique, ri, ci = build_regular_index(data_dir / GRID_FILENAME)
    temp_df = pd.read_csv(reference_csv, encoding="utf-8-sig", usecols=["id", *columns])
    merged = grid_df[["id"]].merge(temp_df, on="id", how="left")

    full = np.full((len(y_unique), len(x_unique), len(columns)), np.nan, dtype=float)
    full[ri, ci, :] = merged[columns].to_numpy(dtype=float)
    coarse = _block_reduce_nanmean_3d(full, stride)
    if coarse.shape[:2] != target_shape:
        raise ValueError(f"Reference grid shape {coarse.shape[:2]} does not match model shape {target_shape}.")
    return _interpolate_reference_depths(coarse, ref_depths, target_depths_km), ref_depths


def _interpolate_reference_depths(ref: np.ndarray, ref_depths_km: np.ndarray, target_depths_km: np.ndarray) -> np.ndarray:
    out = np.full((*ref.shape[:2], target_depths_km.size), np.nan, dtype=float)
    flat_ref = ref.reshape(-1, ref.shape[-1])
    flat_out = out.reshape(-1, out.shape[-1])
    for i, values in enumerate(flat_ref):
        valid = np.isfinite(values)
        if np.count_nonzero(valid) == 0:
            continue
        if np.count_nonzero(valid) == 1:
            depth = ref_depths_km[valid][0]
            exact = np.isclose(target_depths_km, depth, atol=1e-6)
            flat_out[i, exact] = values[valid][0]
            continue
        flat_out[i, :] = np.interp(target_depths_km, ref_depths_km[valid], values[valid], left=np.nan, right=np.nan)
    return out


def _field_stats(values_like: np.ndarray) -> dict[str, float | int | None]:
    values = values_like[np.isfinite(values_like)]
    if values.size == 0:
        return {
            "n_valid": 0,
            "mean": None,
            "mean_abs": None,
            "rms": None,
            "std": None,
            "median": None,
            "min": None,
            "max": None,
            "max_abs": None,
        }
    return {
        "n_valid": int(values.size),
        "mean": float(np.mean(values)),
        "mean_abs": float(np.mean(np.abs(values))),
        "rms": float(np.sqrt(np.mean(values * values))),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "max_abs": float(np.max(np.abs(values))),
    }


def _temperature_diff_stats(diff: np.ndarray) -> dict[str, float | int | None]:
    stats = _field_stats(diff)
    return {
        "n_valid": stats["n_valid"],
        "mean_diff_C": stats["mean"],
        "mean_abs_diff_C": stats["mean_abs"],
        "rmse_C": stats["rms"],
        "std_diff_C": stats["std"],
        "median_diff_C": stats["median"],
        "min_diff_C": stats["min"],
        "max_diff_C": stats["max"],
        "max_abs_diff_C": stats["max_abs"],
    }


def _pearson_r(a: np.ndarray, b: np.ndarray) -> float | None:
    mask = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(mask) < 2:
        return None
    x = a[mask].astype(float)
    y = b[mask].astype(float)
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 0.0 or not np.isfinite(denom):
        return None
    return float(np.dot(x, y) / denom)


def _temperature_difference_metrics(T: np.ndarray, T_ref: np.ndarray, z_nodes_km: np.ndarray) -> tuple[pd.DataFrame, dict[str, Any]]:
    diff = T - T_ref
    rows = []
    for k, depth in enumerate(z_nodes_km):
        rows.append(
            {
                "depth_km": float(depth),
                **_temperature_diff_stats(diff[:, :, k]),
                "pearson_r": _pearson_r(T[:, :, k], T_ref[:, :, k]),
            }
        )
    overall = _temperature_diff_stats(diff)
    overall["pearson_r"] = _pearson_r(T, T_ref)
    return pd.DataFrame(rows), overall


def _nearest_depth_indices(z_nodes_km: np.ndarray, requested_depths_km: list[float] | None) -> list[int]:
    if requested_depths_km is None:
        defaults = [0.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 160.0, 200.0]
        requested_depths_km = [depth for depth in defaults if z_nodes_km[0] <= depth <= z_nodes_km[-1]]
        if len(requested_depths_km) < 4:
            n = min(6, z_nodes_km.size)
            requested_depths_km = np.linspace(z_nodes_km[0], z_nodes_km[-1], n).tolist()

    indices: list[int] = []
    for depth in requested_depths_km:
        idx = int(np.argmin(np.abs(z_nodes_km - depth)))
        if idx not in indices:
            indices.append(idx)
    return indices


def _plot_slices(
    field: np.ndarray,
    grid: LayerGrid,
    z_nodes_km: np.ndarray,
    depth_indices: list[int],
    path: Path,
    *,
    title: str,
    cmap: str,
    center_zero: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    n = len(depth_indices)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.8 * nrows), squeeze=False)
    selected = np.stack([field[:, :, idx] for idx in depth_indices])
    finite = selected[np.isfinite(selected)]
    if finite.size == 0:
        vmin = vmax = None
    elif center_zero:
        vmax_abs = float(np.max(np.abs(finite)))
        vmin, vmax = -vmax_abs, vmax_abs
    else:
        vmin, vmax = float(np.min(finite)), float(np.max(finite))

    extent = [
        float(np.nanmin(grid.x) / 1000.0),
        float(np.nanmax(grid.x) / 1000.0),
        float(np.nanmin(grid.y) / 1000.0),
        float(np.nanmax(grid.y) / 1000.0),
    ]
    image = None
    for ax, idx in zip(axes.ravel(), depth_indices):
        image = ax.imshow(field[:, :, idx], origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"{z_nodes_km[idx]:.1f} km")
        ax.set_xlabel("X (km)")
        ax.set_ylabel("Y (km)")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82, pad=0.01)
    fig.suptitle(title)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_difference_curves(metrics: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    plot_df = metrics[metrics["n_valid"] > 0]
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(plot_df["depth_km"], plot_df["mean_diff_C"], marker="o", label="mean(pred-ref)")
    ax.plot(plot_df["depth_km"], plot_df["mean_abs_diff_C"], marker="o", label="MAE")
    ax.plot(plot_df["depth_km"], plot_df["rmse_C"], marker="o", label="RMSE")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Depth (km)")
    ax.set_ylabel("Temperature difference (°C)")
    ax.set_title("Temperature difference by depth")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_correlation_curve(metrics: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    plot_df = metrics[(metrics["n_valid"] > 1) & metrics["pearson_r"].notna()]
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(plot_df["depth_km"], plot_df["pearson_r"], marker="o", color="tab:green")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Depth (km)")
    ax.set_ylabel("Pearson r")
    ax.set_title("Temperature spatial correlation by depth")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_mean_profiles(T: np.ndarray, T_ref: np.ndarray, z_nodes_km: np.ndarray, path: Path) -> None:
    import matplotlib.pyplot as plt

    pred_mean = np.nanmean(T, axis=(0, 1))
    ref_mean = np.nanmean(T_ref, axis=(0, 1))
    diff_mean = pred_mean - ref_mean

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    axes[0].plot(pred_mean, z_nodes_km, label="pred")
    axes[0].plot(ref_mean, z_nodes_km, label="reference")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Mean temperature (°C)")
    axes[0].set_ylabel("Depth (km)")
    axes[0].set_title("Mean temperature profile")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(diff_mean, z_nodes_km, marker="o", label="pred-ref")
    axes[1].axvline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Mean difference (°C)")
    axes[1].set_ylabel("Depth (km)")
    axes[1].set_title("Mean profile difference")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_difference_histogram(diff: np.ndarray, path: Path) -> None:
    import matplotlib.pyplot as plt

    values = diff[np.isfinite(diff)]
    if values.size == 0:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(values, bins=80, color="tab:blue", alpha=0.8)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Temperature difference pred-ref (°C)")
    ax.set_ylabel("Count")
    ax.set_title("Overall temperature difference distribution")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _compute_heat_flux(solver: HeatSolver, T: np.ndarray, conductivity: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edge_order = 2 if min(T.shape) >= 3 else 1
    dT_dy, dT_dx, dT_dz = np.gradient(T, solver.dy, solver.dx, solver.dz, edge_order=edge_order)
    qx = -conductivity * dT_dx * 1e3
    qy = -conductivity * dT_dy * 1e3
    qz_up = conductivity * dT_dz * 1e3
    return qx, qy, qz_up


def _build_evaluation_state(
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    args: argparse.Namespace,
) -> tuple[LayerGrid, HeatSolver, torch.nn.Module, torch.Tensor | None, np.ndarray, dict[str, Any]]:
    run_options, model_config = _run_context(checkpoint, checkpoint_path)
    arg_source = {
        "data_dir": args.data_dir,
        "stride": args.stride,
        "dz_km": args.dz_km,
        "z_max_km": args.z_max_km,
        "layer_transition_z_km": args.layer_transition_z_km,
        "layer_transition_xy_km": args.layer_transition_xy_km,
    }
    arg_source = {key: value for key, value in arg_source.items() if value is not None}
    sources = [arg_source, checkpoint, model_config, run_options]

    data_dir = Path(_first(sources, ("data_dir",), "Data"))
    stride = int(_first(sources, ("stride",), 10))
    dz_km = float(_first(sources, ("dz_km",), 2.0))
    z_max_km = float(_first(sources, ("z_max_km",), 120.0))
    layer_transition_z_km = float(_first(sources, ("layer_transition_z_km",), 2.0))
    layer_transition_xy_km = float(_first(sources, ("layer_transition_xy_km",), 20.0))
    mode = str(_first([model_config, checkpoint, run_options], ("mode",), "direct"))

    grid = LayerGrid(
        data_dir / GRID_FILENAME,
        data_dir / LAYER_FILENAME,
        stride=stride,
        stratum_csv=data_dir / STRATUM_FILLED_FILENAME,
    )
    surface_temp = load_surface_temperature_grid(data_dir / GRID_FILENAME, data_dir / THERMAL_GRID_FILENAME, stride)
    z_nodes_km = np.linspace(0.0, z_max_km, round(z_max_km / dz_km) + 1)
    refs = build_reference_fields(
        grid,
        z_nodes_km,
        thermal_properties_path=data_dir / THERMAL_PROPERTIES_FILENAME,
        layer_transition_z_km=layer_transition_z_km,
        layer_transition_xy_km=layer_transition_xy_km,
    )
    solver = HeatSolver(
        x=grid.x,
        y=grid.y,
        z_nodes_km=refs.z_nodes_km,
        surface_temp=surface_temp,
        K0_ref=refs.K0_ref,
        c_ref=refs.c_ref,
        A_ref=refs.A_ref,
        layer_masks=refs.layer_masks,
    )

    model_kwargs = {
        "bottom_x": int(_first([model_config, checkpoint, run_options], ("bottom_x", "config_bottom_x"), 3)),
        "bottom_y": int(_first([model_config, checkpoint, run_options], ("bottom_y", "config_bottom_y"), 3)),
        "k_property_x": int(_first([model_config, run_options], ("k_property_x", "k_prop_x", "config_k_prop_x"), 6)),
        "k_property_y": int(_first([model_config, run_options], ("k_property_y", "k_prop_y", "config_k_prop_y"), 6)),
        "a_property_x": int(_first([model_config, run_options], ("a_property_x", "a_prop_x", "config_a_prop_x"), 8)),
        "a_property_y": int(_first([model_config, run_options], ("a_property_y", "a_prop_y", "config_a_prop_y"), 8)),
        "k_property_scale": float(_first([model_config, run_options], ("k_property_scale", "k_prop_scale"), 0.7)),
        "a_property_scale": float(_first([model_config, run_options], ("a_property_scale", "a_prop_scale"), 0.7)),
        "alpha_scale": float(_first([model_config, run_options], ("alpha_scale",), 800.0)),
        "bottom_mode": str(_first([model_config, run_options], ("bottom_mode",), "dct")),
        "k_mode": str(_first([model_config, run_options], ("k_mode",), "fixed")),
        "a_mode": str(_first([model_config, run_options], ("a_mode",), "fixed")),
    }

    features = None
    if mode == "prior_net":
        features, _ = build_prior_feature_pack(data_dir, grid, stride)
        model = PriorParameterModel(
            grid.shape[0],
            grid.shape[1],
            in_channels=features.shape[1],
            width=int(_first([model_config, run_options], ("width", "config_width"), 32)),
            **model_kwargs,
        )
    elif mode == "direct":
        model = DirectParameterModel(grid.shape[0], grid.shape[1], **model_kwargs)
    else:
        raise ValueError(f"Unsupported checkpoint mode: {mode}")

    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device=torch.device("cpu"), dtype=torch.float64).eval()
    metadata = {
        "mode": mode,
        "data_dir": str(data_dir),
        "stride": stride,
        "dz_km": dz_km,
        "z_max_km": z_max_km,
        "layer_transition_z_km": layer_transition_z_km,
        "layer_transition_xy_km": layer_transition_xy_km,
        "model_config": {"mode": mode, **model_kwargs, "width": model_config.get("width")},
    }
    return grid, solver, model, features, z_nodes_km, metadata


def _evaluate_fields(
    solver: HeatSolver,
    model: torch.nn.Module,
    features: torch.Tensor | None,
    *,
    T_ref: float,
    verbose: bool,
) -> dict[str, np.ndarray]:
    dtype = torch.float64
    device = torch.device("cpu")
    feature_t = None if features is None else features.to(device=device, dtype=dtype)
    with torch.no_grad():
        coeffs = model(feature_t)
        T_bot_t, K0_t, A_t, decoded = model.decode_configured_parameter_fields(
            solver,
            coeffs,
            T_ref=T_ref,
            dtype=dtype,
            device=device,
        )
    T_bot = T_bot_t.detach().cpu().numpy()
    K0 = K0_t.detach().cpu().numpy()
    A = A_t.detach().cpu().numpy()
    solver.clear_warm_start()
    T = solver.solve(T_bot, K0=K0, A_source=A, reuse_last=False, verbose=verbose)
    surface_heat_flow = solver.surface_heat_flow_upward_np(T, K0=K0)
    conductivity = HeatSolver._conductivity_from_temperature(K0, solver.c_ref, T)
    heat_flux_x, heat_flux_y, heat_flux_z_up = _compute_heat_flux(solver, T, conductivity)
    return {
        "T": T,
        "K": conductivity,
        "K0": K0,
        "A": A,
        "T_bot": T_bot,
        "surface_heat_flow_mW_m2": surface_heat_flow,
        "heat_flux_x_mW_m2": heat_flux_x,
        "heat_flux_y_mW_m2": heat_flux_y,
        "heat_flux_z_up_mW_m2": heat_flux_z_up,
        "T_bot_delta": decoded["T_bot_delta"].detach().cpu().numpy(),
        "K_log_xy": decoded["K_log_xy"].detach().cpu().numpy(),
        "A_log_xy": decoded["A_log_xy"].detach().cpu().numpy(),
    }


def _write_outputs(
    eval_dir: Path,
    grid: LayerGrid,
    z_nodes_km: np.ndarray,
    fields: dict[str, np.ndarray],
    metadata: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    eval_dir.mkdir(parents=True, exist_ok=True)
    depth_indices = _nearest_depth_indices(z_nodes_km, args.slice_depths)

    npz_payload = {
        "x": grid.x,
        "y": grid.y,
        "lon": grid.lon,
        "lat": grid.lat,
        "z_nodes_km": z_nodes_km,
        **fields,
    }

    reference_summary: dict[str, Any] | None = None
    T_ref_field = None
    if args.skip_reference:
        reference_summary = {
            "reference_temperature_comparison": "skipped",
            "reference_temperature_skip_reason": "disabled by --skip-reference",
        }
    else:
        reference_csv = args.reference_temperature_csv or Path(metadata["data_dir"]) / REFERENCE_TEMPERATURE_FILENAME
        if reference_csv.exists():
            T_ref_field, _ = _load_reference_temperature(
                reference_csv,
                Path(metadata["data_dir"]),
                int(metadata["stride"]),
                z_nodes_km,
                fields["T"].shape[:2],
            )
            diff = fields["T"] - T_ref_field
            metrics, overall = _temperature_difference_metrics(fields["T"], T_ref_field, z_nodes_km)
            metrics.to_csv(eval_dir / "temperature_difference_by_depth.csv", index=False)
            _plot_difference_curves(metrics, eval_dir / "temperature_difference_by_depth.png")
            _plot_correlation_curve(metrics, eval_dir / "temperature_correlation_by_depth.png")
            _plot_mean_profiles(fields["T"], T_ref_field, z_nodes_km, eval_dir / "mean_temperature_profile_comparison.png")
            _plot_difference_histogram(diff, eval_dir / "temperature_difference_histogram.png")
            _plot_slices(
                diff,
                grid,
                z_nodes_km,
                depth_indices,
                eval_dir / "temperature_difference_slices.png",
                title="Temperature difference slices (pred-ref, °C)",
                cmap="coolwarm",
                center_zero=True,
            )
            npz_payload["T_reference"] = T_ref_field
            npz_payload["T_difference"] = diff
            reference_summary = {
                "reference_temperature_csv": str(reference_csv),
                "reference_temperature_comparison": "computed",
                "temperature_correlation_by_depth_png": str(eval_dir / "temperature_correlation_by_depth.png"),
                "overall_temperature_difference": overall,
            }
        elif args.reference_temperature_csv is not None:
            raise FileNotFoundError(f"Reference temperature CSV not found: {reference_csv}")
        else:
            reference_summary = {
                "reference_temperature_csv": str(reference_csv),
                "reference_temperature_comparison": "skipped",
                "reference_temperature_skip_reason": "default reference file not found",
            }

    np.savez_compressed(eval_dir / "evaluated_fields.npz", **npz_payload)

    surface_df = _base_grid_frame(grid)
    surface_df["T_bot_C"] = fields["T_bot"].reshape(-1)
    surface_df["surface_heat_flow_mW_m2"] = fields["surface_heat_flow_mW_m2"].reshape(-1)
    surface_df.to_csv(eval_dir / "surface_fields.csv", index=False)

    if not args.skip_field_csv:
        _wide_field_frame(grid, fields["T"], z_nodes_km, "Temperature", "C").to_csv(eval_dir / "temperature_field.csv", index=False)
        _wide_field_frame(grid, fields["K"], z_nodes_km, "Conductivity", "W_mK").to_csv(eval_dir / "conductivity_field.csv", index=False)
        _wide_field_frame(grid, fields["heat_flux_x_mW_m2"], z_nodes_km, "HeatFluxX", "mW_m2").to_csv(
            eval_dir / "heat_flux_x_field.csv",
            index=False,
        )
        _wide_field_frame(grid, fields["heat_flux_y_mW_m2"], z_nodes_km, "HeatFluxY", "mW_m2").to_csv(
            eval_dir / "heat_flux_y_field.csv",
            index=False,
        )
        _wide_field_frame(grid, fields["heat_flux_z_up_mW_m2"], z_nodes_km, "HeatFluxZUp", "mW_m2").to_csv(
            eval_dir / "heat_flux_z_up_field.csv",
            index=False,
        )

    _plot_slices(
        fields["T"],
        grid,
        z_nodes_km,
        depth_indices,
        eval_dir / "temperature_slices.png",
        title="Predicted temperature slices (°C)",
        cmap="inferno",
    )

    summary = {
        "checkpoint": str(args.checkpoint_path),
        "evaluation_dir": str(eval_dir),
        "field_npz": str(eval_dir / "evaluated_fields.npz"),
        "surface_fields_csv": str(eval_dir / "surface_fields.csv"),
        "temperature_field_csv": None if args.skip_field_csv else str(eval_dir / "temperature_field.csv"),
        "conductivity_field_csv": None if args.skip_field_csv else str(eval_dir / "conductivity_field.csv"),
        "heat_flux_x_field_csv": None if args.skip_field_csv else str(eval_dir / "heat_flux_x_field.csv"),
        "heat_flux_y_field_csv": None if args.skip_field_csv else str(eval_dir / "heat_flux_y_field.csv"),
        "heat_flux_z_up_field_csv": None if args.skip_field_csv else str(eval_dir / "heat_flux_z_up_field.csv"),
        "temperature_slices_png": str(eval_dir / "temperature_slices.png"),
        "slice_depths_km": [float(z_nodes_km[idx]) for idx in depth_indices],
        "field_shapes": {key: list(value.shape) for key, value in fields.items()},
        "surface_heat_flow_mW_m2": _field_stats(fields["surface_heat_flow_mW_m2"]),
        "temperature_C": _field_stats(fields["T"]),
        "conductivity_W_mK": _field_stats(fields["K"]),
        "heat_flux_x_mW_m2": _field_stats(fields["heat_flux_x_mW_m2"]),
        "heat_flux_y_mW_m2": _field_stats(fields["heat_flux_y_mW_m2"]),
        "heat_flux_z_up_mW_m2": _field_stats(fields["heat_flux_z_up_mW_m2"]),
        **metadata,
    }
    if reference_summary is not None:
        summary.update(reference_summary)
    with open(eval_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(description="Evaluate a HeatSolver checkpoint and compare it with reference temperatures.")
    parser.add_argument("output_dir", type=Path, help="Output directory containing checkpoint.pt, or a parent with one checkpoint.pt below it.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint path. Relative paths are resolved under output_dir.")
    parser.add_argument("--eval-dir", type=Path, default=None, help="Directory for evaluation outputs. Defaults to <checkpoint_dir>/evaluation.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Override data directory saved in the checkpoint.")
    parser.add_argument("--stride", type=int, default=None, help="Override grid stride saved in the checkpoint.")
    parser.add_argument("--dz-km", type=float, default=None, help="Override vertical grid spacing saved in the checkpoint.")
    parser.add_argument("--z-max-km", type=float, default=None, help="Override maximum depth saved in the checkpoint.")
    parser.add_argument("--layer-transition-z-km", type=float, default=None)
    parser.add_argument("--layer-transition-xy-km", type=float, default=None)
    parser.add_argument("--t-ref", type=float, default=1300.0, help="Reference bottom temperature used for fixed bottom mode and DCT deltas.")
    parser.add_argument("--slice-depths", "--slice-depths-km", type=float, nargs="*", default=None, help="Depths in km for slice plots.")
    parser.add_argument("--reference-temperature-csv", type=Path, default=None, help="Reference temperature CSV. Defaults to Data/grid_1km_temperature_0_200km.csv.")
    parser.add_argument("--skip-reference", action="store_true", help="Skip reference-temperature comparison.")
    parser.add_argument("--skip-field-csv", action="store_true", help="Only write evaluated_fields.npz and surface_fields.csv for fields.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    output_dir = args.output_dir
    checkpoint_path = _resolve_checkpoint(output_dir, args.checkpoint)
    args.checkpoint_path = checkpoint_path
    eval_dir = args.eval_dir or checkpoint_path.parent / "evaluation"

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    grid, solver, model, features, z_nodes_km, metadata = _build_evaluation_state(checkpoint, checkpoint_path, args)
    if not args.quiet:
        print(f"Evaluating checkpoint: {checkpoint_path}")
        print(f"Grid: {grid.shape[0]}x{grid.shape[1]}x{z_nodes_km.size}; output: {eval_dir}")
    fields = _evaluate_fields(solver, model, features, T_ref=args.t_ref, verbose=not args.quiet)
    summary = _write_outputs(eval_dir, grid, z_nodes_km, fields, metadata, args)
    if not args.quiet:
        print(f"Evaluation summary: {summary['evaluation_dir']}/evaluation_summary.json")


if __name__ == "__main__":
    main()
