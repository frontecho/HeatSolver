from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from evaluation import _build_evaluation_state, _load_json, _nearest_depth_indices, _resolve_checkpoint


DEFAULT_DEPTHS_KM = [20.0, 40.0, 60.0, 80.0, 100.0]


def _channel_group(name: str) -> str:
    lower = name.lower()
    if lower in {"dem", "gebco", "topography", "bathymetry"}:
        return "surface"
    if any(token in lower for token in ("geoid", "vgg", "bouguer", "freeair", "isostatic", "egm", "wgm", "gravity")):
        return "gravity"
    if any(token in lower for token in ("wdmam", "mag", "magnetic")):
        return "magnetic"
    if lower.startswith(("vs_", "vp_")) or "seis" in lower:
        return "seismic"
    if lower.startswith("mt_") or "rho" in lower or "resist" in lower:
        return "mt"
    return "other"


def _load_channel_names(checkpoint: dict[str, Any], checkpoint_path: Path, n_channels: int) -> list[str]:
    names = checkpoint.get("channels")
    if names is None:
        summary = _load_json(checkpoint_path.parent / "summary.json")
        names = (summary.get("prior_network") or {}).get("channels")
    if names is None:
        names = [f"channel_{idx:03d}" for idx in range(n_channels)]
    names = [str(name) for name in names]
    if len(names) != n_channels:
        raise ValueError(f"Expected {n_channels} channel names, got {len(names)}.")
    return names


def _build_arg_source(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        data_dir=args.data_dir,
        stride=args.stride,
        dz_km=args.dz_km,
        z_max_km=args.z_max_km,
        layer_transition_z_km=args.layer_transition_z_km,
        layer_transition_xy_km=args.layer_transition_xy_km,
    )


def _sample_baseline(
    features: torch.Tensor,
    baseline: str,
    generator: torch.Generator,
    noise_std: float,
) -> torch.Tensor:
    if baseline == "zero":
        base = torch.zeros_like(features)
    elif baseline == "noise":
        base = torch.randn(features.shape, dtype=features.dtype, device=features.device, generator=generator) * noise_std
    else:
        raise ValueError(f"Unsupported baseline: {baseline}")
    return base


def _forward_temperature_maps(
    solver: Any,
    model: torch.nn.Module,
    features: torch.Tensor,
    depth_indices: list[int],
    *,
    t_ref: float,
) -> list[torch.Tensor]:
    solver.clear_warm_start()
    coeffs = model(features)
    T_bot, K0, A, _ = model.decode_configured_parameter_fields(
        solver,
        coeffs,
        T_ref=t_ref,
        dtype=features.dtype,
        device=features.device,
    )
    T, _ = solver(T_bot, K0=K0, A_source=A)
    return [T[:, :, idx] for idx in depth_indices]


def _temperature_stats_np(
    solver: Any,
    model: torch.nn.Module,
    features: torch.Tensor,
    depth_indices: list[int],
    *,
    t_ref: float,
) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        maps = _forward_temperature_maps(solver, model, features, depth_indices, t_ref=t_ref)
    mean_values = []
    anomaly_rms_values = []
    for temperature_map in maps:
        values = temperature_map.detach().cpu().numpy()
        anomaly = values - float(np.mean(values))
        mean_values.append(float(np.mean(values)))
        anomaly_rms_values.append(float(np.sqrt(np.mean(anomaly * anomaly))))
    return np.asarray(mean_values, dtype=float), np.asarray(anomaly_rms_values, dtype=float)


def _forward_temperature_anomaly_maps(
    solver: Any,
    model: torch.nn.Module,
    features: torch.Tensor,
    depth_indices: list[int],
    *,
    t_ref: float,
) -> list[torch.Tensor]:
    maps = _forward_temperature_maps(solver, model, features, depth_indices, t_ref=t_ref)
    return [temperature_map - temperature_map.mean() for temperature_map in maps]


def _gradient_shap_temperature_anomalies(
    solver: Any,
    model: torch.nn.Module,
    features: torch.Tensor,
    depth_indices: list[int],
    *,
    t_ref: float,
    n_samples: int,
    baseline: str,
    baseline_noise_std: float,
    path_noise_std: float,
    finite_difference_eps: float,
    seed: int,
    quiet: bool,
) -> np.ndarray:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if finite_difference_eps <= 0.0:
        raise ValueError("finite_difference_eps must be positive.")

    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    generator = torch.Generator(device=features.device)
    generator.manual_seed(seed)
    n_channels = int(features.shape[1])
    map_shape = tuple(features.shape[-2:])
    attr_sums = [torch.zeros((n_channels, *map_shape), dtype=features.dtype, device=features.device) for _ in depth_indices]

    for sample_idx in range(n_samples):
        base = _sample_baseline(features, baseline, generator, baseline_noise_std)
        alpha = torch.rand((), dtype=features.dtype, device=features.device, generator=generator)
        path_input = base + alpha * (features - base)
        if path_noise_std > 0.0:
            path_input = path_input + path_noise_std * torch.randn(
                features.shape,
                dtype=features.dtype,
                device=features.device,
                generator=generator,
            )
        path_input = path_input.detach()
        delta = (features - base).detach()

        with torch.no_grad():
            for channel_idx in range(n_channels):
                channel_delta = torch.zeros_like(features)
                channel_delta[:, channel_idx] = delta[:, channel_idx]
                plus_maps = _forward_temperature_anomaly_maps(
                    solver,
                    model,
                    path_input + finite_difference_eps * channel_delta,
                    depth_indices,
                    t_ref=t_ref,
                )
                minus_maps = _forward_temperature_anomaly_maps(
                    solver,
                    model,
                    path_input - finite_difference_eps * channel_delta,
                    depth_indices,
                    t_ref=t_ref,
                )
                for depth_idx, (plus_map, minus_map) in enumerate(zip(plus_maps, minus_maps, strict=True)):
                    attr_sums[depth_idx][channel_idx] += (plus_map - minus_map) / (2.0 * finite_difference_eps)

        if not quiet and (sample_idx + 1 == n_samples or (sample_idx + 1) % max(1, math.ceil(n_samples / 10)) == 0):
            print(f"Gradient SHAP samples: {sample_idx + 1}/{n_samples}")

    attrs = torch.stack([attr_sum / n_samples for attr_sum in attr_sums], dim=0)
    return attrs.detach().cpu().numpy()


def _channel_rows(
    attrs: np.ndarray,
    channel_names: list[str],
    z_nodes_km: np.ndarray,
    depth_indices: list[int],
    mean_temperature_values: np.ndarray,
    anomaly_rms_values: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for d, z_idx in enumerate(depth_indices):
        depth_km = float(z_nodes_km[z_idx])
        for c, name in enumerate(channel_names):
            values = attrs[d, c]
            rows.append(
                {
                    "depth_km": depth_km,
                    "z_index": int(z_idx),
                    "mean_temperature_C": float(mean_temperature_values[d]),
                    "temperature_anomaly_rms_C": float(anomaly_rms_values[d]),
                    "channel_index": int(c),
                    "channel": name,
                    "group": _channel_group(name),
                    "sum_abs_shap_C": float(np.sum(np.abs(values))),
                    "mean_abs_shap_C": float(np.mean(np.abs(values))),
                    "rms_shap_C": float(np.sqrt(np.mean(values * values))),
                    "signed_sum_shap_C": float(np.sum(values)),
                    "signed_mean_shap_C": float(np.mean(values)),
                    "positive_fraction": float(np.mean(values > 0.0)),
                }
            )
    return pd.DataFrame(rows)


def _group_rows(channel_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (depth_km, group), group_df in channel_df.groupby(["depth_km", "group"], sort=False):
        rows.append(
            {
                "depth_km": float(depth_km),
                "group": str(group),
                "sum_abs_shap_C": float(group_df["sum_abs_shap_C"].sum()),
                "mean_abs_shap_C": float(group_df["mean_abs_shap_C"].mean()),
                "rms_shap_C": float(np.sqrt(np.mean(np.square(group_df["rms_shap_C"].to_numpy(dtype=float))))),
                "signed_sum_shap_C": float(group_df["signed_sum_shap_C"].sum()),
                "n_channels": int(len(group_df)),
            }
        )
    return pd.DataFrame(rows)


def _write_heatmap(
    df: pd.DataFrame,
    row_col: str,
    value_col: str,
    path: Path,
    *,
    title: str,
    figsize_per_row: float = 0.32,
) -> None:
    import matplotlib.pyplot as plt

    pivot = df.pivot_table(index=row_col, columns="depth_km", values=value_col, aggfunc="sum", fill_value=0.0)
    if pivot.empty:
        return
    fig_height = max(4.0, figsize_per_row * len(pivot.index) + 1.8)
    fig, ax = plt.subplots(figsize=(1.25 * len(pivot.columns) + 4.5, fig_height))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{depth:g}" for depth in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Depth (km)")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=value_col)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _safe_depth_stem(depth_km: float) -> str:
    return f"{depth_km:g}".replace(".", "p").replace("-", "m")


def _write_per_depth_importance_bars(
    df: pd.DataFrame,
    row_col: str,
    value_col: str,
    output_dir: Path,
    *,
    filename_prefix: str,
    title_prefix: str,
    max_rows: int | None = None,
) -> None:
    import matplotlib.pyplot as plt

    for depth_km, depth_df in df.groupby("depth_km", sort=True):
        plot_df = depth_df.sort_values(value_col, ascending=True)
        if max_rows is not None:
            plot_df = plot_df.tail(max_rows)
        if plot_df.empty:
            continue
        fig_height = max(4.0, 0.36 * len(plot_df) + 1.4)
        fig, ax = plt.subplots(figsize=(8.5, fig_height))
        values = plot_df[value_col].to_numpy(dtype=float)
        labels = plot_df[row_col].astype(str).tolist()
        bars = ax.barh(np.arange(len(plot_df)), values, color="tab:blue")
        ax.set_yticks(np.arange(len(plot_df)))
        ax.set_yticklabels(labels)
        ax.set_xlabel(value_col)
        ax.set_title(f"{title_prefix} at {float(depth_km):g} km")
        ax.grid(axis="x", alpha=0.25)
        if values.size:
            label_offset = max(float(np.nanmax(values)) * 0.01, 1e-12)
            for bar, value in zip(bars, values, strict=True):
                ax.text(value + label_offset, bar.get_y() + bar.get_height() / 2.0, f"{value:.3g}", va="center", fontsize=8)
        fig.tight_layout()
        path = output_dir / f"{filename_prefix}_{_safe_depth_stem(float(depth_km))}km.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)


def _write_beeswarm_plots(
    attrs: np.ndarray,
    features: np.ndarray,
    channel_df: pd.DataFrame,
    channel_names: list[str],
    z_nodes_km: np.ndarray,
    depth_indices: list[int],
    output_dir: Path,
    *,
    top_k: int,
    max_points_per_channel: int,
    seed: int,
) -> None:
    if top_k <= 0 or max_points_per_channel <= 0:
        return

    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    n_cells = int(np.prod(features.shape[-2:]))
    n_points = min(max_points_per_channel, n_cells)
    cmap = mcolors.LinearSegmentedColormap.from_list("shap_blue_magenta", ["#1685f9", "#7b2ff7", "#ff0051"])

    for d, z_idx in enumerate(depth_indices):
        depth_km = float(z_nodes_km[z_idx])
        depth_df = channel_df[channel_df["depth_km"] == depth_km].sort_values("sum_abs_shap_C", ascending=False)
        channels = [int(idx) for idx in depth_df["channel_index"].head(top_k)]
        if not channels:
            continue

        if n_points < n_cells:
            sample_idx = rng.choice(n_cells, size=n_points, replace=False)
        else:
            sample_idx = np.arange(n_cells)

        plot_channels = list(reversed(channels))
        shap_values_by_channel = []
        feature_values_by_channel = []
        finite_features = []
        for channel_idx in plot_channels:
            shap_values = attrs[d, channel_idx].reshape(-1)[sample_idx]
            feature_values = features[channel_idx].reshape(-1)[sample_idx]
            mask = np.isfinite(shap_values) & np.isfinite(feature_values)
            shap_values_by_channel.append(shap_values[mask])
            feature_values_by_channel.append(feature_values[mask])
            finite_features.append(feature_values[mask])

        all_feature_values = np.concatenate([values for values in finite_features if values.size]) if finite_features else np.array([])
        vmin = float(np.nanpercentile(all_feature_values, 2)) if all_feature_values.size else -1.0
        vmax = float(np.nanpercentile(all_feature_values, 98)) if all_feature_values.size else 1.0
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = -1.0, 1.0

        fig_height = max(3.8, 0.42 * len(plot_channels) + 1.2)
        fig, ax = plt.subplots(figsize=(8.8, fig_height))
        scatter = None
        for row_idx, (shap_values, feature_values) in enumerate(zip(shap_values_by_channel, feature_values_by_channel, strict=True)):
            if shap_values.size == 0:
                continue
            order = np.argsort(np.abs(shap_values))
            shap_values = shap_values[order]
            feature_values = feature_values[order]
            y = row_idx + rng.normal(0.0, 0.09, size=shap_values.shape[0])
            scatter = ax.scatter(
                shap_values,
                y,
                c=feature_values,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                s=12,
                alpha=0.95,
                linewidths=0,
            )
        ax.axvline(0.0, color="0.45", linewidth=0.8)
        ax.set_yticks(np.arange(len(plot_channels)))
        ax.set_yticklabels([channel_names[idx] for idx in plot_channels])
        ax.set_xlabel("SHAP value (impact on local temperature anomaly, °C)")
        ax.set_title(f"Temperature anomaly SHAP summary at {depth_km:g} km")
        ax.grid(axis="y", linestyle=":", alpha=0.28)
        ax.grid(axis="x", alpha=0.08)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", length=0)
        if scatter is not None:
            cbar = fig.colorbar(scatter, ax=ax, pad=0.025)
            cbar.set_label("Feature value")
            cbar.set_ticks([vmin, vmax])
            cbar.set_ticklabels(["Low", "High"])
        fig.tight_layout()
        path = output_dir / f"beeswarm_depth_{_safe_depth_stem(depth_km)}km.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)


def _write_dependence_plots(
    attrs: np.ndarray,
    features: np.ndarray,
    channel_df: pd.DataFrame,
    channel_names: list[str],
    z_nodes_km: np.ndarray,
    depth_indices: list[int],
    output_dir: Path,
    *,
    top_k: int,
    max_points_per_channel: int,
    seed: int,
) -> None:
    if top_k <= 0 or max_points_per_channel <= 0:
        return

    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed + 17)
    n_cells = int(np.prod(features.shape[-2:]))
    n_points = min(max_points_per_channel, n_cells)
    for d, z_idx in enumerate(depth_indices):
        depth_km = float(z_nodes_km[z_idx])
        depth_df = channel_df[channel_df["depth_km"] == depth_km].sort_values("sum_abs_shap_C", ascending=False)
        channels = [int(idx) for idx in depth_df["channel_index"].head(top_k)]
        if not channels:
            continue

        if n_points < n_cells:
            sample_idx = rng.choice(n_cells, size=n_points, replace=False)
        else:
            sample_idx = np.arange(n_cells)

        n_cols = min(4, len(channels))
        n_rows = int(math.ceil(len(channels) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.2 * n_rows), squeeze=False)
        for plot_idx, channel_idx in enumerate(channels):
            ax = axes[plot_idx // n_cols, plot_idx % n_cols]
            feature_values = features[channel_idx].reshape(-1)[sample_idx]
            shap_values = attrs[d, channel_idx].reshape(-1)[sample_idx]
            mask = np.isfinite(feature_values) & np.isfinite(shap_values)
            feature_values = feature_values[mask]
            shap_values = shap_values[mask]
            if feature_values.size == 0:
                ax.axis("off")
                continue

            ax.scatter(feature_values, shap_values, s=7, color="0.25", alpha=0.18, linewidths=0)
            if feature_values.size >= 30:
                quantiles = np.linspace(0.0, 1.0, min(21, max(6, feature_values.size // 20)))
                edges = np.unique(np.quantile(feature_values, quantiles))
                centers = []
                medians = []
                q10 = []
                q90 = []
                for bin_idx, (left, right) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
                    if bin_idx == len(edges) - 2:
                        bin_mask = (feature_values >= left) & (feature_values <= right)
                    else:
                        bin_mask = (feature_values >= left) & (feature_values < right)
                    if np.count_nonzero(bin_mask) < 5:
                        continue
                    values = shap_values[bin_mask]
                    centers.append(float(np.median(feature_values[bin_mask])))
                    medians.append(float(np.median(values)))
                    q10.append(float(np.percentile(values, 10)))
                    q90.append(float(np.percentile(values, 90)))
                if centers:
                    ax.plot(centers, medians, color="#ff0051", linewidth=2.0)
                    ax.fill_between(centers, q10, q90, color="#ff0051", alpha=0.16, linewidth=0)
            ax.axhline(0.0, color="0.45", linewidth=0.8)
            ax.axvline(0.0, color="0.75", linewidth=0.6)
            ax.set_title(channel_names[channel_idx])
            ax.set_xlabel("Normalized feature value")
            ax.set_ylabel("SHAP to local anomaly (°C)")
            ax.grid(alpha=0.16)

        for ax in axes.ravel()[len(channels) :]:
            ax.axis("off")
        fig.suptitle(f"Feature dependence trends for temperature anomalies at {depth_km:g} km")
        fig.tight_layout()
        path = output_dir / f"dependence_depth_{_safe_depth_stem(depth_km)}km.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)



def _write_top_maps(
    attrs: np.ndarray,
    channel_df: pd.DataFrame,
    channel_names: list[str],
    z_nodes_km: np.ndarray,
    depth_indices: list[int],
    output_dir: Path,
    *,
    top_k: int,
    extent: list[float],
) -> None:
    if top_k <= 0:
        return

    import matplotlib.pyplot as plt

    top_k = min(top_k, len(channel_names))
    for d, z_idx in enumerate(depth_indices):
        depth = float(z_nodes_km[z_idx])
        depth_df = channel_df[channel_df["depth_km"] == depth].sort_values("sum_abs_shap_C", ascending=False)
        channels = [int(idx) for idx in depth_df["channel_index"].head(top_k)]
        if not channels:
            continue

        selected_maps = [attrs[d, c] for c in channels]
        finite = np.concatenate([m[np.isfinite(m)].reshape(-1) for m in selected_maps if np.any(np.isfinite(m))]) if selected_maps else np.array([])
        vmax = float(np.percentile(np.abs(finite), 98)) if finite.size else 1.0
        vmax = max(vmax, 1e-12)

        fig, axes = plt.subplots(1, len(channels), figsize=(4.0 * len(channels), 3.6), squeeze=False)
        image = None
        for j, c in enumerate(channels):
            ax = axes[0, j]
            image = ax.imshow(attrs[d, c], origin="lower", extent=extent, cmap="coolwarm", vmin=-vmax, vmax=vmax)
            ax.set_title(channel_names[c])
            ax.set_xlabel("X (km)")
            ax.set_ylabel("Y (km)")
        if image is not None:
            fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.78, pad=0.012, label="SHAP attribution to local anomaly (°C)")
        fig.suptitle(f"Top input attribution maps for temperature anomalies at {depth:g} km")
        path = output_dir / f"top_channel_anomaly_attribution_maps_depth_{_safe_depth_stem(depth)}km.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)


def _write_summary(
    path: Path,
    *,
    checkpoint_path: Path,
    shap_dir: Path,
    z_nodes_km: np.ndarray,
    depth_indices: list[int],
    channel_names: list[str],
    mean_temperature_values: np.ndarray,
    anomaly_rms_values: np.ndarray,
    args: argparse.Namespace,
    metadata: dict[str, Any],
) -> None:
    payload = {
        "method": "manual_gradient_shap_directional_finite_difference",
        "target": "pointwise temperature anomaly T(y,x,z)-mean_xy(T(z)) at selected depths",
        "checkpoint": str(checkpoint_path),
        "shap_dir": str(shap_dir),
        "depths_km": [float(z_nodes_km[idx]) for idx in depth_indices],
        "z_indices": [int(idx) for idx in depth_indices],
        "mean_temperature_C": [float(value) for value in mean_temperature_values],
        "temperature_anomaly_rms_C": [float(value) for value in anomaly_rms_values],
        "n_channels": len(channel_names),
        "channels": channel_names,
        "n_samples": args.n_samples,
        "baseline": args.baseline,
        "baseline_noise_std": args.baseline_noise_std,
        "path_noise_std": args.path_noise_std,
        "finite_difference_eps": args.finite_difference_eps,
        "seed": args.seed,
        **metadata,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(
        description="Run Gradient SHAP attribution from prior-net input channels to pointwise temperature anomalies at selected depths."
    )
    parser.add_argument("output_dir", type=Path, help="Output directory containing checkpoint.pt, or a parent with one checkpoint.pt below it.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint path. Relative paths are resolved under output_dir.")
    parser.add_argument("--shap-dir", type=Path, default=None, help="Attribution output directory. Defaults to <output_dir>/shap_temperature.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Override data directory saved in the checkpoint.")
    parser.add_argument("--stride", type=int, default=None, help="Override grid stride saved in the checkpoint.")
    parser.add_argument("--dz-km", type=float, default=None, help="Override vertical grid spacing saved in the checkpoint.")
    parser.add_argument("--z-max-km", type=float, default=None, help="Override maximum depth saved in the checkpoint.")
    parser.add_argument("--layer-transition-z-km", type=float, default=None)
    parser.add_argument("--layer-transition-xy-km", type=float, default=None)
    parser.add_argument("--t-ref", type=float, default=1300.0, help="Reference bottom temperature used for fixed bottom mode and DCT deltas.")
    parser.add_argument("--depths-km", type=float, nargs="*", default=DEFAULT_DEPTHS_KM, help="Temperature target depths in km.")
    parser.add_argument("--n-samples", type=int, default=32, help="Number of Gradient SHAP path samples.")
    parser.add_argument("--baseline", choices=["zero", "noise"], default="zero", help="Baseline distribution for Gradient SHAP.")
    parser.add_argument("--baseline-noise-std", type=float, default=1.0, help="Std for noise baselines in normalized feature units.")
    parser.add_argument("--path-noise-std", type=float, default=0.0, help="Optional noise added to path samples in normalized feature units.")
    parser.add_argument("--finite-difference-eps", type=float, default=1e-3, help="Step size for channel-wise directional finite differences.")
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--top-k", type=int, default=4, help="Number of top channels to plot per depth in map panels.")
    parser.add_argument("--beeswarm-top-k", type=int, default=16, help="Number of top channels to include in each depth-specific beeswarm plot.")
    parser.add_argument("--beeswarm-max-points", type=int, default=5000, help="Maximum grid points sampled per channel in each beeswarm plot.")
    parser.add_argument("--dependence-top-k", type=int, default=8, help="Number of top channels to include in each depth-specific dependence plot.")
    parser.add_argument("--dependence-max-points", type=int, default=5000, help="Maximum grid points sampled per channel in each dependence plot.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    checkpoint_path = _resolve_checkpoint(args.output_dir, args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_args = _build_arg_source(args)
    grid, solver, model, features, z_nodes_km, metadata = _build_evaluation_state(checkpoint, checkpoint_path, state_args)
    if metadata.get("mode") != "prior_net" or features is None:
        raise ValueError("Gradient SHAP input-feature attribution requires a prior_net checkpoint.")

    features = features.to(device=torch.device("cpu"), dtype=torch.float64).detach()
    channel_names = _load_channel_names(checkpoint, checkpoint_path, int(features.shape[1]))
    depth_indices = _nearest_depth_indices(z_nodes_km, args.depths_km)
    if not depth_indices:
        raise ValueError("No requested depths are within the model depth range.")

    shap_dir = args.shap_dir or args.output_dir / "shap_temperature"
    shap_dir.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        depths = ", ".join(f"{float(z_nodes_km[idx]):g}" for idx in depth_indices)
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Depth targets: {depths} km")
        print(f"Gradient SHAP samples: {args.n_samples}; output: {shap_dir}")

    mean_temperature_values, anomaly_rms_values = _temperature_stats_np(solver, model, features, depth_indices, t_ref=args.t_ref)
    attrs = _gradient_shap_temperature_anomalies(
        solver,
        model,
        features,
        depth_indices,
        t_ref=args.t_ref,
        n_samples=args.n_samples,
        baseline=args.baseline,
        baseline_noise_std=args.baseline_noise_std,
        path_noise_std=args.path_noise_std,
        finite_difference_eps=args.finite_difference_eps,
        seed=args.seed,
        quiet=args.quiet,
    )

    channel_df = _channel_rows(attrs, channel_names, z_nodes_km, depth_indices, mean_temperature_values, anomaly_rms_values)
    group_df = _group_rows(channel_df)
    channel_df.to_csv(shap_dir / "channel_importance_by_depth.csv", index=False)
    group_df.to_csv(shap_dir / "group_importance_by_depth.csv", index=False)

    np.savez_compressed(
        shap_dir / "temperature_anomaly_gradient_shap_maps.npz",
        attributions=attrs,
        features=features[0].detach().cpu().numpy(),
        z_nodes_km=z_nodes_km,
        depth_indices=np.asarray(depth_indices, dtype=int),
        depth_km=np.asarray([float(z_nodes_km[idx]) for idx in depth_indices], dtype=float),
        channel_names=np.asarray(channel_names, dtype=object),
        mean_temperature_C=mean_temperature_values,
        temperature_anomaly_rms_C=anomaly_rms_values,
        x=grid.x,
        y=grid.y,
        lon=grid.lon,
        lat=grid.lat,
    )

    _write_heatmap(
        channel_df,
        "channel",
        "sum_abs_shap_C",
        shap_dir / "channel_importance_heatmap.png",
        title="Input-channel Gradient SHAP importance for temperature anomalies",
    )
    _write_heatmap(
        group_df,
        "group",
        "sum_abs_shap_C",
        shap_dir / "group_importance_heatmap.png",
        title="Grouped Gradient SHAP importance for temperature anomalies",
        figsize_per_row=0.55,
    )
    _write_per_depth_importance_bars(
        channel_df,
        "channel",
        "sum_abs_shap_C",
        shap_dir,
        filename_prefix="channel_importance_depth",
        title_prefix="Input-channel Gradient SHAP importance for temperature anomalies",
    )
    _write_per_depth_importance_bars(
        group_df,
        "group",
        "sum_abs_shap_C",
        shap_dir,
        filename_prefix="group_importance_depth",
        title_prefix="Grouped Gradient SHAP importance for temperature anomalies",
    )
    feature_maps_np = features[0].detach().cpu().numpy()
    _write_beeswarm_plots(
        attrs,
        feature_maps_np,
        channel_df,
        channel_names,
        z_nodes_km,
        depth_indices,
        shap_dir,
        top_k=args.beeswarm_top_k,
        max_points_per_channel=args.beeswarm_max_points,
        seed=args.seed,
    )
    _write_dependence_plots(
        attrs,
        feature_maps_np,
        channel_df,
        channel_names,
        z_nodes_km,
        depth_indices,
        shap_dir,
        top_k=args.dependence_top_k,
        max_points_per_channel=args.dependence_max_points,
        seed=args.seed,
    )
    extent = [
        float(np.nanmin(grid.x) / 1000.0),
        float(np.nanmax(grid.x) / 1000.0),
        float(np.nanmin(grid.y) / 1000.0),
        float(np.nanmax(grid.y) / 1000.0),
    ]
    _write_top_maps(
        attrs,
        channel_df,
        channel_names,
        z_nodes_km,
        depth_indices,
        shap_dir,
        top_k=args.top_k,
        extent=extent,
    )
    _write_summary(
        shap_dir / "shap_temperature_summary.json",
        checkpoint_path=checkpoint_path,
        shap_dir=shap_dir,
        z_nodes_km=z_nodes_km,
        depth_indices=depth_indices,
        channel_names=channel_names,
        mean_temperature_values=mean_temperature_values,
        anomaly_rms_values=anomaly_rms_values,
        args=args,
        metadata=metadata,
    )

    if not args.quiet:
        print(f"Wrote {shap_dir / 'channel_importance_by_depth.csv'}")
        print(f"Wrote {shap_dir / 'temperature_anomaly_gradient_shap_maps.npz'}")


if __name__ == "__main__":
    main()
