from __future__ import annotations

"""Empirical UQ runner for direct and prior-net inversions."""

import argparse
import gc
import json
import math
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
import torch

from data_loading import (
    GRID_FILENAME,
    HEATFLOW_FILENAME,
    LAYER_FILENAME,
    STRATUM_FILLED_FILENAME,
    THERMAL_GRID_FILENAME,
    THERMAL_PROPERTIES_FILENAME,
    load_heat_flow_observations,
    load_surface_temperature_grid,
)
from direct_parameter import DirectParameterModel
from heat_solver import HeatSolver
from inversion import run_inversion
from lithosphere_model import LayerGrid, build_reference_fields
from prior_network import PriorParameterModel, build_prior_feature_pack


BASELINE_REFERENCE_NAME = "prior_sweep_bottom_focus_KA_alpha800_lamb1e6_prop04_reg5e3"

DIRECT_DEFAULTS: dict[str, Any] = {
    "bottom_mode": "dct",
    "k_mode": "dct",
    "a_mode": "dct",
    "bottom_pairs": "3x3",
    "k_prop_pairs": "6x3",
    "a_prop_pairs": "8x8",
    "alpha_scale": 800.0,
    "k_prop_scale": 0.7,
    "a_prop_scale": 0.7,
    "lam_q_stds": "0.3",
    "lam_q_mean": 0.0,
    "steps": 3,
    "optimizer": "lbfgs",
    "lr": 0.1,
    "weight_decay": 0.0,
    "early_stop_patience": 3,
    "early_stop_min_delta": 0.0,
    "early_stop_metric": "loss",
    "n_seeds": 10,
    "seed": 20260520,
    "n_splits": 5,
    "holdout_frac": 0.2,
    "obs_noise_std": 0.0,
    "predictive_noise_std": 5.0,
    "widths": "32",
    "init_std_bottom": 0.05,
    "init_std_k": 0.05,
    "init_std_a": 0.05,
    "workers": 0,
    "isolate_members": True,
}

PRIOR_NET_DEFAULTS: dict[str, Any] = {
    "bottom_mode": "dct",
    "k_mode": "dct",
    "a_mode": "dct",
    "bottom_pairs": "3x3",
    "k_prop_pairs": "6x6",
    "a_prop_pairs": "8x8",
    "alpha_scale": 1200.0,
    "k_prop_scale": 0.2,
    "a_prop_scale": 0.4,
    "lam_q_stds": "0.4",
    "lam_q_mean": 0.0,
    "steps": 100,
    "optimizer": "adamw",
    "lr": 0.005,
    "weight_decay": 1e-4,
    "early_stop_patience": 25,
    "early_stop_min_delta": 0.001,
    "early_stop_metric": "loss",
    "n_seeds": 10,
    "seed": 20260520,
    "n_splits": 5,
    "holdout_frac": 0.2,
    "obs_noise_std": 0.0,
    "predictive_noise_std": 5.0,
    "widths": "32",
    "init_std_bottom": 0.0,
    "init_std_k": 0.0,
    "init_std_a": 0.0,
    "workers": 0,
    "isolate_members": True,
}


@dataclass(frozen=True)
class UQConfig:
    config_id: int
    bottom_x: int
    bottom_y: int
    k_prop_x: int
    k_prop_y: int
    a_prop_x: int
    a_prop_y: int
    lam_q_std: float
    width: int | None


@dataclass(frozen=True)
class MemberSpec:
    member_id: int
    split_id: int
    seed_index: int
    seed: int
    config: UQConfig


@dataclass(frozen=True)
class CommonOptions:
    mode: str
    data_dir: str
    output_dir: str
    stride: int
    dz_km: float
    z_max_km: float
    layer_transition_z_km: float
    layer_transition_xy_km: float
    bottom_mode: str
    k_mode: str
    a_mode: str
    alpha_scale: float
    k_prop_scale: float
    a_prop_scale: float
    lam_q_mean: float
    steps: int
    optimizer: str
    lr: float
    weight_decay: float
    early_stop_patience: int
    early_stop_min_delta: float
    early_stop_metric: str
    init_std_bottom: float
    init_std_k: float
    init_std_a: float
    holdout_frac: float
    split_seed: int
    obs_noise_std: float
    predictive_noise_std: float
    torch_threads: int
    save_temperature: bool
    save_model_state: bool
    quiet: bool


def _parse_float_list(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float")
    return values


def _parse_int_list(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def _parse_pair_list(text: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for item in text.split(","):
        token = item.strip().lower()
        if not token:
            continue
        match = re.fullmatch(r"(\d+)\s*[x:]\s*(\d+)", token)
        if match:
            pairs.append((int(match.group(1)), int(match.group(2))))
            continue
        if token.isdigit():
            value = int(token)
            pairs.append((value, value))
            continue
        raise argparse.ArgumentTypeError(f"invalid DCT pair: {item!r}; use e.g. 5x5, 3x5, or 5")
    if not pairs:
        raise argparse.ArgumentTypeError("expected at least one DCT pair")
    return pairs


def _safe_float_label(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _randomize_direct_model(model: DirectParameterModel, spec: MemberSpec, opts: CommonOptions) -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(spec.seed)

    def add_noise(param: torch.nn.Parameter | None, std: float) -> None:
        if param is None or std <= 0.0:
            return
        noise = torch.randn(param.shape, generator=generator, dtype=param.dtype, device=param.device)
        param.add_(std * noise)

    with torch.no_grad():
        add_noise(model.raw_bottom_field, opts.init_std_bottom)
        add_noise(model.raw_K_field, opts.init_std_k)
        add_noise(model.raw_A_field, opts.init_std_a)


def _make_train_observations(q_obs: np.ndarray, spec: MemberSpec, opts: CommonOptions) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.isfinite(q_obs)
    train = q_obs.copy()
    holdout = np.zeros_like(valid, dtype=bool)

    valid_indices = np.flatnonzero(valid.reshape(-1))
    if opts.holdout_frac > 0.0 and valid_indices.size > 1:
        split_rng = np.random.default_rng(opts.split_seed + 7919 * spec.split_id)
        n_holdout = int(round(valid_indices.size * opts.holdout_frac))
        n_holdout = min(max(n_holdout, 1), valid_indices.size - 1)
        held_flat = split_rng.choice(valid_indices, size=n_holdout, replace=False)
        holdout.reshape(-1)[held_flat] = True
        train[holdout] = np.nan

    train_mask = np.isfinite(train)
    if opts.obs_noise_std > 0.0 and np.any(train_mask):
        noise_rng = np.random.default_rng(spec.seed + 104729)
        train[train_mask] = train[train_mask] + noise_rng.normal(0.0, opts.obs_noise_std, size=int(train_mask.sum()))

    return train, train_mask, holdout


def _member_dir_name(spec: MemberSpec) -> str:
    cfg = spec.config
    width = "" if cfg.width is None else f"_w{cfg.width}"
    return (
        f"m{spec.member_id:04d}_split{spec.split_id}_seed{spec.seed}_"
        f"b{cfg.bottom_x}x{cfg.bottom_y}_k{cfg.k_prop_x}x{cfg.k_prop_y}_"
        f"a{cfg.a_prop_x}x{cfg.a_prop_y}{width}_std{_safe_float_label(cfg.lam_q_std)}"
    )


def _run_member(spec: MemberSpec, opts: CommonOptions) -> dict[str, Any]:
    if opts.torch_threads > 0:
        torch.set_num_threads(opts.torch_threads)
    torch.set_default_dtype(torch.float64)
    _seed_everything(spec.seed)

    data_dir = Path(opts.data_dir)
    thermal_properties_path = data_dir / THERMAL_PROPERTIES_FILENAME

    grid = LayerGrid(
        data_dir / GRID_FILENAME,
        data_dir / LAYER_FILENAME,
        stride=opts.stride,
        stratum_csv=data_dir / STRATUM_FILLED_FILENAME,
    )
    surface_temp = load_surface_temperature_grid(data_dir / GRID_FILENAME, data_dir / THERMAL_GRID_FILENAME, opts.stride)
    q_obs = load_heat_flow_observations(data_dir / GRID_FILENAME, data_dir / HEATFLOW_FILENAME, opts.stride)
    q_train, train_mask, holdout_mask = _make_train_observations(q_obs, spec, opts)

    z_nodes_km = np.linspace(0.0, opts.z_max_km, round(opts.z_max_km / opts.dz_km) + 1)
    refs = build_reference_fields(
        grid,
        z_nodes_km,
        thermal_properties_path=thermal_properties_path,
        layer_transition_z_km=opts.layer_transition_z_km,
        layer_transition_xy_km=opts.layer_transition_xy_km,
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
    cfg = spec.config
    if opts.mode == "direct":
        features = None
        channel_names: list[str] = []
        model = DirectParameterModel(
            grid.shape[0],
            grid.shape[1],
            bottom_x=cfg.bottom_x,
            bottom_y=cfg.bottom_y,
            k_property_x=cfg.k_prop_x,
            k_property_y=cfg.k_prop_y,
            a_property_x=cfg.a_prop_x,
            a_property_y=cfg.a_prop_y,
            k_property_scale=opts.k_prop_scale,
            a_property_scale=opts.a_prop_scale,
            alpha_scale=opts.alpha_scale,
            bottom_mode=opts.bottom_mode,
            k_mode=opts.k_mode,
            a_mode=opts.a_mode,
        )
        _randomize_direct_model(model, spec, opts)
    else:
        if cfg.width is None:
            raise ValueError("prior_net mode requires a width in each UQ config.")
        features, channel_names = build_prior_feature_pack(data_dir, grid, opts.stride)
        model = PriorParameterModel(
            grid.shape[0],
            grid.shape[1],
            in_channels=features.shape[1],
            bottom_x=cfg.bottom_x,
            bottom_y=cfg.bottom_y,
            k_property_x=cfg.k_prop_x,
            k_property_y=cfg.k_prop_y,
            a_property_x=cfg.a_prop_x,
            a_property_y=cfg.a_prop_y,
            k_property_scale=opts.k_prop_scale,
            a_property_scale=opts.a_prop_scale,
            alpha_scale=opts.alpha_scale,
            width=cfg.width,
            bottom_mode=opts.bottom_mode,
            k_mode=opts.k_mode,
            a_mode=opts.a_mode,
        )

    t0 = time.time()
    out = run_inversion(
        solver,
        q_train,
        model,
        features=features,
        steps=opts.steps,
        lr=opts.lr,
        optimizer_name=opts.optimizer,
        weight_decay=opts.weight_decay,
        diagnose_gradients=False,
        early_stop_patience=opts.early_stop_patience,
        early_stop_min_delta=opts.early_stop_min_delta,
        early_stop_metric=opts.early_stop_metric,
        lam_q_mean=opts.lam_q_mean,
        lam_q_std=cfg.lam_q_std,
        verbose=False,
    )
    elapsed_s = time.time() - t0

    q_pred = out["q_pred"]
    valid_original = np.isfinite(q_obs)
    train_rmse = float(np.sqrt(np.mean((q_pred[train_mask] - q_obs[train_mask]) ** 2))) if np.any(train_mask) else math.nan
    holdout_rmse = float(np.sqrt(np.mean((q_pred[holdout_mask] - q_obs[holdout_mask]) ** 2))) if np.any(holdout_mask) else math.nan
    all_obs_rmse = float(np.sqrt(np.mean((q_pred[valid_original] - q_obs[valid_original]) ** 2))) if np.any(valid_original) else math.nan
    q_pred_std_obs = float(np.std(q_pred[valid_original])) if np.any(valid_original) else math.nan
    q_obs_std = float(np.std(q_obs[valid_original])) if np.any(valid_original) else math.nan

    member_dir = Path(opts.output_dir) / "members" / _member_dir_name(spec)
    member_dir.mkdir(parents=True, exist_ok=True)
    npz_payload: dict[str, Any] = {
        "q_obs": q_obs,
        "q_train": q_train,
        "q_pred": q_pred,
        "residual_train": out["residual"],
        "valid_original_mask": valid_original,
        "train_mask": train_mask,
        "holdout_mask": holdout_mask,
        "T_bot": out["T_bot"],
        "K0": out["K0"],
        "A": out["A"],
        "alpha_bot": out["alpha_bot"],
        "K_coeff": out["K_coeff"],
        "A_coeff": out["A_coeff"],
    }
    if opts.save_temperature:
        npz_payload["T"] = out["T"]
    np.savez_compressed(member_dir / "fields.npz", **npz_payload)

    if opts.save_model_state:
        model_config = {
            "mode": opts.mode,
            "bottom_mode": opts.bottom_mode,
            "k_mode": opts.k_mode,
            "a_mode": opts.a_mode,
            "bottom_x": cfg.bottom_x,
            "bottom_y": cfg.bottom_y,
            "k_property_x": cfg.k_prop_x,
            "k_property_y": cfg.k_prop_y,
            "a_property_x": cfg.a_prop_x,
            "a_property_y": cfg.a_prop_y,
            "alpha_scale": opts.alpha_scale,
            "k_property_scale": opts.k_prop_scale,
            "a_property_scale": opts.a_prop_scale,
            "width": cfg.width,
        }
        torch.save(
            {
                "model_state": out["model_state"],
                "mode": opts.mode,
                "model_config": model_config,
                "config": asdict(cfg),
                "seed": spec.seed,
                "channels": channel_names,
            },
            member_dir / "checkpoint.pt",
        )

    last = out["history"][-1] if out["history"] else {}
    summary = {
        "member_id": spec.member_id,
        "mode": opts.mode,
        "split_id": spec.split_id,
        "seed_index": spec.seed_index,
        "seed": spec.seed,
        "member_dir": str(member_dir),
        "fields_npz": str(member_dir / "fields.npz"),
        "checkpoint": str(member_dir / "checkpoint.pt") if opts.save_model_state else None,
        "elapsed_s": elapsed_s,
        "train_rmse_original_obs": train_rmse,
        "holdout_rmse": holdout_rmse,
        "all_obs_rmse": all_obs_rmse,
        "q_pred_std_obs_points": q_pred_std_obs,
        "q_obs_std_obs_points": q_obs_std,
        "std_ratio_obs_points": q_pred_std_obs / q_obs_std if q_obs_std > 0.0 else math.nan,
        "final_loss": float(out["final_loss"]),
        "final_rmse_train_target": float(out["final_rmse"]),
        "early_stopped": bool(out["early_stopped"]),
        "early_stop_best_step": out["early_stop_best_step"],
        "closure_calls": int(out["closure_calls"]),
        "steps_recorded": len(out["history"]),
        "channels": channel_names,
        "last": last,
        **{f"config_{key}": value for key, value in asdict(spec.config).items()},
    }
    with open(member_dir / "member_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def _build_configs(args: argparse.Namespace) -> list[UQConfig]:
    configs: list[UQConfig] = []
    config_id = 0
    widths = args.widths if args.mode == "prior_net" else [None]
    for bottom_x, bottom_y in args.bottom_pairs:
        for k_x, k_y in args.k_prop_pairs:
            for a_x, a_y in args.a_prop_pairs:
                for width in widths:
                    for lam_q_std in args.lam_q_stds:
                        configs.append(
                            UQConfig(
                                config_id=config_id,
                                bottom_x=bottom_x,
                                bottom_y=bottom_y,
                                k_prop_x=k_x,
                                k_prop_y=k_y,
                                a_prop_x=a_x,
                                a_prop_y=a_y,
                                lam_q_std=lam_q_std,
                                width=width,
                            )
                        )
                        config_id += 1
    return configs


def _build_specs(args: argparse.Namespace, configs: list[UQConfig]) -> list[MemberSpec]:
    specs: list[MemberSpec] = []
    member_id = 0
    for split_id in range(args.n_splits):
        for cfg in configs:
            for seed_index in range(args.n_seeds):
                seed = args.seed + 100000 * split_id + 1000 * cfg.config_id + seed_index
                specs.append(MemberSpec(member_id=member_id, split_id=split_id, seed_index=seed_index, seed=seed, config=cfg))
                member_id += 1
    return specs


def _run_member_isolated(spec: MemberSpec, opts: CommonOptions) -> dict[str, Any]:
    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run_member, spec, opts)
        return future.result()


def _load_member_fields(summary: dict[str, Any]) -> dict[str, np.ndarray]:
    with np.load(summary["fields_npz"], allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def _central_interval_coverage(samples: np.ndarray, truth: np.ndarray, level: float) -> float:
    lower_q = 0.5 * (1.0 - level)
    upper_q = 1.0 - lower_q
    lower = np.quantile(samples, lower_q, axis=0)
    upper = np.quantile(samples, upper_q, axis=0)
    return float(np.mean((truth >= lower) & (truth <= upper)))


def _gaussian_interval_coverage(samples: np.ndarray, truth: np.ndarray, level: float, extra_std: float = 0.0) -> float:
    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0, ddof=1) if samples.shape[0] > 1 else np.zeros_like(mean)
    if extra_std > 0.0:
        std = np.sqrt(std * std + extra_std * extra_std)
    z = NormalDist().inv_cdf(0.5 + 0.5 * level)
    lower = mean - z * std
    upper = mean + z * std
    return float(np.mean((truth >= lower) & (truth <= upper)))


def _aggregate_outputs(output_dir: Path, summaries: list[dict[str, Any]], levels: list[float], predictive_noise_std: float) -> dict[str, Any]:
    successful = [row for row in summaries if row.get("status", "ok") == "ok"]
    if not successful:
        aggregate = {"member_count": 0, "failed_count": len(summaries), "error": "no successful members"}
        with open(output_dir / "aggregate_summary.json", "w", encoding="utf-8") as f:
            json.dump(aggregate, f, ensure_ascii=False, indent=2)
        return aggregate

    fields0 = _load_member_fields(successful[0])
    q_obs = fields0["q_obs"]
    valid_original = fields0["valid_original_mask"].astype(bool)

    q_stack = []
    tbot_stack = []
    k_col_stack = []
    a_col_stack = []
    for row in successful:
        fields = _load_member_fields(row)
        q_stack.append(fields["q_pred"])
        tbot_stack.append(fields["T_bot"])
        k0 = fields["K0"]
        a = fields["A"]
        k_col_stack.append(np.mean(k0, axis=2) if k0.ndim == 3 else k0)
        a_col_stack.append(np.mean(a, axis=2) if a.ndim == 3 else a)

    q_arr = np.stack(q_stack, axis=0)
    tbot_arr = np.stack(tbot_stack, axis=0)
    k_col_arr = np.stack(k_col_stack, axis=0)
    a_col_arr = np.stack(a_col_stack, axis=0)

    q_mean = np.mean(q_arr, axis=0)
    q_std = np.std(q_arr, axis=0, ddof=1) if q_arr.shape[0] > 1 else np.zeros_like(q_mean)
    tbot_mean = np.mean(tbot_arr, axis=0)
    tbot_std = np.std(tbot_arr, axis=0, ddof=1) if tbot_arr.shape[0] > 1 else np.zeros_like(tbot_mean)
    k_col_mean = np.mean(k_col_arr, axis=0)
    k_col_std = np.std(k_col_arr, axis=0, ddof=1) if k_col_arr.shape[0] > 1 else np.zeros_like(k_col_mean)
    a_col_mean = np.mean(a_col_arr, axis=0)
    a_col_std = np.std(a_col_arr, axis=0, ddof=1) if a_col_arr.shape[0] > 1 else np.zeros_like(a_col_mean)

    np.savez_compressed(
        output_dir / "aggregate_fields.npz",
        q_obs=q_obs,
        valid_original_mask=valid_original,
        q_pred_mean=q_mean,
        q_pred_std=q_std,
        T_bot_mean=tbot_mean,
        T_bot_std=tbot_std,
        K0_col_mean_mean=k_col_mean,
        K0_col_mean_std=k_col_std,
        A_col_mean_mean=a_col_mean,
        A_col_mean_std=a_col_std,
    )

    valid_rmse = float(np.sqrt(np.mean((q_mean[valid_original] - q_obs[valid_original]) ** 2))) if np.any(valid_original) else math.nan
    q_obs_std = float(np.std(q_obs[valid_original])) if np.any(valid_original) else math.nan
    q_mean_std = float(np.std(q_mean[valid_original])) if np.any(valid_original) else math.nan
    q_ensemble_spread = float(np.mean(q_std[valid_original])) if np.any(valid_original) else math.nan

    calibration_rows: list[dict[str, Any]] = []
    split_ids = sorted({int(row["split_id"]) for row in successful})
    config_ids = sorted({int(row["config_config_id"]) for row in successful})
    for split_id in split_ids:
        split_fields0 = _load_member_fields(next(row for row in successful if int(row["split_id"]) == split_id))
        holdout = split_fields0["holdout_mask"].astype(bool)
        if not np.any(holdout):
            continue
        calibration_groups: list[tuple[str, int | str, list[dict[str, Any]]]] = [
            ("all_configs", "all", [row for row in successful if int(row["split_id"]) == split_id])
        ]
        calibration_groups.extend(
            (
                "single_config",
                config_id,
                [
                    row
                    for row in successful
                    if int(row["split_id"]) == split_id and int(row["config_config_id"]) == config_id
                ],
            )
            for config_id in config_ids
        )
        for group_type, config_id, rows in calibration_groups:
            if len(rows) < 2:
                continue
            samples = np.stack([_load_member_fields(row)["q_pred"] for row in rows], axis=0)[:, holdout]
            truth = q_obs[holdout]
            pred_mean = np.mean(samples, axis=0)
            split_summary = {
                "group_type": group_type,
                "config_id": config_id,
                "split_id": split_id,
                "n_members": len(rows),
                "n_holdout": int(holdout.sum()),
                "holdout_rmse_ensemble_mean": float(np.sqrt(np.mean((pred_mean - truth) ** 2))),
                "holdout_mean_spread": float(np.mean(np.std(samples, axis=0, ddof=1))),
            }
            for level in levels:
                label = _safe_float_label(level)
                split_summary[f"coverage_quantile_{label}"] = _central_interval_coverage(samples, truth, level)
                split_summary[f"coverage_gaussian_{label}"] = _gaussian_interval_coverage(samples, truth, level)
                split_summary[f"coverage_predictive_{label}"] = _gaussian_interval_coverage(
                    samples,
                    truth,
                    level,
                    extra_std=predictive_noise_std,
                )
            calibration_rows.append(split_summary)

    if calibration_rows:
        pd.DataFrame(calibration_rows).to_csv(output_dir / "calibration_by_split.csv", index=False)

    aggregate = {
        "member_count": len(successful),
        "failed_count": len(summaries) - len(successful),
        "aggregate_fields_npz": str(output_dir / "aggregate_fields.npz"),
        "all_obs_rmse_ensemble_mean": valid_rmse,
        "q_obs_std_obs_points": q_obs_std,
        "q_pred_mean_std_obs_points": q_mean_std,
        "std_ratio_ensemble_mean_obs_points": q_mean_std / q_obs_std if q_obs_std > 0.0 else math.nan,
        "mean_ensemble_spread_obs_points": q_ensemble_spread,
        "predictive_noise_std": predictive_noise_std,
        "calibration": calibration_rows,
    }
    with open(output_dir / "aggregate_summary.json", "w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)
    return aggregate


def _write_plots(output_dir: Path, aggregate: dict[str, Any], levels: list[float]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fields_path = aggregate.get("aggregate_fields_npz")
    if not fields_path:
        return
    with np.load(fields_path, allow_pickle=False) as data:
        q_obs = data["q_obs"]
        valid = data["valid_original_mask"].astype(bool)
        q_mean = data["q_pred_mean"]
        q_std = data["q_pred_std"]
        tbot_mean = data["T_bot_mean"]
        tbot_std = data["T_bot_std"]
        k_std = data["K0_col_mean_std"]
        a_std = data["A_col_mean_std"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    panels = [
        (q_obs, "q_obs"),
        (q_mean, "q_pred ensemble mean"),
        (q_std, "q_pred ensemble std"),
        (tbot_mean, "T_bot ensemble mean"),
        (tbot_std, "T_bot ensemble std"),
        (q_mean - q_obs, "q_pred_mean - q_obs"),
    ]
    for ax, (arr, title) in zip(axes.ravel(), panels):
        im = ax.imshow(np.ma.masked_invalid(arr), origin="lower")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.8, pad=0.01, fraction=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "uq_panels.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, arr, title in [
        (axes[0], k_std, "column-mean K0 std"),
        (axes[1], a_std, "column-mean A std"),
    ]:
        im = ax.imshow(np.ma.masked_invalid(arr), origin="lower")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.8, pad=0.01, fraction=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "property_uq_panels.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    if np.any(valid):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.errorbar(q_obs[valid], q_mean[valid], yerr=q_std[valid], fmt="o", ms=3, alpha=0.6, ecolor="0.75")
        lo = float(np.nanmin([np.nanmin(q_obs[valid]), np.nanmin(q_mean[valid])]))
        hi = float(np.nanmax([np.nanmax(q_obs[valid]), np.nanmax(q_mean[valid])]))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel("q_obs (mW/m²)")
        ax.set_ylabel("q_pred ensemble mean (mW/m²)")
        ax.set_title("Observed vs ensemble prediction")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "q_obs_vs_pred.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    calibration_path = output_dir / "calibration_by_split.csv"
    if calibration_path.exists():
        cdf = pd.read_csv(calibration_path)
        plot_df = cdf[cdf["group_type"] == "all_configs"] if "group_type" in cdf else cdf
        if plot_df.empty:
            plot_df = cdf
        nominal = np.asarray(levels, dtype=float)
        actual_cols = [f"coverage_predictive_{_safe_float_label(level)}" for level in levels]
        actual = np.asarray([plot_df[col].mean() for col in actual_cols], dtype=float)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(nominal, actual, marker="o", label="holdout coverage")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
        ax.set_xlabel("Nominal central interval")
        ax.set_ylabel("Actual holdout coverage")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "calibration_curve.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def _save_summaries(output_dir: Path, summaries: list[dict[str, Any]]) -> None:
    ordered = sorted(summaries, key=lambda row: int(row.get("member_id", -1)))
    with open(output_dir / "member_summaries.json", "w", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=False, indent=2)
    pd.DataFrame(ordered).to_csv(output_dir / "member_summaries.csv", index=False)


def _config_group_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ok_rows = [row for row in summaries if row.get("status") == "ok"]
    if not ok_rows:
        return []
    df = pd.DataFrame(ok_rows)
    group_cols = [
        "config_config_id",
        "config_bottom_x",
        "config_bottom_y",
        "config_k_prop_x",
        "config_k_prop_y",
        "config_a_prop_x",
        "config_a_prop_y",
        "config_lam_q_std",
    ]
    if "config_width" in df and not df["config_width"].isna().all():
        group_cols.append("config_width")
    metric_cols = ["train_rmse_original_obs", "holdout_rmse", "all_obs_rmse", "std_ratio_obs_points", "final_loss"]
    rows: list[dict[str, Any]] = []
    for keys, group in df.groupby(group_cols, dropna=False):
        row = {col: value for col, value in zip(group_cols, keys)}
        for col in metric_cols:
            values = pd.to_numeric(group[col], errors="coerce")
            row[f"{col}_mean"] = float(values.mean())
            row[f"{col}_std"] = float(values.std())
            row[f"{col}_min"] = float(values.min())
            row[f"{col}_max"] = float(values.max())
        row["n_members"] = int(len(group))
        rows.append(row)
    return rows


def _write_config_summary(output_dir: Path, summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = _config_group_rows(summaries)
    if rows:
        pd.DataFrame(rows).to_csv(output_dir / "config_summary.csv", index=False)
    return rows


def _format_config(row: dict[str, Any]) -> str:
    width = row.get("config_width")
    width_text = "" if width is None or pd.isna(width) else f"width={int(width)}, "
    return (
        f"config {int(row['config_config_id'])} — "
        f"bottom={int(row['config_bottom_x'])}x{int(row['config_bottom_y'])}, "
        f"K={int(row['config_k_prop_x'])}x{int(row['config_k_prop_y'])}, "
        f"A={int(row['config_a_prop_x'])}x{int(row['config_a_prop_y'])}, "
        f"{width_text}lam_q_std={row['config_lam_q_std']:g}"
    )


def _finite_min(rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    finite = [row for row in rows if np.isfinite(row.get(key, math.nan))]
    return min(finite, key=lambda row: row[key]) if finite else None


def _finite_max(rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    finite = [row for row in rows if np.isfinite(row.get(key, math.nan))]
    return max(finite, key=lambda row: row[key]) if finite else None


def _write_report(
    output_dir: Path,
    *,
    status: str,
    manifest: dict[str, Any],
    summaries: list[dict[str, Any]],
    aggregate: dict[str, Any] | None = None,
    config_rows: list[dict[str, Any]] | None = None,
) -> None:
    config_rows = config_rows or _config_group_rows(summaries)
    options = manifest["options"]
    total = int(manifest["member_count"])
    ok = [row for row in summaries if row.get("status") == "ok"]
    failed = [row for row in summaries if row.get("status") == "failed"]
    mode_label = "Direct inversion" if options["mode"] == "direct" else "Prior-net inversion"
    lines = [
        f"# {mode_label} UQ live report",
        "",
        f"Status: {status}",
        "",
        f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Run",
        "",
        f"- Script: `{manifest['script']}`",
        f"- Run directory: `{output_dir}`",
        f"- Mode: {mode_label} UQ",
        f"- Stride: {options['stride']}",
        f"- Steps per member: {options['steps']}",
        f"- Members: {len(ok)} / {total} completed successfully",
        f"- Failed members: {len(failed)}",
        f"- Ensemble axes: {manifest['n_splits']} holdout split(s) × {len(manifest['configs'])} hyperparameter config(s) × {manifest['n_seeds']} seed(s)",
    ]
    if options["mode"] == "prior_net":
        lines.extend(
            [
                f"- Reference defaults: `{manifest['reference_config_name']}`",
                "- Baseline checkpoint: not loaded; every member starts from an independent random prior-net initialization",
                "- Feature sources: geophys_2d, seismic, MT",
            ]
        )
    else:
        lines.append(
            f"- Direct init std: bottom={options['init_std_bottom']}, K={options['init_std_k']}, A={options['init_std_a']}"
        )

    if ok:
        latest = max(ok, key=lambda row: int(row["member_id"]))
        lines.extend(
            [
                "",
                "## Current progress",
                "",
                f"- Latest successful member: {latest['member_id']} with all-observation RMSE {latest['all_obs_rmse']:.3f} mW/m²",
                f"- Mean successful-member all-observation RMSE: {np.nanmean([row['all_obs_rmse'] for row in ok]):.3f} mW/m²",
                f"- Mean successful-member std ratio: {np.nanmean([row['std_ratio_obs_points'] for row in ok]):.3f}",
            ]
        )

    if failed:
        last_failed = failed[-1]
        lines.extend(["", "## Failures", "", f"- Latest failure: member {last_failed.get('member_id')} — `{last_failed.get('error')}`"])

    if aggregate and aggregate.get("member_count", 0):
        lines.extend(
            [
                "",
                "## Aggregate result",
                "",
                f"- Ensemble-mean all-observation RMSE: {aggregate['all_obs_rmse_ensemble_mean']:.3f} mW/m²",
                f"- Observed heat-flow std: {aggregate['q_obs_std_obs_points']:.3f} mW/m²",
                f"- Ensemble-mean predicted heat-flow std: {aggregate['q_pred_mean_std_obs_points']:.3f} mW/m²",
                f"- Std ratio: {aggregate['std_ratio_ensemble_mean_obs_points']:.3f}",
                f"- Mean ensemble spread at observation points: {aggregate['mean_ensemble_spread_obs_points']:.3f} mW/m²",
                f"- Predictive noise std used for predictive coverage: {aggregate['predictive_noise_std']:.3f} mW/m²",
            ]
        )
        if config_rows:
            best_all = _finite_min(config_rows, "all_obs_rmse_mean")
            best_holdout = _finite_min(config_rows, "holdout_rmse_mean")
            largest_std = _finite_max(config_rows, "std_ratio_obs_points_mean")
            lines.extend(["", "## Best configs", ""])
            if best_all:
                lines.append(f"- Best all-observation RMSE: {_format_config(best_all)}; all RMSE={best_all['all_obs_rmse_mean']:.3f}")
            if best_holdout:
                lines.append(f"- Best holdout RMSE: {_format_config(best_holdout)}; holdout RMSE={best_holdout['holdout_rmse_mean']:.3f}")
            if largest_std:
                lines.append(f"- Largest std ratio: {_format_config(largest_std)}; std ratio={largest_std['std_ratio_obs_points_mean']:.3f}")

        cal_rows = [row for row in aggregate.get("calibration", []) if row.get("group_type") == "all_configs"]
        if cal_rows:
            lines.extend(
                [
                    "",
                    "## All-config holdout calibration",
                    "",
                    "| split | n_members | n_holdout | holdout RMSE | ensemble spread | quantile cov90 | predictive cov90 | predictive cov95 |",
                    "|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for row in cal_rows:
                lines.append(
                    f"| {row['split_id']} | {row['n_members']} | {row['n_holdout']} | "
                    f"{row['holdout_rmse_ensemble_mean']:.3f} | {row['holdout_mean_spread']:.3f} | "
                    f"{row.get('coverage_quantile_0p9', math.nan):.3f} | "
                    f"{row.get('coverage_predictive_0p9', math.nan):.3f} | "
                    f"{row.get('coverage_predictive_0p95', math.nan):.3f} |"
                )

        key_files = [
            output_dir / "aggregate_summary.json",
            output_dir / "aggregate_fields.npz",
            output_dir / "member_summaries.csv",
            output_dir / "config_summary.csv",
            output_dir / "calibration_by_split.csv",
            output_dir / "uq_panels.png",
            output_dir / "property_uq_panels.png",
            output_dir / "q_obs_vs_pred.png",
            output_dir / "calibration_curve.png",
        ]
        lines.extend(["", "## Key output files", ""])
        lines.extend(f"- `{path}`" for path in key_files if path.exists())

    report = "\n".join(lines) + "\n"
    with open(output_dir / "live_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    if status == "finished":
        final_report = report.replace(" UQ live report", " UQ final report", 1)
        with open(output_dir / "final_report.md", "w", encoding="utf-8") as f:
            f.write(final_report)


def _make_parser(mode: str) -> argparse.ArgumentParser:
    defaults = DIRECT_DEFAULTS if mode == "direct" else PRIOR_NET_DEFAULTS
    parser = argparse.ArgumentParser(description="Run empirical UQ ensembles for HeatSolver inversions.")
    parser.add_argument("--mode", choices=["direct", "prior_net"], default=mode)
    parser.add_argument("--data-dir", type=Path, default=Path("Data"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--dz-km", type=float, default=2.0)
    parser.add_argument("--z-max-km", type=float, default=120.0)
    parser.add_argument("--layer-transition-z-km", type=float, default=2.0)
    parser.add_argument("--layer-transition-xy-km", type=float, default=20.0)

    parser.add_argument("--bottom-mode", choices=["fixed", "dct", "full"], default=defaults["bottom_mode"])
    parser.add_argument("--k-mode", choices=["fixed", "dct", "full"], default=defaults["k_mode"])
    parser.add_argument("--a-mode", choices=["fixed", "dct", "full"], default=defaults["a_mode"])
    parser.add_argument("--bottom-pairs", type=_parse_pair_list, default=_parse_pair_list(defaults["bottom_pairs"]))
    parser.add_argument("--k-prop-pairs", type=_parse_pair_list, default=_parse_pair_list(defaults["k_prop_pairs"]))
    parser.add_argument("--a-prop-pairs", type=_parse_pair_list, default=_parse_pair_list(defaults["a_prop_pairs"]))
    parser.add_argument("--alpha-scale", type=float, default=defaults["alpha_scale"])
    parser.add_argument("--k-prop-scale", type=float, default=defaults["k_prop_scale"])
    parser.add_argument("--a-prop-scale", type=float, default=defaults["a_prop_scale"])
    parser.add_argument("--widths", type=_parse_int_list, default=_parse_int_list(defaults["widths"]))

    parser.add_argument("--lam-q-stds", type=_parse_float_list, default=_parse_float_list(defaults["lam_q_stds"]))
    parser.add_argument("--lam-q-mean", type=float, default=defaults["lam_q_mean"])

    parser.add_argument("--steps", "--step", dest="steps", type=int, default=defaults["steps"])
    parser.add_argument("--optimizer", choices=["lbfgs", "adam", "adamw"], default=defaults["optimizer"])
    parser.add_argument("--lr", type=float, default=defaults["lr"])
    parser.add_argument("--weight-decay", type=float, default=defaults["weight_decay"])
    parser.add_argument("--early-stop-patience", type=int, default=defaults["early_stop_patience"])
    parser.add_argument("--early-stop-min-delta", type=float, default=defaults["early_stop_min_delta"])
    parser.add_argument("--early-stop-metric", type=str, default=defaults["early_stop_metric"])

    parser.add_argument("--n-seeds", type=int, default=defaults["n_seeds"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--n-splits", type=int, default=defaults["n_splits"])
    parser.add_argument("--holdout-frac", type=float, default=defaults["holdout_frac"])
    parser.add_argument("--obs-noise-std", type=float, default=defaults["obs_noise_std"])
    parser.add_argument("--predictive-noise-std", type=float, default=defaults["predictive_noise_std"])
    parser.add_argument("--init-std-bottom", type=float, default=defaults["init_std_bottom"])
    parser.add_argument("--init-std-k", type=float, default=defaults["init_std_k"])
    parser.add_argument("--init-std-a", type=float, default=defaults["init_std_a"])
    parser.add_argument("--coverage-levels", type=_parse_float_list, default=_parse_float_list("0.5,0.68,0.9,0.95"))

    parser.add_argument("--workers", type=int, default=defaults["workers"])
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--start-member", type=int, default=0)
    parser.add_argument("--max-members", type=int, default=0)
    parser.add_argument("--isolate-members", action=argparse.BooleanOptionalAction, default=defaults["isolate_members"])
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-temperature", action="store_true")
    parser.add_argument("--save-model-state", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.steps < 1:
        raise SystemExit("--steps must be >= 1")
    if args.stride < 10:
        raise SystemExit("--stride must be >= 10 for UQ runs.")
    if args.n_seeds < 1:
        raise SystemExit("--n-seeds must be >= 1")
    if args.n_splits < 1:
        raise SystemExit("--n-splits must be >= 1")
    if args.start_member < 0:
        raise SystemExit("--start-member must be >= 0")
    if args.max_members < 0:
        raise SystemExit("--max-members must be >= 0")
    if not 0.0 <= args.holdout_frac < 1.0:
        raise SystemExit("--holdout-frac must be in [0, 1)")
    if any(level <= 0.0 or level >= 1.0 for level in args.coverage_levels):
        raise SystemExit("all --coverage-levels must be in (0, 1)")
    if min(args.init_std_bottom, args.init_std_k, args.init_std_a) < 0.0:
        raise SystemExit("direct initialization std values must be non-negative")
    if any(width < 1 for width in args.widths):
        raise SystemExit("all --widths values must be >= 1")

    if args.mode == "direct":
        if args.steps > 3:
            raise SystemExit("direct UQ intentionally limits --steps/--step to <= 3.")
    elif args.bottom_mode != "dct" or args.k_mode not in {"fixed", "dct"} or args.a_mode not in {"fixed", "dct"}:
        raise SystemExit("prior-net UQ supports bottom_mode='dct' and K/A modes in {'fixed', 'dct'} only.")


def main() -> None:
    mode_parser = argparse.ArgumentParser(add_help=False)
    mode_parser.add_argument("--mode", choices=["direct", "prior_net"], default="prior_net")
    mode_args, _ = mode_parser.parse_known_args()
    parser = _make_parser(mode_args.mode)
    args = parser.parse_args()
    _validate_args(args)

    if args.output_dir is not None:
        output_dir = args.output_dir
    elif args.mode == "direct":
        output_dir = Path("outputs") / f"direct_uq_{time.strftime('%Y%m%d_%H%M%S')}"
    else:
        output_dir = Path("outputs") / f"prior_uq_{BASELINE_REFERENCE_NAME}_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = _build_configs(args)
    all_specs = _build_specs(args, configs)
    specs = [spec for spec in all_specs if spec.member_id >= args.start_member]
    if args.max_members > 0:
        specs = specs[: args.max_members]
    workers = args.workers
    if workers <= 0:
        workers = 1 if args.mode == "prior_net" else max(1, min(4, max(1, len(specs)), (os.cpu_count() or 2) // 2 or 1))

    opts = CommonOptions(
        mode=args.mode,
        data_dir=str(args.data_dir),
        output_dir=str(output_dir),
        stride=args.stride,
        dz_km=args.dz_km,
        z_max_km=args.z_max_km,
        layer_transition_z_km=args.layer_transition_z_km,
        layer_transition_xy_km=args.layer_transition_xy_km,
        bottom_mode=args.bottom_mode,
        k_mode=args.k_mode,
        a_mode=args.a_mode,
        alpha_scale=args.alpha_scale,
        k_prop_scale=args.k_prop_scale,
        a_prop_scale=args.a_prop_scale,
        lam_q_mean=args.lam_q_mean,
        steps=args.steps,
        optimizer=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        early_stop_metric=args.early_stop_metric,
        init_std_bottom=args.init_std_bottom,
        init_std_k=args.init_std_k,
        init_std_a=args.init_std_a,
        holdout_frac=args.holdout_frac,
        split_seed=args.seed + 424242,
        obs_noise_std=args.obs_noise_std,
        predictive_noise_std=args.predictive_noise_std,
        torch_threads=args.torch_threads,
        save_temperature=args.save_temperature,
        save_model_state=args.save_model_state,
        quiet=args.quiet,
    )

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script": Path(__file__).name,
        "mode": args.mode,
        "reference_config_name": BASELINE_REFERENCE_NAME if args.mode == "prior_net" else None,
        "baseline_checkpoint_loaded": False,
        "member_count": len(specs),
        "total_possible_members": len(all_specs),
        "start_member": args.start_member,
        "workers": workers,
        "n_splits": args.n_splits,
        "n_seeds": args.n_seeds,
        "options": asdict(opts),
        "configs": [asdict(cfg) for cfg in configs],
        "members": [
            {
                **{key: value for key, value in asdict(spec).items() if key != "config"},
                "config": asdict(spec.config),
            }
            for spec in specs
        ],
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    summaries: list[dict[str, Any]] = []
    _write_report(output_dir, status="dry_run" if args.dry_run else "running", manifest=manifest, summaries=summaries)

    if not args.quiet:
        mode_label = "direct" if args.mode == "direct" else "prior-net"
        print(f"Running {len(specs)} {mode_label} UQ members in {output_dir} with {workers} worker(s).")
        if args.mode == "prior_net":
            print(f"Using defaults from {BASELINE_REFERENCE_NAME}; no summary or checkpoint file is loaded.")
        print(f"Each member uses {args.steps} optimization step(s); configs={len(configs)}, splits={args.n_splits}, seeds/config={args.n_seeds}.")
    if args.dry_run:
        print(f"Dry run only. Manifest saved to {output_dir / 'manifest.json'}")
        print(f"Live report saved to {output_dir / 'live_report.md'}")
        return

    if workers == 1:
        for spec in specs:
            try:
                summary = _run_member_isolated(spec, opts) if args.isolate_members else _run_member(spec, opts)
                summary["status"] = "ok"
                summaries.append(summary)
                if not args.quiet:
                    print(f"[{len(summaries)}/{len(specs)}] ok member={spec.member_id} rmse={summary['all_obs_rmse']:.3f}")
            except Exception as exc:  # noqa: BLE001
                row = {
                    "status": "failed",
                    "member_id": spec.member_id,
                    "split_id": spec.split_id,
                    "seed_index": spec.seed_index,
                    "seed": spec.seed,
                    "error": repr(exc),
                    **{f"config_{key}": value for key, value in asdict(spec.config).items()},
                }
                summaries.append(row)
                print(f"[{len(summaries)}/{len(specs)}] failed member={spec.member_id}: {exc}")
            gc.collect()
            _save_summaries(output_dir, summaries)
            _write_report(output_dir, status="running", manifest=manifest, summaries=summaries)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_run_member, spec, opts): spec for spec in specs}
            for future in as_completed(futures):
                spec = futures[future]
                try:
                    summary = future.result()
                    summary["status"] = "ok"
                    summaries.append(summary)
                    if not args.quiet:
                        print(f"[{len(summaries)}/{len(specs)}] ok member={spec.member_id} rmse={summary['all_obs_rmse']:.3f}")
                except Exception as exc:  # noqa: BLE001
                    row = {
                        "status": "failed",
                        "member_id": spec.member_id,
                        "split_id": spec.split_id,
                        "seed_index": spec.seed_index,
                        "seed": spec.seed,
                        "error": repr(exc),
                        **{f"config_{key}": value for key, value in asdict(spec.config).items()},
                    }
                    summaries.append(row)
                    print(f"[{len(summaries)}/{len(specs)}] failed member={spec.member_id}: {exc}")
                _save_summaries(output_dir, summaries)
                _write_report(output_dir, status="running", manifest=manifest, summaries=summaries)

    summaries = sorted(summaries, key=lambda row: int(row.get("member_id", -1)))
    _save_summaries(output_dir, summaries)
    config_rows = _write_config_summary(output_dir, summaries)
    aggregate = _aggregate_outputs(output_dir, summaries, args.coverage_levels, args.predictive_noise_std)
    if not args.skip_plots:
        try:
            _write_plots(output_dir, aggregate, args.coverage_levels)
        except MemoryError as exc:
            aggregate["plot_error"] = repr(exc)
            with open(output_dir / "aggregate_summary.json", "w", encoding="utf-8") as f:
                json.dump(aggregate, f, ensure_ascii=False, indent=2)
            print(f"Skipping plots after memory error: {exc}")
    _write_report(output_dir, status="finished", manifest=manifest, summaries=summaries, aggregate=aggregate, config_rows=config_rows)

    if not args.quiet:
        print(f"Saved manifest to {output_dir / 'manifest.json'}")
        print(f"Saved live report to {output_dir / 'live_report.md'}")
        print(f"Saved final report to {output_dir / 'final_report.md'}")
        print(f"Saved member summaries to {output_dir / 'member_summaries.csv'}")
        print(f"Saved aggregate summary to {output_dir / 'aggregate_summary.json'}")
        if aggregate.get("member_count", 0):
            print(
                "Aggregate: "
                f"members={aggregate['member_count']}, "
                f"all_obs_rmse_mean={aggregate['all_obs_rmse_ensemble_mean']:.3f}, "
                f"std_ratio={aggregate['std_ratio_ensemble_mean_obs_points']:.3f}, "
                f"mean_spread={aggregate['mean_ensemble_spread_obs_points']:.3f}"
            )


if __name__ == "__main__":
    main()
