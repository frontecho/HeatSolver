from __future__ import annotations

"""Unified inversion entry point.

Two parameter sources share the same PDE-constrained optimization loop:
1. `direct`: optimize thermal parameters directly.
2. `prior_net`: predict parameters from multi-source geophysical features.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

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
from lithosphere_model import LayerGrid, build_reference_fields
from parameterization import PARAMETER_MODES
from prior_network import PriorParameterModel, build_prior_feature_pack


def _build_optimizer(name: str, params: list[nn.Parameter], lr: float, weight_decay: float) -> torch.optim.Optimizer:
    if name == "lbfgs":
        return torch.optim.LBFGS(
            params,
            lr=lr,
            max_iter=20,
            tolerance_grad=1e-6,
            tolerance_change=1e-10,
            line_search_fn="strong_wolfe",
        )
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.zeros(0, dtype=float)


def _output_gradient_groups(coeffs: dict[str, torch.Tensor]) -> dict[str, list[torch.Tensor]]:
    return {
        "bottom": [coeffs["bottom_coeff"]] if coeffs["bottom_coeff"].numel() > 0 else [],
        "K": [coeffs["K_coeff"]] if coeffs["K_coeff"].numel() > 0 else [],
        "A": [coeffs["A_coeff"]] if coeffs["A_coeff"].numel() > 0 else [],
    }


def _grad_group_norm(loss: torch.Tensor, tensors: list[torch.Tensor]) -> float:
    tensors = [tensor for tensor in tensors if tensor.requires_grad and tensor.numel() > 0]
    if not tensors or not loss.requires_grad:
        return 0.0
    grads = torch.autograd.grad(loss, tensors, retain_graph=True, allow_unused=True)
    total = torch.zeros((), dtype=loss.dtype, device=loss.device)
    for grad in grads:
        if grad is not None:
            total = total + torch.sum(grad.detach() * grad.detach())
    return float(torch.sqrt(total).cpu())


def run_inversion(
    solver: HeatSolver,
    q_obs: np.ndarray,
    model: nn.Module,
    *,
    features: torch.Tensor | None = None,
    T_ref: float = 1300.0,
    steps: int = 40,
    lr: float = 1.0,
    optimizer_name: str = "lbfgs",
    weight_decay: float = 1e-4,
    history_every: int = 1,
    diagnose_gradients: bool = False,
    early_stop_patience: int = 0,
    early_stop_min_delta: float = 0.0,
    early_stop_metric: str = "loss",
    lam_q_mean: float = 0.0,
    lam_q_std: float = 0.0,
    verbose: bool = True,
) -> dict[str, Any]:
    if early_stop_patience < 0:
        raise ValueError("early_stop_patience must be non-negative.")
    if early_stop_min_delta < 0.0:
        raise ValueError("early_stop_min_delta must be non-negative.")
    if lam_q_mean < 0.0 or lam_q_std < 0.0:
        raise ValueError("lam_q_mean and lam_q_std must be non-negative.")

    valid = np.isfinite(q_obs)
    if not np.any(valid):
        raise ValueError("No valid heat-flow observations.")

    device = torch.device("cpu")
    dtype = torch.float64
    q_obs_t = torch.as_tensor(q_obs, dtype=dtype, device=device)
    valid_t = torch.as_tensor(valid, dtype=torch.bool, device=device)
    q_obs_valid_t = q_obs_t[valid_t]
    q_obs_mean_t = torch.mean(q_obs_valid_t)
    q_obs_std_t = torch.std(q_obs_valid_t, unbiased=False)
    feature_t = None if features is None else features.to(device=device, dtype=dtype)

    model = model.to(device=device, dtype=dtype)
    solver = solver.to(device=device, dtype=dtype)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = None if not trainable_params else _build_optimizer(optimizer_name, trainable_params, lr=lr, weight_decay=weight_decay)

    history: list[dict[str, float]] = []
    closure_calls = 0
    final_coeffs: dict[str, torch.Tensor] = {}
    final_decoded: dict[str, torch.Tensor | None] = {}
    final_fields: dict[str, torch.Tensor] = {}
    final_metrics: dict[str, float] | None = None
    early_stop_enabled = early_stop_patience > 0
    early_stop_best_metric = float("nan")
    early_stop_best_step = 0
    early_stop_wait = 0
    early_stopped = False
    early_stop_step: int | None = None
    best_model_state: dict[str, torch.Tensor] | None = None

    def copy_model_state() -> dict[str, torch.Tensor]:
        return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    def forward_state() -> tuple[
        dict[str, torch.Tensor],
        dict[str, torch.Tensor | None],
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]:
        coeffs = model(feature_t)
        T_bot, K0, A, decoded = model.decode_configured_parameter_fields(
            solver,
            coeffs,
            T_ref=T_ref,
            dtype=dtype,
            device=device,
        )
        T, q_pred = solver(T_bot, K0=K0, A_source=A)
        fields = {"T_bot": T_bot, "K0": K0, "A": A, "T": T, "q_pred": q_pred}
        q_pred_valid = q_pred[valid_t]
        residual = q_pred_valid - q_obs_valid_t
        return coeffs, decoded, fields, q_pred_valid, residual

    def compute_loss() -> tuple[
        torch.Tensor,
        dict[str, float],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor | None],
        dict[str, torch.Tensor],
    ]:
        coeffs, decoded, fields, q_pred_valid, residual = forward_state()
        data_misfit = 0.5 * torch.mean(residual * residual)
        data_misfit_sum = 0.5 * torch.dot(residual, residual)
        q_pred_mean = torch.mean(q_pred_valid)
        q_pred_std = torch.std(q_pred_valid, unbiased=False)
        moment_mean = 0.5 * lam_q_mean * (q_pred_mean - q_obs_mean_t) * (q_pred_mean - q_obs_mean_t)
        moment_std = 0.5 * lam_q_std * (q_pred_std - q_obs_std_t) * (q_pred_std - q_obs_std_t)
        moment = moment_mean + moment_std
        loss = data_misfit + moment
        metrics = {
            "loss": float(loss.detach().cpu()),
            "data_misfit": float(data_misfit.detach().cpu()),
            "data_misfit_sum": float(data_misfit_sum.detach().cpu()),
            "moment": float(moment.detach().cpu()),
            "moment_mean": float(moment_mean.detach().cpu()),
            "moment_std": float(moment_std.detach().cpu()),
            "q_pred_mean_obs_points": float(q_pred_mean.detach().cpu()),
            "q_pred_std_obs_points": float(q_pred_std.detach().cpu()),
            "q_obs_mean_obs_points": float(q_obs_mean_t.detach().cpu()),
            "q_obs_std_obs_points": float(q_obs_std_t.detach().cpu()),
            "rmse": float(torch.sqrt(torch.mean(residual * residual)).detach().cpu()),
        }
        if diagnose_gradients:
            for group_name, tensors in _output_gradient_groups(coeffs).items():
                metrics[f"grad_data_{group_name}"] = _grad_group_norm(data_misfit, tensors)
        return loss, metrics, coeffs, decoded, fields

    def update_early_stop(row: dict[str, float]) -> bool:
        nonlocal early_stop_best_metric, early_stop_best_step, early_stop_wait, early_stopped, early_stop_step, best_model_state
        if not early_stop_enabled:
            return False
        if early_stop_metric not in row:
            raise ValueError(f"early_stop_metric must be one of the recorded history keys; got {early_stop_metric!r}.")
        metric = row[early_stop_metric]
        if not np.isfinite(metric):
            early_stop_wait += 1
        elif not np.isfinite(early_stop_best_metric) or metric < early_stop_best_metric - early_stop_min_delta:
            early_stop_best_metric = metric
            early_stop_best_step = int(row["step"])
            early_stop_wait = 0
            best_model_state = copy_model_state()
            return False
        else:
            early_stop_wait += 1

        if early_stop_wait >= early_stop_patience:
            early_stopped = True
            early_stop_step = int(row["step"])
            return True
        return False

    def record_row(step: int) -> dict[str, float]:
        nonlocal final_coeffs, final_decoded, final_fields, final_metrics
        with torch.set_grad_enabled(diagnose_gradients):
            _, metrics, coeffs, decoded, fields = compute_loss()
        final_coeffs = coeffs
        final_decoded = decoded
        final_fields = fields
        final_metrics = metrics
        solve_info = solver.last_solve_info or {}
        bottom_coeff = coeffs["bottom_coeff"]
        K_log_xy = decoded["K_log_xy"]
        A_log_xy = decoded["A_log_xy"]
        row = {
            "step": float(step),
            "alpha_norm": float(torch.linalg.vector_norm(bottom_coeff).detach().cpu()) if bottom_coeff.numel() > 0 else 0.0,
            "K_log_maxabs": float(torch.max(torch.abs(K_log_xy)).detach().cpu()) if isinstance(K_log_xy, torch.Tensor) and K_log_xy.numel() > 0 else 0.0,
            "A_log_maxabs": float(torch.max(torch.abs(A_log_xy)).detach().cpu()) if isinstance(A_log_xy, torch.Tensor) and A_log_xy.numel() > 0 else 0.0,
            "closure_calls": float(closure_calls),
            "picard_iters": float(solve_info.get("picard_iters", float("nan"))),
            "cg_iters_total": float(solve_info.get("cg_iters_total", float("nan"))),
            "solve_time_s": float(solve_info.get("total_time_s", float("nan"))),
            **metrics,
        }
        history.append(row)
        if verbose:
            message = (
                f"[step {step:03d}] "
                f"loss={row['loss']:.6e} "
                f"rmse={row['rmse']:.3f} "
                f"data={row['data_misfit']:.6e} "
                f"moment={row['moment']:.6e} "
                f"|alpha|={row['alpha_norm']:.3f} "
                f"max|K_log|={row['K_log_maxabs']:.3f} "
                f"max|A_log|={row['A_log_maxabs']:.3f} "
                f"picard={int(row['picard_iters']) if np.isfinite(row['picard_iters']) else -1} "
                f"cg={int(row['cg_iters_total']) if np.isfinite(row['cg_iters_total']) else -1} "
                f"closures={int(row['closure_calls'])}"
            )
            if diagnose_gradients:
                message += (
                    f" | grad_data(b/K/A)="
                    f"{row.get('grad_data_bottom', 0.0):.3e}/"
                    f"{row.get('grad_data_K', 0.0):.3e}/"
                    f"{row.get('grad_data_A', 0.0):.3e}"
                )
            print(message)
        return row

    t0 = time.time()
    update_early_stop(record_row(0))

    if optimizer is None:
        if verbose:
            print("No trainable parameters; evaluated the fixed model only.")
    elif optimizer_name == "lbfgs":
        for step in range(1, steps + 1):
            def closure() -> torch.Tensor:
                nonlocal closure_calls, final_coeffs, final_decoded, final_fields, final_metrics
                closure_calls += 1
                optimizer.zero_grad()
                loss, metrics, coeffs, decoded, fields = compute_loss()
                loss.backward()
                final_coeffs = coeffs
                final_decoded = decoded
                final_fields = fields
                final_metrics = metrics
                return loss

            optimizer.step(closure)
            if step % history_every == 0 or step == steps:
                if update_early_stop(record_row(step)):
                    break
    else:
        for step in range(1, steps + 1):
            closure_calls += 1
            optimizer.zero_grad()
            loss, metrics, coeffs, decoded, fields = compute_loss()
            final_coeffs = coeffs
            final_decoded = decoded
            final_fields = fields
            final_metrics = metrics
            loss.backward()
            optimizer.step()
            if step % history_every == 0 or step == steps:
                if update_early_stop(record_row(step)):
                    break

    if early_stopped:
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            _, final_metrics, final_coeffs, final_decoded, final_fields = compute_loss()
        if verbose:
            print(
                f"Early stopped at step {early_stop_step}; "
                f"restored step {early_stop_best_step} "
                f"with {early_stop_metric}={early_stop_best_metric:.6e}."
            )

    elapsed = time.time() - t0
    if not final_fields:
        _, final_metrics, final_coeffs, final_decoded, final_fields = compute_loss()

    q_pred = final_fields["q_pred"].detach().cpu().numpy()
    residual = np.full_like(q_obs, np.nan, dtype=float)
    residual[valid] = q_pred[valid] - q_obs[valid]
    alpha_bot = final_coeffs["bottom_coeff"] if model.bottom_mode == "dct" else None

    return {
        "elapsed_s": elapsed,
        "closure_calls": closure_calls,
        "history": history,
        "T_bot": final_fields["T_bot"].detach().cpu().numpy(),
        "K0": final_fields["K0"].detach().cpu().numpy(),
        "A": final_fields["A"].detach().cpu().numpy(),
        "T": final_fields["T"].detach().cpu().numpy(),
        "q_pred": q_pred,
        "residual": residual,
        "valid_mask": valid,
        "alpha_bot": _tensor_to_numpy(alpha_bot),
        "T_bot_delta": _tensor_to_numpy(final_decoded["T_bot_delta"]),
        "K_coeff": _tensor_to_numpy(final_coeffs["K_coeff"]),
        "A_coeff": _tensor_to_numpy(final_coeffs["A_coeff"]),
        "model_state": model.state_dict(),
        "bottom_mode": model.bottom_mode,
        "k_mode": model.k_mode,
        "a_mode": model.a_mode,
        "dct_x": getattr(getattr(model, "bottom_dct", None), "x", None),
        "dct_y": getattr(getattr(model, "bottom_dct", None), "y", None),
        "optimizer": optimizer_name,
        "early_stopped": early_stopped,
        "early_stop_step": early_stop_step,
        "early_stop_best_step": early_stop_best_step,
        "early_stop_metric": early_stop_metric,
        "early_stop_best_metric": early_stop_best_metric,
        "early_stop_patience": early_stop_patience,
        "early_stop_min_delta": early_stop_min_delta,
        "final_loss": final_metrics["loss"] if final_metrics is not None else (history[-1]["loss"] if history else float("nan")),
        "final_rmse": final_metrics["rmse"] if final_metrics is not None else (history[-1]["rmse"] if history else float("nan")),
        "lam_q_mean": lam_q_mean,
        "lam_q_std": lam_q_std,
    }


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["direct", "prior_net"], default="direct")
    parser.add_argument("--data-dir", type=Path, default=Path("Data"))
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--dz-km", type=float, default=2.0)
    parser.add_argument("--z-max-km", type=float, default=120.0)
    parser.add_argument("--layer-transition-xy-km", type=float, default=20.0)
    parser.add_argument("--layer-transition-z-km", type=float, default=2.0)
    parser.add_argument("--bottom-mode", choices=PARAMETER_MODES, default="dct")
    parser.add_argument("--k-mode", choices=PARAMETER_MODES, default="fixed")
    parser.add_argument("--a-mode", choices=PARAMETER_MODES, default="fixed")
    parser.add_argument("--bottom-x", type=int, default=3)
    parser.add_argument("--bottom-y", type=int, default=3)
    parser.add_argument("--k-prop-x", type=int, default=6)
    parser.add_argument("--k-prop-y", type=int, default=6)
    parser.add_argument("--a-prop-x", type=int, default=8)
    parser.add_argument("--a-prop-y", type=int, default=8)
    parser.add_argument("--alpha-scale", type=float, default=800.0)
    parser.add_argument("--k-prop-scale", type=float, default=0.7)
    parser.add_argument("--a-prop-scale", type=float, default=0.7)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--optimizer", choices=["lbfgs", "adam", "adamw"], default="lbfgs")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.01)
    parser.add_argument("--early-stop-metric", type=str, default="loss")
    parser.add_argument("--lam-q-mean", type=float, default=0.0)
    parser.add_argument("--lam-q-std", type=float, default=0.0)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output-prefix", type=str, default=None)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--diagnose-gradients", action="store_true")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    data_dir = args.data_dir
    thermal_properties_path = data_dir / THERMAL_PROPERTIES_FILENAME

    if args.mode == "prior_net":
        if args.bottom_mode == "fixed":
            raise ValueError("prior_net requires bottom_mode in {'dct', 'full'}.")
        grid = LayerGrid(
            data_dir / GRID_FILENAME,
            data_dir / LAYER_FILENAME,
            stride=args.stride,
            stratum_csv=data_dir / STRATUM_FILLED_FILENAME,
        )
        surface_temp = load_surface_temperature_grid(data_dir / GRID_FILENAME, data_dir / THERMAL_GRID_FILENAME, args.stride)
        q_obs = load_heat_flow_observations(data_dir / GRID_FILENAME, data_dir / HEATFLOW_FILENAME, args.stride)
        features, channel_names = build_prior_feature_pack(data_dir, grid, args.stride)
        z_nodes_km = np.linspace(0.0, args.z_max_km, round(args.z_max_km / args.dz_km) + 1)
        refs = build_reference_fields(
            grid,
            z_nodes_km,
            thermal_properties_path=thermal_properties_path,
            layer_transition_z_km=args.layer_transition_z_km,
            layer_transition_xy_km=args.layer_transition_xy_km,
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
        model = PriorParameterModel(
            grid.shape[0],
            grid.shape[1],
            in_channels=features.shape[1],
            width=args.width,
            bottom_x=args.bottom_x,
            bottom_y=args.bottom_y,
            k_property_x=args.k_prop_x,
            k_property_y=args.k_prop_y,
            a_property_x=args.a_prop_x,
            a_property_y=args.a_prop_y,
            k_property_scale=args.k_prop_scale,
            a_property_scale=args.a_prop_scale,
            alpha_scale=args.alpha_scale,
            bottom_mode=args.bottom_mode,
            k_mode=args.k_mode,
            a_mode=args.a_mode,
        )
    else:
        grid = LayerGrid(
            data_dir / GRID_FILENAME,
            data_dir / LAYER_FILENAME,
            stride=args.stride,
            stratum_csv=data_dir / STRATUM_FILLED_FILENAME,
        )
        surface_temp = load_surface_temperature_grid(data_dir / GRID_FILENAME, data_dir / THERMAL_GRID_FILENAME, args.stride)
        q_obs = load_heat_flow_observations(data_dir / GRID_FILENAME, data_dir / HEATFLOW_FILENAME, args.stride)
        features = None
        channel_names = []
        z_nodes_km = np.linspace(0.0, args.z_max_km, round(args.z_max_km / args.dz_km) + 1)
        refs = build_reference_fields(
            grid,
            z_nodes_km,
            thermal_properties_path=thermal_properties_path,
            layer_transition_z_km=args.layer_transition_z_km,
            layer_transition_xy_km=args.layer_transition_xy_km,
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
        model = DirectParameterModel(
            grid.shape[0],
            grid.shape[1],
            bottom_x=args.bottom_x,
            bottom_y=args.bottom_y,
            k_property_x=args.k_prop_x,
            k_property_y=args.k_prop_y,
            a_property_x=args.a_prop_x,
            a_property_y=args.a_prop_y,
            k_property_scale=args.k_prop_scale,
            a_property_scale=args.a_prop_scale,
            alpha_scale=args.alpha_scale,
            bottom_mode=args.bottom_mode,
            k_mode=args.k_mode,
            a_mode=args.a_mode,
        )

    if args.init_checkpoint is not None:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"])

    early_stop_patience = args.early_stop_patience if args.early_stop else 0

    out = run_inversion(
        solver,
        q_obs,
        model,
        features=features,
        steps=args.steps,
        lr=args.lr,
        optimizer_name=args.optimizer,
        weight_decay=args.weight_decay,
        diagnose_gradients=args.diagnose_gradients,
        early_stop_patience=early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        early_stop_metric=args.early_stop_metric,
        lam_q_mean=args.lam_q_mean,
        lam_q_std=args.lam_q_std,
        verbose=not args.quiet,
    )

    if args.output_prefix:
        output_dir = Path(args.output_prefix)
    else:
        output_dir = Path("outputs") / time.strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = output_dir / "checkpoint.pt"
    summary_path = output_dir / "summary.json"
    train_log_json_path = output_dir / "train_log.json"
    fig_path = output_dir / "panels.png"
    loss_fig_path = output_dir / "loss_curve.png"
    model_config = {
        "mode": args.mode,
        "bottom_mode": args.bottom_mode,
        "k_mode": args.k_mode,
        "a_mode": args.a_mode,
        "bottom_x": args.bottom_x,
        "bottom_y": args.bottom_y,
        "k_property_x": args.k_prop_x,
        "k_property_y": args.k_prop_y,
        "a_property_x": args.a_prop_x,
        "a_property_y": args.a_prop_y,
        "alpha_scale": args.alpha_scale,
        "k_property_scale": args.k_prop_scale,
        "a_property_scale": args.a_prop_scale,
        "width": args.width if args.mode == "prior_net" else None,
    }
    torch.save(
        {
            "model_state": out["model_state"],
            "mode": args.mode,
            "optimizer": args.optimizer,
            "data_dir": str(data_dir),
            "stride": args.stride,
            "dz_km": args.dz_km,
            "z_max_km": args.z_max_km,
            "layer_transition_z_km": args.layer_transition_z_km,
            "layer_transition_xy_km": args.layer_transition_xy_km,
            "bottom_x": args.bottom_x,
            "bottom_y": args.bottom_y,
            "dct_x": out["dct_x"],
            "dct_y": out["dct_y"],
            "model_config": model_config,
            "diagnose_gradients": args.diagnose_gradients,
            "lam_q_mean": args.lam_q_mean,
            "lam_q_std": args.lam_q_std,
            "early_stop": args.early_stop,
            "early_stop_patience": early_stop_patience,
            "early_stop_min_delta": args.early_stop_min_delta,
            "early_stop_metric": args.early_stop_metric,
            "channels": channel_names,
            "init_checkpoint": None if args.init_checkpoint is None else str(args.init_checkpoint),
        },
        ckpt_path,
    )
    history_df = pd.DataFrame(out["history"])
    with open(train_log_json_path, "w", encoding="utf-8") as f:
        json.dump(out["history"], f, ensure_ascii=False, indent=2)

    summary = {
        "run": {
            "mode": args.mode,
            "optimizer": args.optimizer,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "steps": args.steps,
            "heatflow_csv": str(data_dir / HEATFLOW_FILENAME),
            "layer_csv": str(data_dir / LAYER_FILENAME),
            "stratum_csv": str(data_dir / STRATUM_FILLED_FILENAME),
            "stride": args.stride,
            "dz_km": args.dz_km,
            "z_max_km": args.z_max_km,
            "layer_transition_z_km": args.layer_transition_z_km,
            "layer_transition_xy_km": args.layer_transition_xy_km,
            "early_stop": args.early_stop,
            "early_stop_patience": early_stop_patience,
            "early_stop_min_delta": args.early_stop_min_delta,
            "early_stop_metric": args.early_stop_metric,
            "init_checkpoint": None if args.init_checkpoint is None else str(args.init_checkpoint),
        },
        "loss_options": {
            "lam_q_mean": out["lam_q_mean"],
            "lam_q_std": out["lam_q_std"],
        },
        "model_config": model_config,
    }
    if args.mode == "prior_net":
        summary["prior_network"] = {
            "width": args.width,
            "channels": channel_names,
        }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    loss_fig, loss_ax = plt.subplots(figsize=(7, 4))
    loss_ax.plot(history_df["step"], history_df["loss"], marker="o", label="loss")
    loss_ax.plot(history_df["step"], history_df["data_misfit"], marker="o", label="data_misfit")
    loss_ax.set_xlabel("step")
    loss_ax.set_ylabel("objective")
    loss_ax.set_title("Training loss")
    loss_ax.grid(True, alpha=0.3)
    loss_ax.legend()
    loss_fig.tight_layout()
    loss_fig.savefig(loss_fig_path, dpi=180, bbox_inches="tight")
    plt.close(loss_fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [
        (q_obs, "q_obs"),
        (out["q_pred"], "q_pred"),
        (out["residual"], "residual"),
        (out["T_bot"], "T_bot"),
    ]
    for ax, (data, title) in zip(axes.ravel(), panels):
        im = ax.imshow(data, origin="lower")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.8, pad=0.01, fraction=0.04)
    fig.subplots_adjust(wspace=0.02, hspace=0.28)
    fig.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
