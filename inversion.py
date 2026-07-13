"""Direct PDE-constrained geothermal inversion."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

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
from heat_solver import HeatSolver
from lithosphere_model import LayerGrid, build_reference_fields
from parameterization import PARAMETER_MODES, DirectParameterization

CHECKPOINT_FORMAT_VERSION = 2
INPUT_FILENAMES = (
    GRID_FILENAME,
    LAYER_FILENAME,
    STRATUM_FILLED_FILENAME,
    THERMAL_GRID_FILENAME,
    THERMAL_PROPERTIES_FILENAME,
    HEATFLOW_FILENAME,
)


def hash_input_files(data_dir: Path) -> dict[str, str]:
    hashes = {}
    for filename in INPUT_FILENAMES:
        with (data_dir / filename).open("rb") as stream:
            hashes[filename] = hashlib.file_digest(stream, "sha256").hexdigest()
    return hashes


def verify_input_hashes(expected: dict[str, str], actual: dict[str, str]) -> None:
    if actual != expected:
        raise ValueError("input files differ from the checkpoint")


def build_problem(
    data_dir: Path,
    *,
    stride: int,
    dz_km: float,
    z_max_km: float,
    layer_transition_xy_km: float,
    layer_transition_z_km: float,
) -> tuple[LayerGrid, HeatSolver, np.ndarray, dict[str, str]]:
    input_sha256 = hash_input_files(data_dir)
    vertical_intervals = round(z_max_km / dz_km)
    if not np.isclose(vertical_intervals * dz_km, z_max_km):
        raise ValueError("z_max_km must be divisible by dz_km")
    grid = LayerGrid(
        data_dir / GRID_FILENAME,
        data_dir / LAYER_FILENAME,
        stride=stride,
        stratum_csv=data_dir / STRATUM_FILLED_FILENAME,
    )
    surface_temp = load_surface_temperature_grid(
        data_dir / GRID_FILENAME,
        data_dir / THERMAL_GRID_FILENAME,
        stride,
    )
    q_obs = load_heat_flow_observations(
        data_dir / GRID_FILENAME,
        data_dir / HEATFLOW_FILENAME,
        stride,
    )
    refs = build_reference_fields(
        grid,
        np.linspace(0.0, z_max_km, vertical_intervals + 1),
        thermal_properties_path=data_dir / THERMAL_PROPERTIES_FILENAME,
        layer_transition_xy_km=layer_transition_xy_km,
        layer_transition_z_km=layer_transition_z_km,
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
    return grid, solver, q_obs, input_sha256


def _coefficient_regularization(
    parameters: np.ndarray,
    parameterization: DirectParameterization,
    weights: dict[str, float],
) -> tuple[float, np.ndarray, dict[str, float]]:
    gradient = np.zeros_like(parameters)
    metrics: dict[str, float] = {}
    total = 0.0
    groups = {
        "bottom": (parameterization.bottom_slice, parameterization.bottom_mode),
        "K": (parameterization.k_slice, parameterization.k_mode),
        "A": (parameterization.a_slice, parameterization.a_mode),
    }
    for name, (parameter_slice, mode) in groups.items():
        values = parameters[parameter_slice]
        weight = weights[name]
        scale = (
            1.0 / (parameterization.ny * parameterization.nx)
            if mode == "full"
            else 1.0
        )
        value = 0.5 * weight * scale * float(np.sum(values**2))
        metrics[f"{name}_regularization"] = value
        total += value
        gradient[parameter_slice] = weight * scale * values
    metrics["coefficient_regularization"] = total
    return total, gradient, metrics


def evaluate_objective(
    solver: HeatSolver,
    q_obs: np.ndarray,
    parameterization: DirectParameterization,
    parameters: np.ndarray,
    *,
    T_ref: float,
    lam_bottom: float = 0.0,
    lam_k: float = 0.0,
    lam_a: float = 0.0,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    valid = np.isfinite(q_obs)
    if not np.any(valid):
        raise ValueError("no valid heat-flow observations")

    state = parameterization.decode(solver, parameters, T_ref=T_ref)
    T = solver.solve(state["T_bot"], K0=state["K0"], A_source=state["A"])
    q_pred = solver.surface_heat_flow(T, state["K0"])
    observed = q_obs[valid]
    predicted = q_pred[valid]
    residual = predicted - observed
    count = residual.size

    data_misfit = 0.5 * float(np.mean(residual**2))
    grad_q_valid = residual / count
    predicted_mean = float(np.mean(predicted))
    observed_mean = float(np.mean(observed))
    predicted_std = float(np.std(predicted))
    observed_std = float(np.std(observed))

    regularization, regularization_gradient, regularization_metrics = _coefficient_regularization(
        parameters,
        parameterization,
        {"bottom": lam_bottom, "K": lam_k, "A": lam_a},
    )
    loss = data_misfit + regularization

    if parameterization.size:
        grad_q = np.zeros_like(q_pred)
        grad_q[valid] = grad_q_valid
        grad_T_bot, grad_K0, grad_A = solver.adjoint_gradient(
            T,
            state["K0"],
            grad_q=grad_q,
        )
        gradient = parameterization.pullback(
            solver,
            state,
            grad_T_bot,
            grad_K0,
            grad_A,
        )
        gradient += regularization_gradient
    else:
        gradient = np.zeros(0, dtype=np.float64)

    residual_field = np.full_like(q_obs, np.nan, dtype=np.float64)
    residual_field[valid] = residual
    metrics = {
        "loss": loss,
        "data_misfit": data_misfit,
        "rmse_mW_m2": float(np.sqrt(np.mean(residual**2))),
        "predicted_mean_mW_m2": predicted_mean,
        "predicted_std_mW_m2": predicted_std,
        "observed_mean_mW_m2": observed_mean,
        "observed_std_mW_m2": observed_std,
        **regularization_metrics,
    }
    state.update(
        {
            "T": T,
            "q_pred": q_pred,
            "residual": residual_field,
            "metrics": metrics,
        }
    )
    return loss, gradient, state


def run_inversion(
    solver: HeatSolver,
    q_obs: np.ndarray,
    parameterization: DirectParameterization,
    *,
    T_ref: float = 1300.0,
    maxiter: int = 40,
    initial_parameters: np.ndarray | None = None,
    lam_bottom: float = 0.0,
    lam_k: float = 0.0,
    lam_a: float = 0.0,
    verbose: bool = True,
) -> dict[str, Any]:
    weights = (lam_bottom, lam_k, lam_a)
    if any(weight < 0.0 for weight in weights):
        raise ValueError("regularization weights must be non-negative")
    parameters = parameterization.initial() if initial_parameters is None else np.asarray(initial_parameters, dtype=np.float64)
    if parameters.shape != (parameterization.size,):
        raise ValueError(f"initial_parameters must have shape {(parameterization.size,)}")

    cache_parameters: np.ndarray | None = None
    cache_evaluation: tuple[float, np.ndarray, dict[str, Any]] | None = None
    evaluations = 0

    def evaluate(x: np.ndarray) -> tuple[float, np.ndarray, dict[str, Any]]:
        nonlocal cache_parameters, cache_evaluation, evaluations
        if cache_parameters is None or not np.array_equal(x, cache_parameters):
            cache_parameters = x.copy()
            cache_evaluation = evaluate_objective(
                solver,
                q_obs,
                parameterization,
                x,
                T_ref=T_ref,
                lam_bottom=lam_bottom,
                lam_k=lam_k,
                lam_a=lam_a,
            )
            evaluations += 1
        return cache_evaluation

    history: list[dict[str, float | int]] = []
    recorded_parameters: list[np.ndarray] = []

    def record(iteration: int, x: np.ndarray) -> None:
        loss, gradient, state = evaluate(x)
        row: dict[str, float | int] = {
            "iteration": iteration,
            **state["metrics"],
            "gradient_norm": float(np.linalg.norm(gradient)),
            "pde_evaluations": evaluations,
        }
        if solver.last_solve_info is not None:
            row.update(solver.last_solve_info)
        history.append(row)
        recorded_parameters.append(x.copy())
        if verbose:
            print(
                f"iter={iteration:03d} loss={loss:.6e} "
                f"rmse={state['metrics']['rmse_mW_m2']:.4f} "
                f"|g|={row['gradient_norm']:.3e}"
            )

    start = time.perf_counter()
    record(0, parameters)
    if parameterization.size and maxiter > 0:
        iteration = 0

        def objective(x: np.ndarray) -> tuple[float, np.ndarray]:
            loss, gradient, _ = evaluate(x)
            return loss, gradient

        def callback(x: np.ndarray) -> None:
            nonlocal iteration
            iteration += 1
            record(iteration, x)

        optimization = minimize(
            objective,
            parameters,
            method="L-BFGS-B",
            jac=True,
            bounds=parameterization.bounds(),
            callback=callback,
            options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-6, "maxls": 30},
        )
        parameters = optimization.x
        optimization_summary = {
            "success": bool(optimization.success),
            "status": int(optimization.status),
            "message": str(optimization.message),
            "iterations": int(optimization.nit),
            "function_evaluations": int(optimization.nfev),
        }
    else:
        optimization_summary = {
            "success": True,
            "status": 0,
            "message": "fixed model" if parameterization.size == 0 else "maxiter is zero",
            "iterations": 0,
            "function_evaluations": 1,
        }

    final_loss, final_gradient, final_state = evaluate(parameters)
    if not np.array_equal(parameters, recorded_parameters[-1]):
        record(optimization_summary["iterations"], parameters)
    return {
        **final_state,
        "parameters": parameters.copy(),
        "history": history,
        "optimization": optimization_summary,
        "elapsed_s": time.perf_counter() - start,
        "pde_evaluations": evaluations,
        "final_loss": final_loss,
        "final_gradient_norm": float(np.linalg.norm(final_gradient)),
    }


def save_checkpoint(path: Path, parameters: np.ndarray, metadata: dict[str, Any]) -> None:
    np.savez_compressed(
        path,
        parameters=np.asarray(parameters, dtype=np.float64),
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )


def load_checkpoint(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as checkpoint:
        parameters = checkpoint["parameters"].astype(np.float64, copy=True)
        metadata = json.loads(str(checkpoint["metadata"].item()))
    if metadata["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported checkpoint format")
    return parameters, metadata


def load_initial_checkpoint(
    path: Path,
    data_config: dict[str, Any],
    parameterization_config: dict[str, Any],
) -> np.ndarray:
    parameters, metadata = load_checkpoint(path)
    if metadata["data"] != data_config:
        raise ValueError("initial checkpoint uses different data settings")
    if metadata["parameterization"] != parameterization_config:
        raise ValueError("initial checkpoint uses a different parameterization")
    return parameters


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def save_run(
    output_dir: Path,
    result: dict[str, Any],
    q_obs: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_checkpoint(output_dir / "checkpoint.npz", result["parameters"], metadata)
    np.savez_compressed(
        output_dir / "surface_fields.npz",
        q_obs=q_obs,
        q_pred=result["q_pred"],
        residual=result["residual"],
        T_bot=result["T_bot"],
    )
    _write_json(output_dir / "train_log.json", result["history"])
    _write_json(
        output_dir / "summary.json",
        {
            "metadata": metadata,
            "optimization": result["optimization"],
            "metrics": result["metrics"],
            "elapsed_s": result["elapsed_s"],
            "pde_evaluations": result["pde_evaluations"],
            "final_gradient_norm": result["final_gradient_norm"],
        },
    )

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, field, title in zip(
        axes.ravel(),
        (q_obs, result["q_pred"], result["residual"], result["T_bot"]),
        ("Observed heat flow", "Predicted heat flow", "Residual", "Bottom temperature"),
        strict=True,
    ):
        image = axis.imshow(field, origin="lower")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, shrink=0.8)
    figure.tight_layout()
    figure.savefig(output_dir / "fields.png", dpi=180)
    plt.close(figure)

    history = result["history"]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(
        [row["iteration"] for row in history],
        [row["loss"] for row in history],
        marker="o",
        label="objective",
    )
    axis.plot(
        [row["iteration"] for row in history],
        [row["data_misfit"] for row in history],
        marker="o",
        label="data misfit",
    )
    axis.set_xlabel("L-BFGS-B iteration")
    axis.set_ylabel("Objective")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "loss_curve.png", dpi=180)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("Data"))
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--dz-km", type=float, default=2.0)
    parser.add_argument("--z-max-km", type=float, default=120.0)
    parser.add_argument("--t-ref", type=float, default=1300.0)
    parser.add_argument("--layer-transition-xy-km", type=float, default=20.0)
    parser.add_argument("--layer-transition-z-km", type=float, default=2.0)

    parser.add_argument("--bottom-mode", choices=PARAMETER_MODES, default="dct")
    parser.add_argument("--k-mode", choices=PARAMETER_MODES, default="fixed")
    parser.add_argument("--a-mode", choices=PARAMETER_MODES, default="fixed")
    parser.add_argument("--bottom-dct-x", type=int, default=3)
    parser.add_argument("--bottom-dct-y", type=int, default=3)
    parser.add_argument("--k-dct-x", type=int, default=5)
    parser.add_argument("--k-dct-y", type=int, default=5)
    parser.add_argument("--a-dct-x", type=int, default=5)
    parser.add_argument("--a-dct-y", type=int, default=5)
    parser.add_argument("--bottom-bound-c", type=float, default=800.0)
    parser.add_argument("--k-log-bound", type=float, default=0.7)
    parser.add_argument("--a-log-bound", type=float, default=0.7)

    parser.add_argument("--maxiter", type=int, default=40)
    parser.add_argument("--lam-bottom", type=float, default=0.0)
    parser.add_argument("--lam-k", type=float, default=0.0)
    parser.add_argument("--lam-a", type=float, default=0.0)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    grid, solver, q_obs, input_sha256 = build_problem(
        args.data_dir,
        stride=args.stride,
        dz_km=args.dz_km,
        z_max_km=args.z_max_km,
        layer_transition_xy_km=args.layer_transition_xy_km,
        layer_transition_z_km=args.layer_transition_z_km,
    )
    parameterization = DirectParameterization(
        *grid.shape,
        bottom_mode=args.bottom_mode,
        k_mode=args.k_mode,
        a_mode=args.a_mode,
        bottom_dct_x=args.bottom_dct_x,
        bottom_dct_y=args.bottom_dct_y,
        k_dct_x=args.k_dct_x,
        k_dct_y=args.k_dct_y,
        a_dct_x=args.a_dct_x,
        a_dct_y=args.a_dct_y,
        bottom_bound_C=args.bottom_bound_c,
        k_log_bound=args.k_log_bound,
        a_log_bound=args.a_log_bound,
    )
    data_config = {
        "data_dir": str(args.data_dir.resolve()),
        "stride": args.stride,
        "dz_km": args.dz_km,
        "z_max_km": args.z_max_km,
        "t_ref": args.t_ref,
        "layer_transition_xy_km": args.layer_transition_xy_km,
        "layer_transition_z_km": args.layer_transition_z_km,
        "coarsening": "complete_blocks",
        "input_sha256": input_sha256,
    }
    initial_parameters = None
    if args.initial_checkpoint is not None:
        initial_parameters = load_initial_checkpoint(
            args.initial_checkpoint,
            data_config,
            parameterization.config(),
        )

    output_dir = args.output_dir or Path("outputs") / time.strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True)

    result = run_inversion(
        solver,
        q_obs,
        parameterization,
        T_ref=args.t_ref,
        maxiter=args.maxiter,
        initial_parameters=initial_parameters,
        lam_bottom=args.lam_bottom,
        lam_k=args.lam_k,
        lam_a=args.lam_a,
        verbose=not args.quiet,
    )
    metadata = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "data": data_config,
        "parameterization": parameterization.config(),
        "objective": {
            "lam_bottom": args.lam_bottom,
            "lam_k": args.lam_k,
            "lam_a": args.lam_a,
        },
        "run": {
            "maxiter": args.maxiter,
            "initial_checkpoint": (
                None
                if args.initial_checkpoint is None
                else str(args.initial_checkpoint.resolve())
            ),
        },
    }
    save_run(output_dir, result, q_obs, metadata)
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
