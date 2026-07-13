"""Empirical direct-inversion ensemble from a template checkpoint."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from inversion import (
    CHECKPOINT_FORMAT_VERSION,
    build_problem,
    load_checkpoint,
    run_inversion,
    save_checkpoint,
    verify_input_hashes,
)
from parameterization import DirectParameterization


def split_observations(
    q_obs: np.ndarray,
    holdout_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid_indices = np.flatnonzero(np.isfinite(q_obs).ravel())
    holdout_count = round(holdout_fraction * valid_indices.size)
    if not 0 < holdout_count < valid_indices.size:
        raise ValueError("holdout split must leave both training and holdout observations")
    rng = np.random.default_rng(seed)
    held_indices = rng.choice(valid_indices, size=holdout_count, replace=False)
    train = q_obs.copy()
    holdout_mask = np.zeros(q_obs.shape, dtype=bool)
    holdout_mask.ravel()[held_indices] = True
    train[holdout_mask] = np.nan
    return train, np.isfinite(train), holdout_mask


def _update_moments(
    moments: dict[str, tuple[np.ndarray, np.ndarray]],
    fields: dict[str, np.ndarray],
    count: int,
) -> None:
    for name, value in fields.items():
        if name not in moments:
            moments[name] = (value.copy(), np.zeros_like(value))
            continue
        mean, m2 = moments[name]
        delta = value - mean
        mean += delta / count
        m2 += delta * (value - mean)


def _finalize_moments(
    moments: dict[str, tuple[np.ndarray, np.ndarray]],
    count: int,
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for name, (mean, m2) in moments.items():
        output[f"{name}_mean"] = mean
        output[f"{name}_std"] = np.sqrt(m2 / (count - 1)) if count > 1 else np.zeros_like(mean)
    return output


def run_uq(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    members: int,
    holdout_fraction: float,
    seed: int,
    initial_noise_fraction: float,
    maxiter: int,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    if members < 1:
        raise ValueError("members must be positive")
    if initial_noise_fraction < 0.0 or maxiter < 0:
        raise ValueError("initial noise and maxiter must be non-negative")
    _, metadata = load_checkpoint(checkpoint_path)
    data = metadata["data"]
    grid, solver, q_obs, input_sha256 = build_problem(
        Path(data["data_dir"]),
        stride=data["stride"],
        dz_km=data["dz_km"],
        z_max_km=data["z_max_km"],
        layer_transition_xy_km=data["layer_transition_xy_km"],
        layer_transition_z_km=data["layer_transition_z_km"],
    )
    verify_input_hashes(data["input_sha256"], input_sha256)
    parameterization = DirectParameterization.from_config(metadata["parameterization"])
    if grid.shape != (parameterization.ny, parameterization.nx):
        raise ValueError("checkpoint and data grid shapes differ")
    objective = metadata["objective"]
    bounds = np.asarray(parameterization.bounds(), dtype=np.float64)
    output_dir.mkdir(parents=True)
    members_dir = output_dir / "members"
    members_dir.mkdir(exist_ok=True)

    summaries: list[dict[str, Any]] = []
    moments: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    start = time.perf_counter()
    for member_index in range(members):
        member_seed = seed + member_index
        train_obs, train_mask, holdout_mask = split_observations(
            q_obs,
            holdout_fraction,
            member_seed,
        )
        rng = np.random.default_rng(member_seed)
        if parameterization.size:
            scales = bounds[:, 1]
            initial_parameters = np.clip(
                rng.normal(size=parameterization.size)
                * initial_noise_fraction
                * scales,
                bounds[:, 0],
                bounds[:, 1],
            )
        else:
            initial_parameters = parameterization.initial()

        solver.clear_warm_start()
        result = run_inversion(
            solver,
            train_obs,
            parameterization,
            T_ref=data["t_ref"],
            maxiter=maxiter,
            initial_parameters=initial_parameters,
            verbose=False,
            **objective,
        )
        holdout_residual = result["q_pred"][holdout_mask] - q_obs[holdout_mask]
        holdout_rmse = float(np.sqrt(np.mean(holdout_residual**2)))
        member_dir = members_dir / f"member_{member_index:03d}"
        member_dir.mkdir()
        member_metadata = {
            **metadata,
            "run": {
                "maxiter": maxiter,
                "initial_checkpoint": None,
                "template_checkpoint": str(checkpoint_path.resolve()),
                "uq_member": {
                    "index": member_index,
                    "seed": member_seed,
                    "holdout_fraction": holdout_fraction,
                    "initial_noise_fraction": initial_noise_fraction,
                },
            },
        }
        save_checkpoint(
            member_dir / "checkpoint.npz",
            result["parameters"],
            member_metadata,
        )
        np.savez_compressed(
            member_dir / "surface_fields.npz",
            q_obs=q_obs,
            q_pred=result["q_pred"],
            residual=result["q_pred"] - q_obs,
            train_mask=train_mask,
            holdout_mask=holdout_mask,
            T_bot=result["T_bot"],
        )
        summary = {
            "member": member_index,
            "seed": member_seed,
            "train_count": int(np.count_nonzero(train_mask)),
            "holdout_count": int(np.count_nonzero(holdout_mask)),
            "train_rmse_mW_m2": result["metrics"]["rmse_mW_m2"],
            "holdout_rmse_mW_m2": holdout_rmse,
            "loss": result["final_loss"],
            "gradient_norm": result["final_gradient_norm"],
            "iterations": result["optimization"]["iterations"],
            "success": result["optimization"]["success"],
            "status": result["optimization"]["status"],
            "message": result["optimization"]["message"],
            "pde_evaluations": result["pde_evaluations"],
            "elapsed_s": result["elapsed_s"],
        }
        with (member_dir / "summary.json").open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(summary, stream, indent=2)
        summaries.append(summary)
        _update_moments(
            moments,
            {
                "T": result["T"],
                "T_bot": result["T_bot"],
                "K0": result["K0"],
                "A": result["A"],
                "q_pred": result["q_pred"],
            },
            member_index + 1,
        )
        if verbose:
            print(
                f"member={member_index:03d} train_rmse={summary['train_rmse_mW_m2']:.4f} "
                f"holdout_rmse={holdout_rmse:.4f}"
            )

    aggregate_fields = _finalize_moments(moments, members)
    np.savez_compressed(
        output_dir / "aggregate_fields.npz",
        z_nodes_km=solver.z_nodes_km,
        **aggregate_fields,
    )
    pd.DataFrame(summaries).to_csv(output_dir / "member_summaries.csv", index=False)
    aggregate_summary = {
        "members": members,
        "train_rmse_mean_mW_m2": float(np.mean([row["train_rmse_mW_m2"] for row in summaries])),
        "holdout_rmse_mean_mW_m2": float(np.mean([row["holdout_rmse_mW_m2"] for row in summaries])),
        "success_count": sum(row["success"] for row in summaries),
        "elapsed_s": time.perf_counter() - start,
    }
    with (output_dir / "aggregate_summary.json").open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(aggregate_summary, stream, indent=2)
    manifest = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "template_checkpoint": str(checkpoint_path.resolve()),
        "data": data,
        "parameterization": parameterization.config(),
        "objective": objective,
        "ensemble": {
            "members": members,
            "holdout_fraction": holdout_fraction,
            "seed": seed,
            "initial_noise_fraction": initial_noise_fraction,
            "maxiter": maxiter,
        },
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("template", type=Path, help="Run directory or checkpoint.npz")
    parser.add_argument("--members", type=int, default=8)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--initial-noise-fraction", type=float, default=0.02)
    parser.add_argument("--maxiter", type=int, default=40)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    checkpoint_path = args.template / "checkpoint.npz" if args.template.is_dir() else args.template
    output_dir = args.output_dir or Path("outputs") / f"uq_{time.strftime('%Y%m%d_%H%M%S')}"
    run_uq(
        checkpoint_path,
        output_dir,
        members=args.members,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
        initial_noise_fraction=args.initial_noise_fraction,
        maxiter=args.maxiter,
        verbose=not args.quiet,
    )
    print(f"UQ directory: {output_dir}")


if __name__ == "__main__":
    main()
