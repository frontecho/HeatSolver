from __future__ import annotations

"""Evaluate a direct-inversion checkpoint."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from inversion import build_problem, load_checkpoint
from parameterization import DirectParameterization


def _field_stats(field: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(field)[np.isfinite(field)]
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def evaluate_checkpoint(checkpoint_path: Path) -> tuple[Any, Any, dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    parameters, metadata = load_checkpoint(checkpoint_path)
    data = metadata["data"]
    grid, solver, q_obs = build_problem(
        Path(data["data_dir"]),
        stride=data["stride"],
        dz_km=data["dz_km"],
        z_max_km=data["z_max_km"],
        layer_transition_xy_km=data["layer_transition_xy_km"],
        layer_transition_z_km=data["layer_transition_z_km"],
    )
    parameterization = DirectParameterization.from_config(metadata["parameterization"])
    fields = parameterization.decode(solver, parameters, T_ref=data["t_ref"])
    fields["T"] = solver.solve(
        fields["T_bot"],
        K0=fields["K0"],
        A_source=fields["A"],
    )
    fields["q_pred"] = solver.surface_heat_flow(fields["T"], fields["K0"])
    fields["residual"] = fields["q_pred"] - q_obs
    fields["residual"][~np.isfinite(q_obs)] = np.nan
    return grid, solver, fields, q_obs, metadata


def save_evaluation(
    output_dir: Path,
    checkpoint_path: Path,
    grid: Any,
    solver: Any,
    fields: dict[str, np.ndarray],
    q_obs: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "evaluated_fields.npz",
        z_nodes_km=solver.z_nodes_km,
        T=fields["T"],
        T_bot=fields["T_bot"],
        K0=fields["K0"],
        A=fields["A"],
        q_obs=q_obs,
        q_pred=fields["q_pred"],
        residual=fields["residual"],
    )

    rows, columns = np.indices(grid.shape)
    pd.DataFrame(
        {
            "row": rows.ravel(),
            "column": columns.ravel(),
            "lon": grid.lon.ravel(),
            "lat": grid.lat.ravel(),
            "X": grid.x.ravel(),
            "Y": grid.y.ravel(),
            "T_bot_C": fields["T_bot"].ravel(),
            "q_obs_mW_m2": q_obs.ravel(),
            "q_pred_mW_m2": fields["q_pred"].ravel(),
            "residual_mW_m2": fields["residual"].ravel(),
        }
    ).to_csv(output_dir / "surface_fields.csv", index=False)

    summary = {
        "checkpoint": str(checkpoint_path.resolve()),
        "metadata": metadata,
        "shape": list(fields["T"].shape),
        "solver": solver.last_solve_info,
        "statistics": {
            "temperature_C": _field_stats(fields["T"]),
            "bottom_temperature_C": _field_stats(fields["T_bot"]),
            "conductivity_W_mK": _field_stats(fields["K0"]),
            "heat_production_W_m3": _field_stats(fields["A"]),
            "predicted_heat_flow_mW_m2": _field_stats(fields["q_pred"]),
            "residual_mW_m2": _field_stats(fields["residual"]),
        },
    }
    with (output_dir / "evaluation_summary.json").open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, field, title in zip(
        axes.ravel(),
        (q_obs, fields["q_pred"], fields["residual"], fields["T_bot"]),
        ("Observed heat flow", "Predicted heat flow", "Residual", "Bottom temperature"),
        strict=True,
    ):
        image = axis.imshow(field, origin="lower")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, shrink=0.8)
    figure.tight_layout()
    figure.savefig(output_dir / "surface_diagnostics.png", dpi=180)
    plt.close(figure)

    depth_indices = np.unique(
        np.linspace(0, solver.nz - 1, min(6, solver.nz), dtype=int)
    )
    figure, axes = plt.subplots(1, depth_indices.size, figsize=(3.3 * depth_indices.size, 3.3))
    axes = np.atleast_1d(axes)
    for axis, depth_index in zip(axes, depth_indices, strict=True):
        image = axis.imshow(fields["T"][:, :, depth_index], origin="lower")
        axis.set_title(f"{solver.z_nodes_km[depth_index]:g} km")
        figure.colorbar(image, ax=axis, shrink=0.75)
    figure.tight_layout()
    figure.savefig(output_dir / "temperature_slices.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Run directory or checkpoint.npz")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    checkpoint_path = args.run / "checkpoint.npz" if args.run.is_dir() else args.run
    grid, solver, fields, q_obs, metadata = evaluate_checkpoint(checkpoint_path)
    output_dir = args.output_dir or checkpoint_path.parent / "evaluation"
    save_evaluation(
        output_dir,
        checkpoint_path,
        grid,
        solver,
        fields,
        q_obs,
        metadata,
    )
    print(f"Evaluation directory: {output_dir}")


if __name__ == "__main__":
    main()
