# Reproducing the NCC thermal model

## Environment

```powershell
uv sync --locked
uv run python -B -m pytest -q -p no:cacheprovider
```

Python dependencies are pinned by `uv.lock`. The maintained numerical code is
split across `heat_solver.py`, `nonlinear_heat_solver.py`, `heat_flux.py`,
`observations.py` and `spatial_basis.py`. Generic error statistics are in
`fit_metrics.py`, with NCC assumptions in `experiments/model.py`;
tests cover analytical solutions, PDE and adjoint behavior,
sampling, physical bounds, source budgets, refinement and current-model gradients.
GMT and TeX are additional requirements for publication graphics and documents.
New fit manifests record the split source files in `source_sha256`; a hash of
`heat_solver.py` alone no longer identifies the complete numerical implementation.

## Numerical inputs and identity

The selected model pointer is
`experiments/configs/selected_model.json`. Its HDF5 SHA256 is
`683a5edd6d096448c1b770a88dcbf5dffe57818fa45f4ba7263f5883e9909b0c`.
The model is under `experiments/results/main/product/`; the fitted state is under
`experiments/results/main/fit/`.

`experiments/inputs/problem/` contains the processed inverse
problem: grid geometry, material reference branches, prescribed boundaries,
retained observations, sparse operators, uncertainty weights, magnetic precision,
selected coefficients and a reference state. Its manifest verifies input hashes.
Loading this bundle does not require executing historical fitting scripts.
The original data and their acquisition/processing code remain under `Data/`.
The processed bundle complements those provenance records; it is not a raw-data source.

Historical manifests and `reproducibility/ncc/` preserve original run identities.
Maintained code uses explicit current paths without runtime path conversion.
The retired mapping and migration evidence are archived under
`experiments/legacy/path_routing_20260923/`. Source snapshots are retained under
the NCC exploratory archive. Recorded source hashes describe the original run,
not the refactored code.

## Experiments

The [experiment index](../experiments/README.md) and
`experiments/configs/index.json` identify the main fit, parameter
comparisons, spatial folds, basal-boundary tests, factorial controls and synthetic
recovery. Initial coefficients, tolerances and iteration budgets are explicit.

```powershell
uv run python -B -m experiments.run --config experiments/configs/main/AK170_budget_refined.json --check
uv run python -B -m experiments.run --config experiments/configs/main/AK170_budget_refined.json --output experiments/results/replays/main
uv run python -B -m experiments.analyze experiments/results/main/fit
```

`--check` loads inputs without solving. Fitting requires a fresh output directory
and is expensive. Existing evidence is never overwritten by this entry point.
The selected fit starts from its recorded preceding budget-constrained solution;
its temperature is frozen under `experiments/inputs/initial_states/` with provenance,
so no preceding experiment must be run first. The earlier initialization stages
have separate configurations. Spatial-validation
configs start from zero corrections. Validation remains conditional on common
preprocessing, geometry, calibration and boundaries. Training-fold sensitivity
is not a posterior covariance or total uncertainty.

The matched-physics synthetic experiment uses the same 170-mode mapping and
material laws on a small idealized domain. It does not establish regional model
error or independent geological validation.

## Manuscript

English and Chinese sources are under
`docs/ncc_thermal_paper/refinement_20260917/` and its `_zh` counterpart.
Publication scripts are under `docs/ncc_thermal_paper/scripts/`.

```powershell
uv run python -B -m docs.ncc_thermal_paper.scripts.ak170_refinement_assemble
./docs/ncc_thermal_paper/build.ps1 -Document manuscript
./docs/ncc_thermal_paper/build.ps1 -Document supporting
```

Figure sources, prepared cartographic layers and GMT/TeX installations are
required to rebuild publication artifacts. Moving or refactoring code does not
constitute a PDF rebuild. Author, acknowledgment and public data-release fields
still require author-supplied information before submission.

The current `.gitignore` excludes `Data/*`, `docs/*` and `experiments/*`.
Select maintained scripts, configurations and manuscript sources explicitly for
publication; do not infer Git tracking from their presence in the working tree.
