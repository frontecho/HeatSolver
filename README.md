# HeatSolver

HeatSolver is a NumPy/SciPy implementation of direct, PDE-constrained 3-D
steady-state geothermal inversion for the Ordos Basin.

The inversion optimizes a bounded low-dimensional representation of bottom
temperature, conductivity, and heat production. The finite-volume heat solver
always computes the temperature field. Gradients use one exact nonlinear
adjoint solve per objective evaluation.

## Setup and tests

```powershell
uv sync
uv run python -B -m pytest -q
```

The input datasets are local and are not distributed by Git. Before running an
inversion, populate `Data/` with `grid_1km.csv`, `grid_1km_layer.csv`,
`grid_1km_stratum_filled.csv`, `grid_1km_thermal.csv`,
`thermal_properties.json`, and `sample_heatflow_processed.csv`.

## Direct inversion

The default inversion uses a `3 x 3` DCT bottom-temperature field with fixed
reference conductivity and heat production.

Coarsening keeps complete `stride x stride` blocks and discards incomplete
blocks at the high-X/high-Y edges, so the finite-volume grid remains uniform.
DCT fields use RMS-normalized modes: the first coefficient is the field mean,
and the remaining coefficient bounds share one aggregate spatial budget equal
to the configured bound. In `full` mode, the bound applies to each grid cell.

```powershell
uv run python inversion.py --maxiter 40 --output-dir outputs/direct
```

Conductivity and heat production can also be inverted with layered DCT or full
2-D log-multiplier fields.

```powershell
uv run python inversion.py --bottom-mode dct --k-mode dct --a-mode dct --maxiter 40 --output-dir outputs/direct_k_a
```

Each run writes `checkpoint.npz`, `summary.json`, `train_log.json`, surface
fields, and diagnostic plots. Checkpoints record SHA-256 hashes for all six
inputs; evaluation and UQ reject changed input files.

## Evaluation

```powershell
uv run python evaluation.py outputs/direct
```

Evaluation reconstructs the model from checkpoint metadata and writes full
temperature/property fields under `outputs/direct/evaluation/`.

## Empirical uncertainty ensemble

Use a completed direct checkpoint as the ensemble template.

```powershell
uv run python inversion_uq.py outputs/direct --members 8 --holdout-fraction 0.2 --maxiter 40
```

The ensemble varies observation holdouts and initial coefficients, saves each
member checkpoint, and aggregates field means and sample standard deviations.
The template supplies configuration only; members start independently from zero
coefficients plus seeded noise, so held-out observations do not leak through a
previously fitted parameter vector.
