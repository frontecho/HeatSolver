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

## Direct inversion

The default inversion uses a `3 x 3` DCT bottom-temperature field with fixed
reference conductivity and heat production.

```powershell
uv run python inversion.py --maxiter 40 --output-dir outputs/direct
```

Conductivity and heat production can also be inverted with layered DCT or full
2-D log-multiplier fields.

```powershell
uv run python inversion.py --bottom-mode dct --k-mode dct --a-mode dct --maxiter 40 --output-dir outputs/direct_k_a
```

Each run writes `checkpoint.npz`, `summary.json`, `train_log.json`, surface
fields, and diagnostic plots.

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
