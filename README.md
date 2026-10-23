# HeatSolver

Code for three-dimensional thermal structure inversion of the North China Craton.
Surface heat flow, borehole temperature gradients, and Curie depths jointly constrain heat production and thermal conductivity in each layer and the resulting temperature field.

## Environment and data preparation

Use Python 3.12 and uv. Run all commands below from the project root.

```bash
uv sync --locked
```

Place the raw data in `inputs/`. These data must be prepared separately.

## Running the main experiment

Each command accepts an experiment root directory and locates its `problem/`, `references/`, `fit/`, `uncertainty/`, and other subdirectories automatically. Run the following commands in order, using `results/run01/` as an example. Use a different directory for a new run: input preparation, fitting, and uncertainty calculations do not overwrite existing output directories.

```bash
# Generate regional boundaries
uv run python -B -m inputs.regions.extract_boundaries

# Prepare the grid, boundary temperatures, and observations
uv run python -B prepare.py results/run01

# Build reference heat production and thermal conductivity fields
uv run python -B references.py results/run01

# Run the inversion starting from zero coefficients
uv run python -B fit.py results/run01

# Propagate errors in observations, boundary temperatures, and reference properties, and check nonlinear perturbations
uv run python -B uncertainty.py results/run01
```

Edit `config.py` to set the grid, input selection, property bounds, parameterization, regularization, and optimization settings.
The configuration used for fitting is saved to `fit/training_config.json`; the stopping reason and fit metrics are saved to `fit/summary.json`.

Uncertainty sampling and error magnitudes are configured through `CALIBRATION_SETTINGS` and `FIELD_SETTINGS` in `config.py`. The uncertainty script runs observation error propagation, reference field calibration sampling, external input propagation, combination of three-dimensional T/K/A variances, and positive and negative perturbation checks in that order. Final fields and depth statistics are written directly to `uncertainty/`; intermediate outputs are stored in its `linear/`, `inputs/`, and `external/` subdirectories.

Each experiment uses the same output structure:

```text
results/run01/
  problem/       Grid, boundary temperatures, and observation tables
  references/    Reference property fields, calibration relationships, and sample tables
  fit/           Fitted fields, observation predictions, metrics, and training configuration
  uncertainty/   Uncertainty fields, source contributions, depth statistics, and intermediate outputs
  si/            Supplementary experiment results
```

## Supplementary experiments

Run the following modules from the project root, for example:

```bash
uv run python -B -m si.synthetic results/run01
```

| Experiment | Module |
|---|---|
| Synthetic data experiments | `si.synthetic` |
| Basal temperature comparisons: smoothed, uniform, and published models | `si.basal` |
| Spatial cross-validation | `si.cv` |
| Temperature attribution and thermal conductivity comparisons | `si.attribution`, `si.conductivity` |

All modules accept the experiment root directory. The basal temperature comparison prepares boundaries, fits each case, and computes statistics in a single run:

```bash
uv run python -B -m si.basal results/run01
```

The six basal temperature cases are `original`, `smoothed`, `uniform`, `sun2022_c`, `zhang2025`, and `yan2019`. Each case stores its fitted results directly in `si/basal/<case>/`. The cases share `boundaries.npz`, `metrics.csv`, `depth_profiles.csv`, and `layer_properties.csv`; regional boundaries used for statistics are saved in `regions.geojson`. The `original` case reuses the main fit. The remaining cases are fitted sequentially from zero coefficients using the main model's training configuration.

## Tests

```bash
uv run python -B -m pytest -q
```
