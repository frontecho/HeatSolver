# HeatSolver

Three-dimensional thermal inversion for the North China Craton. A finite-volume
heat-conduction solver computes temperature; bounded conductivity and radiogenic
heat-production parameters are constrained by surface heat flow, interval
gradients and magnetic depths. Derivatives use the discrete implicit adjoint.

## Layout

| Location | Purpose |
|---|---|
| `heat_solver.py` | Finite-volume matrix assembly, conduction solves and built-in-law adjoints |
| `nonlinear_heat_solver.py` | Nonlinear solve and implicit derivatives for a supplied conductivity law |
| `heat_flux.py` | Heat flux and conductivity/temperature derivatives |
| `observations.py` | Grid sampling, interval gradients and weighted observation losses |
| `spatial_basis.py` | DCT spatial basis and transpose operators |
| `fit_metrics.py` | Weighted error and correlation statistics extracted from paper analysis |
| `tests/` | Core numerical tests |
| `project.py` | Repository root, SHA256 and JSON helpers; no path translation |
| `experiments/` | NCC model definitions in `model.py`, configurations, inputs, runners, tests and results |
| `experiments/exploratory/` | Earlier NCC alternatives, their source code, tests and results, classified by scientific purpose |
| `experiments/legacy/` | Earlier non-NCC studies |
| `Data/processed/thermal_properties/` | k/A data, calibration and property-model studies |
| `Data/processed/ncc_preparation/` | Regional source preprocessing |
| `docs/ncc_thermal_paper/` | Manuscript sources, figures and publication scripts |

## Environment and checks

```powershell
uv sync --locked
uv run python -B -m pytest -q -p no:cacheprovider
uv run python -B -m experiments.run --config experiments/configs/main/AK170_budget_refined.json --check
```

The tests use synthetic fixtures. The last command requires the local processed
input bundle and validates loading without solving or fitting. A regional fit
requires an explicit fresh output directory; see
[reproduction instructions](docs/reproducibility.md).

The selected model has 99 heat-production and 71 conductivity coefficients,
pointwise material bounds and fixed reference layer-production budgets. Surface
and 100-km temperatures are prescribed. Conditional validation does not measure
total uncertainty or resolve the uniqueness of deep thermal structure.

The authoritative artifact pointer is
`experiments/configs/selected_model.json`.
See the [paper experiment index](experiments/README.md).

Maintained readers use current paths directly. The paper experiment keeps its
fixed inputs and initialization fields in its own `inputs/` directory and does
not import earlier experiments. Historical exploratory groups still contain
cross-group code reuse; their complete isolation has not been validated.

Import numerical components directly from their owning modules. The root modules
contain no NCC bounds, layer definitions, data loading or experiment configuration;
those remain in the experiment group. There is no old-import forwarding layer.

The current `.gitignore` excludes `Data/*`, `docs/*` and `experiments/*`.
Maintained source files and configurations in those directories must be selected
explicitly before publication. Numerical payloads and historical archives remain
local; cloning the repository alone does not provide the regional data.
