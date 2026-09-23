# Current paper experiments

The current experiment's code, configurations, inputs and results are directly
under `experiments/`. Earlier NCC studies are under `exploratory/`, while earlier
non-NCC studies and migration records are under `legacy/`.

`configs/index.json` indexes 31 explicit experiment configurations. Fit options,
initial coefficients, observation folds and regularization are taken from the
recorded runs. Configurations point to retained results and their original
manifests; they do not claim that those runs were repeated during code reorganization.

| Location | Contents |
|---|---|
| `configs/main/` | The selected 170-parameter fit |
| `configs/comparisons/` | A-only, joint and budget-constrained alternatives, including initialization stages |
| `configs/spatial_cv/` | Three recorded spatial folds for the compared parameterizations |
| `configs/boundary/` | Original, smoothed and uniform basal fields, including continuation controls |
| `configs/synthetic.json` | Matched-physics recovery, noise and data-removal experiment |
| `configs/attribution.json` | Reference/fitted A and K factorial forward controls |
| `configs/baselines.json` | Training-only mean and IDW32 on the recorded spatial folds |
| `inputs/problem/` | Checksummed grid, material references, observations, operators, weights and selected state |
| `inputs/spatial_split.npz` | Recorded 150-km blocks with 40-km exclusion |
| `inputs/initial_states/` | Four fixed starting temperature fields, with source hashes and provenance |
| `results/main/` | Selected model and fitted state |
| `results/comparisons/` | Parameter comparisons and bounded reference |
| `results/spatial_cv/` | Held-out predictions, training-fold models and regional summaries |
| `results/boundary/` | Basal-temperature refits |
| `results/attribution/` | Factorial property controls |
| `results/synthetic/` | Matched-physics truth, fitted fields and data-removal comparisons |
| `results/property_checks/`, `results/regional/` | Conductivity consistency and regional analyses |
| `results/reporting/` | Numerical tables and analyses retained from the manuscript revisions |

`model.py` defines this paper's material units, K/A bounds, spatial mode layouts,
source-budget constraints, observation objective and basal scenarios. Its
`InverseProblem` loads the prepared inputs and constructs the generic solver.
These assumptions are not defaults of `heat_solver.py`.

`run.py` is the unified entry point for fitting, reference evaluation, boundary
tests, factorial controls and synthetic recovery. `analyze.py` exports physical
checks and run summaries. `synthetic.py` defines the synthetic domain and its
matched-physics protocol. Both regional and synthetic experiments use `model.py`
and the root solver, flux, sampling and basis modules; none imports the
historical experimental implementations.

The root `fit_metrics.py` supplies generic prediction statistics for `analyze.py`.
NCC material definitions stay in `model.py`; run orchestration, synthetic scenarios,
regional budgets and output formats stay in this experiment group.

All 31 action configurations load their bundles, splits and initial temperatures
from this group's `inputs/`. Initialization fields retain the preceding runs'
numerical values, but execution does not read those runs' result directories.
Paths are explicit; there is no runtime location mapping.

From the repository root:

```powershell
uv run python -B -m experiments.run --config experiments/configs/main/AK170_budget_refined.json --check
uv run python -B -m experiments.run --config experiments/configs/main/AK170_budget_refined.json --output experiments/results/replays/main
uv run python -B -m experiments.analyze experiments/results/main/fit
```

The first command only loads inputs. The second starts an expensive regional fit
and requires a new output directory. It uses the recorded initial coefficients;
the initialization-stage configurations are indexed separately. Continuation
configs likewise preserve the recorded preceding solution, not an arbitrary new
run's endpoint. Basal tests preserve the recorded starting temperature field and
its boundary adjustment as well as the initial coefficients. Full fits and sweeps
are not part of the test suite.

Training-only mean and IDW32 predictions, physical-model comparisons and their
paired spatial folds are retained under `results/comparisons/physical_bounds/`.
Manuscript plots and publication code are under `docs/ncc_thermal_paper/`.
