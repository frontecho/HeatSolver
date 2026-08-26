# HeatSolver

HeatSolver is a PDE-constrained framework for three-dimensional steady-state
geothermal inversion. It combines surface heat-flow observations, layered
thermal properties, and optional multi-source geophysical features to constrain
bottom temperature, thermal conductivity, and radiogenic heat production. The
temperature field itself is always computed by the 3-D steady-state heat
conduction solver.

The Ordos Basin in the western North China Craton is included as [a case study](docs/final%20report.md).
The example dataset and commands below prepare its regional grid and
observations, run a parameterized inversion, and export the resulting
temperature, property, and heat-flow fields for evaluation.

## Reproducing commands

[Download Dataset](https://drive.hfecho.com/d/CloudflareR2/Guest/Data.zip?sign=CEstFDPhk1TQQgGKtjLZhuPBTimHoyqsZV1wU5sHMmo=:0)

Run a prior-network inversion with DCT parameterization for bottom temperature, conductivity, and heat production.

```bash
uv run python inversion.py --mode prior_net --optimizer adamw --lr 0.005 --steps 100 --k-mode dct --a-mode dct --lam-q-std 0.4 --alpha-scale 1200 --k-prop-scale 0.3 --a-prop-scale 0.5 --t-ref 1250
```

Evaluate a saved inversion checkpoint and write diagnostic fields and plots.

```bash
uv run python evaluation.py outputs/20260522_112830
```

Run the uncertainty-quantification workflow using the prior-network inversion mode.

```bash
uv run python inversion_uq.py --mode prior_net
```
