# HeatSolver

HeatSolver is a 3D steady-state geothermal inversion framework.

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
