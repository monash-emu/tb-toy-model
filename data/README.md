# Dummy data

All files in this folder are **placeholder / dummy** inputs, provided so that the
model and the notebooks run out of the box. They are **not** real epidemiological
data and should be replaced before drawing any conclusion.

| File | Content |
| --- | --- |
| `parameters.csv` | Constant parameters. Rows with a `distribution` entry are treated as calibrated parameters (priors); the `value` column is used as a default/fixed value otherwise. |
| `time_variant_parameters.csv` | Time-variant parameters (one column per parameter, `year` as index). Currently only the historical TB treatment success percentage. |
| `calibration_targets.csv` | Dummy calibration targets. One row per (output, year) pair. `tol_perc` sets the width of the normal likelihood (95% of the density within +/- `tol_perc`% of the mean of the target series). |

Demography is **not** read from a file: the population trajectory is generated
analytically from the model configuration (reference population size, reference
year and growth rate) — see `tbtoy/demography.py`.
