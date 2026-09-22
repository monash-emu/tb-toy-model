# Tuberculosis Toy Model

A deliberately simplified ("toy") dynamic transmission model of tuberculosis (TB), built with
[summer2](https://github.com/monash-emu/summer2) and calibrated with
[estival](https://github.com/monash-emu/estival) / [pymc](https://www.pymc.io/).

The model is a stripped-down, country-agnostic version of a full TB screening model. It keeps the
natural history structure, the parameterisation approach and the screening/treatment structure of
the full model, but removes everything that is not needed to compare programmatic and screening
strategies at an aggregate level:

| Feature | Full model | This toy model |
| --- | --- | --- |
| Age stratification and empirical contact matrices | Yes | **No** (homogeneous mixing) |
| Stratification by screening reachability | Yes | **No** |
| Age-specific outputs | Yes | **No** (aggregated outputs only) |
| Country-specific UN demography | Yes | **No** (analytic population trajectory) |
| Screening algorithms | PEARL-specific | Generic, configurable algorithms |
| Screening volume / test count outputs | Partial | **Yes**, by test type |

> **All data in this repository are dummy placeholders**, including the calibration targets and the
> historical treatment success series. They exist so the code runs end to end; replace them with real
> data before interpreting any result.

## Model structure

Ten compartments describe the TB infection and disease spectrum:

```
                     ┌──────────► cleared ◄──── (preventive therapy)
                     │                │
mtb_naive ──► incipient ◄──► contained│
                 │   ▲                │
                 │   └────────────────┘
                 ▼
         subclin_lowinf ◄──► subclin_inf          (asymptomatic disease)
               ▲ │                ▲ │
               │ ▼                │ ▼
           clin_lowinf ◄────► clin_inf            (symptomatic disease)
                 │                │
                 └──► treatment ──┴──► recovered
```

* **Disease states** are cross-classified by symptom status (`subclin` / `clin`) and infectiousness
  (`lowinf` / `inf`), with transitions in both directions.
* **Detection** happens either passively (routine health services) or actively (screening campaigns).
* **Deaths** (background and TB-related) return individuals to `mtb_naive`; net population growth is
  then added as an importation flow, so the modelled population follows the configured trajectory.

### Scenario levers

Two parameters drive the *programmatic* scenarios and ramp in from `scenario_start_time`:

* `future_detection_rate_rel` — future passive detection rate, relative to the current one;
* `future_tx_success_prop` — future TB treatment success proportion.

*Screening* scenarios are built from `ScreeningProgram` objects, each combining a screening tool
(algorithm), a time window and a total coverage. The tools provided are:

| Tool | Pathway | Tests counted per person screened |
| --- | --- | --- |
| `SSX` | Symptom screening → treatment | 1 symptom screen |
| `CXR` | Universal chest X-ray → treatment | 1 symptom screen, 1 CXR |
| `CXR_XPERT` | CXR triage → Xpert confirmation → treatment | 1 symptom screen, 1 CXR, `cxr_positivity_prop` Xpert |
| `TST_TPT` | TST → preventive therapy | 1 TST |

"No screening" is simply a scenario with no screening program, and acts as the comparator.

## Outputs

All outputs are aggregated (no age breakdown) and available from
`model.get_derived_outputs_df()`. They include:

* **Burden** — `tb_incidence(_per100k)`, `tb_prevalence(_per100k)`, `tbi_prevalence_perc`,
  `viable_tbi_prevalence_perc`, `tst_positivity_perc`, `tb_mortality(_per100k)`,
  `perc_prev_subclinical`, `perc_prev_infectious`.
* **Care cascade** — `tb_notifications`, `screening_tb_detections`, **`tb_treatment_starts`**,
  `n_on_treatment`, `tpt_completions`, `case_detection_prop`.
* **Screening volume** — `n_screening_encounters`, **`n_tests`** and `n_tests_{symptom_screen, cxr,
  xpert, tst}`.
* **Cumulative versions** of the above (`cum_*`), accumulated from `scenario_start_time`, for
  scenario comparison and resource-use summaries.

## Repository structure

```
├── code/
│   ├── pyproject.toml
│   └── tbtoy/
│       ├── compartments.py    # compartment definitions
│       ├── config.py          # default model and analysis configurations
│       ├── demography.py      # generic population trajectory and background mortality
│       ├── interventions.py   # screening tools, screening programs, scenario container
│       ├── model.py           # model construction
│       ├── outputs.py         # derived output requests
│       ├── paths.py           # repository paths
│       ├── plotting.py        # figures for single runs, calibration and scenarios
│       ├── runner_tools.py    # parameters, priors, targets, calibration, scenario runs
│       └── scenarios.py       # the two sets of scenarios
├── data/                      # dummy parameters, time-variant parameters, calibration targets
└── notebooks/
    ├── 01_single_run.ipynb            # build and run the model once
    ├── 02_calibration.ipynb           # MLE search and Metropolis calibration
    ├── 03_programmatic_scenarios.ipynb# future passive detection and treatment success
    └── 04_screening_scenarios.ipynb   # alternative screening algorithms and coverage
```

## Installation

The environment is managed with [pixi](https://pixi.sh/):

```bash
pixi install
pixi run jupyter lab
```

Alternatively, install the package into an existing environment:

```bash
pip install -e ./code
```

## Quick start

```python
import tbtoy.runner_tools as rt
from tbtoy.config import DEFAULT_MODEL_CONFIG
from tbtoy.model import get_tb_model

params, priors, tv_params = rt.get_parameters_and_priors()

model = get_tb_model(DEFAULT_MODEL_CONFIG, tv_params)
model.run(parameters=params)
outputs = model.get_derived_outputs_df()
```

Running a scenario comparison:

```python
from tbtoy.scenarios import SCREENING_SCENARIOS

bcm_dict = rt.build_bcm_dict(SCREENING_SCENARIOS)
results = rt.run_scenarios_single_params(bcm_dict, params)
```

## Workflow

1. **Parameterise** — edit `data/parameters.csv`. A row with a `distribution` becomes a calibration
   prior; the `value` column is always used as the default/fixed value.
2. **Set targets** — edit `data/calibration_targets.csv`. Each `(output, year)` row becomes part of a
   normal likelihood term, with `tol_perc` controlling its standard deviation.
3. **Calibrate** — `notebooks/02_calibration.ipynb` finds a maximum likelihood estimate with
   nevergrad, then samples the posterior with `DEMetropolisZ`.
4. **Compare scenarios** — notebooks 03 and 04 run every scenario over posterior samples and
   summarise burden averted and resource use.

## Notes and caveats

* This is a **toy** model: homogeneous mixing and the absence of age structure mean that screening
  impact, preventive therapy benefits and the natural history of infection are all approximated.
* Because there is no age structure, background mortality is a single constant rate derived from the
  configured life expectancy.
* The `n_screening_encounters` output sums across screening programs, so concurrent programs
  covering the same people (e.g. disease screening plus infection screening) are counted once each.
  The `n_tests_*` outputs are the meaningful measures of resource use.
* On macOS, multi-core pymc sampling can deadlock because `os.fork()` is incompatible with JAX's
  threads. Use `cores=1` there (as `TEST_ANALYSIS_CONFIG` does), or run the calibration on Linux.

## License

BSD 2-Clause — see [LICENSE](LICENSE).
