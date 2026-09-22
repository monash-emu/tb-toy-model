"""
Tools to parameterise, calibrate and run the model.

The approach mirrors the full model:
  * fixed parameter values and prior distributions are read from a single
    parameter table;
  * calibration targets are read from a target table and turned into normal
    likelihood terms;
  * calibration uses `estival` / `pymc` (DEMetropolisZ by default);
  * scenarios are run by sampling from the calibration posterior, and summarised
    as quantiles.
"""

from pathlib import Path
from time import time

import arviz as az
import pandas as pd
import pymc as pm

from estival import priors as esp
from estival import targets as est
from estival.model import BayesianCompartmentalModel
from estival.sampling import tools as esamp
from estival.wrappers import pymc as epm

from tbtoy.config import DEFAULT_ANALYSIS_CONFIG, DEFAULT_MODEL_CONFIG
from tbtoy.model import get_tb_model
from tbtoy.paths import OUTPUT_PARENT_FOLDER, PARAMS_PATH, TARGETS_PATH, TV_PARAMS_PATH
from tbtoy.scenarios import BASELINE

# Outputs summarised when comparing scenarios against a reference scenario
DIFF_OUTPUTS = {
    "TB_averted": "cum_tb_incidence",
    "deaths_averted": "cum_tb_mortality",
}

# Cumulative outputs reported for each scenario (burden, care cascade and resource use)
CUMULATIVE_OUTPUTS = [
    "cum_tb_incidence",
    "cum_tb_mortality",
    "cum_tb_treatment_starts",
    "cum_tpt_completions",
    "cum_n_screening_encounters",
    "cum_n_tests",
    "cum_n_tests_symptom_screen",
    "cum_n_tests_cxr",
    "cum_n_tests_xpert",
    "cum_n_tests_tst",
]


"""
    Inputs
"""


def get_parameters_and_priors(params_path: Path = PARAMS_PATH, tv_params_path: Path = TV_PARAMS_PATH):
    """
    Read parameter values and prior distributions.

    Rows of the parameter table that specify a `distribution` are turned into
    calibration priors; the `value` column provides the default (fixed) value of
    every parameter.

    Args:
        params_path: path to the constant parameter table.
        tv_params_path: path to the time-variant parameter table.

    Returns:
        params: dictionary of parameter values.
        priors: list of estival prior objects.
        tv_params: dictionary of pandas Series (indexed by year).
    """
    df = pd.read_csv(params_path)
    df = df.where(pd.notna(df), None)

    params = dict(zip(df["parameter"], df["value"]))

    priors_df = df[df["distribution"].notnull()]
    priors = [
        get_prior(row["parameter"], row["distribution"], row["distri_param1"], row["distri_param2"])
        for _, row in priors_df.iterrows()
    ]

    tv_df = pd.read_csv(tv_params_path, index_col=0)
    tv_params = {col: tv_df[col].dropna() for col in tv_df.columns}

    return params, priors, tv_params


def get_prior(param_name: str, distribution: str, distri_param1, distri_param2=None):
    if distribution == "uniform":
        return esp.UniformPrior(param_name, [distri_param1, distri_param2])
    if distribution == "truncnormal":
        # distri_param1/2 are the mean and standard deviation; truncated at zero
        return esp.TruncNormalPrior(param_name, distri_param1, distri_param2, (0.0, float("inf")))
    raise ValueError(f"'{distribution}' is not currently a supported prior distribution")


def get_targets(targets_path: Path = TARGETS_PATH) -> list:
    """
    Read calibration targets and convert them to normal likelihood terms.

    The `tol_perc` column controls the standard deviation, such that 95% of the
    likelihood density lies within +/- `tol_perc`% of the mean of the target series.

    Args:
        targets_path: path to the target table.

    Returns:
        List of estival target objects.
    """
    df = pd.read_csv(targets_path)

    targets = []
    for output, group in df.groupby("output", sort=False):
        data = pd.Series(data=group["value"].values, index=group["year"].values)
        targets.append(get_normal_target(output, data, tol_perc=float(group["tol_perc"].iloc[0])))

    return targets


def get_normal_target(name: str, data: pd.Series, tol_perc: float = 20.0):
    return est.NormalTarget(
        name=name,
        data=data,
        stdev=(tol_perc / 100.0) / 1.96 * data.mean(),
    )


"""
    Model / BCM building
"""


def build_bcm(
    scenario=BASELINE,
    model_config: dict = DEFAULT_MODEL_CONFIG,
    params: dict = None,
    priors: list = None,
    targets: list = None,
    tv_params: dict = None,
) -> BayesianCompartmentalModel:
    """
    Build a `BayesianCompartmentalModel` for a given scenario.

    Args:
        scenario: the scenario to model (defaults to the status quo baseline).
        model_config: model configuration dictionary.
        params, priors, targets, tv_params: analysis inputs; read from the data
            folder when not provided.

    Returns:
        The Bayesian compartmental model.
    """
    if params is None or priors is None or tv_params is None:
        default_params, default_priors, default_tv_params = get_parameters_and_priors()
        params = default_params if params is None else params
        priors = default_priors if priors is None else priors
        tv_params = default_tv_params if tv_params is None else tv_params
    targets = get_targets() if targets is None else targets

    model = get_tb_model(model_config, tv_params, scenario.screening_programs)
    return BayesianCompartmentalModel(model, params | scenario.params_ow, priors, targets)


def build_bcm_dict(scenarios: list, **kwargs) -> dict:
    """Build one BCM per scenario, keyed by scenario id."""
    sc_ids = [scenario.sc_id for scenario in scenarios]
    assert len(sc_ids) == len(set(sc_ids)), "Please use unique scenario ids."
    return {scenario.sc_id: build_bcm(scenario, **kwargs) for scenario in scenarios}


def run_scenarios_single_params(bcm_dict: dict, params: dict) -> dict:
    """
    Run every scenario once, using a single parameter set (e.g. the MLE).

    Args:
        bcm_dict: scenario id to BCM mapping.
        params: values for the calibrated parameters.

    Returns:
        Scenario id to derived outputs DataFrame mapping.
    """
    return {sc_id: bcm.run(params).derived_outputs for sc_id, bcm in bcm_dict.items()}


"""
    Calibration
"""


def find_mle(bcm: BayesianCompartmentalModel, budget: int = 2000, opt_class=None) -> dict:
    """
    Find a maximum likelihood estimate of the calibrated parameters, using nevergrad.

    Args:
        bcm: the Bayesian compartmental model.
        budget: number of objective function evaluations.
        opt_class: nevergrad optimiser class (defaults to `ng.optimizers.NGOpt`).

    Returns:
        Dictionary of best-fitting parameter values.
    """
    import nevergrad as ng
    from estival.wrappers.nevergrad import optimize_model

    opt_class = ng.optimizers.NGOpt if opt_class is None else opt_class
    orunner = optimize_model(bcm, opt_class=opt_class)
    rec = orunner.minimize(budget)
    return rec.value[1]


def run_metropolis_calibration(
    bcm: BayesianCompartmentalModel,
    draws: int = 10000,
    tune: int = 2000,
    cores: int = 4,
    chains: int = 4,
    method: str = "DEMetropolisZ",
    initvals=None,
) -> az.InferenceData:
    """
    Run Bayesian sampling using pymc.

    Args:
        bcm: estival calibration object containing the model, priors and targets.
        draws: number of iterations per chain.
        tune: number of tuning iterations (in addition to `draws`).
        cores: number of cores.
        chains: number of chains.
        method: pymc sampling algorithm ("DEMetropolis" or "DEMetropolisZ").
        initvals: optional initial values for the chains.

    Returns:
        The calibration inference data.
    """
    samplers = {"DEMetropolis": pm.DEMetropolis, "DEMetropolisZ": pm.DEMetropolisZ}
    if method not in samplers:
        raise ValueError(f"Requested sampling method '{method}' not currently supported.")

    with pm.Model():
        variables = epm.use_model(bcm)
        idata = pm.sample(
            step=[samplers[method](variables)],
            draws=draws,
            tune=tune,
            cores=cores,
            chains=chains,
            initvals=initvals,
            progressbar=False,
        )

    return idata


def run_full_runs(bcm_dict: dict, idata: az.InferenceData, burn_in: int, full_runs_samples: int):
    """
    Run every scenario over a sample of the calibration posterior.

    Args:
        bcm_dict: scenario id to BCM mapping.
        idata: calibration inference data.
        burn_in: number of draws discarded from the start of each chain.
        full_runs_samples: number of posterior samples used.

    Returns:
        full_runs: scenario id to raw sampled results.
        unc_dfs: scenario id to output quantiles DataFrame.
    """
    chain_length = idata.sample_stats.sizes["draw"]
    assert full_runs_samples <= chain_length - burn_in, "Too many full-run samples requested."

    burnt_idata = idata.sel(draw=range(burn_in, chain_length))
    full_run_params = az.extract(burnt_idata, num_samples=full_runs_samples)

    full_runs, unc_dfs = {}, {}
    for sc_id, bcm in bcm_dict.items():
        full_run = esamp.model_results_for_samples(full_run_params, bcm, include_extras=False)
        unc_df = esamp.quantiles_for_results(full_run.results, [0.025, 0.25, 0.5, 0.75, 0.975])
        # avoid float column names, which are not parquet-compatible
        unc_df.columns = unc_df.columns.set_levels(
            [str(q) for q in unc_df.columns.levels[1]], level=1
        )
        full_runs[sc_id] = full_run
        unc_dfs[sc_id] = unc_df

    return full_runs, unc_dfs


"""
    Scenario summaries
"""


def calculate_diff_output_quantiles(
    full_runs: dict, quantiles=(0.025, 0.25, 0.5, 0.75, 0.975), ref_sc: str = "baseline", end_year: int = None
) -> dict:
    """
    Quantiles of the health gains of each scenario relative to a reference scenario.

    Args:
        full_runs: scenario id to sampled results (from `run_full_runs`).
        quantiles: quantiles to report.
        ref_sc: reference scenario id.
        end_year: year at which cumulative outputs are read (defaults to the last modelled year).

    Returns:
        Scenario id to DataFrame of absolute and relative differences.
    """
    quantiles = list(quantiles)
    ref_results = full_runs[ref_sc].results
    end_year = ref_results.index.max() if end_year is None else end_year
    ref_latest = ref_results.loc[end_year]

    diff_output_quantiles = {}
    for sc_id, full_run in full_runs.items():
        if sc_id == ref_sc:
            continue
        sc_latest = full_run.results.loc[end_year]
        abs_diff = ref_latest - sc_latest
        rel_diff = abs_diff / ref_latest

        abs_df = pd.DataFrame(
            index=quantiles,
            data={name: abs_diff[output].quantile(quantiles) for name, output in DIFF_OUTPUTS.items()},
        )
        rel_df = pd.DataFrame(
            index=quantiles,
            data={
                f"{name}_relative": rel_diff[output].quantile(quantiles)
                for name, output in DIFF_OUTPUTS.items()
            },
        )
        diff_output_quantiles[sc_id] = pd.concat([abs_df, rel_df], axis=1)

    return diff_output_quantiles


def calculate_cumulative_output_quantiles(
    full_runs: dict, outputs: list = None, quantiles=(0.025, 0.5, 0.975), end_year: int = None
) -> pd.DataFrame:
    """
    Quantiles of cumulative outputs (burden, treatment starts, tests) for each scenario.

    Args:
        full_runs: scenario id to sampled results (from `run_full_runs`).
        outputs: cumulative outputs to summarise.
        quantiles: quantiles to report.
        end_year: year at which cumulative outputs are read (defaults to the last modelled year).

    Returns:
        DataFrame indexed by (scenario, quantile), with one column per output.
    """
    outputs = CUMULATIVE_OUTPUTS if outputs is None else outputs
    quantiles = list(quantiles)

    frames = {}
    for sc_id, full_run in full_runs.items():
        end = full_run.results.index.max() if end_year is None else end_year
        latest = full_run.results.loc[end]
        frames[sc_id] = pd.DataFrame(
            index=quantiles, data={output: latest[output].quantile(quantiles) for output in outputs}
        )

    return pd.concat(frames, names=["scenario", "quantile"])


"""
    End-to-end analysis
"""


def run_full_analysis(
    scenarios: list,
    model_config: dict = DEFAULT_MODEL_CONFIG,
    analysis_config: dict = DEFAULT_ANALYSIS_CONFIG,
    output_folder: Path = None,
    idata_path: Path = None,
):
    """
    Run a complete analysis: calibration, scenario runs and output summaries.

    Args:
        scenarios: scenarios to run (the first one is used as the reference).
        model_config: model configuration dictionary.
        analysis_config: sampling and full-run configuration.
        output_folder: where results are written (nothing is written if None).
        idata_path: folder containing a previously saved `idata.nc`, to skip calibration.

    Returns:
        idata, full_runs, unc_dfs
    """
    a_c = analysis_config
    params, priors, tv_params = get_parameters_and_priors()
    targets = get_targets()

    inputs = dict(model_config=model_config, params=params, priors=priors, targets=targets, tv_params=tv_params)
    bcm_dict = build_bcm_dict(scenarios, **inputs)
    ref_sc = scenarios[0].sc_id

    if output_folder:
        output_folder.mkdir(parents=True, exist_ok=True)

    print(">>> Running Metropolis sampling")
    t0 = time()
    if idata_path:
        idata = az.from_netcdf(Path(idata_path) / "idata.nc")
    else:
        idata = run_metropolis_calibration(
            bcm_dict[ref_sc], draws=a_c["draws"], tune=a_c["tune"], cores=a_c["cores"], chains=a_c["chains"]
        )
        if output_folder:
            az.to_netcdf(idata, output_folder / "idata.nc")
    print(f"    ... completed in {round(time() - t0)} sec")

    print(">>> Running scenarios over posterior samples")
    t0 = time()
    full_runs, unc_dfs = run_full_runs(bcm_dict, idata, a_c["burn_in"], a_c["full_runs_samples"])
    print(f"    ... completed in {round(time() - t0)} sec")

    if output_folder:
        for sc_id, unc_df in unc_dfs.items():
            unc_df.to_parquet(output_folder / f"uncertainty_df_{sc_id}.parquet")
        for sc_id, diff_df in calculate_diff_output_quantiles(full_runs, ref_sc=ref_sc).items():
            diff_df.to_parquet(output_folder / f"diff_quantiles_df_{sc_id}.parquet")
        calculate_cumulative_output_quantiles(full_runs).to_parquet(
            output_folder / "cumulative_quantiles_df.parquet"
        )

    return idata, full_runs, unc_dfs


def create_output_dir(analysis_name: str) -> Path:
    output_dir = OUTPUT_PARENT_FOLDER / analysis_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
