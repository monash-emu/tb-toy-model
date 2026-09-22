"""
Core transmission model.

This is a simplified version of the Kiribati TB model: no age stratification, no
stratification by screening reachability, and aggregated (non age-specific)
outputs only. The natural history structure, the parameterisation approach and
the screening/treatment structure are unchanged.
"""

from jax import numpy as jnp
import pandas as pd

from summer2 import CompartmentalModel
from summer2.functions import time as stf
from summer2.parameters import Function, Parameter, Time

from tbtoy.compartments import (
    ACTIVE_COMPS,
    COMPARTMENTS,
    INFECTION_SOURCE_COMPS,
    INFECTIOUS_COMPS,
)
from tbtoy.demography import (
    get_background_death_rate,
    get_population_entry_series,
    get_population_series,
)
from tbtoy.interventions import get_screening_flow_name, get_screening_rate_output_name
from tbtoy.outputs import request_model_outputs


def get_tb_model(model_config: dict, tv_params: dict, screening_programs=()) -> CompartmentalModel:
    """
    Build the full (toy) TB model, including outputs.

    Args:
        model_config: model configuration dictionary (see `tbtoy.config`).
        tv_params: time-variant parameters, as a dict of pandas Series indexed by year.
        screening_programs: iterable of `ScreeningProgram` objects to apply.

    Returns:
        The built summer2 model, ready to be run.
    """
    screening_programs = list(screening_programs)
    prog_names = [prog.name for prog in screening_programs]
    assert len(prog_names) == len(set(prog_names)), "Screening program names must be unique"

    pop_series = get_population_series(model_config)
    bckd_death_rate = get_background_death_rate(model_config)
    tv_tsr = get_time_variant_tsr(tv_params, model_config)

    model = build_natural_history_model(model_config, pop_series)
    add_detection_and_treatment(model, model_config, tv_tsr, bckd_death_rate, screening_programs)
    add_births_and_deaths(model, pop_series, bckd_death_rate, tv_tsr, screening_programs)

    request_model_outputs(model, model_config, screening_programs)

    return model


def build_natural_history_model(model_config: dict, pop_series: pd.Series) -> CompartmentalModel:
    """Set up the model and add all natural history (i.e. non-intervention) flows."""

    model = CompartmentalModel(
        times=(model_config["start_time"], model_config["end_time"]),
        compartments=COMPARTMENTS,
        infectious_compartments=INFECTIOUS_COMPS,
    )

    model.set_initial_population(
        distribution={
            "mtb_naive": pop_series.iloc[0] - model_config["seed"],
            "clin_inf": model_config["seed"],
        },
    )

    """
        Transmission (including reinfection)
    """
    infection_pop_scale = Parameter("infection_pop_scale")
    # Rescale the raw transmission rate using the population size at the reference year, so that
    # "raw_transmission_rate" keeps the same order of magnitude regardless of the transmission mode
    # (density- vs frequency-dependent), which is controlled by `infection_pop_scale`.
    reference_pop = float(pop_series.loc[model_config["reference_year"]])
    rescaled_transmission_rate = Parameter("raw_transmission_rate") * Function(
        _power, [reference_pop, infection_pop_scale]
    )

    for susceptible_comp in INFECTION_SOURCE_COMPS:
        if susceptible_comp == "mtb_naive":
            rel_susceptibility = 1.0
        elif susceptible_comp in ("cleared", "recovered"):
            # common parameter for relative susceptibility after clearance and after recovery
            rel_susceptibility = Parameter("rel_sus_cleared")
        else:
            rel_susceptibility = Parameter(f"rel_sus_{susceptible_comp}")

        model.add_infection_generalised_flow(
            name=f"infection_from_{susceptible_comp}",
            gen_infection_exp=infection_pop_scale,
            contact_rate=rescaled_transmission_rate * rel_susceptibility,
            source=susceptible_comp,
            dest="incipient",
        )

    """
        Early TB infection dynamics
    """
    model.add_transition_flow(
        name="containment",
        fractional_rate=Parameter("containment_rate"),
        source="incipient",
        dest="contained",
    )
    model.add_transition_flow(
        name="clearance",
        fractional_rate=Parameter("clearance_rate"),
        source="contained",
        dest="cleared",
    )
    model.add_transition_flow(
        name="breakdown",
        fractional_rate=Parameter("breakdown_rate"),
        source="contained",
        dest="incipient",
    )

    # Progression to disease, split between low- and high-infectiousness (all initially subclinical)
    model.add_transition_flow(
        name="progression_lowinf",
        fractional_rate=Parameter("progression_rate") * (1.0 - Parameter("progression_prop_infectious")),
        source="incipient",
        dest="subclin_lowinf",
    )
    model.add_transition_flow(
        name="progression_inf",
        fractional_rate=Parameter("progression_rate") * Parameter("progression_prop_infectious"),
        source="incipient",
        dest="subclin_inf",
    )

    """
        Active TB dynamics
    """
    # Symptom onset and resolution
    for infectious_status in ["inf", "lowinf"]:
        model.add_transition_flow(
            name=f"clinical_progression_{infectious_status}",
            fractional_rate=Parameter("clinical_progression_rate"),
            source=f"subclin_{infectious_status}",
            dest=f"clin_{infectious_status}",
        )
        model.add_transition_flow(
            name=f"clinical_regression_{infectious_status}",
            fractional_rate=Parameter("clinical_regression_rate"),
            source=f"clin_{infectious_status}",
            dest=f"subclin_{infectious_status}",
        )

    # Infectiousness gain and loss
    for clinical_status in ["clin", "subclin"]:
        model.add_transition_flow(
            name=f"infectiousness_gain_{clinical_status}",
            fractional_rate=Parameter("infectiousness_gain_rate"),
            source=f"{clinical_status}_lowinf",
            dest=f"{clinical_status}_inf",
        )
        model.add_transition_flow(
            name=f"infectiousness_loss_{clinical_status}",
            fractional_rate=Parameter("infectiousness_loss_rate"),
            source=f"{clinical_status}_inf",
            dest=f"{clinical_status}_lowinf",
        )

    # Self-recovery (only from subclinical disease)
    for infectious_status in ["inf", "lowinf"]:
        model.add_transition_flow(
            name=f"self_recovery_{infectious_status}",
            fractional_rate=Parameter("self_recovery_rate"),
            source=f"subclin_{infectious_status}",
            dest="recovered",
        )

    return model


def add_detection_and_treatment(
    model: CompartmentalModel, model_config: dict, tv_tsr, bckd_death_rate: float, screening_programs: list
):
    """
    Add passive case detection, active screening and TB treatment outcomes.

    Args:
        model: the model under construction.
        model_config: model configuration dictionary.
        tv_tsr: time-variant treatment success proportion.
        bckd_death_rate: background (non-TB) death rate.
        screening_programs: screening campaigns to apply.
    """

    """
        Passive case detection
    """
    historical_profile = Function(
        tanh_based_scaleup,
        [
            Time,
            Parameter("passive_detection_shape"),
            Parameter("passive_detection_inflection"),
            Parameter("passive_detection_past_frac"),  # past detection rate, relative to current
            1.0,
        ],
    )
    # Scenario lever: future passive detection rate, relative to the current one
    future_profile = get_scenario_ramp_func(model_config, Parameter("future_detection_rate_rel"))

    tv_detection_rate = Parameter("recent_detection_rate") * historical_profile * future_profile

    for active_comp in ACTIVE_COMPS:
        multiplier = Parameter("rel_detection_subclin") if active_comp.startswith("subclin_") else 1.0
        model.add_transition_flow(
            name=f"passive_detection_{active_comp}",
            fractional_rate=multiplier * tv_detection_rate,
            source=active_comp,
            dest="treatment",
        )

    # Track the detection rate and the treatment success proportion so they can be output
    model.add_computed_value_func("passive_detection_rate", tv_detection_rate)
    model.add_computed_value_func("treatment_success_prop", tv_tsr)

    """
        Active screening
    """
    for prog in screening_programs:
        for source_comp, sensitivity in prog.tool.sensitivities.items():
            model.add_transition_flow(
                name=get_screening_flow_name(prog, source_comp),
                fractional_rate=sensitivity * prog.tool.success_prop * prog.screening_rate_func,
                source=source_comp,
                dest=prog.tool.dest_comp,
            )
        # Per-capita screening rate, used to derive the number of people screened / tests performed
        model.add_computed_value_func(get_screening_rate_output_name(prog), prog.screening_rate_func)

    """
        TB treatment outcomes (death on treatment is handled with the other death flows)
    """
    model.add_transition_flow(
        name="tx_recovery",
        fractional_rate=tv_tsr / Parameter("tx_duration"),
        source="treatment",
        dest="recovered",
    )
    model.add_transition_flow(
        name="tx_relapse",
        fractional_rate=get_relapse_rate(tv_tsr, bckd_death_rate),
        source="treatment",
        dest="subclin_lowinf",
    )


def add_births_and_deaths(
    model: CompartmentalModel, pop_series: pd.Series, bckd_death_rate: float, tv_tsr, screening_programs: list
):
    """
    Add mortality and births.

    All deaths are modelled as transitions back to the `mtb_naive` compartment,
    which on its own would keep the population size constant. Extra births are
    then injected so that the modelled population follows the requested
    trajectory. Background deaths occurring in the `mtb_naive` compartment are
    not modelled explicitly, since they would be exactly offset by the
    replacement births entering the same compartment.
    """

    # Background (non-TB) mortality
    for compartment in COMPARTMENTS:
        if compartment == "mtb_naive":
            continue
        model.add_transition_flow(
            name=f"background_mortality_{compartment}",
            fractional_rate=bckd_death_rate,
            source=compartment,
            dest="mtb_naive",
        )

    # TB mortality among untreated (symptomatic) disease
    for infectious_status in ["inf", "lowinf"]:
        model.add_transition_flow(
            name=f"tb_mortality_{infectious_status}",
            fractional_rate=Parameter(f"tb_mortality_rate_{infectious_status}"),
            source=f"clin_{infectious_status}",
            dest="mtb_naive",
        )

    # TB mortality on treatment
    model.add_transition_flow(
        name="tx_death",
        fractional_rate=get_tx_death_rate(tv_tsr, bckd_death_rate),
        source="treatment",
        dest="mtb_naive",
    )

    # Extra births, capturing population growth
    pop_entry = get_population_entry_series(pop_series)
    entry_rate = stf.get_sigmoidal_interpolation_function(
        [pop_entry.index.min() - 1] + pop_entry.index.to_list(),
        [0.0] + pop_entry.to_list(),
    )
    model.add_importation_flow("births", entry_rate, dest="mtb_naive", split_imports=False)


def get_time_variant_tsr(tv_params: dict, model_config: dict):
    """
    Build the time-variant TB treatment success proportion.

    Historical values are read from the time-variant parameter data; from
    `scenario_start_time` the proportion ramps towards the (scenario-dependent)
    `future_tx_success_prop` parameter.
    """
    observed = tv_params["tx_success_pct"].dropna() / 100.0
    scenario_start = model_config["scenario_start_time"]

    observed = observed[observed.index < scenario_start]
    assert len(observed) > 0, "No historical treatment success data before the scenario start time"

    times = observed.index.to_list() + [scenario_start, scenario_start + model_config["scenario_ramp_years"]]
    values = observed.to_list() + [observed.iloc[-1], Parameter("future_tx_success_prop")]

    return stf.get_linear_interpolation_function(times, values)


def get_scenario_ramp_func(model_config: dict, target_value):
    """Linear ramp from 1 to `target_value`, starting at the scenario start time."""
    scenario_start = model_config["scenario_start_time"]
    return stf.get_linear_interpolation_function(
        [scenario_start, scenario_start + model_config["scenario_ramp_years"]],
        [1.0, target_value],
    )


def get_relapse_rate(tv_tsr, bckd_death_rate: float):
    """Rate of relapse (i.e. return to disease) following an unsuccessful treatment episode."""
    return Function(
        _relapse_rate,
        [tv_tsr, Parameter("tx_duration"), Parameter("pct_neg_tx_death"), bckd_death_rate],
    )


def get_tx_death_rate(tv_tsr, bckd_death_rate: float):
    """Rate of TB-attributable death while on treatment."""
    return Function(
        _tx_death_rate,
        [tv_tsr, Parameter("tx_duration"), Parameter("pct_neg_tx_death"), bckd_death_rate],
    )


def _prop_death_from_treatment(tsr, tx_duration, pct_neg_tx_death, bckd_death_rate):
    """Proportion of treatment episodes ending in a TB-attributable death, with a floor of zero."""
    prop_natural_death_on_tx = 1.0 - jnp.exp(-tx_duration * bckd_death_rate)
    requested_prop_death = (1.0 - tsr) * pct_neg_tx_death / 100.0
    return jnp.max(jnp.array((requested_prop_death - prop_natural_death_on_tx, 0.0)))


def _tx_death_rate(tsr, tx_duration, pct_neg_tx_death, bckd_death_rate):
    return _prop_death_from_treatment(tsr, tx_duration, pct_neg_tx_death, bckd_death_rate) / tx_duration


def _relapse_rate(tsr, tx_duration, pct_neg_tx_death, bckd_death_rate):
    prop_natural_death_on_tx = 1.0 - jnp.exp(-tx_duration * bckd_death_rate)
    prop_death_from_tx = _prop_death_from_treatment(tsr, tx_duration, pct_neg_tx_death, bckd_death_rate)
    relapse_prop = 1.0 - tsr - prop_death_from_tx - prop_natural_death_on_tx
    return relapse_prop / tx_duration


def _power(base, exponent):
    return base**exponent


def tanh_based_scaleup(
    t: float,
    shape: float,
    inflection_time: float,
    start_asymptote: float,
    end_asymptote: float = 1.0,
) -> float:
    """
    Smooth transition between two asymptotes, based on the hyperbolic tangent:
        f(t) = (tanh(shape * (t - inflection_time)) / 2 + 0.5) * (end - start) + start

    Args:
        t: time at which to evaluate the function.
        shape: steepness of the transition (higher is steeper).
        inflection_time: midpoint of the transition.
        start_asymptote: value before the transition.
        end_asymptote: value after the transition.

    Returns:
        The scaled value at time t.
    """
    rng = end_asymptote - start_asymptote
    return (jnp.tanh(shape * (t - inflection_time)) / 2.0 + 0.5) * rng + start_asymptote
