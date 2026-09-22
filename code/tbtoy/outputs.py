"""
Model outputs.

Only aggregated (i.e. non age-specific) outputs are requested, covering:
  * demography and TB burden (incidence, prevalence, mortality);
  * the care cascade (notifications, TB treatment starts, preventive therapy);
  * screening volumes (people screened and number of tests performed, by test type).
"""

from summer2 import CompartmentalModel
from summer2.parameters import DerivedOutput, Parameter

from tbtoy.compartments import (
    ACTIVE_COMPS,
    COMPARTMENTS,
    LATENT_COMPS,
    VIABLE_INFECTION_COMPS,
)
from tbtoy.interventions import (
    TEST_TYPES,
    get_screening_flow_name,
    get_screening_rate_output_name,
)

TB_DEATH_FLOWS = ["tb_mortality_inf", "tb_mortality_lowinf", "tx_death"]

COMPUTED_VALUE_OUTPUTS = ["passive_detection_rate", "treatment_success_prop"]


def request_model_outputs(model: CompartmentalModel, model_config: dict, screening_programs: list):
    """
    Request all model outputs, retrievable through `model.get_derived_outputs_df()`.

    Args:
        model: the fully built TB model.
        model_config: model configuration dictionary.
        screening_programs: screening campaigns applied to this model.
    """
    cum_start = model_config["scenario_start_time"]

    request_demography_outputs(model)
    request_prevalence_outputs(model)
    request_incidence_outputs(model, cum_start)
    request_care_cascade_outputs(model, screening_programs, cum_start)
    request_screening_volume_outputs(model, screening_programs, cum_start)
    request_mortality_outputs(model, cum_start)

    for computed_value in COMPUTED_VALUE_OUTPUTS:
        model.request_computed_value_output(computed_value)


def request_demography_outputs(model: CompartmentalModel):
    model.request_output_for_compartments(name="population", compartments=COMPARTMENTS)
    model.request_output_for_flow(name="births", flow_name="births")


def request_prevalence_outputs(model: CompartmentalModel):
    """True (and TST-measured) prevalence of TB disease and TB infection."""

    for comp in COMPARTMENTS:
        model.request_output_for_compartments(name=f"prev_{comp}", compartments=[comp], save_results=False)

    # TB disease prevalence
    model.request_aggregate_output(name="tb_prevalence", sources=[f"prev_{c}" for c in ACTIVE_COMPS])
    request_per_capita_output(model, "tb_prevalence", per=100000.0)

    # TB infection prevalence (any past infection, and infection still viable)
    model.request_aggregate_output(name="tbi_prevalence", sources=[f"prev_{c}" for c in LATENT_COMPS])
    request_per_capita_output(model, "tbi_prevalence", per=100.0)
    model.request_aggregate_output(
        name="viable_tbi_prevalence", sources=[f"prev_{c}" for c in VIABLE_INFECTION_COMPS]
    )
    request_per_capita_output(model, "viable_tbi_prevalence", per=100.0)

    # Composition of prevalent TB disease
    model.request_aggregate_output(
        name="subclin_tb_prevalence",
        sources=[f"prev_{c}" for c in ACTIVE_COMPS if c.startswith("subclin_")],
        save_results=False,
    )
    model.request_aggregate_output(
        name="infectious_tb_prevalence",
        sources=[f"prev_{c}" for c in ACTIVE_COMPS if c.endswith("_inf")],
        save_results=False,
    )
    model.request_function_output(
        name="perc_prev_subclinical",
        func=100.0 * DerivedOutput("subclin_tb_prevalence") / DerivedOutput("tb_prevalence"),
    )
    model.request_function_output(
        name="perc_prev_infectious",
        func=100.0 * DerivedOutput("infectious_tb_prevalence") / DerivedOutput("tb_prevalence"),
    )

    # TST positivity, i.e. TB infection prevalence as it would be measured in a survey
    for comp in COMPARTMENTS:
        model.request_function_output(
            name=f"tst_pos_{comp}",
            func=DerivedOutput(f"prev_{comp}") * get_tst_sensitivity(comp),
            save_results=False,
        )
    model.request_aggregate_output(name="tst_positivity", sources=[f"tst_pos_{c}" for c in COMPARTMENTS])
    request_per_capita_output(model, "tst_positivity", per=100.0)

    # Number of people currently on TB treatment
    model.request_output_for_compartments(name="n_on_treatment", compartments=["treatment"])


def get_tst_sensitivity(comp: str):
    """Probability of a positive tuberculin skin test for individuals in `comp`."""
    if comp in VIABLE_INFECTION_COMPS:
        return Parameter(f"prev_se_{comp}_tst")
    if comp in ("cleared", "recovered"):
        return Parameter("prev_se_cleared_tst")
    if comp == "mtb_naive":
        return 0.0
    return 1.0  # active disease or on treatment


def request_incidence_outputs(model: CompartmentalModel, cum_start: float):
    for inf_cat in ["lowinf", "inf"]:
        model.request_output_for_flow(
            name=f"tb_incidence_{inf_cat}", flow_name=f"progression_{inf_cat}", save_results=False
        )
    model.request_aggregate_output(
        name="tb_incidence", sources=[f"tb_incidence_{inf_cat}" for inf_cat in ["lowinf", "inf"]]
    )
    request_per_capita_output(model, "tb_incidence", per=100000.0)
    model.request_cumulative_output(name="cum_tb_incidence", source="tb_incidence", start_time=cum_start)


def request_care_cascade_outputs(model: CompartmentalModel, screening_programs: list, cum_start: float):
    """Passively detected notifications, screening detections, treatment and preventive therapy starts."""

    # Passive case detection
    for comp in ACTIVE_COMPS:
        model.request_output_for_flow(
            name=f"notifications_{comp}", flow_name=f"passive_detection_{comp}", save_results=False
        )
    model.request_aggregate_output(
        name="tb_notifications", sources=[f"notifications_{comp}" for comp in ACTIVE_COMPS]
    )
    model.request_function_output(
        name="perc_notifications_clin",
        func=100.0
        * (DerivedOutput("notifications_clin_lowinf") + DerivedOutput("notifications_clin_inf"))
        / DerivedOutput("tb_notifications"),
    )

    # Screen-detected TB, and preventive therapy delivered through infection screening
    tb_screening_flows, tpt_flows = [], []
    for prog in screening_programs:
        for source_comp in prog.tool.sensitivities:
            flow_name = get_screening_flow_name(prog, source_comp)
            model.request_output_for_flow(name=flow_name, flow_name=flow_name, save_results=False)
            target_list = tb_screening_flows if prog.tool.dest_comp == "treatment" else tpt_flows
            target_list.append(flow_name)

    request_aggregate_or_zero(model, "screening_tb_detections", tb_screening_flows)
    request_aggregate_or_zero(model, "tpt_completions", tpt_flows)

    # People started on TB treatment, from any pathway
    model.request_aggregate_output(
        name="tb_treatment_starts", sources=["tb_notifications", "screening_tb_detections"]
    )
    model.request_cumulative_output(
        name="cum_tb_treatment_starts", source="tb_treatment_starts", start_time=cum_start
    )
    model.request_cumulative_output(name="cum_tpt_completions", source="tpt_completions", start_time=cum_start)

    # Proportion of incident TB that gets started on treatment (approximate case detection ratio)
    model.request_function_output(
        name="case_detection_prop",
        func=DerivedOutput("tb_treatment_starts") / DerivedOutput("tb_incidence"),
    )


def request_screening_volume_outputs(model: CompartmentalModel, screening_programs: list, cum_start: float):
    """
    Number of screening encounters and number of tests performed, by test type.

    Note that `n_screening_encounters` sums over screening programs, so that
    concurrent programs covering the same people (e.g. disease screening plus
    infection screening) are counted once each.
    """

    screened_outputs = []
    tests_by_type = {test: [] for test in TEST_TYPES}

    for prog in screening_programs:
        rate_name = get_screening_rate_output_name(prog)
        model.request_computed_value_output(rate_name)

        n_screened_name = f"n_screenedX{prog.name}"
        model.request_function_output(
            name=n_screened_name, func=DerivedOutput(rate_name) * DerivedOutput("population")
        )
        screened_outputs.append(n_screened_name)

        for test, n_per_screen in prog.tool.tests_per_screen.items():
            test_output_name = f"n_tests_{test}X{prog.name}"
            model.request_function_output(
                name=test_output_name,
                func=DerivedOutput(n_screened_name) * n_per_screen,
                save_results=False,
            )
            tests_by_type[test].append(test_output_name)

    request_aggregate_or_zero(model, "n_screening_encounters", screened_outputs)
    for test, sources in tests_by_type.items():
        request_aggregate_or_zero(model, f"n_tests_{test}", sources)
    model.request_aggregate_output(name="n_tests", sources=[f"n_tests_{test}" for test in TEST_TYPES])

    for output in ["n_screening_encounters", "n_tests"] + [f"n_tests_{test}" for test in TEST_TYPES]:
        model.request_cumulative_output(name=f"cum_{output}", source=output, start_time=cum_start)


def request_mortality_outputs(model: CompartmentalModel, cum_start: float):
    for flow in TB_DEATH_FLOWS:
        model.request_output_for_flow(name=flow, flow_name=flow, save_results=False)
    model.request_aggregate_output(name="tb_mortality", sources=TB_DEATH_FLOWS)
    request_per_capita_output(model, "tb_mortality", per=100000.0)
    model.request_cumulative_output(name="cum_tb_mortality", source="tb_mortality", start_time=cum_start)


def request_aggregate_or_zero(model: CompartmentalModel, name: str, sources: list):
    """
    Aggregate `sources`, or request an all-zero output if there is no source.

    This keeps the set of model outputs identical across scenarios, including
    scenarios without any screening program (the no-screening comparator).
    """
    if sources:
        model.request_aggregate_output(name=name, sources=sources)
    else:
        model.request_function_output(name=name, func=0.0 * DerivedOutput("population"))


def request_per_capita_output(model: CompartmentalModel, output: str, per=100.0, denominator_output="population"):
    if per == 100.0:
        suffix = "perc"
    elif per == 100000.0:
        suffix = "per100k"
    else:
        suffix = f"per{per}"

    model.request_function_output(
        name=f"{output}_{suffix}",
        func=per * DerivedOutput(output) / DerivedOutput(denominator_output),
    )
