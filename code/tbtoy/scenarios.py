"""
Scenario definitions.

Two sets of scenarios are provided, matching the two applications of the toy model:

1. `PROGRAMMATIC_SCENARIOS` — future changes to routine TB services, i.e. the
   passive case detection rate and the treatment success proportion.
2. `SCREENING_SCENARIOS` — alternative active screening algorithms and coverage
   levels, with "no screening" as the comparator.

Both sets start with the same `baseline` scenario (status quo continued, no
screening), which acts as the common comparator.
"""

from tbtoy.config import DEFAULT_MODEL_CONFIG
from tbtoy.interventions import CXR, CXR_XPERT, SSX, TST_TPT, Scenario, ScreeningProgram

SCREENING_START = DEFAULT_MODEL_CONFIG["scenario_start_time"]
SCREENING_END = SCREENING_START + 2  # two-year campaign
DEFAULT_COVERAGE_PERC = 70.0

BASELINE = Scenario(
    sc_id="baseline",
    sc_name="Status quo",
    screening_programs=[],
    params_ow={},
    description="Current passive case detection and treatment success maintained, no active screening.",
)


"""
    Application 1: future passive detection rate and treatment success rate
"""
PROGRAMMATIC_SCENARIOS = [
    BASELINE,
    Scenario(
        sc_id="detection_down",
        sc_name="Detection -30%",
        params_ow={"future_detection_rate_rel": 0.7},
        description="Passive case detection rate declines by 30% (e.g. service disruption).",
    ),
    Scenario(
        sc_id="detection_up",
        sc_name="Detection +50%",
        params_ow={"future_detection_rate_rel": 1.5},
        description="Passive case detection rate improves by 50%.",
    ),
    Scenario(
        sc_id="detection_up_strong",
        sc_name="Detection x2",
        params_ow={"future_detection_rate_rel": 2.0},
        description="Passive case detection rate doubles.",
    ),
    Scenario(
        sc_id="tsr_up",
        sc_name="Treatment success 95%",
        params_ow={"future_tx_success_prop": 0.95},
        description="Treatment success proportion improves to 95%.",
    ),
    Scenario(
        sc_id="tsr_down",
        sc_name="Treatment success 75%",
        params_ow={"future_tx_success_prop": 0.75},
        description="Treatment success proportion deteriorates to 75%.",
    ),
    Scenario(
        sc_id="detection_and_tsr_up",
        sc_name="Detection x2 + TSR 95%",
        params_ow={"future_detection_rate_rel": 2.0, "future_tx_success_prop": 0.95},
        description="Combined improvement of passive detection and treatment success.",
    ),
]


"""
    Application 2: alternative screening approaches
"""


def make_screening_scenario(
    sc_id: str,
    sc_name: str,
    tools: list,
    coverage_perc: float = DEFAULT_COVERAGE_PERC,
    start_time: float = SCREENING_START,
    end_time: float = SCREENING_END,
    description: str = "",
) -> Scenario:
    """
    Build a screening scenario from a list of screening tools sharing the same
    time window and coverage.

    Args:
        sc_id: unique scenario identifier.
        sc_name: display name.
        tools: screening tools applied concurrently (e.g. disease screening plus
            infection screening).
        coverage_perc: percentage of the population screened over the campaign.
        start_time: campaign start.
        end_time: campaign end.
        description: free-text description.

    Returns:
        The scenario.
    """
    programs = [
        ScreeningProgram(
            name=f"{sc_id}_{tool.name}",
            tool=tool,
            start_time=start_time,
            end_time=end_time,
            coverage_perc=coverage_perc,
        )
        for tool in tools
    ]
    return Scenario(sc_id=sc_id, sc_name=sc_name, screening_programs=programs, description=description)


SCREENING_SCENARIOS = [
    BASELINE,
    make_screening_scenario(
        sc_id="ssx",
        sc_name="Symptom screening",
        tools=[SSX],
        description="Symptom screening only; symptomatic screen-positives are started on treatment.",
    ),
    make_screening_scenario(
        sc_id="cxr",
        sc_name="Universal CXR",
        tools=[CXR],
        description="Universal chest X-ray, treatment started on radiological diagnosis.",
    ),
    make_screening_scenario(
        sc_id="cxr_xpert",
        sc_name="CXR + Xpert",
        tools=[CXR_XPERT],
        description="Universal chest X-ray used as a triage test, with Xpert confirmation.",
    ),
    make_screening_scenario(
        sc_id="cxr_xpert_tpt",
        sc_name="CXR + Xpert + TST/TPT",
        tools=[CXR_XPERT, TST_TPT],
        description="Disease screening (CXR triage with Xpert confirmation) plus TST-guided preventive therapy.",
    ),
]

# Coverage sensitivity for the most complete algorithm
SCREENING_SCENARIOS += [
    make_screening_scenario(
        sc_id=f"cxr_xpert_tpt_cov{int(coverage)}",
        sc_name=f"CXR + Xpert + TST/TPT ({int(coverage)}%)",
        tools=[CXR_XPERT, TST_TPT],
        coverage_perc=coverage,
        description=f"Full algorithm delivered at {int(coverage)}% coverage.",
    )
    for coverage in (40.0, 90.0)
]


SCENARIO_SETS = {
    "programmatic": PROGRAMMATIC_SCENARIOS,
    "screening": SCREENING_SCENARIOS,
}


def get_scenario_names(scenarios: list) -> dict:
    """Map scenario ids to display names, for use in plots and tables."""
    return {scenario.sc_id: scenario.sc_name for scenario in scenarios}
