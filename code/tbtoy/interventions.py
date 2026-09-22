"""
Screening interventions and scenario definitions.

The screening structure of the full model is retained (so that alternative
screening algorithms can be compared), but simplified: there is no age or
reachability stratification, so a screening program applies uniformly to the
whole population.

A screening program is characterised by:
  * a screening **tool** (i.e. algorithm), describing which compartments it can
    detect, with which sensitivity, where detected individuals are moved to, and
    how many tests of each type are consumed per person screened;
  * a **time window** and a **total coverage** achieved over that window.
"""

from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp

from summer2.functions import time as stf
from summer2.parameters import Function, Parameter

from tbtoy.compartments import ACTIVE_COMPS, VIABLE_INFECTION_COMPS

# Test types tracked as model outputs (number of tests performed)
TEST_TYPES = ("symptom_screen", "cxr", "xpert", "tst")


@dataclass(frozen=True)
class ScreeningTool:
    """
    A screening algorithm.

    Attributes:
        name: short identifier of the algorithm.
        sensitivities: probability of a positive screen, by source compartment.
        dest_comp: compartment that screen-positive individuals are moved to
            ("treatment" for TB disease screening, "cleared" for infection
            screening followed by preventive therapy).
        success_prop: probability of successfully completing the care pathway
            (treatment initiation, or preventive therapy completion) once
            screened positive.
        tests_per_screen: expected number of tests of each type consumed per
            person screened (values may be model parameters).
    """

    name: str
    sensitivities: dict
    dest_comp: str
    success_prop: Any = 1.0
    tests_per_screen: dict = field(default_factory=dict)

    def __post_init__(self):
        unknown = set(self.tests_per_screen) - set(TEST_TYPES)
        assert not unknown, f"Unknown test types {unknown}. Allowed: {TEST_TYPES}"


def _se(test: str) -> dict:
    """Compartment-specific sensitivity parameters for a TB disease screening test."""
    return {comp: Parameter(f"prev_se_{comp}_{test}") for comp in ACTIVE_COMPS}


# Symptom screening only
SSX = ScreeningTool(
    name="ssx",
    sensitivities=_se("ssx"),
    dest_comp="treatment",
    success_prop=1.0,
    tests_per_screen={"symptom_screen": 1.0},
)

# Universal chest X-ray (clinical diagnosis, no bacteriological confirmation)
CXR = ScreeningTool(
    name="cxr",
    sensitivities=_se("cxr"),
    dest_comp="treatment",
    success_prop=1.0,
    tests_per_screen={"symptom_screen": 1.0, "cxr": 1.0},
)

# Universal chest X-ray used as a triage test, with Xpert confirmation of CXR-positives.
# Individuals must be both CXR-positive and Xpert-positive to be started on treatment.
CXR_XPERT = ScreeningTool(
    name="cxr_xpert",
    sensitivities={
        comp: Parameter(f"prev_se_{comp}_cxr") * Parameter(f"prev_se_{comp}_xpert")
        for comp in ACTIVE_COMPS
    },
    dest_comp="treatment",
    success_prop=1.0,
    tests_per_screen={
        "symptom_screen": 1.0,
        "cxr": 1.0,
        # only those with an abnormal CXR proceed to Xpert
        "xpert": Parameter("cxr_positivity_prop"),
    },
)

# TB infection screening (TST) followed by TB preventive therapy
TST_TPT = ScreeningTool(
    name="tst_tpt",
    sensitivities={comp: Parameter(f"prev_se_{comp}_tst") for comp in VIABLE_INFECTION_COMPS},
    dest_comp="cleared",
    success_prop=Parameter("tpt_completion_perc") / 100.0,
    tests_per_screen={"tst": 1.0},
)

SCREENING_TOOLS = {tool.name: tool for tool in (SSX, CXR, CXR_XPERT, TST_TPT)}


class ScreeningProgram:
    """
    A time-limited screening campaign applied to the whole modelled population.

    Args:
        name: unique identifier (used to name model flows and outputs).
        tool: the screening algorithm applied.
        start_time: campaign start (model time units, i.e. years).
        end_time: campaign end.
        coverage_perc: percentage of the population screened over the campaign.
    """

    def __init__(self, name: str, tool: ScreeningTool, start_time: float, end_time: float, coverage_perc: float):
        assert end_time > start_time, "Screening end time must be after its start time"
        assert 0.0 < coverage_perc < 100.0, "Screening coverage must be within (0, 100)%"

        self.name = name
        self.tool = tool
        self.start_time = start_time
        self.end_time = end_time
        self.coverage_perc = coverage_perc
        self.screening_rate_func = self._build_screening_rate_func()

    def _build_screening_rate_func(self):
        """
        Constant per-capita screening rate that achieves the requested total
        coverage over the campaign duration, switched on between start and end
        times: `rate = -log(1 - coverage) / duration`.
        """
        duration = self.end_time - self.start_time
        scr_rate = Function(_coverage_to_rate, [self.coverage_perc, duration])
        return stf.get_linear_interpolation_function(
            [self.start_time - 0.01, self.start_time, self.end_time - 0.01, self.end_time],
            [0.0, scr_rate, scr_rate, 0.0],
        )

    def __repr__(self):
        return (
            f"ScreeningProgram({self.name}, tool={self.tool.name}, "
            f"{self.start_time}-{self.end_time}, coverage={self.coverage_perc}%)"
        )


def _coverage_to_rate(coverage_perc, duration):
    return -jnp.log(1.0 - coverage_perc / 100.0) / duration


def get_screening_flow_name(prog: ScreeningProgram, source_comp: str) -> str:
    """Name of the model flow representing screening of `source_comp` by `prog`."""
    return f"screening_{prog.name}_{source_comp}"


def get_screening_rate_output_name(prog: ScreeningProgram) -> str:
    """Name of the computed value holding the per-capita screening rate of `prog`."""
    return f"screening_rate_{prog.name}"


@dataclass
class Scenario:
    """
    A modelled scenario: a set of screening programs and/or parameter overrides.

    Attributes:
        sc_id: unique identifier, used for file names and dictionary keys.
        sc_name: display name used in figures and tables.
        screening_programs: screening campaigns active in this scenario.
        params_ow: parameter overrides applied on top of the calibrated values
            (used e.g. for future passive detection or treatment success).
        description: free-text description.
    """

    sc_id: str
    sc_name: str
    screening_programs: list = field(default_factory=list)
    params_ow: dict = field(default_factory=dict)
    description: str = ""
