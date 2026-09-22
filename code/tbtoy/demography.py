"""
Generic (i.e. not country-specific) demography for the toy model.

Rather than reading country-specific UN data, the population trajectory is
generated analytically from the model configuration. The approach used by the
model is otherwise unchanged:
  * background deaths are modelled as transitions back to the `mtb_naive`
    compartment, which on its own keeps the population size constant;
  * additional births are then injected as an importation flow so that the
    modelled population follows the requested trajectory.
"""

import numpy as np
import pandas as pd


def get_population_series(model_config: dict) -> pd.Series:
    """
    Build the target total population size over time (one value per year).

    An exponential growth profile is anchored on a reference population size at a
    reference year, i.e. `pop(t) = ref_pop * exp(growth_rate * (t - ref_year))`.

    Args:
        model_config: model configuration dictionary.

    Returns:
        Population size indexed by (integer) year.
    """
    years = np.arange(int(model_config["start_time"]), int(model_config["end_time"]) + 1)
    pop = model_config["reference_population"] * np.exp(
        model_config["population_growth_rate"] * (years - model_config["reference_year"])
    )
    return pd.Series(data=pop, index=pd.Index(years, name="year"), name="population")


def get_background_death_rate(model_config: dict) -> float:
    """
    Constant background (non-TB) mortality rate, derived from life expectancy.

    Args:
        model_config: model configuration dictionary.

    Returns:
        Per-capita, per-year background death rate.
    """
    return 1.0 / model_config["life_expectancy"]


def get_population_entry_series(pop_series: pd.Series) -> pd.Series:
    """
    Yearly net population increments implied by the target population trajectory.

    Args:
        pop_series: total population size indexed by year.

    Returns:
        Net number of extra individuals entering the population each year.
    """
    return pop_series.diff().dropna()
