"""Default model and analysis configurations."""

DEFAULT_MODEL_CONFIG = {
    # Simulation window
    "start_time": 1850,
    "end_time": 2040,
    # Number of infectious individuals seeded at the start of the simulation
    "seed": 100,
    # Demography (generic, not country-specific)
    "reference_population": 100_000.0,
    "reference_year": 2020,
    "population_growth_rate": 0.015,  # per year, exponential
    "life_expectancy": 65.0,  # years, used to derive the background (non-TB) death rate
    # Scenario levers become active from this time, ramping over `scenario_ramp_years`
    "scenario_start_time": 2026,
    "scenario_ramp_years": 2.0,
}

DEFAULT_ANALYSIS_CONFIG = {
    # Metropolis sampling
    "chains": 4,
    "cores": 4,
    "tune": 2000,
    "draws": 10000,
    # Full (sampled) runs
    "burn_in": 2000,
    "full_runs_samples": 500,
}

# Quick settings used by the notebooks. `cores=1` avoids a known deadlock between pymc's
# multiprocessing (os.fork) and JAX's threads on macOS; increase it on Linux.
TEST_ANALYSIS_CONFIG = DEFAULT_ANALYSIS_CONFIG | {
    "chains": 2,
    "cores": 1,
    "tune": 50,
    "draws": 200,
    "burn_in": 50,
    "full_runs_samples": 50,
}
