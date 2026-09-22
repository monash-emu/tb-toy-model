"""Plotting helpers for single runs, calibration diagnostics and scenario comparisons."""

from copy import copy
from math import ceil

import arviz as az
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from estival import priors as esp

TITLE_LOOKUP = {
    "population": "Population size",
    "births": "Births (n/y)",
    "tb_incidence": "TB incidence (n/y)",
    "tb_incidence_per100k": "TB incidence (/100k/y)",
    "tb_prevalence_per100k": "TB prevalence (/100k)",
    "tbi_prevalence_perc": "TB infection prevalence (%)",
    "viable_tbi_prevalence_perc": "Viable TB infection prevalence (%)",
    "tst_positivity_perc": "TST positivity (%)",
    "tb_mortality": "TB deaths (n/y)",
    "tb_mortality_per100k": "TB mortality (/100k/y)",
    "perc_prev_subclinical": "Asymptomatic TB (% of prevalent TB)",
    "perc_prev_infectious": "More infectious TB (% of prevalent TB)",
    "tb_notifications": "TB notifications (n/y)",
    "perc_notifications_clin": "Symptomatic notifications (%)",
    "screening_tb_detections": "Screen-detected TB (n/y)",
    "tb_treatment_starts": "TB treatment starts (n/y)",
    "n_on_treatment": "People on TB treatment (n)",
    "tpt_completions": "Preventive therapy courses completed (n/y)",
    "case_detection_prop": "Treatment starts / incident TB",
    "n_screening_encounters": "Screening encounters (n/y)",
    "n_tests": "Screening tests performed (n/y)",
    "n_tests_symptom_screen": "Symptom screens (n/y)",
    "n_tests_cxr": "Chest X-rays (n/y)",
    "n_tests_xpert": "Xpert tests (n/y)",
    "n_tests_tst": "Tuberculin skin tests (n/y)",
    "passive_detection_rate": "Passive detection rate (/y)",
    "treatment_success_prop": "Treatment success (proportion)",
    "cum_tb_incidence": "Cumulative TB episodes",
    "cum_tb_mortality": "Cumulative TB deaths",
    "cum_tb_treatment_starts": "Cumulative TB treatment starts",
    "cum_tpt_completions": "Cumulative preventive therapy courses",
    "cum_n_screening_encounters": "Cumulative screening encounters",
    "cum_n_tests": "Cumulative screening tests",
    "cum_n_tests_symptom_screen": "Cumulative symptom screens",
    "cum_n_tests_cxr": "Cumulative chest X-rays",
    "cum_n_tests_xpert": "Cumulative Xpert tests",
    "cum_n_tests_tst": "Cumulative tuberculin skin tests",
    "TB_averted": "TB episodes averted",
    "TB_averted_relative": "TB episodes averted (proportion)",
    "deaths_averted": "TB deaths averted",
    "deaths_averted_relative": "TB deaths averted (proportion)",
}

SC_COLOURS = (
    "black",
    "tab:blue",
    "tab:red",
    "tab:green",
    "tab:orange",
    "tab:purple",
    "tab:brown",
    "tab:pink",
)


def get_title(output_name: str) -> str:
    return TITLE_LOOKUP.get(output_name, output_name)


"""
    Single runs
"""


def plot_outputs(derived_outputs, outputs: list, n_col: int = 3, x_lim: tuple = None, targets: dict = None):
    """
    Plot a selection of model outputs from a single run.

    Args:
        derived_outputs: DataFrame of derived outputs (indexed by time).
        outputs: names of the outputs to plot.
        n_col: number of columns in the figure grid.
        x_lim: time range to display.
        targets: optional mapping of output name to estival target, overlaid as points.

    Returns:
        The figure.
    """
    n_row = ceil(len(outputs) / n_col)
    fig, axes = plt.subplots(n_row, n_col, figsize=(4.6 * n_col, 3.2 * n_row), squeeze=False)
    axes = axes.flatten()

    for i, output in enumerate(outputs):
        ax = axes[i]
        series = derived_outputs[output]
        if x_lim:
            series = series.loc[x_lim[0] : x_lim[1]]
        ax.plot(series.index, series.values, color="tab:blue", label="Model")

        if targets and output in targets:
            target_data = copy(targets[output].data)
            ax.scatter(target_data.index, target_data.values, color="black", s=18, zorder=10, label="Target")
            ax.legend(fontsize=8)

        ax.set_title(get_title(output), fontsize=10)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0.0)

    for j in range(len(outputs), len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    return fig


def plot_single_fit(bcm, params: dict, n_col: int = 3, x_lim: tuple = (1990, 2030)):
    """Run the model for one parameter set and compare all targeted outputs with their targets."""
    derived_outputs = bcm.run(params).derived_outputs
    return plot_outputs(derived_outputs, list(bcm.targets.keys()), n_col=n_col, x_lim=x_lim, targets=bcm.targets)


"""
    Calibration diagnostics
"""


def plot_traces(idata, bcm, burn_in: int = 0, n_col: int = 3):
    """Plot MCMC traces, one panel per calibrated parameter and one colour per chain."""
    posterior = idata.posterior
    if burn_in > 0 and "draw" in posterior.dims:
        posterior = posterior.isel(draw=slice(burn_in, None))

    trace_params = [p for p in bcm.priors.keys() if p in posterior.data_vars]
    n_row = ceil(len(trace_params) / n_col)
    fig, axes = plt.subplots(n_row, n_col, figsize=(5.5 * n_col, 2.6 * n_row), squeeze=False)
    axes = axes.flatten()

    chain_ids = sorted(list(posterior[trace_params[0]]["chain"].values))
    base_colors = list(plt.cm.tab10.colors) + list(plt.cm.Set2.colors)
    if len(chain_ids) > len(base_colors):
        base_colors = (base_colors * ceil(len(chain_ids) / len(base_colors)))[: len(chain_ids)]
    chain_color_map = {chain_id: base_colors[i] for i, chain_id in enumerate(chain_ids)}

    for i, param_name in enumerate(trace_params):
        ax = axes[i]
        da = posterior[param_name]
        for chain_value in chain_ids:
            ax.plot(
                da["draw"].values,
                da.sel(chain=chain_value).values,
                linewidth=0.8,
                alpha=0.85,
                color=chain_color_map[chain_value],
            )
        ax.set_title(param_name, fontsize=10)
        ax.set_xlabel("Draw")
        ax.grid(alpha=0.3)

    for j in range(len(trace_params), len(axes)):
        fig.delaxes(axes[j])

    fig.legend(
        handles=[
            mlines.Line2D([], [], color=chain_color_map[c], linewidth=1.8, label=f"chain {c}")
            for c in chain_ids
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=min(8, len(chain_ids)),
        frameon=False,
    )
    fig.tight_layout()
    return fig


def plot_post_prior_comparison(idata, priors: list, burn_in: int = 0, n_col: int = 4, kde_bw="silverman"):
    """
    Compare posterior densities with the corresponding prior distributions.

    Args:
        idata: calibration inference data.
        priors: list of estival prior objects.
        burn_in: number of draws discarded from the start of each chain.
        n_col: number of columns in the figure grid.
        kde_bw: bandwidth rule used for the posterior KDE.

    Returns:
        The figure.
    """
    chain_length = idata.sample_stats.sizes["draw"]
    burnt_idata = idata.sel(draw=range(burn_in, chain_length))

    req_vars = [p.name for p in priors if p.name in burnt_idata.posterior.data_vars]
    n_row = ceil(len(req_vars) / n_col)
    fig, axes = plt.subplots(n_row, n_col, figsize=(4.0 * n_col, 2.8 * n_row), squeeze=False)
    axes = axes.flatten()

    prior_lookup = {p.name: p for p in priors}

    for i_ax, param in enumerate(req_vars):
        ax = axes[i_ax]
        values = burnt_idata.posterior[param].values.flatten()
        az.plot_kde(
            values,
            ax=ax,
            bw=kde_bw,
            plot_kwargs={"color": "tab:blue", "linewidth": 1.8},
            fill_kwargs={"alpha": 0.3, "color": "tab:blue"},
        )

        distri = prior_lookup.get(param)
        if distri is not None and not isinstance(distri, esp.TruncNormalPrior):
            x_min, x_max = float(np.nanmin(values)), float(np.nanmax(values))
            if np.isfinite(x_min) and np.isfinite(x_max) and x_max > x_min:
                pad = 0.15 * (x_max - x_min)
                x_vals = np.linspace(x_min - pad, x_max + pad, 200)
                y_vals = np.exp(distri.logpdf(x_vals))
                ax.plot(x_vals, y_vals, color="k", linewidth=1.2, alpha=0.9)
                ax.fill_between(x_vals, y_vals, color="k", alpha=0.15)

        ax.set_title(param, fontsize=10)
        ax.set_ylabel("Density")
        ax.grid(alpha=0.3)

    for j in range(len(req_vars), len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    return fig


"""
    Uncertainty and scenario comparison
"""


def plot_model_fit_with_uncertainty(axis, uncertainty_df, output_name: str, bcm=None, x_lim=None, colour="#B22222"):
    """Plot the median and credible intervals of one output, with calibration targets if available."""
    df = uncertainty_df[output_name]
    if x_lim:
        df = df.loc[x_lim[0] : x_lim[1]]

    if bcm is not None and output_name in bcm.targets:
        target_data = copy(bcm.targets[output_name].data)
        axis.scatter(list(target_data.index), target_data, marker=".", color="black", zorder=11, s=30, label="Target")

    axis.plot(df.index, df["0.5"], color=colour, zorder=10, label="Model (median)")
    axis.fill_between(df.index, df["0.25"], df["0.75"], color=colour, alpha=0.35, edgecolor=None, label="Model (IQR)")
    axis.fill_between(
        df.index, df["0.025"], df["0.975"], color=colour, alpha=0.2, edgecolor=None, label="Model (95% CrI)"
    )

    axis.set_ylabel(get_title(output_name))
    axis.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    axis.set_ylim(0.0, 1.2 * axis.get_ylim()[1])


def plot_all_model_fits(uncertainty_df, bcm, n_col: int = 3, x_lim: tuple = (1990, 2030)):
    """Plot the model fit against every calibration target."""
    outputs = list(bcm.targets.keys())
    n_row = ceil(len(outputs) / n_col)
    fig, axes = plt.subplots(n_row, n_col, figsize=(4.8 * n_col, 3.4 * n_row), squeeze=False)
    axes = axes.flatten()

    for i, output in enumerate(outputs):
        plot_model_fit_with_uncertainty(axes[i], uncertainty_df, output, bcm, x_lim=x_lim)
        axes[i].set_title(get_title(output), fontsize=10)
        if i == 0:
            axes[i].legend(fontsize=8)

    for j in range(len(outputs), len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    return fig


def plot_scenario_comparison(
    axis, unc_dfs: dict, output_name: str, sc_names: dict, x_lim: tuple, include_unc: bool = True
):
    """Overlay the same output across scenarios, with optional uncertainty ribbons."""
    for i_sc, (sc_id, unc_df) in enumerate(unc_dfs.items()):
        df = unc_df[output_name].loc[x_lim[0] : x_lim[1]]
        colour = SC_COLOURS[i_sc % len(SC_COLOURS)]
        axis.plot(df.index, df["0.5"], color=colour, label=sc_names.get(sc_id, sc_id), zorder=10 - i_sc)
        if include_unc:
            axis.fill_between(df.index, df["0.025"], df["0.975"], color=colour, alpha=0.15, edgecolor=None)

    axis.set_ylabel(get_title(output_name))
    axis.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    axis.set_ylim(bottom=0.0)
    axis.grid(alpha=0.3)


def plot_scenarios(unc_dfs: dict, outputs: list, sc_names: dict, x_lim: tuple, n_col: int = 2, include_unc: bool = True):
    """Grid of scenario comparison panels, one per requested output."""
    n_row = ceil(len(outputs) / n_col)
    fig, axes = plt.subplots(n_row, n_col, figsize=(5.4 * n_col, 3.6 * n_row), squeeze=False)
    axes = axes.flatten()

    for i, output in enumerate(outputs):
        plot_scenario_comparison(axes[i], unc_dfs, output, sc_names, x_lim, include_unc=include_unc)
        axes[i].set_title(get_title(output), fontsize=11)

    axes[0].legend(fontsize=8)
    for j in range(len(outputs), len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    return fig


def plot_diff_outputs(diff_quantiles: dict, output: str, sc_names: dict, ax=None):
    """
    Horizontal bar chart of scenario impact (e.g. TB episodes or deaths averted).

    Args:
        diff_quantiles: scenario id to difference-quantile DataFrame.
        output: column of the difference DataFrame to display.
        sc_names: scenario id to display name mapping.
        ax: optional existing axis.

    Returns:
        The figure.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 0.6 * len(diff_quantiles) + 1.5))

    sc_ids = list(diff_quantiles.keys())
    medians = [diff_quantiles[sc][output].loc[0.5] for sc in sc_ids]
    lows = [medians[i] - diff_quantiles[sc][output].loc[0.025] for i, sc in enumerate(sc_ids)]
    highs = [diff_quantiles[sc][output].loc[0.975] - medians[i] for i, sc in enumerate(sc_ids)]

    y_pos = np.arange(len(sc_ids))
    ax.barh(y_pos, medians, xerr=[lows, highs], color="tab:blue", alpha=0.7, capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([sc_names.get(sc, sc) for sc in sc_ids])
    ax.invert_yaxis()
    ax.set_xlabel(get_title(output))
    ax.grid(alpha=0.3, axis="x")

    return ax.get_figure()


def plot_cumulative_outputs(cumulative_df, outputs: list, sc_names: dict, n_col: int = 2):
    """
    Bar charts of cumulative outputs (burden and resource use) by scenario.

    Args:
        cumulative_df: output of `calculate_cumulative_output_quantiles`.
        outputs: cumulative outputs to display.
        sc_names: scenario id to display name mapping.
        n_col: number of columns in the figure grid.

    Returns:
        The figure.
    """
    sc_ids = list(cumulative_df.index.get_level_values("scenario").unique())
    n_row = ceil(len(outputs) / n_col)
    fig, axes = plt.subplots(n_row, n_col, figsize=(5.6 * n_col, 0.45 * len(sc_ids) * n_row + 2.0 * n_row), squeeze=False)
    axes = axes.flatten()

    for i, output in enumerate(outputs):
        ax = axes[i]
        medians = [cumulative_df.loc[(sc, 0.5), output] for sc in sc_ids]
        lows = [medians[j] - cumulative_df.loc[(sc, 0.025), output] for j, sc in enumerate(sc_ids)]
        highs = [cumulative_df.loc[(sc, 0.975), output] - medians[j] for j, sc in enumerate(sc_ids)]

        y_pos = np.arange(len(sc_ids))
        ax.barh(y_pos, medians, xerr=[lows, highs], color="tab:green", alpha=0.7, capsize=3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([sc_names.get(sc, sc) for sc in sc_ids], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(get_title(output), fontsize=10)
        ax.grid(alpha=0.3, axis="x")

    for j in range(len(outputs), len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    return fig


"""
    Scenario comparison for a single parameter set
"""


def plot_scenarios_single_params(scenario_outputs: dict, outputs: list, sc_names: dict, x_lim: tuple, n_col: int = 2):
    """
    Grid of scenario comparison panels for a single parameter set (no uncertainty).

    Args:
        scenario_outputs: scenario id to derived outputs mapping.
        outputs: names of the outputs to plot.
        sc_names: scenario id to display name mapping.
        x_lim: time range to display.
        n_col: number of columns in the figure grid.

    Returns:
        The figure.
    """
    n_row = ceil(len(outputs) / n_col)
    fig, axes = plt.subplots(n_row, n_col, figsize=(5.4 * n_col, 3.6 * n_row), squeeze=False)
    axes = axes.flatten()

    for i, output in enumerate(outputs):
        ax = axes[i]
        for i_sc, (sc_id, derived_outputs) in enumerate(scenario_outputs.items()):
            series = derived_outputs[output].loc[x_lim[0] : x_lim[1]]
            ax.plot(
                series.index,
                series.values,
                color=SC_COLOURS[i_sc % len(SC_COLOURS)],
                label=sc_names.get(sc_id, sc_id),
                zorder=10 - i_sc,
            )
        ax.set_ylabel(get_title(output))
        ax.set_title(get_title(output), fontsize=11)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.set_ylim(bottom=0.0)
        ax.grid(alpha=0.3)

    axes[0].legend(fontsize=8)
    for j in range(len(outputs), len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    return fig


def plot_diff_outputs_single_params(diff_df, output: str, sc_names: dict, ax=None):
    """
    Horizontal bar chart of scenario impact for a single parameter set.

    Args:
        diff_df: output of `calculate_diff_outputs_single_params`.
        output: column of the difference DataFrame to display.
        sc_names: scenario id to display name mapping.
        ax: optional existing axis.

    Returns:
        The figure.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 0.6 * len(diff_df) + 1.5))

    sc_ids = list(diff_df.index)
    y_pos = np.arange(len(sc_ids))
    ax.barh(y_pos, diff_df[output].values, color="tab:blue", alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([sc_names.get(sc, sc) for sc in sc_ids])
    ax.invert_yaxis()
    ax.set_xlabel(get_title(output))
    ax.grid(alpha=0.3, axis="x")

    return ax.get_figure()


def plot_cumulative_outputs_single_params(cumulative_df, outputs: list, sc_names: dict, n_col: int = 2):
    """
    Bar charts of cumulative outputs by scenario, for a single parameter set.

    Args:
        cumulative_df: output of `calculate_cumulative_outputs_single_params`.
        outputs: cumulative outputs to display.
        sc_names: scenario id to display name mapping.
        n_col: number of columns in the figure grid.

    Returns:
        The figure.
    """
    sc_ids = list(cumulative_df.index)
    n_row = ceil(len(outputs) / n_col)
    fig, axes = plt.subplots(n_row, n_col, figsize=(5.6 * n_col, 0.45 * len(sc_ids) * n_row + 2.0 * n_row), squeeze=False)
    axes = axes.flatten()

    y_pos = np.arange(len(sc_ids))
    for i, output in enumerate(outputs):
        ax = axes[i]
        ax.barh(y_pos, cumulative_df[output].values, color="tab:green", alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([sc_names.get(sc, sc) for sc in sc_ids], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(get_title(output), fontsize=10)
        ax.grid(alpha=0.3, axis="x")

    for j in range(len(outputs), len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    return fig


def visualise_mle_params(priors: dict, mle_params: dict, n_col: int = 4):
    """Show where the maximum likelihood estimates sit within their prior ranges."""
    param_names = [p for p in mle_params if p in priors]
    n_row = ceil(len(param_names) / n_col)
    fig, axes = plt.subplots(n_row, n_col, figsize=(3.6 * n_col, 1.6 * n_row), squeeze=False)
    axes = axes.flatten()

    for i, param in enumerate(param_names):
        ax = axes[i]
        bounds = priors[param].bounds()
        ax.plot(bounds, [0, 0], color="lightgrey", linewidth=6, solid_capstyle="butt")
        ax.scatter([mle_params[param]], [0], color="crimson", zorder=5)
        ax.set_yticks([])
        ax.set_title(f"{param}\n{mle_params[param]:.4g}", fontsize=8)
        ax.set_xlim(bounds)

    for j in range(len(param_names), len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    return fig
