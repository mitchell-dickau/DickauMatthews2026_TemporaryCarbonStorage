import logging
import os
import string
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.lines import Line2D

# Add parent directory to path so we can import utils and plotting_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import plotting_utils  # noqa: E402
import utils  # noqa: E402

logger = logging.getLogger(__name__)


def generate_figure_1(
    ds_tsi: Optional[xr.Dataset] = None,
    ds_hist: Optional[xr.Dataset] = None,
    anomaly: Optional[float] = None,
    output_dir: Optional[Path] = None,
    file_format: Optional[str] = None,
) -> None:
    """
    Generates and saves Figure 1 of the paper.

    This figure displays:
      - Column 1: Annual CO2 emissions (Subplot A) and difference from baseline (Subplot D).
      - Column 2: Cumulative CO2 emissions (Subplot B) and difference from baseline (Subplot E).
      - Column 3: Temperature anomaly (Subplot C) and difference from baseline (Subplot F).

    For the difference plots (D & E), SSP1-1.9 is plotted as a proxy for emission reduction
    differences, while Subplot F displays temperature differences across all scenarios.

    Parameters
    ----------
    ds_tsi : xr.Dataset, optional
        The main timeseries dataset. If not provided, it will be loaded from the default path.
    ds_hist : xr.Dataset, optional
        The historical dataset used for calculating the temperature baseline.
        If not provided, it will be loaded from the default path.
    anomaly : float, optional
        The pre-industrial baseline temperature anomaly. If not provided, it will be calculated.
    output_dir : Path, optional
        Directory where the figure will be saved. Defaults to utils.FIGURE_DIR.
    file_format : str, optional
        Format of the output file (e.g., 'pdf', 'png'). Defaults to fig_params['file_format'].
    """
    logger.info("Plotting Figure 1...")

    # Load configuration parameters
    fig_params = plotting_utils.fig_params
    colours = fig_params["colours"]
    plot_config = fig_params["plot_config"]

    t_font = plot_config.get("subplot_title_fontsize", 12)
    leg_font = plot_config.get("legend_fontsize", 11)
    leg_t_font = plot_config.get("legend_title_fontsize", 12)
    e_col = plot_config.get("legend_edge_color", "black")

    # Load datasets if not provided
    if ds_tsi is None or ds_hist is None:
        try:
            loaded_tsi, loaded_hist = utils.load_main_datasets()
            ds_tsi = ds_tsi if ds_tsi is not None else loaded_tsi
            ds_hist = ds_hist if ds_hist is not None else loaded_hist
        except FileNotFoundError as e:
            logger.error(
                f"Required data not found. Ensure files are in the data directory. {e}"
            )
            raise

    # Calculate baseline temperature anomaly if not provided
    if anomaly is None:
        anomaly = utils.calculate_historical_anomaly(ds_hist)

    # Resolve output directory and file format
    if output_dir is None:
        output_dir = utils.FIGURE_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if file_format is None:
        file_format = fig_params.get("file_format", "pdf")
    fig_dpi = fig_params.get("fig_dpi", 300)

    # Prepare active variants and scenarios
    all_variants = fig_params["temp_variants"]
    variants_to_exclude = fig_params["perm_variants"]
    plot_variants = [v for v in all_variants if v not in variants_to_exclude]
    active_vars = [v for v in plot_variants if v != "base"]

    # Sort scenarios using the canonical order defined in utils
    scenarios = utils._sort_ssp(ds_tsi.scenario.values)

    # Retrieve line styles and SSP patch handles
    variant_styles, patch_handles = plotting_utils.get_plot_elements(
        ds_tsi, plot_variants, scenarios
    )

    # Identify SSP1-1.9 scenario (used as grey proxy in difference subplots D & E)
    ssp119_sce = [s for s in scenarios if "119" in str(s)][0]

    # Initialize the figure with a 3-row grid: 2 rows of plots and 1 row for legends
    fig = plt.figure(constrained_layout=True, figsize=(14, 8.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.3], wspace=0.02)
    axes = [fig.add_subplot(gs[row, col]) for row in [0, 1] for col in range(3)]

    # Bottom legend axis
    l_ax = fig.add_subplot(gs[2, :])
    l_ax.axis("off")

    years = ds_tsi.time.values
    cumulative_co2 = utils.calc_cumulative(ds_tsi)

    # Subplots A, B, C: Plotting raw emission timeseries, cumulative emissions, and temperature anomalies
    for sce in scenarios:
        for var in plot_variants:
            alpha = 1.0 if var == "base" else 0.5
            # Subplot A: Annual CO2 emissions
            axes[0].plot(
                years,
                ds_tsi.F_co2emit.sel(scenario=sce, variant=var),
                color=colours[sce],
                linestyle=variant_styles[var],
                alpha=alpha,
            )
            # Subplot B: Cumulative emissions
            axes[1].plot(
                years,
                cumulative_co2.sel(scenario=sce, variant=var),
                color=colours[sce],
                linestyle=variant_styles[var],
                alpha=alpha if var == "base" else 0.3,
            )
            # Subplot C: Temperature anomaly (w.r.t pre-industrial baseline)
            temp_anom = ds_tsi.A_sat.sel(scenario=sce, variant=var) - anomaly
            axes[2].plot(
                years,
                temp_anom,
                color=colours[sce],
                linestyle=variant_styles[var],
                alpha=alpha,
            )

    # Add a subtle horizontal baseline at y=0 on the emissions and cumulative emissions axes
    for ax in axes[:2]:
        ax.axhline(0, color="grey", lw=0.8, alpha=0.3, zorder=0)

    # Subplots D & E: Difference relative to baseline for the SSP1-1.9 proxy
    emit_diff = ds_tsi.F_co2emit - ds_tsi.F_co2emit.sel(variant="base")
    cumul_diff = cumulative_co2 - cumulative_co2.sel(variant="base")
    for var in active_vars:
        # Subplot D: Annual emissions difference relative to baseline
        axes[3].plot(
            years,
            emit_diff.sel(scenario=ssp119_sce, variant=var),
            color="grey",
            linestyle=variant_styles[var],
        )
        # Subplot E: Cumulative emissions difference relative to baseline
        axes[4].plot(
            years,
            cumul_diff.sel(scenario=ssp119_sce, variant=var),
            color="grey",
            linestyle=variant_styles[var],
        )

    # Subplot F: Temperature difference relative to baseline for all scenarios
    temp_diff = ds_tsi.A_sat - ds_tsi.A_sat.sel(variant="base")
    for sce in scenarios:
        for var in active_vars:
            axes[5].plot(
                years,
                temp_diff.sel(scenario=sce, variant=var),
                color=colours[sce],
                linestyle=variant_styles[var],
                linewidth=0.5,
            )

    # --- Legends ---
    # Legend 1 (Bottom line of the legend area): SSP Scenario Patches + Baseline handle
    baseline_handle = Line2D(
        [0],
        [0],
        color="black",
        linestyle=variant_styles["base"],
        label="Baseline",
        lw=2,
    )

    first_legend = l_ax.legend(
        handles=patch_handles + [baseline_handle],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.1),
        title="$\mathbf{SSPs:}$",
        ncol=len(patch_handles) + 1,
        frameon=True,
        edgecolor=e_col,
        fontsize=leg_font,
        title_fontsize=leg_t_font,
    )
    l_ax.add_artist(first_legend)

    # Legend 2 (Top line of the legend area): Temporary Storage Styles (sce1 to sce6)
    temp_vars = [
        v for v in active_vars if v in ["sce1", "sce2", "sce3", "sce4", "sce5", "sce6"]
    ]
    temp_h = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=variant_styles[v],
            label=f"Storage {v[-1]}",
        )
        for v in temp_vars
    ]

    l_ax.legend(
        handles=temp_h,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.05),
        title=r"$\mathbf{Temporary\ Storage:}$",
        ncol=6,
        frameon=True,
        edgecolor=e_col,
        fontsize=leg_font,
    )

    # --- Subplot Labels and Titles ---
    titles = [
        "Annual CO$_2$ emissions",
        "Cumulative emissions (from 2015)",
        "Temperature anomaly\n(w.r.t 1850-1900)",
        "Annual emissions difference\nrelative to baseline",
        "Cumulative emissions difference\nrelative to baseline",
        "Temperature difference\nrelative to baseline",
    ]

    for i, ax in enumerate(axes):
        ax.set_title(titles[i], fontsize=t_font)
        ax.set_xlim(2010, 2300.5)
        # Add subplot letter label (e.g., A), B), C))
        ax.text(
            -0.05,
            1.1,
            f"{string.ascii_uppercase[i]})",
            transform=ax.transAxes,
            fontsize=t_font,
            fontweight="bold",
            va="top",
            ha="right",
        )

    # Save the figure
    output_path = output_dir / f"figure1.{file_format}"
    plt.savefig(output_path, dpi=fig_dpi)
    plt.close()
    logger.info(f"Figure successfully saved to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    # Ensure directories are set up prior to running
    utils.setup_output_directories()
    generate_figure_1()
