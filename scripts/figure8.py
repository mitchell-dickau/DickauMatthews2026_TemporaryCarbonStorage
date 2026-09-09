import logging
import os
import string
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# Add parent directory to path so we can import utils and plotting_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import plotting_utils
import utils

logger = logging.getLogger(__name__)


def generate_figure_8() -> None:
    """
    Generates and saves Figure 8 of the paper (Added Benefit of Permanence).

    This figure displays a grid showing:
      - Rows: Irreversible variables (defined in utils.vars_irrev by default).
      - Columns: Pairwise comparisons of equivalent carbon storage durations
        (Duration 1: sce7 - sce2, Duration 2: sce8 - sce4, Duration 3: sce9 - sce6).
      - The y-axis represents the difference (Permanent Removal - Temporary Storage).
      - Column titles show cumulative emissions for the permanent removal variants.

    """
    logger.info("Plotting Figure 8...")

    # Load configuration parameters
    fig_params = plotting_utils.fig_params
    colours = fig_params["colours"]
    plot_config = fig_params["plot_config"]

    l_font = plot_config.get("label_fontsize", 11)
    t_font = plot_config.get("subplot_title_fontsize", 12)
    leg_font = plot_config.get("legend_fontsize", 11) + 2
    e_col = plot_config.get("legend_edge_color", "black")

    # Define the pairings: (Permanent, Temporary)
    pairings = [("sce7", "sce2"), ("sce8", "sce4"), ("sce9", "sce6")]

    # Load dataset
    try:
        ds_tsi, _ = utils.load_main_datasets()
    except FileNotFoundError as e:
        logger.error(
            f"Required data not found. Ensure files are in the data directory. {e}"
        )
        raise

    # Default to all irreversibility variables
    vars_irrev = utils.vars_irrev

    # Calculate cumulative CO2
    cumulative_co2 = utils.calc_cumulative(ds_tsi)

    # Resolve output directory, file format, and resolution
    output_dir = Path(utils.FIGURE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_output_dir = Path(utils.DATA_OUTPUT_DIR)
    data_output_dir.mkdir(parents=True, exist_ok=True)

    file_format = fig_params.get("file_format", "pdf")
    fig_dpi = fig_params.get("fig_dpi")

    plot_start_year = 2015
    years = ds_tsi.time.values
    x_limit = max(years)

    # Sort scenarios using the canonical order defined in utils
    scenarios = utils._sort_ssp(ds_tsi.scenario.values)

    n_rows = len(vars_irrev)
    n_cols = 3  # Fixed to the 3 duration comparisons

    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(3.8 * n_cols, 2.1 * n_rows), sharex=True, sharey="row"
    )

    fig_data = {}
    # --- Plotting Loop ---
    for r, var in enumerate(vars_irrev):
        fig_data[var] = {"time": years}
        for c, (p_perm, p_temp) in enumerate(pairings):
            ax = axs[r, c]
            ax.axhline(0, color="black", lw=1.0, alpha=0.5)

            # Calculate the added benefit: (Permanent Removal - Temporary Storage)
            diff_ds = ds_tsi[var].sel(variant=p_perm) - ds_tsi[var].sel(variant=p_temp)

            for sce in scenarios:
                data_to_plot = diff_ds.sel(scenario=sce)
                ax.plot(years, data_to_plot, color=colours[sce], lw=1.5, alpha=0.8)

                # Calculate cumulative temporary storage/permanent removal difference
                # Use the last scenario to calculate equivalent cumulative storage
                last_sce = scenarios[-1]
                cum_emis = cumulative_co2.sel(
                    scenario=last_sce, variant=p_perm, time=2300.5
                ) - cumulative_co2.sel(scenario=last_sce, variant="base", time=2300.5)
                # Format column name in fig data
                col_title = f"{int(abs(np.round(cum_emis.values, decimals=0)))} GtCO2 temporary storage vs. permanent removal"
                # save figure data
                fig_data[var][col_title] = data_to_plot

            # --- Aesthetics ---
            ax.tick_params(labelsize=l_font - 1)
            ax.set_xlim(plot_start_year, x_limit)

            # Column Titles (Top Row Only)
            if r == 0:
                # Format title with cumulative removal total
                col_title = f"{int(abs(np.round(cum_emis.values, decimals=0)))} GtCO$_2$ temporary storage \n vs. permanent removal "
                ax.set_title(col_title, fontweight="bold", fontsize=t_font, pad=15)

            # Row Labels (First Column Only)
            if c == 0:
                unit = ds_tsi[var].attrs.get("units", "units").replace("C", "°C")
                name = ds_tsi[var].attrs.get("long_name", var).capitalize()
                label_text = f"{name}\n({unit})"
                ax.set_ylabel(
                    "\n".join(textwrap.wrap(label_text, width=15)),
                    fontweight="bold",
                    fontsize=l_font,
                )

            # Panel Indexing (A), B), C), etc.)
            panel_idx = r * n_cols + c
            ax.text(
                -0.02,
                1.03,
                f"{string.ascii_uppercase[panel_idx]})",
                transform=ax.transAxes,
                fontsize=t_font,
                fontweight="bold",
                va="bottom",
                ha="right",
            )

            # X-Axis labels for bottom row
            if r == n_rows - 1:
                ax.set_xlabel("Year", fontsize=l_font)

    # Recreate scenario legend patches
    ssp_handles = [
        Patch(color=colours[s], label=f"SSP{s[-3]}-{s[-2]}.{s[-1]}") for s in scenarios
    ]

    fig.legend(
        handles=ssp_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.07),
        ncol=min(len(scenarios), 4),
        frameon=True,
        edgecolor=e_col,
        fontsize=leg_font,
    )

    plt.tight_layout()
    # Extra space for the bottom legend
    plt.subplots_adjust(bottom=0.12, hspace=0.15)

    output_path = output_dir / f"figure8.{file_format}"
    plt.savefig(
        output_path,
        dpi=fig_dpi,
        bbox_inches="tight",
    )
    plt.close()
    logger.info(f"Figure successfully saved to {output_path}")

    # Save figure data
    for k, v in fig_data.items():
        pd.DataFrame(v).to_csv(data_output_dir / f"Fig8_{k}.csv", index=False)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    # Ensure directories are set up prior to running
    utils.setup_output_directories()
    generate_figure_8()
