import logging
import os
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add parent directory to path so we can import utils and plotting_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import plotting_utils
import utils

logger = logging.getLogger(__name__)


def generate_figure_4() -> None:
    """
    Generates and saves Figure 4 of the paper (timeseries grid).

    This figure displays a grid showing:
      - Rows: Irreversibility variables (defined in utils.vars_irrev by default).
      - Columns: Active storage variants (excluding 'base' and any excluded variants).
      - Each grid cell contains two segments: a main timeline (2015-2100) and a
        compressed timeline (2100-2300) to represent a broken axis.
    """
    logger.info("Plotting Figure 4...")

    # Load configuration parameters
    fig_params = plotting_utils.fig_params
    colours = fig_params["colours"]
    plot_config = fig_params["plot_config"]

    l_font = plot_config.get("label_fontsize", 11)
    t_font = plot_config.get("subplot_title_fontsize", 12)
    leg_font = plot_config.get("legend_fontsize", 11)
    e_col = plot_config.get("legend_edge_color", "black")

    # Load dataset
    try:
        ds_tsi, _ = utils.load_main_datasets()
    except FileNotFoundError as e:
        logger.error(
            f"Required data not found. Ensure files are in the data directory. {e}"
        )
        raise

    # Load vars_irrev from utils if not provided
    vars_irrev = utils.vars_irrev

    # Resolve output directory, file format, and resolution
    output_dir = Path(utils.FIGURE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_output_dir = Path(utils.DATA_OUTPUT_DIR)
    data_output_dir.mkdir(parents=True, exist_ok=True)

    file_format = fig_params.get("file_format", "pdf")
    fig_dpi = fig_params.get("fig_dpi")

    # Resolve time slice
    years = ds_tsi.time.values

    split_yr = 2100
    plot_start_year = 2015
    x_limit = max(years)

    # Resolve active variants to plot i.e., only temp. storage variants
    active_variants = [v for v in utils.temp_variants if v != "base"]

    # Sort scenarios using the canonical order defined in utils
    scenarios = utils._sort_ssp(ds_tsi.scenario.values)

    # Retrieve SSP patch handles (variants argument is set to ["base"] to avoid index errors)
    _, patch_handles = plotting_utils.get_plot_elements(ds_tsi, ["base"], scenarios)

    # Ensure patches are sorted
    patch_handles = utils._sort_ssp(patch_handles)

    n_rows = len(vars_irrev)
    n_cols = len(active_variants)

    # Grid proportions: Wide segment (18 units) + gap (1) + Narrow segment (8)
    col_per_var = 31
    fig = plt.figure(figsize=(2 * n_cols + 1, 1.6 * n_rows))
    gs = fig.add_gridspec(n_rows, n_cols * col_per_var)

    # Calculate difference from baseline
    ds_diff = ds_tsi - ds_tsi.sel(variant="base")

    # Track axes for sharing logic
    axes_main = np.empty((n_rows, n_cols), dtype=object)
    axes_comp = np.empty((n_rows, n_cols), dtype=object)

    # dictionary to save figure data
    fig_data = {}

    # --- Plotting Loop ---
    for r, var in enumerate(vars_irrev):
        fig_data[var] = {"time": years}
        for c, v_name in enumerate(active_variants):
            start_col = c * col_per_var

            # Subplot Creation with sharing
            share_y = axes_main[r, 0] if c > 0 else None
            share_x_m = axes_main[0, c] if r > 0 else None
            share_x_c = axes_comp[0, c] if r > 0 else None

            # Main segment (2015-2100)
            ax1 = fig.add_subplot(
                gs[r, start_col : start_col + 18], sharey=share_y, sharex=share_x_m
            )
            # Compressed segment (2100-2300)
            ax2 = fig.add_subplot(
                gs[r, start_col + 19 : start_col + 27], sharey=ax1, sharex=share_x_c
            )

            axes_main[r, c] = ax1
            axes_comp[r, c] = ax2

            # Plotting (Reversed scenario order to show top scenarios on top)
            for sce in scenarios[::-1]:
                diff_data = ds_diff[var].sel(scenario=sce, variant=v_name)
                ax1.plot(years, diff_data, color=colours[sce], lw=1)
                ax2.plot(years, diff_data, color=colours[sce], lw=1)

                # save figure data
                fig_data[var][f"{sce} - {v_name}"] = diff_data.values

            # --- Aesthetics & Break Marks ---
            for ax in [ax1, ax2]:
                ax.axhline(0, color="grey", lw=0.8, alpha=0.3)
                ax.tick_params(labelsize=l_font - 2, pad=8)

            ax1.set_xlim(plot_start_year, split_yr)
            ax2.set_xlim(split_yr, x_limit)

            # Use tick_params instead of set_yticklabels to avoid sharing issues
            ax2.tick_params(labelleft=False)

            ax1.spines["right"].set_visible(False)
            ax2.spines["left"].set_visible(False)
            ax2.yaxis.set_ticks_position("none")

            # Vertical break marks
            d = 0.03
            kw = {"color": "k", "clip_on": False, "lw": 1.0}
            ax1.plot([1, 1], [-d, d], transform=ax1.transAxes, **kw)
            ax1.plot([1, 1], [1 - d, 1 + d], transform=ax1.transAxes, **kw)
            ax2.plot([0, 0], [-d, d], transform=ax2.transAxes, **kw)
            ax2.plot([0, 0], [1 - d, 1 + d], transform=ax2.transAxes, **kw)

            # --- Titles & Labels ---
            if r == 0:
                title_prefix = "Storage"
                ax1.set_title(
                    f"{title_prefix} {v_name[-1]}",
                    fontweight="bold",
                    fontsize=t_font - 2,
                    x=0.6,
                )

            if c == 0:
                unit = (
                    ds_tsi[var].attrs.get("units", "units").replace("°C", r"$^\circ$C")
                )
                name = ds_tsi[var].attrs.get("long_name", var).capitalize()
                ax1.set_ylabel(
                    "\n".join(textwrap.wrap(f"{name}\n({unit})", width=16)),
                    fontweight="bold",
                    fontsize=l_font - 2,
                )
                # Force labels ON for the first column
                ax1.tick_params(labelleft=True)
            else:
                # Force labels OFF for internal columns
                ax1.tick_params(labelleft=False)

            # --- Fix X-Axis Visibility & Ticks (Bottom Row Only) ---
            if r < n_rows - 1:
                ax1.tick_params(labelbottom=False)
                ax2.tick_params(labelbottom=False)
            else:
                ax1.tick_params(labelbottom=True)
                ax2.tick_params(labelbottom=True)
                ax1.set_xticks([2050, 2100])
                ax2.set_xticks([2300])

            # add xlabel in bottom row
            if r == n_rows - 1:
                x_lab_ob = ax1.set_xlabel("Year", fontsize=l_font)
                x_lab_ob.set_position((0.75, 0))

    fig.subplots_adjust(hspace=0.25, wspace=0.0)

    # --- Legend & Final Save ---
    fig.legend(
        handles=patch_handles,
        loc="center left",
        bbox_to_anchor=(0.9, 0.5),
        title=r"$\mathbf{SSP:}$",
        frameon=True,
        edgecolor=e_col,
        fontsize=leg_font,
    )
    output_path = output_dir / f"figure4.{file_format}"
    plt.savefig(
        output_path,
        dpi=fig_dpi,
        bbox_inches="tight",
    )
    plt.close()
    logger.info(f"Figure successfully saved to {output_path}")

    # Save figure data
    for k, v in fig_data.items():
        pd.DataFrame(v).to_csv(data_output_dir / f"Fig4_{k}.csv", index=False)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    # Ensure directories are set up prior to running
    utils.setup_output_directories()
    generate_figure_4()
