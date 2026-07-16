import logging
import os
import sys
import textwrap
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Add parent directory to path so we can import utils and plotting_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import plotting_utils  # noqa: E402
import utils  # noqa: E402

logger = logging.getLogger(__name__)


def generate_figure_4(
    ds_tsi: Optional[xr.Dataset] = None,
    vars_irrev: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    file_format: Optional[str] = None,
    variants_to_exclude: Optional[List[str]] = plotting_utils.fig_params.get(
        "perm_variants"
    ),
    end_year: Optional[int] = None,
) -> None:
    """
    Generates and saves Figure 4 of the paper (timeseries grid).

    This figure displays a grid showing:
      - Rows: Irreversibility variables (defined in utils.vars_irrev by default).
      - Columns: Active storage variants (excluding 'base' and any excluded variants).
      - Each grid cell contains two segments: a main timeline (2015-2100) and a
        compressed timeline (2100-2300) to represent a broken axis.

    Parameters
    ----------
    ds_tsi : xr.Dataset, optional
        The main timeseries dataset. If not provided, it will be loaded from the default path.
    vars_irrev : list of str, optional
        List of variables to plot. Defaults to utils.vars_irrev.
    output_dir : Path, optional
        Directory where the figure will be saved. Defaults to utils.FIGURE_DIR.
    file_format : str, optional
        Format of the output file (e.g., 'pdf', 'png'). Defaults to fig_params['file_format'].
    variants_to_exclude : list of str, optional
        List of variants to exclude from plotting. Defaults to empty list.
    end_year : int, optional
        The end year for plotting. Defaults to the maximum year in the dataset.
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

    # Load dataset if not provided
    if ds_tsi is None:
        try:
            ds_tsi, _ = utils.load_main_datasets()
        except FileNotFoundError as e:
            logger.error(
                f"Required data not found. Ensure files are in the data directory. {e}"
            )
            raise

    # Load vars_irrev from utils if not provided
    if vars_irrev is None:
        vars_irrev = getattr(utils, "vars_irrev", [])

    # Resolve output directory, file format, and resolution
    if output_dir is None:
        output_dir = utils.FIGURE_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if file_format is None:
        file_format = fig_params.get("file_format", "pdf")
    fig_dpi = fig_params.get("fig_dpi", 300)

    # Resolve time slice
    years = ds_tsi.time.values
    if end_year:
        ds_tsi = ds_tsi.sel(time=slice(None, end_year))
        years = years[years <= end_year]

    split_yr = 2100
    plot_start_year = 2015
    x_limit = end_year if end_year else max(years)

    # Resolve active variants to plot
    active_variants = [
        v
        for v in ds_tsi.variant.values
        if v not in (variants_to_exclude or []) and v != "base"
    ]

    # Sort scenarios using the canonical order defined in utils
    scenarios = utils._sort_ssp(ds_tsi.scenario.values)

    # Retrieve SSP patch handles (variants argument is set to ["base"] to avoid index errors)
    _, patch_handles = plotting_utils.get_plot_elements(ds_tsi, ["base"], scenarios)

    # Ensure patches are canonically sorted
    patch_handles = utils._sort_ssp(patch_handles)

    n_rows = len(vars_irrev)
    n_cols = len(active_variants)

    # Grid proportions: Wide segment (18 units) + gap (1) + Narrow segment (8)
    col_per_var = 31
    fig = plt.figure(figsize=(2.8 * n_cols + 1, 2.0 * n_rows))
    gs = fig.add_gridspec(n_rows, n_cols * col_per_var)

    # Calculate difference from baseline
    ds_diff = ds_tsi - ds_tsi.sel(variant="base")

    # Track axes for sharing logic
    axes_main = np.empty((n_rows, n_cols), dtype=object)
    axes_comp = np.empty((n_rows, n_cols), dtype=object)

    # --- Plotting Loop ---
    for r, var in enumerate(vars_irrev):
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
                ax1.plot(years, diff_data, color=colours[sce], lw=1.2)
                ax2.plot(years, diff_data, color=colours[sce], lw=1.2)

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
            kw = dict(color="k", clip_on=False, lw=1.0)
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

    fig.subplots_adjust(hspace=0.25, wspace=0.0)

    # --- Legend & Final Save ---
    fig.legend(
        handles=patch_handles,
        loc="center left",
        bbox_to_anchor=(0.9, 0.5),
        title="$\mathbf{SSP:}$",
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


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    # Ensure directories are set up prior to running
    utils.setup_output_directories()
    generate_figure_4()
