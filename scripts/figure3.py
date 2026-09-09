import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from matplotlib.lines import Line2D

# Add parent directory to path so we can import utils and plotting_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import plotting_utils
import utils

logger = logging.getLogger(__name__)


def generate_figure_3(
    ds_tsi: xr.Dataset | None = None,
    storage_years: xr.DataArray | None = None,
    degree_years: xr.DataArray | None = None,
    output_dir: Path | None = None,
    file_format: str | None = None,
    variants_to_exclude: list[str] | None = None,
    end_year: int | None = None,
) -> None:
    """
    Generates and saves Figure 3 of the paper.

    This figure shows the temporal evolution and relationships of:
      - Panel A: Storage-years
      - Panel B: Degree-years
      - Panel C: Degree-years vs Storage-years, with snapshot
        markers at specific years (2100 and 2300).

    Panels A and B are split into two subplots to represent a broken axis (2015-2100 and 2100-2300).

    Parameters
    ----------
    ds_tsi : xr.Dataset, optional
        The main timeseries dataset. If not provided, it will be loaded from the default path.
    storage_years : xr.DataArray, optional
        Avoided carbon burden integrated over time. If not provided, it will be calculated.
    degree_years : xr.DataArray, optional
        Temperature difference integrated over time. If not provided, it will be calculated.
    output_dir : Path, optional
        Directory where the figure will be saved. Defaults to utils.FIGURE_DIR.
    file_format : str, optional
        Format of the output file (e.g., 'pdf', 'png'). Defaults to fig_params['file_format'].
    variants_to_exclude : list of str, optional
        List of variants to exclude from plotting. Defaults to fig_params['perm_variants'].
    end_year : int, optional
        The end year for plotting. Defaults to the maximum year in the dataset.
    """
    logger.info("Plotting Figure 3...")

    # Load configuration parameters
    fig_params = plotting_utils.fig_params
    colours = fig_params["colours"]
    plot_config = fig_params["plot_config"]

    l_font = plot_config.get("label_fontsize", 11)
    t_font = plot_config.get("subplot_title_fontsize", 12)
    leg_font = plot_config.get("legend_fontsize", 11) - 2
    h_len = plot_config.get("handle_length", 3.5)
    e_col = plot_config.get("legend_edge_color", "black")

    # Load dataset
    try:
        ds_tsi, _ = utils.load_main_datasets()
    except FileNotFoundError as e:
        logger.error(
            f"Required data not found. Ensure files are in the data directory. {e}"
        )
        raise

    # output directory, file format, and resolution
    output_dir = Path(utils.FIGURE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_output_dir = Path(utils.DATA_OUTPUT_DIR)
    data_output_dir.mkdir(parents=True, exist_ok=True)

    file_format = fig_params.get("file_format", "pdf")
    fig_dpi = fig_params.get("fig_dpi")

    # Calculate storage-years and degree-years if not provided
    years = ds_tsi.time.values
    ds_base = ds_tsi.sel(variant="base")
    storage_years = utils.calc_sy(ds_tsi, ds_base, years=years)
    degree_years = utils.calc_dy(ds_base, ds_tsi, years=years)

    split_yr = 2100
    plot_start_year = 2015
    x_limit = end_year if end_year else max(years)

    # Resolve variants to plot
    all_variants = ds_tsi.variant.values
    variants_to_exclude = utils.perm_variants  # exclude permanent removal variatns
    active_vars = [
        v for v in all_variants if v not in variants_to_exclude and v != "base"
    ]  # subset temporary storage variants

    # Sort scenarios using the canonical order defined in utils
    scenarios = utils._sort_ssp(ds_tsi.scenario.values)

    # Retrieve line styles and SSP patch handles
    variant_styles, patch_handles = plotting_utils.get_plot_elements(
        ds_tsi, active_vars, scenarios
    )

    # Ensure patches are sorted
    patch_handles = utils._sort_ssp(patch_handles)

    # --- Setup Figure (Equalized Grid) ---
    fig = plt.figure(constrained_layout=False, figsize=(16, 5.5))
    gs = fig.add_gridspec(1, 100)

    # Panel A: Inputs
    axA1 = fig.add_subplot(gs[0, 0:19])
    axA2 = fig.add_subplot(gs[0, 20:28])
    # Panel B: Response
    axB1 = fig.add_subplot(gs[0, 33:52])
    axB2 = fig.add_subplot(gs[0, 53:61])
    # Panel C: Relationship
    axC = fig.add_subplot(gs[0, 66:94])

    # data dictionary to store figure data for export
    fig_data = {}
    fig_data["storage_years"] = {"time": years}
    fig_data["degree_years"] = {"time": years}

    # --- Plotting Execution ---
    LW = 1.8
    # Reversing order so that the first items in the sorted lists (e.g., SSP1-1.9) plot last (on top)
    for var in active_vars[::-1]:
        for sce in scenarios[::-1]:
            sy = storage_years.sel(scenario=sce, variant=var)
            dy = degree_years.sel(scenario=sce, variant=var)

            fig_data["storage_years"][f"{sce} - {var}"] = sy.values
            fig_data["degree_years"][f"{sce} - {var}"] = dy.values

            # Subplot A : storage years
            for pair_ax in [axA1, axA2]:
                pair_ax.plot(
                    years,
                    sy,
                    color="black",
                    linestyle=variant_styles[var],
                    alpha=0.6,
                    lw=LW,
                )

            # Subplot B : degree years
            for pair_ax in [axB1, axB2]:
                pair_ax.plot(
                    years,
                    dy,
                    color=colours[sce],
                    linestyle=variant_styles[var],
                    alpha=0.6,
                    lw=LW,
                )

            # Subplot C: storage-years vs degree years
            axC.plot(
                sy,
                dy,
                color=colours[sce],
                linestyle=variant_styles[var],
                alpha=0.6,
                lw=LW,
                zorder=2,
            )

            # Snapshot Markers (Hollow with bold edge)
            sy_2100 = float(sy.sel(time=2100.5, method="nearest"))
            dy_2100 = float(dy.sel(time=2100.5, method="nearest"))
            axC.scatter(
                sy_2100,
                dy_2100,
                facecolors="none",
                edgecolors=colours[sce],
                marker="*",
                s=70,
                zorder=3,
                linewidths=1,
                alpha=0.8,
            )

            if x_limit >= 2300:
                sy_2300 = float(sy.sel(time=2300.5, method="nearest"))
                dy_2300 = float(dy.sel(time=2300.5, method="nearest"))
                axC.scatter(
                    sy_2300,
                    dy_2300,
                    facecolors="none",
                    edgecolors=colours[sce],
                    marker="o",
                    s=60,
                    zorder=3,
                    linewidths=1,
                    alpha=0.8,
                )

    # --- Broken Axis Aesthetics ---
    for pair in [(axA1, axA2), (axB1, axB2)]:
        pair[0].set_xlim(plot_start_year, split_yr)
        pair[1].set_xlim(split_yr, x_limit)
        pair[1].set_xticks([2200, 2300])
        pair[1].set_yticklabels([])
        pair[0].spines["right"].set_visible(False)
        pair[1].spines["left"].set_visible(False)
        pair[1].yaxis.set_ticks_position("none")
        d = 0.02
        kw = {"color": "k", "clip_on": False, "lw": 1.2}
        pair[0].plot([1, 1], [-d, d], transform=pair[0].transAxes, **kw)
        pair[0].plot([1, 1], [1 - d, 1 + d], transform=pair[0].transAxes, **kw)
        pair[1].plot([0, 0], [-d, d], transform=pair[1].transAxes, **kw)
        pair[1].plot([0, 0], [1 - d, 1 + d], transform=pair[1].transAxes, **kw)

    # --- Internal Legend System ---
    clean_line_handles = plotting_utils.get_line_legend_handles(
        variant_styles, active_vars, no_titles=True
    )

    time_handles = [
        Line2D(
            [0],
            [0],
            marker="*",
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=1,
            label="2100",
            markersize=8,
            linestyle="None",
            alpha=0.8,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=1,
            label="2300",
            markersize=8,
            linestyle="None",
            alpha=0.8,
        ),
    ]

    L_KWARGS = {
        "frameon": True,
        "edgecolor": e_col,
        "fontsize": leg_font - 1,
        "facecolor": "white",
        "framealpha": 0.9,
    }

    axA1.legend(
        handles=clean_line_handles, loc="upper left", handlelength=h_len, **L_KWARGS
    )

    leg_b_ssp = axB1.legend(handles=patch_handles, loc="upper left", ncol=2, **L_KWARGS)
    axB1.add_artist(leg_b_ssp)
    axB1.legend(
        handles=clean_line_handles,
        loc="upper left",
        bbox_to_anchor=(0, 0.81),
        ncol=1,
        handlelength=h_len,
        **L_KWARGS,
    )

    leg_c_ssp = axC.legend(handles=patch_handles, loc="upper left", ncol=2, **L_KWARGS)
    axC.add_artist(leg_c_ssp)
    leg_c_store = axC.legend(
        handles=clean_line_handles,
        loc="upper left",
        bbox_to_anchor=(0, 0.81),
        ncol=1,
        handlelength=h_len,
        **L_KWARGS,
    )
    axC.add_artist(leg_c_store)
    axC.legend(
        handles=time_handles, loc="upper center", bbox_to_anchor=(0.7, 1), **L_KWARGS
    )

    # Setting labels
    axA1.set_ylabel("Storage-years (Gt CO$_2$-yr)", fontsize=l_font)
    axB1.set_ylabel("Degree-years of avoided warming (°C-yr)", fontsize=l_font)
    axC.set_ylabel("Degree-years of avoided warming  (°C-yr)", fontsize=l_font)
    axC.set_xlabel("Storage-years (Gt CO$_2$-yr)", fontsize=l_font)
    axA1_xlab = axA1.set_xlabel("Year", fontsize=l_font)
    axA1_xlab.set_position((0.75, 0))
    axB1_xlab = axB1.set_xlabel("Year", fontsize=l_font)
    axB1_xlab.set_position((0.75, 0))

    # labelling subpanels
    for label, ax_ref in zip(["A)", "B)", "C)"], [axA1, axB1, axC]):
        ax_ref.text(
            -0.15,
            1.04,
            label,
            transform=ax_ref.transAxes,
            fontsize=t_font,
            fontweight="bold",
            va="top",
        )
        ax_ref.tick_params(labelsize=l_font - 1, pad=8)

    axA2.tick_params(labelsize=l_font - 1, pad=8)
    axB2.tick_params(labelsize=l_font - 1, pad=8)

    # save figure
    output_path = output_dir / f"figure3.{file_format}"
    plt.savefig(
        output_path,
        dpi=fig_dpi,
        bbox_inches="tight",
    )
    plt.close()
    logger.info(f"Figure successfully saved to {output_path}")

    # Save figure data
    for k, v in fig_data.items():
        pd.DataFrame(v).to_csv(data_output_dir / f"Fig3_{k}.csv", index=False)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    # Ensure directories are set up prior to running
    utils.setup_output_directories()
    generate_figure_3()
