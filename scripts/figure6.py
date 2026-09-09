import logging
import os
import string
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


def generate_figure_6() -> None:
    """
    Generates and saves Figure 6 of the paper (Scenario-Dependent Regression Grid).

    This figure displays:
      - Column 1: Snapshot regression at year 2100.
      - Column 2: Snapshot regression at year 2300.
      - Rows: Different variables included in 'vars_include' (defaults to utils.vars_sce_dependent).
      - Fits a separate regression line for each scenario individually.
    """
    logger.info("Plotting Figure 6...")

    # Load configuration parameters
    fig_params = plotting_utils.fig_params
    colours = fig_params["colours"]
    plot_config = fig_params["plot_config"]

    l_font = plot_config.get("label_fontsize", 11)
    t_font = plot_config.get("subplot_title_fontsize", 12)
    leg_font = plot_config.get("legend_fontsize", 11)
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

    # Load vars_include from utils
    vars_include = utils.vars_sce_dependent

    # Calculate x_metric (default is Storage-years)
    ds_base = ds_tsi.sel(variant="base")
    x_metric = utils.calc_sy(ds_tsi, ds_base, years=ds_tsi.time.values)

    # Resolve output directory, file format, and resolution
    output_dir = Path(utils.FIGURE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_output_dir = Path(utils.DATA_OUTPUT_DIR)
    data_output_dir.mkdir(parents=True, exist_ok=True)

    file_format = fig_params.get("file_format", "pdf")
    fig_dpi = fig_params.get("fig_dpi")

    # Resolve time slice
    years = ds_tsi.time.values

    # Resolve active variants to plot
    active_variants = [v for v in utils.temp_variants if v != "base"]

    # Sort scenarios using the canonical order defined in utils
    scenarios = utils._sort_ssp(ds_tsi.scenario.values)

    # Retrieve line styles and SSP patch handles
    variant_styles, patch_handles = plotting_utils.get_plot_elements(
        ds_tsi, active_variants, scenarios
    )

    # Ensure patches are canonically sorted
    patch_handles = utils._sort_ssp(patch_handles)

    n_cols = len(vars_include)
    fig, axs = plt.subplots(2, n_cols, figsize=(10, 2.5 * n_cols), sharex="col")
    if n_cols == 1:
        axs = np.expand_dims(axs, axis=0)

    ds_diff = ds_tsi - ds_tsi.sel(variant="base")
    time_snapshots = [2100.5, 2300.5]

    # dicitonary to save figure data
    fig_data = {}

    # --- Plotting Loop ---
    for c, t_snap in enumerate(time_snapshots):
        for r, var in enumerate(vars_include):
            ax = axs[r, c]
            ax.axhline(0, color="grey", lw=0.8, alpha=0.3)

            global_x, global_y, global_colors = [], [], []

            # save data to 2300
            if t_snap == 2300.5:
                fig_data[var] = {"time": years}

            # Filter data to the specific time snapshot
            for v_name in active_variants:
                for sce in scenarios:
                    y_full = ds_diff[var].sel(scenario=sce, variant=v_name)
                    x_full = x_metric.sel(scenario=sce, variant=v_name)

                    try:
                        y_val = float(y_full.sel(time=t_snap, method="nearest"))
                        x_val = float(x_full.sel(time=t_snap, method="nearest"))

                        # Plot trajectories leading to snapshot
                        ax.plot(
                            x_full.sel(time=slice(None, t_snap)),
                            y_full.sel(time=slice(None, t_snap)),
                            color=colours[sce],
                            linestyle=variant_styles[v_name],
                            alpha=0.6,
                            lw=1,
                        )

                        # save data to 2300
                        if t_snap == 2300.5:
                            fig_data[var][f"{sce} - {v_name}"] = y_val

                        global_x.append(x_val)
                        global_y.append(y_val)
                        global_colors.append(colours[sce])
                    except KeyError:
                        continue

            # Plot snapshot markers
            ax.scatter(
                global_x,
                global_y,
                c=global_colors,
                marker=".",
                s=100,
                zorder=10,
                alpha=0.8,
            )

            # Scenario-dependent regression fitting and annotations
            if var in ["L_permacarb", "O_motmax"]:
                y_pos = 0.98
                for idx_sce, sce in enumerate(scenarios):
                    # Add a blank space after the 5th scenario label for O_motmax
                    if var == "O_motmax" and t_snap == 2300.5 and idx_sce == 4:
                        y_pos -= 0.48

                    idx = [
                        j
                        for j, color_ref in enumerate(global_colors)
                        if color_ref == colours[sce]
                    ]
                    if not idx:
                        continue

                    xs_sub = [global_x[j] for j in idx]
                    ys_sub = [global_y[j] for j in idx]
                    m, ci, r2, rmse = plotting_utils.plot_lr(  # noqa: RUF059
                        xs_sub,
                        ys_sub,
                        ax,
                        color=colours[sce],
                        fill_between=False,
                        line_width=1,
                    )

                    ax.text(
                        0.02,
                        y_pos,
                        f"SSP{sce[-3]}-{sce[-2]}.{sce[-1]} $R^2$:{r2:.3f}",
                        transform=ax.transAxes,
                        # color=colours[sce],
                        fontsize=l_font - 3,
                        # fontweight="bold",
                        va="top",
                    )
                    y_pos -= 0.06

            # Formatting titles and labels
            if r == 0:
                ax.set_title(f"Year {int(t_snap)}", fontweight="bold", fontsize=t_font)

            unit = ds_tsi[var].attrs.get("units", "units").replace("°C", r"$^\circ$C")
            name = ds_tsi[var].attrs.get("long_name", var).capitalize()
            ax.set_ylabel(
                "\n".join(textwrap.wrap(f"{name}\n({unit})", width=25)),
                fontsize=l_font,
            )

            if r == len(vars_include) - 1:
                ax.set_xlabel("Storage-years (Gt CO$_2$-yr)", fontsize=l_font - 1)
            ax.tick_params(labelsize=l_font - 2)

            # Letter labels for panels (e.g. A), B))
            panel_idx = r * 2 + c
            ax.text(
                0,
                1.1,
                f"{string.ascii_uppercase[panel_idx]})",
                transform=ax.transAxes,
                fontsize=t_font,
                fontweight="bold",
                va="top",
                ha="right",
            )

    # --- External Legend Placement ---
    clean_line_handles = plotting_utils.get_line_legend_handles(
        variant_styles, active_variants, no_titles=True
    )

    fig.legend(
        handles=patch_handles,
        loc="upper right",
        bbox_to_anchor=(0.51, 0.05),
        ncol=4,
        frameon=True,
        edgecolor=e_col,
        fontsize=leg_font - 2,
    )

    fig.legend(
        handles=clean_line_handles,
        loc="upper left",
        bbox_to_anchor=(0.53, 0.05),
        ncol=3,
        frameon=True,
        edgecolor=e_col,
        fontsize=leg_font - 2,
        handlelength=h_len,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    output_path = output_dir / f"figure6.{file_format}"
    plt.savefig(output_path, dpi=fig_dpi, bbox_inches="tight")
    plt.close()
    logger.info(f"Figure successfully saved to {output_path}")

    # save figure data
    for k, v in fig_data.items():
        pd.DataFrame(v).to_csv(data_output_dir / f"Fig6_{k}.csv", index=False)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    # Ensure directories are set up prior to running
    utils.setup_output_directories()
    generate_figure_6()
