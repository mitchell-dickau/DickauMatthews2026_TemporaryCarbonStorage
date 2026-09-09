import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea, VPacker

# Add parent directory to path so we can import utils and plotting_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import plotting_utils
import utils

logger = logging.getLogger(__name__)


def generate_si_figure_3() -> None:
    """
    Generates and saves SI Figure 3 of the paper (Two-Panel Regression Comparison).

    This figure shows the relationship between Degree-years and Storage-years:
      - Panel A: Global regressions across all scenarios for year 2100 (dashed black)
                 and year 2300 (solid black) with respective R^2 values in the legend.
      - Panel B: Global regression for year 2100 (dashed black) alongside separate
                 scenario ensemble regression lines for year 2300 (colored by scenario),
                 with scatter symbols only in the top legend and individual scenario
                 R^2 values annotated in the lower right corner with matching SSP colors.
    """
    logger.info("Plotting SI Figure 3 (Two-panel Degree-years vs Storage-years)...")

    # Load configuration parameters
    fig_params = plotting_utils.fig_params
    colours = fig_params["colours"]
    plot_config = fig_params["plot_config"]

    l_font = 10
    t_font = 11
    leg_font = 8.5
    e_col = plot_config.get("legend_edge_color", "black")

    # Load dataset
    try:
        ds_tsi, _ = utils.load_main_datasets()
    except FileNotFoundError as e:
        logger.error(
            f"Required data not found. Ensure files are in the data directory. {e}"
        )
        raise

    # Resolve output directory, file format, and resolution
    output_dir = Path(utils.FIGURE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_format = fig_params.get("file_format", "pdf")
    fig_dpi = fig_params.get("fig_dpi")

    # Calculate storage-years and degree-years
    years = ds_tsi.time.values
    ds_base = ds_tsi.sel(variant="base")
    storage_years = utils.calc_sy(ds_tsi, ds_base, years=years)
    degree_years = utils.calc_dy(ds_base, ds_tsi, years=years)

    # Resolve variants to plot
    active_vars = [v for v in utils.temp_variants if v != "base"]

    # Sort scenarios using the canonical order defined in utils
    scenarios = utils._sort_ssp(ds_tsi.scenario.values)

    # Retrieve SSP patch handles
    _, patch_handles = plotting_utils.get_plot_elements(ds_tsi, active_vars, scenarios)

    # Ensure patches are sorted
    patch_handles = utils._sort_ssp(patch_handles)

    # --- Setup Two-Panel Figure (Width reduced by ~25%) ---
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.5, 5.0), sharey=True)

    # --- Collect Data & Plot Snapshot Scatter Points for Both Panels ---
    x_2100, y_2100 = [], []
    x_2300, y_2300 = [], []
    x_2300_by_sce = {sce: [] for sce in scenarios}
    y_2300_by_sce = {sce: [] for sce in scenarios}

    # Reversing order so that the first items in the sorted lists (e.g., SSP1-1.9) plot last (on top)
    for var in active_vars[::-1]:
        for sce in scenarios[::-1]:
            sy = storage_years.sel(scenario=sce, variant=var)
            dy = degree_years.sel(scenario=sce, variant=var)

            # Snapshot Markers (Hollow with scenario edge color)
            sy_2100 = float(sy.sel(time=2100.5))
            dy_2100 = float(dy.sel(time=2100.5))
            x_2100.append(sy_2100)
            y_2100.append(dy_2100)

            for ax in [axA]:
                ax.scatter(
                    sy_2100,
                    dy_2100,
                    facecolors="none",
                    edgecolors=colours[sce],
                    marker="*",
                    s=70,
                    zorder=3,
                    linewidths=1.2,
                    alpha=0.9,
                )

            sy_2300 = float(sy.sel(time=2300.5, method="nearest"))
            dy_2300 = float(dy.sel(time=2300.5, method="nearest"))
            x_2300.append(sy_2300)
            y_2300.append(dy_2300)
            x_2300_by_sce[sce].append(sy_2300)
            y_2300_by_sce[sce].append(dy_2300)

            for ax in [axA, axB]:
                ax.scatter(
                    sy_2300,
                    dy_2300,
                    facecolors="none",
                    edgecolors=colours[sce],
                    marker="o",
                    s=60,
                    zorder=3,
                    linewidths=1.2,
                    alpha=0.9,
                )

    # --- Panel A Regressions (Global 2100 and Global 2300, on top with zorder=5) ---
    m2100 = sm.OLS(y_2100, x_2100).fit()
    r2_2100 = 1 - (m2100.ssr / m2100.centered_tss)
    X_plot_2100 = np.linspace(0, max(x_2100), 100)
    y_plot_2100 = m2100.predict(X_plot_2100)

    axA.plot(
        X_plot_2100,
        y_plot_2100,
        color="black",
        linestyle="--",
        lw=2.0,
        alpha=0.95,
        zorder=5,
    )

    fit_handles_A = [
        Line2D(
            [0],
            [0],
            marker="*",
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=1.2,
            label="2100",
            markersize=7,
            linestyle="None",
            alpha=0.9,
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            lw=1.8,
            label=f"2100 fit ($R^2 = {r2_2100:.4f}$)",
        ),
    ]

    if x_2300:
        m2300 = sm.OLS(y_2300, x_2300).fit()
        r2_2300 = 1 - (m2300.ssr / m2300.centered_tss)
        X_plot_2300 = np.linspace(0, max(x_2300), 100)
        y_plot_2300 = m2300.predict(X_plot_2300)

        axA.plot(
            X_plot_2300,
            y_plot_2300,
            color="black",
            linestyle="-",
            lw=2.0,
            alpha=0.9,
            zorder=5,
        )

        fit_handles_A.extend(
            [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    markerfacecolor="none",
                    markeredgecolor="black",
                    markeredgewidth=1.2,
                    label="2300",
                    markersize=7,
                    linestyle="None",
                    alpha=0.9,
                ),
                Line2D(
                    [0],
                    [0],
                    color="black",
                    linestyle="-",
                    lw=1.8,
                    alpha=0.9,
                    label=f"2300 fit ($R^2 = {r2_2300:.4f}$)",
                ),
            ]
        )

    # --- Panel B Regressions (Global 2100 and Scenario-specific 2300, on top with zorder=5) ---

    r2_by_sce = {}
    if x_2300:
        for sce in scenarios:
            xs = x_2300_by_sce[sce]
            ys = y_2300_by_sce[sce]
            m_sce = sm.OLS(ys, xs).fit()
            r2_sce = 1 - (m_sce.ssr / m_sce.centered_tss)
            r2_by_sce[sce] = r2_sce
            X_plot_sce = np.linspace(0, max(xs), 100)
            y_plot_sce = m_sce.predict(X_plot_sce)
            axB.plot(
                X_plot_sce,
                y_plot_sce,
                color=colours[sce],
                linestyle="-",
                lw=2.0,
                alpha=0.5,
                zorder=5,
            )

    # Panel B top legend: ONLY scatterplot symbols
    marker_handles_B = [
        Line2D(
            [0],
            [0],
            marker="o",
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=1.2,
            label="2300",
            markersize=7,
            linestyle="None",
            alpha=0.9,
        ),
    ]

    # --- Legends ---
    L_KWARGS = {
        "frameon": True,
        "edgecolor": e_col,
        "fontsize": leg_font,
        "facecolor": "white",
        "framealpha": 0.9,
    }

    # Panel A Legends
    leg_ssp_A = axA.legend(handles=patch_handles, loc="upper left", ncol=2, **L_KWARGS)
    axA.add_artist(leg_ssp_A)
    axA.legend(
        handles=fit_handles_A,
        loc="lower right",
        ncol=2,
        **L_KWARGS,
    )

    # Panel B Legends
    leg_ssp_B = axB.legend(handles=patch_handles, loc="upper left", ncol=2, **L_KWARGS)
    axB.add_artist(leg_ssp_B)
    axB.legend(
        handles=marker_handles_B,
        bbox_to_anchor=(0.0, 0.8),
        loc="upper left",
        ncol=2,
        **L_KWARGS,
    )

    # --- Panel B: Colored SSP R2 Annotations in Lower Right Corner ---
    if x_2300:
        col1_sces = scenarios[:4]
        col2_sces = scenarios[4:]

        col1_boxes = [
            TextArea(
                f"SSP{s[3]}-{s[4]}.{s[5]}: $R^2={r2_by_sce[s]:.4f}$",
                textprops={
                    "color": colours[s],
                    "fontsize": leg_font - 0.5,
                    "fontweight": "bold",
                },
            )
            for s in col1_sces
        ]
        col2_boxes = [
            TextArea(
                f"SSP{s[3]}-{s[4]}.{s[5]}: $R^2={r2_by_sce[s]:.4f}$",
                textprops={
                    "color": colours[s],
                    "fontsize": leg_font - 0.5,
                    "fontweight": "bold",
                },
            )
            for s in col2_sces
        ]

        vbox1 = VPacker(children=col1_boxes, align="left", pad=0, sep=2)
        vbox2 = VPacker(children=col2_boxes, align="left", pad=0, sep=2)
        hbox = HPacker(children=[vbox1, vbox2], align="baseline", pad=0, sep=10)
        full_vbox = VPacker(children=[hbox], align="left", pad=0, sep=3)

        anchored_box = AnchoredOffsetbox(
            loc="lower right",
            child=full_vbox,
            pad=0.4,
            frameon=True,
            bbox_to_anchor=(0.99, 0.01),
            bbox_transform=axB.transAxes,
            borderpad=0.3,
        )
        anchored_box.patch.set_boxstyle("round,pad=0.35")
        anchored_box.patch.set_facecolor("white")
        anchored_box.patch.set_edgecolor(e_col)
        anchored_box.patch.set_alpha(0.9)
        anchored_box.set_zorder(10)
        axB.add_artist(anchored_box)

    # --- Subplot Labels & Aesthetics ---
    for label, ax_ref in zip(["A)", "B)"], [axA, axB]):
        ax_ref.text(
            -0.13,
            1.04,
            label,
            transform=ax_ref.transAxes,
            fontsize=t_font,
            fontweight="bold",
            va="top",
        )
        ax_ref.set_xlabel("Storage-years (Gt CO$_2$-yr)", fontsize=l_font)
        ax_ref.tick_params(labelsize=l_font - 1, pad=6)

    axA.set_ylabel("Degree-years of avoided warming (°C-yr)", fontsize=l_font)

    fig.tight_layout()
    output_path = output_dir / f"si_figure3.{file_format}"
    fig.savefig(
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
    generate_si_figure_3()
