import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.lines import Line2D

# Add parent directory to path so we can import utils and plotting_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import plotting_utils  # noqa: E402
import utils  # noqa: E402

logger = logging.getLogger(__name__)


def generate_figure_3(
    ds_tsi: Optional[xr.Dataset] = None,
    ds_hist: Optional[xr.Dataset] = None,
    anomaly_base: Optional[float] = None,
    output_dir: Optional[Path] = None,
    file_format: Optional[str] = None,
    variants_to_exclude: Optional[List[str]] = plotting_utils.fig_params.get(
        "perm_variants"
    ),
) -> None:
    """
    Generates and saves Figure 3 of the paper (Peak Warming difference).

    This figure shows:
      - Peak warming difference relative to baseline (°C) for each scenario and active variant.
      - Uses horizontal jittering to separate points for different variants.

    Parameters
    ----------
    ds_tsi : xr.Dataset, optional
        The main timeseries dataset. If not provided, it will be loaded from the default path.
    ds_hist : xr.Dataset, optional
        The historical dataset used for calculating the temperature baseline.
        If not provided, it will be loaded from the default path.
    anomaly_base : float, optional
        The pre-industrial baseline temperature anomaly. If not provided, it will be calculated.
    output_dir : Path, optional
        Directory where the figure will be saved. Defaults to utils.FIGURE_DIR.
    file_format : str, optional
        Format of the output file (e.g., 'pdf', 'png'). Defaults to fig_params['file_format'].
    variants_to_exclude : list of str, optional
        List of variants to exclude from plotting. Defaults to permanent variants.
    """
    logger.info("Plotting Figure 3...")

    # Load configuration parameters
    fig_params = plotting_utils.fig_params
    colours = fig_params["colours"]
    plot_config = fig_params["plot_config"]

    l_font = plot_config.get("label_fontsize", 11)
    leg_font = plot_config.get("legend_fontsize", 11) - 2
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
    if anomaly_base is None:
        anomaly_base = utils.calculate_historical_anomaly(ds_hist)

    # Resolve output directory, file format, and resolution
    if output_dir is None:
        output_dir = utils.FIGURE_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if file_format is None:
        file_format = fig_params.get("file_format", "pdf")
    fig_dpi = fig_params.get("fig_dpi", 300)

    # Sort scenarios using the order defined in utils
    scenarios_to_plot = utils._sort_ssp(ds_tsi.scenario.values)

    # Resolve active variants to plot
    active_variants = [
        v
        for v in ds_tsi.variant.values
        if v not in (variants_to_exclude or []) and v != "base"
    ]

    # Map each variant to a marker
    all_variants_no_base = [v for v in fig_params["temp_variants"] if v != "base"]
    markers = ["o", "s", "v", "^", "D", "*", "1", "3", "x"]
    marker_map = {v: m for v, m in zip(all_variants_no_base, markers)}

    # Initialize plotting
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    # Set grid for orientation
    ax.grid(True, which="both", axis="y", ls="-", alpha=0.1)
    ax.axhline(0, color="black", lw=1.2, alpha=0.5, zorder=1)

    # Pre-calculate jitter offsets based on the number of active variants
    jitter_offsets = np.linspace(-0.4, 0.4, num=len(active_variants))

    # Plotting loop
    for x_idx, sce in enumerate(scenarios_to_plot):
        base_peak = (
            ds_tsi.sel(variant="base", scenario=sce).A_sat.values.max() - anomaly_base
        )
        for v_idx, vari in enumerate(active_variants):
            # Apply the specific offset for this variant to the scenario's base x-index
            x_pos = x_idx + jitter_offsets[v_idx]

            ax.scatter(
                x_pos,
                (
                    ds_tsi.sel(variant=vari, scenario=sce).A_sat.values.max()
                    - anomaly_base
                )
                - base_peak,
                color=colours[sce],
                marker=marker_map[vari],
                s=65,
                alpha=0.75,
                linewidths=1.5,
            )

    ax.set_ylabel("Peak warming relative to baseline (°C)", fontsize=l_font)

    # --- Formatting X-ticks ---
    formatted_labels = [
        f"SSP{s[3]}-{s[4]}.{s[5:]}"
        if str(s).startswith("ssp") and len(str(s)) >= 6
        else s
        for s in scenarios_to_plot
    ]

    ax.set_xticks(range(len(scenarios_to_plot)))

    # Set labels and rotate them vertically
    ax.set_xticklabels(
        formatted_labels, fontsize=l_font, fontweight="bold", rotation=90
    )

    # Color the x-axis tick labels to match their respective scenario colors
    for tick_label, sce in zip(ax.get_xticklabels(), scenarios_to_plot):
        tick_label.set_color(colours[sce])

    # Variant Legend
    sh = []
    for v in active_variants:
        sh.append(
            Line2D(
                [0],
                [0],
                marker=marker_map[v],
                color="black",
                linestyle="None",
                label=f"Storage {v[-1]}",
                markersize=8,
            )
        )

    ax.legend(
        handles=sh,
        loc="lower right",
        bbox_to_anchor=(1, 0),
        frameon=True,
        edgecolor=e_col,
        ncol=3,
        fontsize=leg_font,
    )

    output_path = output_dir / f"figure3.{file_format}"
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
    generate_figure_3()
