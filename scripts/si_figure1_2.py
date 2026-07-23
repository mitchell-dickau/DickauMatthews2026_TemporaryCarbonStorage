import logging
import os
import sys
from pathlib import Path
from typing import Optional

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr

# Add parent directory to path so we can import utils and plotting_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import plotting_utils  # noqa: E402
import utils  # noqa: E402

logger = logging.getLogger(__name__)


def generate_si_figure_1_2(
    ds: Optional[xr.Dataset] = None,
    output_dir: Optional[Path] = None,
    file_format: Optional[str] = None,
) -> None:
    """
    Generates and saves SI Figure 1-2 (Spatial Surface Air Temperature Difference).

    This figure plots a grid of maps showing the spatial differences in GSAT (A_sat)
    for 6 variants (sce1 to sce6, as rows) and 8 scenarios (columns), projected
    onto a Robinson map projection.

    Parameters
    ----------
    ds : xr.Dataset, optional
        Dataset containing spatial differences. If not provided, it will be loaded from
        the default file (A_sat_diff.nc).
    output_dir : Path, optional
        Directory where the figure will be saved. Defaults to utils.FIGURE_DIR.
    file_format : str, optional
        Format of the output file (e.g., 'pdf', 'png'). Defaults to fig_params['file_format'].
    """
    logger.info("Reading in spatial diff data from A_sat_diff.nc...")

    # Load dataset if not provided
    if ds is None:
        try:
            ds = xr.open_dataset(utils.DATA_DIR / "A_sat_diff.nc")
        except FileNotFoundError as e:
            logger.error(f"Required spatial difference dataset not found. {e}")
            raise

    fig_params = plotting_utils.fig_params

    if output_dir is None:
        output_dir = utils.FIGURE_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if file_format is None:
        file_format = fig_params.get("file_format", "pdf")

    scenarios = utils._sort_ssp(ds.scenario.values)

    # Assuming the NetCDF has 'target_year' and 'midpoint_year' coordinates
    for yr, midpoint_yr in zip(ds.target_year.values, ds.midpoint_year.values):
        logger.info(f"Plotting spatial A_sat diff for year {yr}....")

        # Subplots grid mapping 6 variants (rows) and 8 scenarios (columns)
        fig, axs = plt.subplots(
            6,
            8,
            subplot_kw={"projection": ccrs.Robinson(central_longitude=0.0)},
            figsize=(35, 15),
        )

        for c, ssp in enumerate(scenarios):
            logger.info(f"Plotting {ssp}...")
            for r, variant in enumerate(ds.variant.values[:6]):
                ax = axs[r, c]
                ax.set_global()

                da = ds.A_sat.sel(scenario=ssp, target_year=yr, variant=variant)
                plot1 = da.plot(
                    ax=ax,
                    vmin=-0.06,
                    vmax=0.06,
                    cmap="RdBu_r",
                    add_colorbar=False,
                    transform=ccrs.PlateCarree(),
                )
                ax.coastlines()
                ax.set_title(
                    f"SSP{ssp[3]}-{ssp[-2]}.{ssp[-1]} - Storage {variant[-1]}",
                    fontsize=12,
                )

        fig.subplots_adjust(right=0.85, hspace=0)
        cbar_ax = fig.add_axes([0.9, 0.25, 0.01, 0.5])  # [left, bottom, width, height]
        cb = fig.colorbar(plot1, cax=cbar_ax)
        cb.ax.tick_params(labelsize=15)

        # Extract limits and apply custom labels to colorbar
        vmin, vmax = plot1.get_clim()
        raw_ticks = cb.get_ticks()
        valid_ticks = [t for t in raw_ticks if vmin <= t <= vmax]
        cb.ax.set_yticks(valid_ticks)
        cb.ax.set_yticklabels([f"{t:.2f}°C" for t in valid_ticks])

        # Subplot Titles and Row Labels
        for c, ssp in enumerate(scenarios):
            axs[0, c].text(
                0.5,
                1.2,
                f"SSP{ssp[3]}-{ssp[-2]}.{ssp[-1]}",
                transform=axs[0, c].transAxes,
                fontsize=16,
                fontweight="bold",
                va="bottom",
                ha="center",
            )

        for r, key in enumerate(["sce1", "sce2", "sce3", "sce4", "sce5", "sce6"]):
            axs[r, 0].text(
                -0.1,
                0.5,
                f"Storage {key[-1]}",
                transform=axs[r, 0].transAxes,
                fontsize=16,
                fontweight="bold",
                va="center",
                ha="right",
                rotation=90,
            )

        logger.info("Saving figure...")
        if yr == 2100.5:
            output_path = output_dir / f"si_figure1.{file_format}"
        if yr == 2300.5:
            output_path = output_dir / f"si_figure2.{file_format}"
        fig.savefig(
            output_path,
            format=file_format,
            dpi=500,
            bbox_inches="tight",
            pad_inches=0.1,
        )
        plt.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    # Ensure directories are set up prior to running
    utils.setup_output_directories()
    generate_si_figure_1_2()
