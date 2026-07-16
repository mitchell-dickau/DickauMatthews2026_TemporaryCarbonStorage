import logging
import os
import sys
from pathlib import Path
from typing import Optional

import xarray as xr

# Add parent directory to path so we can import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import utils  # noqa: E402

logger = logging.getLogger(__name__)


def calculate_spatial_asat_diff(
    data: Optional[xr.Dataset] = None,
    save_path: Optional[Path] = None,
    var: str = "A_sat",
) -> None:
    """
    Computes the spatial difference between scenario variants and a baseline
    for specified target years, averaged over a given interval.
    Time slicing dynamically adjusts based on the target year (leading, trailing, or centered).

    Parameters
    ----------
    data : xr.Dataset, optional
        Dataset containing spatial variables (e.g. tavg_A_sat.nc).
        If not provided, it will be loaded from the default path with decode_times=False.
    save_path : Path, optional
        Directory where A_sat_diff.nc will be saved. Defaults to utils.OUTPUT_DIR.
    var : str, optional
        Variable to analyze. Defaults to "A_sat".
    """
    # Resolve default paths
    if save_path is None:
        save_path = utils.OUTPUT_DIR
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    file_path = save_path / "A_sat_diff.nc"

    # Load dataset if not provided (using decode_times=False to handle years calendar)
    if data is None:
        try:
            data = xr.open_dataset(utils.DATA_DIR / "tavg_A_sat.nc", decode_times=False)
        except FileNotFoundError as e:
            logger.error(
                f"Required spatial temperature dataset 'tavg_A_sat.nc' not found. {e}"
            )
            raise

    target_years = [2100.5, 2300.5]
    interval = 20

    logger.info(
        f"Calculating spatial differences for '{var}' across years: {target_years} with a {interval}-year interval."
    )

    if var not in data.variables:
        raise ValueError(f"Variable '{var}' not found in the dataset.")

    da = data[var]
    mean_list = []

    for target_year in target_years:
        # Custom Boundary Logic based on target_year
        if target_year == 2100.5:
            # interval years AFTER 2100.5
            start = 2100.5
            end = start + interval
            midpoint = start + (interval / 2)

        elif target_year == 2300.5:
            # interval years LEADING UP TO 2300
            start = 2300.5 - interval + 1
            end = 2300.5
            midpoint = 2300.5 - (interval / 2)

        # Slice and average over time
        time_slice = da.sel(time=slice(start, end))
        year_mean = time_slice.mean(dim="time")

        # Add Nominal Dimension and Midpoint Coordinate
        year_mean = year_mean.expand_dims(target_year=[target_year])
        year_mean = year_mean.assign_coords(midpoint_year=("target_year", [midpoint]))

        mean_list.append(year_mean)

    # Combine all target years into a single master DataArray
    final_da = xr.concat(mean_list, dim="target_year")

    # Vectorized Difference Calculation
    base_da = final_da.sel(variant="base")
    diff_da = final_da - base_da
    diff_da = diff_da.drop_sel(variant="base")

    # Convert back into a Dataset
    diff_ds = diff_da.to_dataset(name=var)

    # --- THE METADATA NOTES ---
    # Injecting global attributes into the dataset
    diff_ds.attrs["description"] = (
        f"Spatial difference of {var} from the baseline scenario."
    )
    diff_ds.attrs["averaging_interval_years"] = interval
    diff_ds.attrs["target_years_nominal"] = ", ".join(map(str, target_years))
    diff_ds.attrs["time_slice_methodology"] = (
        "2101 uses trailing years. 2300 uses leading years. "
    )

    # Injecting specific metadata into the variable itself
    diff_ds[var].attrs["long_name"] = (
        f"{var} diff from baseline averaged over {interval} years"
    )
    # --------------------------

    # Explicitly assign the necessary plotting coordinates (enables Panoply compatibility)
    coords_to_add = {}
    for coord in ["longitude_edges", "latitude_edges"]:
        if coord in data.variables:
            coords_to_add[coord] = data[coord]

    if coords_to_add:
        diff_ds = diff_ds.assign_coords(coords_to_add)

    # Save to disk
    diff_ds.to_netcdf(path=file_path)
    logger.info(f"Spatial differences calculated and saved to {file_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    calculate_spatial_asat_diff()
