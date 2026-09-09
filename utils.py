from pathlib import Path

import numpy as np
import xarray as xr

#### GLOBAL PATHS AND SETTINGS ####
# Since utils.py is in the root, its parent is the Project Root
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "analysis_output"
FIGURE_DIR = OUTPUT_DIR / "figures"
DATA_OUTPUT_DIR = OUTPUT_DIR / "figure_data"

# variable declaration
vars_irrev = ["L_permacarb", "O_dsealev", "O_motmax", "O_temp", "O_totcarb"]
vars_sce_independent = ["O_dsealev", "O_temp", "O_totcarb"]
vars_sce_dependent = [
    "L_permacarb",
    "O_motmax",
]

# temporary storage variants of SSPs + baseline (referecence) SSP
temp_variants = [
    "base",
    "sce1",
    "sce2",
    "sce3",
    "sce4",
    "sce5",
    "sce6",
]
# permanent removal variants
perm_variants = ["sce7", "sce8", "sce9"]


# funciton to set up output direcetories
def setup_output_directories():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(FIGURE_DIR).mkdir(parents=True, exist_ok=True)
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)


#### HELPER FUNCTIONS ####


# load data function
def load_main_datasets():
    """Loads the heavy NetCDF files for timeseries/spatial plots."""
    ds_tsi = xr.open_dataset(DATA_DIR / "tsi_data.nc", decode_times=False)
    ds_hist = xr.open_dataset(DATA_DIR / "tsi_hist.nc", decode_times=False)
    return ds_tsi, ds_hist


# sort SSPs function
def _sort_ssp(items):
    """Sorts scenarios or legend handles based on the paper's canonical SSP order."""
    order = ["119", "126", "534", "434", "245", "460", "370", "585"]

    def get_index(item):
        # Handle either a string (scenario) or a patch object (legend handle)
        label = item if isinstance(item, (str, np.str_)) else item.get_label()
        # Normalize labels like 'SSP1-1.9' or 'ssp119' into '119'
        clean_label = label.replace("-", "").replace(".", "")
        for i, code in enumerate(order):
            if code in clean_label:
                return i
        return 999  # Fallback for unexpected labels

    return sorted(items, key=get_index)


# calculate historical anomaly
def calculate_historical_anomaly(ds_hist):
    """Computes the pre-industrial temperature baseline (years < 1901)."""
    pre_ind_temp = ds_hist.A_sat.where(ds_hist.time < 1901, drop=True)
    return pre_ind_temp.mean().item()


# calculate cumulative emissions
def calc_cumulative(ds, var_name="F_co2emit", years=None):
    """
    Calculates the cumulative integral of emissions over time.
    Args:
        ds: xarray Dataset or DataArray.
        var_name: The variable to integrate (ignored if ds is a DataArray).
        years: Optional array of time points. Defaults to ds.time.
    Returns:
        xr.DataArray/Dataset: The cumulative sum, preserving all dimensions.
    """
    # 1. Handle input types (Dataset vs DataArray)
    da = ds[var_name] if isinstance(ds, xr.Dataset) else ds

    # 2. Coordinate handling
    if years is None:
        years = da.time

    # 3. Calculate the time step (dt)
    # Using np.gradient or diff ensures we handle the spacing correctly
    dt = np.gradient(years)

    # Convert dt to an xarray object so it aligns automatically during multiplication
    dt_da = xr.DataArray(dt, coords={"time": da.time}, dims=["time"])

    # 4. Integrate: (Value * dt) then cumulative sum
    # This replaces the need for scipy.integrate.cumulative_trapezoid
    cumulative = (da * dt_da).cumsum(dim="time")

    # 5. Metadata cleanup
    cumulative.attrs.update(
        {
            "units": "Gt CO2",
            "long_name": f"Cumulative {da.attrs.get('long_name', var_name)}",
        }
    )

    return cumulative


# calculate storage years
def calc_sy(ds, ds_base, var_name="F_co2emit", years=None):
    """
    Calculates Storage-Years: The integral of the difference in
    cumulative emissions between a baseline and a variant.
    """
    # 1. Calculate cumulative emissions for both (Mass of Carbon)
    ce_variant = calc_cumulative(ds, var_name, years)
    ce_base = calc_cumulative(ds_base, var_name, years)

    # 2. Calculate the avoided carbon burden (The difference)
    # Xarray handles the broadcasting: ce_base is subtracted from
    # every variant in ce_variant automatically.
    avoided_burden = ce_base - ce_variant

    # 3. Integrate the avoided burden over time (Mass * Time)
    sy = calc_cumulative(avoided_burden, years=years)

    sy.attrs.update({"units": "Gt CO2-yr", "long_name": "Storage-Years"})
    return sy


# calculate degree years
def calc_dy(ds_base, ds_variant, var_name="A_sat", years=None):
    """
    Calculates Degree-Years: The integral of the temperature difference
    between a baseline and a variant.
    """
    # 1. Calculate the temperature difference (Cooling effect)
    temp_diff = ds_base[var_name] - ds_variant[var_name]

    # 2. Integrate the difference over time
    dy = calc_cumulative(temp_diff, years=years)

    dy.attrs.update({"units": "°C-yr", "long_name": "Degree-Years"})
    return dy
