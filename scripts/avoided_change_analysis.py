import logging
import os
import sys
from pathlib import Path

import pandas as pd

# Add parent directory to path so we can import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import utils

logger = logging.getLogger(__name__)


def run_avoided_change_analysis() -> pd.DataFrame:
    """
    Calcualtes relative avoided change of temporary carbon storage scenarios
    relative to reference SSPs.

    For each variable, year, scenario, and variant, it calculates the percentage of change
    avoided relative to the baseline scenario (no storage):

        avoided_change (%) = ((e_v_b - v_r) / abs_c) * 100

    where:
      - s_v: baseline value at starting year 2015.5
      - e_v_b: baseline value at snapshot year
      - v_r: variant value at snapshot year
      - abs_c: baseline absolute change (e_v_b - s_v)

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame containing the analysis results.
    """
    logger.info("Running avoided change analysis...")

    # Load dataset
    try:
        ds_tsi, _ = utils.load_main_datasets()
    except FileNotFoundError as e:
        logger.error(
            f"Required data not found. Ensure files are in the data directory. {e}"
        )
        raise

    # Default to all irreversibility variables
    vars_include = utils.vars_irrev

    # Default to utils.OUTPUT_DIR
    data_output_dir = Path(utils.DATA_OUTPUT_DIR)
    data_output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = utils._sort_ssp(ds_tsi.scenario.values)
    all_variants = [v for v in ds_tsi.variant.values if v != "base"]
    time_snapshots = [2100.5, 2300.5]

    records = []

    # loop through time snapshots, variables, and scenarios to calculate avoided change
    for yr in time_snapshots:
        for var in vars_include:
            for ssp in scenarios:
                # Baseline start value at 2015.5
                s_v = float(ds_tsi[var].sel(scenario=ssp, variant="base", time=2015.5))
                # Baseline end value at snapshot year
                e_v_b = float(ds_tsi[var].sel(scenario=ssp, variant="base", time=yr))
                # Absolute change under baseline
                abs_c = e_v_b - s_v

                record = {
                    "year": int(yr),
                    "variable": var,
                    "scenario": ssp,
                }

                for v_n in all_variants:
                    v_r = float(ds_tsi[var].sel(scenario=ssp, variant=v_n, time=yr))
                    # Avoid division by zero
                    try:
                        val = (e_v_b - v_r) / abs_c * 100
                        record[v_n] = val
                    except ZeroDivisionError as e:
                        print(f"Zero division error: {e}")

                records.append(record)

    df_results = pd.DataFrame(records)

    # Reorder columns to have year, variable, scenario first, followed by variants
    cols_order = ["year", "variable", "scenario"] + all_variants
    df_results = df_results[cols_order]

    # Save to CSV
    output_path = data_output_dir / "Fig7_avoided_change_results.csv"
    df_results.to_csv(output_path, index=False)
    logger.info(f"Avoided change analysis results saved to {output_path}")

    return df_results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    run_avoided_change_analysis()
