# Temporary carbon storage can mitigate slow-responding climate changes

**Author:** Mitchell Dickau  
**Email:** mitchell.dickau@mail.concordia.ca , dickaumitch@gmail.com

Repository containing the data processing and visualization code for the manuscript:  
*Dickau & Matthews (2026). Temporary carbon storage can mitigate slow-responding climate changes.*

## Overview
This repository contains the Python scripts required to process UVic-ESCM climate model outputs and generate the figures presented in the manuscript. The analysis investigates the extent to which temporary carbon storage (and subsequent release) can mitigate slow-responding (irreversible or lag-response) climate system variables compared to permanent carbon dioxide removal (CDR) and baseline overshoot scenarios.

## Citation
If you use this code or data in your research, please cite the associated paper:
> Dickau, M., and Matthews, H. D. (2026). Temporary carbon storage can mitigate slow-responding climate changes. Nature Communications (In Review).

### Data Dictionary
Ensure the `data/` directory is populated with the following NetCDF input files before running the analysis:

| File Name | Description |
| :--- | :--- |
| `tsi_data.nc` | Main time-series outputs for baseline, temporary storage, and permanent removal scenarios. |
| `tsi_hist.nc` | Historical simulation outputs used for calculating pre-industrial baseline anomalies. |
| `tavg_A_sat.nc` | Spatial-temporal gridded surface air temperature dataset used to compute regional differences. |

## Installation and Setup
The analysis was performed using Python 3.11.8. It is highly recommended to run this code within a virtual environment.

1. **Create and activate a virtual environment:**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows use: env\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Running the Analysis and making figures:**
   The figure scripts rely on shared utility functions located in `utils.py` and styling variables in `plotting_utils.py`.

   * **Run the Entire Pipeline:**
     To run the entire analysis sequentially—including all processing, regressions, and figure plotting—execute the master pipeline script:
     ```bash
     python scripts/run_analysis.py
     ```

   * **Run Individual Steps (Optional):**
     Alternatively, you can run individual parts of the analysis or create specific plots independently:

     **1. Pre-processing & Analysis Steps:**
     ```bash
     # Calculate spatial surface air temperature difference maps (saves A_sat_diff.nc)
     python scripts/calc_asat_spatial_diff.py

     # Compute relative avoided change values (saves avoided_change_results.csv)
     python scripts/avoided_change_analysis.py
     ```

     **2. Generate Individual Figures:**
     ```bash
     # Manuscript Figures
     python scripts/figure1.py
     python scripts/figure2.py
     python scripts/figure3.py
     python scripts/figure4.py
     python scripts/figure5.py
     python scripts/figure6.py
     python scripts/figure7.py
     python scripts/figure8.py

     # Supplementary Information Figures
     python scripts/si_figure1_2.py  # Plots SI Figures 1 & 2 (requires calc_asat_spatial_diff.py)
     python scripts/si_figure3.py    # Plots SI Figure 3 (requires avoided_change_analysis.py)
     ```

4. **Outputs:**
   * Generated figures will be saved to `analysis_output/figures/`.
   * Spatial difference NetCDF datasets and regression CSV files will be saved directly to `analysis_output/`.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

**Disclaimer:** This code is provided "as-is" and is specifically intended to reproduce the results and figures presented in the manuscript. It is shared for the purposes of scientific transparency and reproducibility, and is not actively maintained as a general-purpose software package.
