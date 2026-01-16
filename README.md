# On Strategic Conduct Under Emissions Caps

This repository contains the full codebase and datasets used in the bachelor’s thesis  
**“On Strategic Conduct Under Emissions Caps: A Game-Theoretic Model of Price Formation in the EU Emissions Trading System.”**

The project studies how departures from price-taking behavior affect emissions, permit prices, and price dynamics in the EU Emissions Trading System (EU ETS), using a reduced-form repeated-game framework calibrated to Phase III data.

All results in the thesis are fully reproducible using the scripts provided here.

## Repository Overview

The repository implements the complete empirical and numerical pipeline:

- Cleaning and construction of EU ETS price and emissions data  
- Estimation of a reduced-form permit pricing rule  
- Extraction and estimation of a stochastic shock process  
- Calibration of a quadratic abatement technology under the competitive benchmark  
- Simulation of emissions and prices under varying degrees of strategic conduct  

All raw data are included, and all intermediate objects are generated programmatically.

## Repository Structure

.
├── scripts/                 # Main scripts (run in order)
│   ├── 00_config.py
│   ├── 01_clean_and_merge_data.py
│   ├── 02_estimate_price_rule.py
│   ├── 03_construct_shock_and_estimate_ar1.py
│   ├── 04_calibrate_quadratic_parameters.py
│   ├── 05_simulate_replay_and_montecarlo.py
│   └── 06_export_ch4_inputs_tables.py
│
├── data_clean/              # Cleaned datasets (generated)
├── objects/                 # Estimated parameters and model objects (generated)
├── sim_replay/              # Shock-replay simulation outputs (generated)
├── sim_mc/                  # Monte Carlo simulation outputs (generated)
├── tables_ch4/              # LaTeX tables for Chapter 4 (generated)
├── logs/                    # Optional logs
│
├── Carbon_Emissions_Futures_Historical_Data.csv
├── Historical_emissions_.csv
└── README.md

## Data

The analysis uses two datasets:

- **EU ETS allowance futures prices**  
  Daily EUA futures prices for Phase III, sourced from Investing.com.

- **EU ETS verified emissions**  
  Annual verified emissions data from the European Environment Agency.

No external downloads are required; all raw data are included in this repository.

## Script Workflow

Scripts should be run **in numerical order**.

### `00_config.py`
Defines fixed model and simulation settings (sample window, number of firms, discounting, seeds) and creates all required directories.

### `01_clean_and_merge_data.py`
Cleans raw price and emissions data, constructs the daily Phase III panel, and computes the reference scarcity level used as a cap proxy.

### `02_estimate_price_rule.py`
Estimates the reduced-form pricing rule  
p_t = α₀ + α₁ (E_t − Ē) + ε_t  
imposing the model restriction α₁ > 0. A descriptive OLS regression is reported for reference.

### `03_construct_shock_and_estimate_ar1.py`
Constructs the daily shock series as pricing-rule residuals and estimates an AR(1) shock process. The persistence parameter is clamped to ensure stationarity.

### `04_calibrate_quadratic_parameters.py`
Calibrates quadratic technology parameters under the competitive benchmark (σ = 0) using closed-form equilibrium expressions and price volatility matching.

### `05_simulate_replay_and_montecarlo.py`
Runs the main simulations:
- Shock replay using the empirical shock path
- Monte Carlo simulations using synthetic shock paths

Outputs emissions, prices, cost measures, summary statistics, and figures.

### `06_export_ch4_inputs_tables.py`
Exports LaTeX tables summarizing data moments, estimated parameters, and calibrated model inputs used in Chapter 4 of the thesis.

## Reproducibility

- All random seeds are fixed  
- All intermediate objects are saved to disk  
- The simulation pipeline is deterministic  
- Results can be replicated exactly by running the scripts in order  

The code relies only on standard Python scientific libraries (`numpy`, `pandas`, `statsmodels`, `scipy`, `matplotlib`).

## Notes

- The model is intentionally reduced-form and designed for transparency and calibration rather than structural estimation.
- Permit trading is not modeled explicitly; prices are linked to emissions through a maintained pricing rule consistent with EU ETS scarcity logic.
- Strategic behavior is captured by a conduct parameter that measures partial internalization of perceived price effects.

## Citation

If you use this code or data, please cite the associated thesis:

Villiam Axtelius and Nora Gullhav (2025)  
*On Strategic Conduct Under Emissions Caps: A Game-Theoretic Model of Price Formation in the EU Emissions Trading System*  
Bachelor’s Thesis, University of Gothenburg
