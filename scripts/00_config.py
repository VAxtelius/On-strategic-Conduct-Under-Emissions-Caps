"""
00_config.py

Defines all fixed constants for Chapter 4 and writes objects/config.json.
This script must be run first.

Model period: 1 trading day.
Phase III window: 2013-01-02 to 2020-12-31.
"""

from __future__ import annotations

import json
from pathlib import Path
from math import pow

def main() -> None:
    # Project root = "THESIS PAPER/"
    base_dir = Path(__file__).resolve().parents[1]

    # Output directories (created automatically)
    data_clean = base_dir / "data_clean"
    objects_dir = base_dir / "objects"
    sim_replay = base_dir / "sim_replay"
    sim_mc = base_dir / "sim_mc"
    tables_ch4 = base_dir / "tables_ch4"
    logs_dir = base_dir / "logs"

    for d in [data_clean, objects_dir, sim_replay, sim_mc, tables_ch4, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Fixed sample window (Phase III)
    start_date = "2013-01-02"
    end_date = "2020-12-31"

    # Fixed model design choices
    N = 10
    sigma_grid = [round(x / 10, 1) for x in range(0, 11)]
    M = 5000
    burn_in = 252
    seed = 20260102

    # Discounting
    annual_discount_rate = 0.03
    trading_days_per_year = 252
    beta = pow(1.0 / (1.0 + annual_discount_rate), 1.0 / trading_days_per_year)

    # Scaling for emissions
    emissions_scale = 1e9

    # ✅ Raw data location (your actual folder structure)
    raw_dir = base_dir / "Simulations" / "Data"

    price_filename = "Carbon_Emissions_Futures_Historical_Data.csv"
    emissions_filename = "Historical_emissions_.csv"

    config = {
        "paths": {
            "base_dir": str(base_dir),
            "data_clean": str(data_clean),
            "objects": str(objects_dir),
            "sim_replay": str(sim_replay),
            "sim_mc": str(sim_mc),
            "tables_ch4": str(tables_ch4),
            "logs": str(logs_dir),
        },
        "raw_files": {
            "price_csv": str(raw_dir / price_filename),
            "emissions_csv": str(raw_dir / emissions_filename),
        },
        "sample": {
            "start_date": start_date,
            "end_date": end_date,
            "years": list(range(2013, 2021)),
        },
        "model": {
            "N": N,
            "sigma_grid": sigma_grid,
            "emissions_scale": emissions_scale,
            "annual_discount_rate": annual_discount_rate,
            "trading_days_per_year": trading_days_per_year,
            "beta": beta,
            "burn_in": burn_in,
            "seed": seed,
            "M": M,
        },
    }

    out_path = objects_dir / "config.json"
    out_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"[00_config] Wrote config to: {out_path}")
    print("[00_config] Using raw data files:")
    print(f"  - {raw_dir / price_filename}")
    print(f"  - {raw_dir / emissions_filename}")

if __name__ == "__main__":
    main()
