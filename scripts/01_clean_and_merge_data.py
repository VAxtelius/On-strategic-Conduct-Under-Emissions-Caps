"""
01_clean_and_merge_data.py

Reads the two raw datasets, cleans them, constructs the daily panel for Phase III,
and writes cleaned CSV outputs + barE_gt.json.

Outputs:
- data_clean/price_daily_phaseIII.csv
- data_clean/emissions_annual_phaseIII.csv
- data_clean/panel_daily_phaseIII.csv
- objects/annual_targets.csv
- objects/barE_gt.json
"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import numpy as np

def _load_config(base_dir: Path) -> dict:
    cfg_path = base_dir / "objects" / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError("Missing objects/config.json. Run scripts/00_config.py first.")
    return json.loads(cfg_path.read_text(encoding="utf-8"))

def _parse_price_to_float(x) -> float:
    # Investing.com exports are typically numeric already, but keep robust parsing.
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    s = s.replace(",", "")  # remove thousand separators if any
    return float(s)

def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    cfg = _load_config(base_dir)

    price_csv = Path(cfg["raw_files"]["price_csv"])
    emissions_csv = Path(cfg["raw_files"]["emissions_csv"])
    start_date = cfg["sample"]["start_date"]
    end_date = cfg["sample"]["end_date"]
    years = set(cfg["sample"]["years"])
    scale = float(cfg["model"]["emissions_scale"])

    data_clean = Path(cfg["paths"]["data_clean"])
    objects_dir = Path(cfg["paths"]["objects"])

    if not price_csv.exists():
        raise FileNotFoundError(f"Price CSV not found: {price_csv}")
    if not emissions_csv.exists():
        raise FileNotFoundError(f"Emissions CSV not found: {emissions_csv}")

    # --- Prices (daily) ---
    dfp = pd.read_csv(price_csv, sep=";")
    if "Date" not in dfp.columns or "Price" not in dfp.columns:
        raise ValueError("Price CSV must contain columns 'Date' and 'Price'.")

    dfp["Date"] = pd.to_datetime(dfp["Date"], format="%m/%d/%Y", errors="raise")
    dfp["Price"] = dfp["Price"].apply(_parse_price_to_float)
    dfp = dfp.sort_values("Date").reset_index(drop=True)

    dfp = dfp[(dfp["Date"] >= pd.to_datetime(start_date)) & (dfp["Date"] <= pd.to_datetime(end_date))].copy()
    dfp["Year"] = dfp["Date"].dt.year.astype(int)

    # enforce Phase III years only
    dfp = dfp[dfp["Year"].isin(years)].copy()

    # Export daily price
    out_price = data_clean / "price_daily_phaseIII.csv"
    dfp_out = dfp[["Date", "Year", "Price"]].copy()
    dfp_out["Date"] = dfp_out["Date"].dt.strftime("%Y-%m-%d")
    dfp_out.to_csv(out_price, index=False)

    # --- Emissions (annual) ---
    dfe = pd.read_csv(emissions_csv, sep=";")
    if "Year" not in dfe.columns:
        raise ValueError("Emissions CSV must contain a 'Year' column.")
    # Column name in your file:
    if "Verified emissions" not in dfe.columns:
        raise ValueError("Emissions CSV must contain 'Verified emissions' column.")

    dfe["Year"] = dfe["Year"].astype(int)
    dfe = dfe[dfe["Year"].isin(years)].copy()
    dfe = dfe.sort_values("Year").reset_index(drop=True)

    dfe["Emissions_Gt"] = dfe["Verified emissions"].astype(float) / scale

    out_em = data_clean / "emissions_annual_phaseIII.csv"
    dfe[["Year", "Verified emissions", "Emissions_Gt"]].to_csv(out_em, index=False)

    # --- Construct barE (cap proxy / reference scarcity level) ---
    barE_gt = float(dfe["Emissions_Gt"].mean())
    (objects_dir / "barE_gt.json").write_text(json.dumps({"barE_gt": barE_gt}, indent=2), encoding="utf-8")

    # --- Merge: map annual emissions to daily trading days ---
    emissions_map = dfe.set_index("Year")["Emissions_Gt"].to_dict()
    dfp["Emissions_Gt"] = dfp["Year"].map(emissions_map).astype(float)
    dfp["X"] = dfp["Emissions_Gt"] - barE_gt  # X_t = E_t - barE

    out_panel = data_clean / "panel_daily_phaseIII.csv"
    df_panel = dfp[["Date", "Year", "Price", "Emissions_Gt", "X"]].copy()
    df_panel["Date"] = df_panel["Date"].dt.strftime("%Y-%m-%d")
    df_panel.to_csv(out_panel, index=False)

    # Annual targets (for Script 02)
    annual_price = dfp.groupby("Year")["Price"].mean().rename("Price_Mean")
    annual = dfe.set_index("Year")[["Emissions_Gt"]].join(annual_price, how="inner")
    annual["X"] = annual["Emissions_Gt"] - barE_gt
    annual = annual.reset_index()

    out_targets = objects_dir / "annual_targets.csv"
    annual.to_csv(out_targets, index=False)

    print("[01_clean_and_merge_data] Wrote:")
    print(f"  - {out_price}")
    print(f"  - {out_em}")
    print(f"  - {out_panel}")
    print(f"  - {out_targets}")
    print(f"  - {objects_dir / 'barE_gt.json'}")
    print(f"[01_clean_and_merge_data] barE_gt = {barE_gt:.6f} Gt")

if __name__ == "__main__":
    main()
