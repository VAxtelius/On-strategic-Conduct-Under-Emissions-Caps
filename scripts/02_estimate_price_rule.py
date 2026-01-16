"""
02_estimate_price_rule.py

Constructs price-rule parameters (alpha0, alpha1) consistent with Chapter 3:
p_t = alpha0 + alpha1*(E_t - barE) + eps_t, with alpha1 > 0.

Because annual emissions are the only emissions data available, we use annual means:
pbar_y = mean(p_t in year y)

Fixed rule (no tuning):
- alpha0 = mean(pbar_y)
- alpha1 = sd(pbar_y) / sd(E_y - barE)   (positive scale normalization)

Also computes OLS slope/intercept as a descriptive reference (not used for simulation).

Outputs:
- objects/price_rule_params.json
- tables_ch4/table_price_rule_estimation.tex
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

def _load_config(base_dir: Path) -> dict:
    cfg_path = base_dir / "objects" / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError("Missing objects/config.json. Run scripts/00_config.py first.")
    return json.loads(cfg_path.read_text(encoding="utf-8"))

def _latex_escape(s: str) -> str:
    return s.replace("_", "\\_")

def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    cfg = _load_config(base_dir)

    objects_dir = Path(cfg["paths"]["objects"])
    tables_dir = Path(cfg["paths"]["tables_ch4"])
    tables_dir.mkdir(parents=True, exist_ok=True)

    annual_targets_path = objects_dir / "annual_targets.csv"
    if not annual_targets_path.exists():
        raise FileNotFoundError("Missing objects/annual_targets.csv. Run scripts/01_clean_and_merge_data.py first.")

    annual = pd.read_csv(annual_targets_path)
    if not {"Year", "Emissions_Gt", "Price_Mean", "X"}.issubset(set(annual.columns)):
        raise ValueError("annual_targets.csv missing required columns.")

    pbar = annual["Price_Mean"].to_numpy(dtype=float)
    X = annual["X"].to_numpy(dtype=float)  # E_y - barE

    alpha0 = float(pbar.mean())
    sd_p = float(np.std(pbar, ddof=1))
    sd_x = float(np.std(X, ddof=1))

    if sd_x <= 0:
        # Degenerate case (should not happen with real data)
        alpha1 = 1.0
    else:
        alpha1 = float(sd_p / sd_x)

    # Descriptive OLS for reference (can be negative due to endogeneity)
    X_ols = sm.add_constant(X)
    ols_res = sm.OLS(pbar, X_ols).fit(cov_type="HC1")

    out_params = {
        "alpha0": alpha0,
        "alpha1": alpha1,
        "construction_rule": {
            "alpha0": "mean of annual mean prices",
            "alpha1": "sd(annual mean prices) / sd(annual emissions deviations)",
            "model_restriction": "alpha1 > 0 required by Chapter 3 price rule"
        },
        "descriptive_ols": {
            "const": float(ols_res.params[0]),
            "slope": float(ols_res.params[1]),
            "se_const_hc1": float(ols_res.bse[0]),
            "se_slope_hc1": float(ols_res.bse[1]),
            "r2": float(ols_res.rsquared),
            "n_obs": int(ols_res.nobs),
        }
    }

    params_path = objects_dir / "price_rule_params.json"
    params_path.write_text(json.dumps(out_params, indent=2), encoding="utf-8")

    # --- LaTeX table for Chapter 4 ---
    tex_path = tables_dir / "table_price_rule_estimation.tex"

    lines = []
    lines.append("\\begin{table}[!ht]\n\\centering")
    lines.append("\\caption{Pricing rule parameters (Phase III, annual aggregation)}")
    lines.append("\\label{tab:price_rule_params}")
    lines.append("\\begin{tabular}{lrr}")
    lines.append("\\toprule")
    lines.append("Parameter & Value used & Descriptive OLS \\\\")
    lines.append("\\midrule")
    lines.append(f"$\\alpha_0$ & {alpha0:.6f} & {ols_res.params[0]:.6f} \\\\")
    lines.append(f"$\\alpha_1$ & {alpha1:.6f} & {ols_res.params[1]:.6f} \\\\")
    lines.append("\\midrule")
    lines.append(f"HC1 s.e. (slope) & -- & {ols_res.bse[1]:.6f} \\\\")
    lines.append(f"$R^2$ & -- & {ols_res.rsquared:.3f} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\begin{flushleft}\\footnotesize")
    lines.append("Notes: Values used for simulation impose the Chapter 3 restriction $\\alpha_1>0$. "
                 "Because verified emissions are annual and endogenous to economic conditions, the OLS slope is reported only as a descriptive comparison.")
    lines.append("\\end{flushleft}")
    lines.append("\\end{table}\n")

    tex_path.write_text("\n".join(lines), encoding="utf-8")

    print("[02_estimate_price_rule] Wrote:")
    print(f"  - {params_path}")
    print(f"  - {tex_path}")
    print(f"[02_estimate_price_rule] alpha0={alpha0:.6f}, alpha1={alpha1:.6f} (positive by construction)")
    print(f"[02_estimate_price_rule] descriptive OLS slope={ols_res.params[1]:.6f}")

if __name__ == "__main__":
    main()
