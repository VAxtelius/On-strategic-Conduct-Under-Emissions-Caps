"""
03_construct_shock_and_estimate_ar1.py

Constructs the daily shock series as the residual in the Chapter 3 price rule:
eps_hat_t = p_t - alpha0 - alpha1*(E_t - barE)

Then demeans eps_hat and estimates AR(1):
eps_{t+1} = rho * eps_t + eta_t
with rho clamped into (-0.999, 0.999) to satisfy |rho|<1 as assumed in Chapter 3.

Outputs:
- objects/shock_series.csv
- objects/shock_ar1_params.json
- tables_ch4/table_shock_ar1_estimation.tex
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

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    cfg = _load_config(base_dir)

    data_clean = Path(cfg["paths"]["data_clean"])
    objects_dir = Path(cfg["paths"]["objects"])
    tables_dir = Path(cfg["paths"]["tables_ch4"])
    tables_dir.mkdir(parents=True, exist_ok=True)

    panel_path = data_clean / "panel_daily_phaseIII.csv"
    price_params_path = objects_dir / "price_rule_params.json"
    barE_path = objects_dir / "barE_gt.json"

    if not panel_path.exists():
        raise FileNotFoundError("Missing data_clean/panel_daily_phaseIII.csv. Run script 01 first.")
    if not price_params_path.exists():
        raise FileNotFoundError("Missing objects/price_rule_params.json. Run script 02 first.")
    if not barE_path.exists():
        raise FileNotFoundError("Missing objects/barE_gt.json. Run script 01 first.")

    panel = pd.read_csv(panel_path)
    alpha = json.loads(price_params_path.read_text(encoding="utf-8"))
    barE_gt = json.loads(barE_path.read_text(encoding="utf-8"))["barE_gt"]

    alpha0 = float(alpha["alpha0"])
    alpha1 = float(alpha["alpha1"])

    # Construct eps_hat and demean
    eps = panel["Price"].to_numpy(dtype=float) - alpha0 - alpha1 * (panel["Emissions_Gt"].to_numpy(dtype=float) - float(barE_gt))
    eps = eps - eps.mean()

    # Estimate AR(1) without intercept: eps_{t+1} = rho * eps_t + eta_t
    eps_t = eps[:-1]
    eps_tp1 = eps[1:]

    res = sm.OLS(eps_tp1, eps_t).fit()
    rho_hat = float(res.params[0])
    rho = clamp(rho_hat, -0.999, 0.999)

    eta = eps_tp1 - rho * eps_t
    sigma_eta = float(np.sqrt(np.mean(eta**2)))

    out_series = objects_dir / "shock_series.csv"
    pd.DataFrame({"Date": panel["Date"].values, "eps_hat": eps}).to_csv(out_series, index=False)

    out_params = {
        "rho_hat_ols": rho_hat,
        "rho_used": rho,
        "sigma_eta": sigma_eta,
        "eps_sd": float(np.std(eps, ddof=1)),
        "n_obs": int(len(eps)),
        "note": "rho is clamped to satisfy |rho|<1 as assumed in Chapter 3."
    }
    out_params_path = objects_dir / "shock_ar1_params.json"
    out_params_path.write_text(json.dumps(out_params, indent=2), encoding="utf-8")

    # LaTeX table
    tex_path = tables_dir / "table_shock_ar1_estimation.tex"
    lines = []
    lines.append("\\begin{table}[!ht]\n\\centering")
    lines.append("\\caption{Estimated shock process (daily, Phase III)}")
    lines.append("\\label{tab:shock_ar1}")
    lines.append("\\begin{tabular}{lr}")
    lines.append("\\toprule")
    lines.append("Quantity & Value \\\\")
    lines.append("\\midrule")
    lines.append(f"$\\hat\\rho$ (OLS) & {rho_hat:.6f} \\\\")
    lines.append(f"$\\rho$ used ($|\\rho|<1$) & {rho:.6f} \\\\")
    lines.append(f"$\\hat\\sigma_\\eta$ & {sigma_eta:.6f} \\\\")
    lines.append(f"$\\mathrm{{sd}}(\\hat\\varepsilon_t)$ & {np.std(eps, ddof=1):.6f} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\begin{flushleft}\\footnotesize")
    lines.append("Notes: The shock series $\\hat\\varepsilon_t$ is constructed as the residual in the Chapter 3 pricing rule. "
                 "The AR(1) coefficient is clamped to ensure stationarity ($|\\rho|<1$), consistent with equation (\\ref{eq:shock_process}).")
    lines.append("\\end{flushleft}")
    lines.append("\\end{table}\n")
    tex_path.write_text("\n".join(lines), encoding="utf-8")

    print("[03_construct_shock_and_estimate_ar1] Wrote:")
    print(f"  - {out_series}")
    print(f"  - {out_params_path}")
    print(f"  - {tex_path}")
    print(f"[03_construct_shock_and_estimate_ar1] rho_hat={rho_hat:.6f}, rho_used={rho:.6f}, sigma_eta={sigma_eta:.6f}")

if __name__ == "__main__":
    main()
