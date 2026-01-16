"""
06_export_ch4_inputs_tables.py

Exports Chapter 4 "inputs tables" only (no simulation results narrative):
- Table 4.1: Data moments used as targets (mean emissions, mean price, sd price, T)
- Table 4.2: Price rule parameters used (alpha0, alpha1) + descriptive OLS
- Table 4.3: Shock AR(1) parameters (rho, sigma_eta, sd(eps_hat))
- Table 4.4: Full parameter set used in simulations (params_full.json)

Outputs: LaTeX tables in tables_ch4/
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

def _load_config(base_dir: Path) -> dict:
    cfg_path = base_dir / "objects" / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError("Missing objects/config.json. Run scripts/00_config.py first.")
    return json.loads(cfg_path.read_text(encoding="utf-8"))

def write_table(path: Path, caption: str, label: str, rows: list[tuple[str, str]], notes: str) -> None:
    lines = []
    lines.append("\\begin{table}[!ht]\n\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\begin{tabular}{lr}")
    lines.append("\\toprule")
    lines.append("Quantity & Value \\\\")
    lines.append("\\midrule")
    for k, v in rows:
        lines.append(f"{k} & {v} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    if notes.strip():
        lines.append("\\begin{flushleft}\\footnotesize")
        lines.append(f"Notes: {notes}")
        lines.append("\\end{flushleft}")
    lines.append("\\end{table}\n")
    path.write_text("\n".join(lines), encoding="utf-8")

def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    cfg = _load_config(base_dir)

    data_clean = Path(cfg["paths"]["data_clean"])
    objects_dir = Path(cfg["paths"]["objects"])
    tables_dir = Path(cfg["paths"]["tables_ch4"])
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Load core objects
    barE = json.loads((objects_dir / "barE_gt.json").read_text(encoding="utf-8"))["barE_gt"]
    price_params = json.loads((objects_dir / "price_rule_params.json").read_text(encoding="utf-8"))
    shock_params = json.loads((objects_dir / "shock_ar1_params.json").read_text(encoding="utf-8"))
    params_full = json.loads((objects_dir / "params_full.json").read_text(encoding="utf-8"))

    # Data moments
    dfp = pd.read_csv(data_clean / "price_daily_phaseIII.csv")
    mean_price = float(dfp["Price"].mean())
    sd_price = float(np.std(dfp["Price"].to_numpy(dtype=float), ddof=1))
    T = int(len(dfp))

    # Table 4.1
    write_table(
        tables_dir / "table_4_1_data_moments.tex",
        "Data moments used as calibration targets (Phase III)",
        "tab:data_moments",
        [
            ("$\\bar E$ (Gt, mean verified emissions)", f"{barE:.6f}"),
            ("Mean daily price", f"{mean_price:.6f}"),
            ("sd(daily price)", f"{sd_price:.6f}"),
            ("Trading days $T$", f"{T:d}"),
        ],
        notes="Verified emissions are annual and mapped to trading days within each year. $\\bar E$ is a fixed reference scarcity level constructed from Phase III verified emissions."
    )

    # Table 4.2 (price rule)
    ols = price_params["descriptive_ols"]
    write_table(
        tables_dir / "table_4_2_price_rule.tex",
        "Pricing rule parameters used in simulation",
        "tab:price_rule",
        [
            ("$\\alpha_0$ (used)", f"{price_params['alpha0']:.6f}"),
            ("$\\alpha_1$ (used, $>0$)", f"{price_params['alpha1']:.6f}"),
            ("OLS slope (descriptive)", f"{ols['slope']:.6f}"),
            ("OLS $R^2$ (descriptive)", f"{ols['r2']:.3f}"),
        ],
        notes="Values used impose the Chapter 3 restriction $\\alpha_1>0$. OLS is reported only as a descriptive comparison."
    )

    # Table 4.3 (shock AR1)
    write_table(
        tables_dir / "table_4_3_shock_ar1.tex",
        "Estimated shock process parameters",
        "tab:shock_process",
        [
            ("$\\hat\\rho$ (OLS)", f"{shock_params['rho_hat_ols']:.6f}"),
            ("$\\rho$ used ($|\\rho|<1$)", f"{shock_params['rho_used']:.6f}"),
            ("$\\hat\\sigma_\\eta$", f"{shock_params['sigma_eta']:.6f}"),
            ("sd($\\hat\\varepsilon_t$)", f"{shock_params['eps_sd']:.6f}"),
        ],
        notes="The shock series is the residual in the pricing rule. The AR(1) coefficient is clamped to satisfy $|\\rho|<1$ as assumed in Chapter 3."
    )

    # Table 4.4 (full params)
    pf = params_full
    write_table(
        tables_dir / "table_4_4_full_params.tex",
        "Full calibrated parameter set used for simulation",
        "tab:full_params",
        [
            ("$N$", f"{int(pf['N'])}"),
            ("$\\bar E$ (Gt)", f"{pf['barE_gt']:.6f}"),
            ("$\\alpha_0$", f"{pf['alpha0']:.6f}"),
            ("$\\alpha_1$", f"{pf['alpha1']:.6f}"),
            ("$\\rho$", f"{pf['rho']:.6f}"),
            ("$\\sigma_\\eta$", f"{pf['sigma_eta']:.6f}"),
            ("$a_1$", f"{pf['a1']:.6f}"),
            ("$a_2$", f"{pf['a2']:.1f}"),
            ("$c_2$", f"{pf['c2']:.6f}"),
            ("$\\bar e$ (Gt)", f"{pf['bar_e']:.6f}"),
            ("$\\beta$ (daily)", f"{pf['beta']:.8f}"),
            ("$T$", f"{int(pf['T'])}"),
            ("$M$", f"{int(pf['M'])}"),
            ("Burn-in", f"{int(pf['burn_in'])}"),
            ("Seed", f"{int(pf['seed'])}"),
        ],
        notes="These inputs are produced by Scripts 01--04 and are required to replicate the simulation outputs used in Chapter 5."
    )

    print("[06_export_ch4_inputs_tables] Wrote LaTeX tables to:", tables_dir)

if __name__ == "__main__":
    main()
