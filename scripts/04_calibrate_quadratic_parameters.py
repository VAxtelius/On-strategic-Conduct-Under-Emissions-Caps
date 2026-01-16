"""
04_calibrate_quadratic_parameters.py

Calibrates quadratic technology parameters under the competitive benchmark (sigma=0),
strictly using Chapter 3 closed forms and only the provided datasets.

Fixed:
- a2 = 0
- N from config (N=10)
- alpha0, alpha1 from Script 02
- barE from Script 01
- (rho, sigma_eta) from Script 03

Targets (fixed):
1) E*(eps=0, sigma=0) = barE  (implemented by setting a1 given c2)
2) sd(p*_t | sigma=0) matches sd(p_t data)

Calibration chooses c2 > 0 by minimizing squared deviation in price sd.
We use:
- coarse log-grid search (200 points) on c2 in [1e-3, 1e5]
- bounded refinement in log10-space with Brent (scipy)

Outputs:
- objects/tech_params_calibrated.json
- objects/params_full.json
- tables_ch4/table_calibration_inputs.tex
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

def _load_config(base_dir: Path) -> dict:
    cfg_path = base_dir / "objects" / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError("Missing objects/config.json. Run scripts/00_config.py first.")
    return json.loads(cfg_path.read_text(encoding="utf-8"))

def simulate_price_sd(
    *,
    eps: np.ndarray,
    alpha0: float,
    alpha1: float,
    barE: float,
    N: int,
    a2: float,
    c2: float,
    bar_e: float
) -> float:
    # Competitive benchmark: sigma=0
    denom = a2 + c2 + alpha1 * (N + 0.0)
    # a1 chosen to hit E*(eps=0)=barE (given a2=0 this simplifies)
    a1 = alpha0 + (barE / N) * c2

    e = (a1 - alpha0 + alpha1 * barE - eps) / denom
    e = np.clip(e, 0.0, bar_e)
    E = N * e
    p = alpha0 + alpha1 * (E - barE) + eps
    return float(np.std(p, ddof=1))

def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    cfg = _load_config(base_dir)

    data_clean = Path(cfg["paths"]["data_clean"])
    objects_dir = Path(cfg["paths"]["objects"])
    tables_dir = Path(cfg["paths"]["tables_ch4"])
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Load required objects
    price_params = json.loads((objects_dir / "price_rule_params.json").read_text(encoding="utf-8"))
    shock_params = json.loads((objects_dir / "shock_ar1_params.json").read_text(encoding="utf-8"))
    barE = json.loads((objects_dir / "barE_gt.json").read_text(encoding="utf-8"))["barE_gt"]

    alpha0 = float(price_params["alpha0"])
    alpha1 = float(price_params["alpha1"])
    rho = float(shock_params["rho_used"])
    sigma_eta = float(shock_params["sigma_eta"])

    N = int(cfg["model"]["N"])
    T = 2066  # fixed by sample; also in config implicitly
    burn = int(cfg["model"]["burn_in"])
    seed = int(cfg["model"]["seed"])

    # bar_e fixed rule: 3*(barE/N), non-binding by construction
    bar_e = float(3.0 * (barE / N))

    # Observed target volatility
    price_daily_path = data_clean / "price_daily_phaseIII.csv"
    dfp = pd.read_csv(price_daily_path)
    sd_price_data = float(np.std(dfp["Price"].to_numpy(dtype=float), ddof=1))

    # Pre-generate innovations once for deterministic objective
    rng = np.random.default_rng(seed)
    eta = rng.normal(loc=0.0, scale=sigma_eta, size=(T + burn,))
    eps = np.zeros((T + burn,), dtype=float)
    for t in range(1, T + burn):
        eps[t] = rho * eps[t - 1] + eta[t]
    eps = eps[burn:]  # drop burn-in, length T

    a2 = 0.0

    # Coarse grid search over c2
    grid = np.logspace(-3, 5, 200)
    obj_vals = []
    for c2 in grid:
        sd_sim = simulate_price_sd(
            eps=eps,
            alpha0=alpha0,
            alpha1=alpha1,
            barE=float(barE),
            N=N,
            a2=a2,
            c2=float(c2),
            bar_e=bar_e,
        )
        obj = (sd_sim - sd_price_data) ** 2
        obj_vals.append(obj)

    obj_vals = np.array(obj_vals)
    idx_best = int(np.argmin(obj_vals))
    c2_best_grid = float(grid[idx_best])

    # Refine around best grid point in log10 space
    log_best = float(np.log10(c2_best_grid))
    log_lo = max(-3.0, log_best - 1.0)
    log_hi = min(5.0, log_best + 1.0)

    def obj_logc(logc: float) -> float:
        c2 = 10.0 ** logc
        sd_sim = simulate_price_sd(
            eps=eps,
            alpha0=alpha0,
            alpha1=alpha1,
            barE=float(barE),
            N=N,
            a2=a2,
            c2=float(c2),
            bar_e=bar_e,
        )
        return (sd_sim - sd_price_data) ** 2

    res = minimize_scalar(obj_logc, bounds=(log_lo, log_hi), method="bounded")
    c2_cal = float(10.0 ** float(res.x))

    # Given c2, compute a1 to enforce E*(eps=0)=barE
    a1_cal = float(alpha0 + (float(barE) / N) * c2_cal)

    tech_params = {
        "a1": a1_cal,
        "a2": 0.0,
        "c2": c2_cal,
        "bar_e": bar_e,
        "targets": {
            "E_at_eps0_sigma0_equals_barE": True,
            "sd_price_data": sd_price_data,
        },
        "calibration": {
            "grid_c2_min": 1e-3,
            "grid_c2_max": 1e5,
            "grid_points": 200,
            "refinement_log10_bounds": [log_lo, log_hi],
            "objective": "minimize (sd(p_sim) - sd(p_data))^2 under sigma=0",
            "seed": seed
        }
    }

    (objects_dir / "tech_params_calibrated.json").write_text(json.dumps(tech_params, indent=2), encoding="utf-8")

    params_full = {
        "N": N,
        "barE_gt": float(barE),
        "alpha0": alpha0,
        "alpha1": alpha1,
        "rho": rho,
        "sigma_eta": sigma_eta,
        "a1": a1_cal,
        "a2": 0.0,
        "c2": c2_cal,
        "bar_e": bar_e,
        "beta": float(cfg["model"]["beta"]),
        "sigma_grid": cfg["model"]["sigma_grid"],
        "T": T,
        "burn_in": burn,
        "M": int(cfg["model"]["M"]),
        "seed": seed,
    }
    (objects_dir / "params_full.json").write_text(json.dumps(params_full, indent=2), encoding="utf-8")

    # LaTeX table: calibration inputs
    tex_path = tables_dir / "table_calibration_inputs.tex"
    lines = []
    lines.append("\\begin{table}[!ht]\n\\centering")
    lines.append("\\caption{Calibrated model inputs used in simulations}")
    lines.append("\\label{tab:calibrated_inputs}")
    lines.append("\\begin{tabular}{lr}")
    lines.append("\\toprule")
    lines.append("Parameter & Value \\\\")
    lines.append("\\midrule")
    lines.append(f"$N$ & {N:d} \\\\")
    lines.append(f"$\\bar E$ (Gt) & {float(barE):.6f} \\\\")
    lines.append(f"$\\alpha_0$ & {alpha0:.6f} \\\\")
    lines.append(f"$\\alpha_1$ & {alpha1:.6f} \\\\")
    lines.append(f"$\\rho$ & {rho:.6f} \\\\")
    lines.append(f"$\\sigma_\\eta$ & {sigma_eta:.6f} \\\\")
    lines.append(f"$a_1$ & {a1_cal:.6f} \\\\")
    lines.append(f"$a_2$ & {0.0:.1f} \\\\")
    lines.append(f"$c_2$ & {c2_cal:.6f} \\\\")
    lines.append(f"$\\bar e$ (Gt) & {bar_e:.6f} \\\\")
    lines.append(f"$\\beta$ (daily) & {float(cfg['model']['beta']):.8f} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\begin{flushleft}\\footnotesize")
    lines.append("Notes: $a_2$ is fixed at zero. Given $c_2$, $a_1$ is set to enforce $E^*(\\varepsilon=0,\\sigma=0)=\\bar E$. "
                 "The remaining free parameter $c_2$ is chosen to match the standard deviation of daily prices under the competitive benchmark.")
    lines.append("\\end{flushleft}")
    lines.append("\\end{table}\n")
    tex_path.write_text("\n".join(lines), encoding="utf-8")

    print("[04_calibrate_quadratic_parameters] Wrote:")
    print(f"  - {objects_dir / 'tech_params_calibrated.json'}")
    print(f"  - {objects_dir / 'params_full.json'}")
    print(f"  - {tex_path}")
    print(f"[04_calibrate_quadratic_parameters] sd(price_data)={sd_price_data:.6f}, c2={c2_cal:.6f}, a1={a1_cal:.6f}")

if __name__ == "__main__":
    main()
