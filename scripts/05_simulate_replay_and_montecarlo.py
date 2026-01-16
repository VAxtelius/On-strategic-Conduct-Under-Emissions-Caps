"""
05_simulate_replay_and_montecarlo.py

Runs two simulation modes using Chapter 3 closed forms:

A) Shock replay:
   Uses empirical eps_hat_t from Script 03 as the shock path.
   For each sigma in grid, computes e_t*, E_t*, p_t*, K_t.

B) Monte Carlo:
   Generates M synthetic shock paths from eps_{t+1} = rho eps_t + eta_t, eta ~ N(0, sigma_eta^2).
   For each sigma, computes replication-level moments and exports:
   - sim_mc/mc_replication_moments.csv
   - sim_mc/mc_summary_by_sigma.csv

Extended outputs (no more than 5 figures and 5 tables):
- Replay summary tables and effects vs sigma=0
- Monte Carlo effects vs sigma=0
- 5 figures (3 MC + 2 replay comparisons)

LaTeX export is OPTIONAL and skipped if jinja2 is not installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Utilities
# ----------------------------
def _load_config(base_dir: Path) -> dict:
    cfg_path = base_dir / "objects" / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError("Missing objects/config.json. Run scripts/00_config.py first.")
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def ar1_coeff_per_replication(p: np.ndarray) -> np.ndarray:
    x = p[:, :-1]
    y = p[:, 1:]
    x_mean = x.mean(axis=1, keepdims=True)
    y_mean = y.mean(axis=1, keepdims=True)
    cov = ((x - x_mean) * (y - y_mean)).mean(axis=1)
    varx = ((x - x_mean) ** 2).mean(axis=1)
    return np.where(varx > 0, cov / varx, 0.0)


def ar1_coeff_1d(x: np.ndarray) -> float:
    return float(ar1_coeff_per_replication(x[None, :])[0])


def pct_change(x: pd.Series, base: float) -> pd.Series:
    denom = abs(base)
    if denom == 0:
        return pd.Series(np.nan, index=x.index)
    return 100.0 * (x - base) / denom


def df_to_latex_table(df: pd.DataFrame, path: Path, float_format: str = "{:.4f}") -> None:
    """
    Optional LaTeX export.
    If jinja2 is missing, skip LaTeX output gracefully.
    """
    try:
        latex = df.to_latex(index=False, float_format=lambda v: float_format.format(v))
        path.write_text(latex, encoding="utf-8")
    except ImportError:
        print(f"[05_simulate] Skipping LaTeX export (missing jinja2): {path.name}")


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    cfg = _load_config(base_dir)

    objects_dir = Path(cfg["paths"]["objects"])
    sim_replay = Path(cfg["paths"]["sim_replay"])
    sim_mc = Path(cfg["paths"]["sim_mc"])
    ensure_dirs(sim_replay, sim_mc)

    replay_fig = sim_replay / "figures"
    replay_tab = sim_replay / "tables"
    mc_fig = sim_mc / "figures"
    mc_tab = sim_mc / "tables"
    ensure_dirs(replay_fig, replay_tab, mc_fig, mc_tab)

    P = json.loads((objects_dir / "params_full.json").read_text(encoding="utf-8"))
    shock_df = pd.read_csv(objects_dir / "shock_series.csv")

    N = int(P["N"])
    barE = float(P["barE_gt"])
    alpha0 = float(P["alpha0"])
    alpha1 = float(P["alpha1"])
    rho = float(P["rho"])
    sigma_eta = float(P["sigma_eta"])
    a1 = float(P["a1"])
    a2 = float(P["a2"])
    c2 = float(P["c2"])
    bar_e = float(P["bar_e"])
    beta = float(P["beta"])
    sigma_grid = [float(x) for x in P["sigma_grid"]]
    T = int(P["T"])
    burn = int(P["burn_in"])
    M = int(P["M"])
    seed = int(P["seed"])

    beta_pows = beta ** np.arange(T)

    # ----------------------------
    # A) Shock replay
    # ----------------------------
    dates = shock_df["Date"].values
    eps_replay = shock_df["eps_hat"].to_numpy()

    replay_paths: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
    replay_rows = []

    for sigma in sigma_grid:
        denom = a2 + c2 + alpha1 * (N + sigma)
        e = (a1 - alpha0 + alpha1 * barE - eps_replay) / denom
        e = np.clip(e, 0.0, bar_e)
        E = N * e
        p = alpha0 + alpha1 * (E - barE) + eps_replay
        K = N * 0.5 * c2 * e ** 2

        pd.DataFrame({
            "Date": dates, "eps": eps_replay, "e": e, "E": E, "p": p, "K": K
        }).to_csv(sim_replay / f"replay_sigma_{sigma:.1f}.csv", index=False)

        replay_rows.append({
            "sigma": sigma,
            "mean_p": p.mean(),
            "sd_p": p.std(ddof=1),
            "ar1_p": ar1_coeff_1d(p),
            "mean_E": E.mean(),
            "mean_K": K.mean(),
            "pv_K": np.sum(K * beta_pows),
        })

        if sigma in (0.0, 1.0):
            replay_paths[sigma] = (E.copy(), p.copy())

    replay_summary = pd.DataFrame(replay_rows).sort_values("sigma")
    replay_summary.to_csv(replay_tab / "replay_summary_by_sigma.csv", index=False)
    df_to_latex_table(replay_summary, replay_tab / "replay_summary_by_sigma.tex", "{:.6f}")

    base = replay_summary.loc[replay_summary["sigma"] == 0.0].iloc[0]
    replay_effects = replay_summary.copy()
    for c in ["mean_p", "sd_p", "ar1_p", "mean_E", "mean_K", "pv_K"]:
        replay_effects[f"pctchg_{c}"] = pct_change(replay_effects[c], base[c])
    replay_effects.to_csv(replay_tab / "replay_effects_vs_sigma0.csv", index=False)
    df_to_latex_table(replay_effects, replay_tab / "replay_effects_vs_sigma0.tex")

    if 0.0 in replay_paths and 1.0 in replay_paths:
        E0, p0 = replay_paths[0.0]
        E1, p1 = replay_paths[1.0]

        plt.figure()
        plt.plot(p0, label="σ=0")
        plt.plot(p1, label="σ=1")
        plt.legend()
        plt.title("Replay price paths")
        plt.tight_layout()
        plt.savefig(replay_fig / "fig_replay_price_sigma0_vs_sigma1.png", dpi=200)
        plt.close()

        plt.figure()
        plt.plot(E0, label="σ=0")
        plt.plot(E1, label="σ=1")
        plt.legend()
        plt.title("Replay emissions paths")
        plt.tight_layout()
        plt.savefig(replay_fig / "fig_replay_emissions_sigma0_vs_sigma1.png", dpi=200)
        plt.close()

    # ----------------------------
    # B) Monte Carlo
    # ----------------------------
    rng = np.random.default_rng(seed)
    eta = rng.normal(0.0, sigma_eta, size=(M, T + burn))
    eps = np.zeros_like(eta)
    for t in range(1, T + burn):
        eps[:, t] = rho * eps[:, t - 1] + eta[:, t]
    eps = eps[:, burn:]

    rows = []
    for sigma in sigma_grid:
        denom = a2 + c2 + alpha1 * (N + sigma)
        e = (a1 - alpha0 + alpha1 * barE - eps) / denom
        e = np.clip(e, 0.0, bar_e)
        E = N * e
        p = alpha0 + alpha1 * (E - barE) + eps
        K = N * 0.5 * c2 * e ** 2

        rows.append(pd.DataFrame({
            "replication": np.arange(M),
            "sigma": sigma,
            "mean_p": p.mean(axis=1),
            "sd_p": p.std(axis=1, ddof=1),
            "ar1_p": ar1_coeff_per_replication(p),
            "mean_E": E.mean(axis=1),
            "mean_K": K.mean(axis=1),
            "pv_K": (K * beta_pows).sum(axis=1),
        }))

    mc_rep = pd.concat(rows, ignore_index=True)
    mc_rep.to_csv(sim_mc / "mc_replication_moments.csv", index=False)

    summary = mc_rep.groupby("sigma").mean().reset_index()
    summary.to_csv(sim_mc / "mc_summary_by_sigma.csv", index=False)
    df_to_latex_table(summary, mc_tab / "mc_summary_by_sigma.tex", "{:.6f}")

    base = summary.loc[summary["sigma"] == 0.0].iloc[0]
    mc_eff = summary.copy()
    for c in ["mean_p", "sd_p", "ar1_p", "mean_E", "pv_K"]:
        mc_eff[f"pctchg_{c}"] = pct_change(mc_eff[c], base[c])
    mc_eff.to_csv(mc_tab / "mc_effects_vs_sigma0.csv", index=False)
    df_to_latex_table(mc_eff, mc_tab / "mc_effects_vs_sigma0.tex")

    plt.figure()
    plt.plot(summary["sigma"], summary["mean_p"])
    plt.title("MC mean price vs σ")
    plt.tight_layout()
    plt.savefig(mc_fig / "fig_mc_mean_price_vs_sigma.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(summary["sigma"], summary["sd_p"])
    plt.title("MC price volatility vs σ")
    plt.tight_layout()
    plt.savefig(mc_fig / "fig_mc_sd_price_vs_sigma.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(summary["sigma"], summary["ar1_p"])
    plt.title("MC price persistence vs σ")
    plt.tight_layout()
    plt.savefig(mc_fig / "fig_mc_ar1_price_vs_sigma.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
