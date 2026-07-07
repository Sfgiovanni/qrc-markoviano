#!/usr/bin/env python3
"""Phase 3C (item 7, OPTIONAL) — memory-scaling law with 8-12 log-spaced gamma points.

PREPARED BUT GUARDED. This is NOT a central claim of the study, so per the task spec it
must not run without an explicit go-ahead. Running it bare only PRINTS the plan + a GPU
time estimate and exits. Pass --run to actually execute.

Method (matches the v6 4-point refit so points are comparable): for each gamma, build the
dissipative channel at that gamma (the disk/GPU channel cache is keyed on gamma, so this
is safe), drive AB-embedded over N_SEEDS seeds, measure tau_mem = largest delay tau whose
mean STM capacity C(tau) >= THRESH (0.1). Then fit log(tau_mem) ~ log(1/gamma):
tau_mem ~ (1/gamma)^p, with a bootstrap CI on p. Writes results_review/scaling_sweep_*.

Run from repo root:  python3 experiments_review/phase3c_scaling_sweep.py --run
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import qrc_pipeline as p  # renamed module in this mirror  # noqa: E402

OUT = ROOT / "results_review"
GAMMAS = [round(g, 4) for g in np.geomspace(0.01, 0.30, 10)]   # 10 log-spaced points
N_SEEDS = 12
TAU_MAX = 60
THRESH = 0.10
WASHOUT, TRAIN, TEST = 300, 900, 600


def tau_mem_for_gamma(gamma: float, seeds) -> dict:
    p.CFG.gamma = gamma
    slices = p.split_slices(WASHOUT, TRAIN, TEST)
    seq_len = WASHOUT + TRAIN + TEST
    horizons = []
    for seed in seeds:
        grid = p.build_channel_grid_gpu(seed, p.CFG.n_a)     # cache keyed on gamma (safe)
        seq = p.iid_inputs(seed, seq_len)
        model = p.make_embedded_model("AB-embedded", p.best_params("AB-embedded")).reset()
        feats = p.drive_features(model, seq, grid, register="A")
        caps = []
        for tau in range(TAU_MAX + 1):
            c, _ = p.evaluate_capacity_from_features(feats, seq, p.stm_target(seq, tau), slices, alpha=1e-6)
            caps.append(max(0.0, float(c)))
        caps = np.array(caps)
        h = int(np.max(np.where(caps >= THRESH)[0])) if np.any(caps >= THRESH) else 0
        horizons.append(h)
    return {"gamma": gamma, "tau_mem_mean": float(np.mean(horizons)),
            "tau_mem_median": float(np.median(horizons)), "tau_mem_std": float(np.std(horizons)),
            "n_seeds": len(seeds), "horizons": horizons}


def fit_power(gammas, taus):
    x = np.log(1.0 / np.asarray(gammas))
    y = np.log(np.asarray(taus))
    p_hat, b = np.polyfit(x, y, 1)
    yhat = p_hat * x + b
    r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
    rng = np.random.default_rng(20260707)
    boots = []
    n = len(x)
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        boots.append(np.polyfit(x[idx], y[idx], 1)[0])
    return float(p_hat), float(r2), [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def main():
    run = "--run" in sys.argv
    print(f"[Phase 3C] gammas={GAMMAS}  seeds={N_SEEDS}  tau_max={TAU_MAX}  thresh={THRESH}")
    if not run:
        print("GUARD: dry run. Estimated GPU cost ~ len(gammas)*N_SEEDS AB-embedded drives")
        print(f"       ~ {len(GAMMAS)*N_SEEDS} drives of {WASHOUT+TRAIN+TEST} steps "
              f"(AB=8 qubits, ~3.5 ms/step) ~= "
              f"{len(GAMMAS)*N_SEEDS*(WASHOUT+TRAIN+TEST)*0.0035/60:.0f} min + channel builds.")
        print("       Re-run with --run to execute (needs explicit go-ahead).")
        return
    t0 = time.time()
    seeds = list(range(N_SEEDS))
    rows = [tau_mem_for_gamma(g, seeds) for g in GAMMAS]
    for r in rows:
        print(f"  gamma={r['gamma']:.4f}  tau_mem={r['tau_mem_mean']:.1f} "
              f"(median {r['tau_mem_median']:.0f})")
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "horizons"} for r in rows])
    df.to_csv(OUT / "scaling_sweep_tau_mem.csv", index=False)
    p_hat, r2, ci = fit_power(df.gamma.values, df.tau_mem_mean.values)
    summary = {"n_points": len(df), "p": p_hat, "r2": r2, "p_ci95_bootstrap": ci,
               "gammas": GAMMAS, "n_seeds": N_SEEDS, "threshold": THRESH,
               "wall_seconds": round(time.time() - t0, 1)}
    (OUT / "scaling_sweep_fit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\ntau_mem ~ (1/gamma)^p : p={p_hat:.3f}, R2={r2:.3f}, CI95={ci}, n={len(df)}")
    print(f"wrote {OUT/'scaling_sweep_tau_mem.csv'} and scaling_sweep_fit.json "
          f"({(time.time()-t0)/60:.1f} min)")
    p.CFG.gamma = 0.1  # restore default


if __name__ == "__main__":
    main()
