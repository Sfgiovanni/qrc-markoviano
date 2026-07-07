#!/usr/bin/env python3
"""Phase 3A (item 6) — do the two memory scales actually exist? GPU.

The production multiscale phase (run_multiscale_and_ipc) evaluated STM on only 7 coarse
delays and detected peaks on the *mean* curve, and characterised the A/B/C layer
timescales on only 2 seeds. That is too thin to claim "R-ABC has two separated memory
scales at tau_B and tau_C vs R1's single scale". This script redoes it properly over ALL
CFG.eval_seeds (20), with:

  (1) STM capacity curve on a FINE delay grid (tau = 0..TAU_MAX) per seed, for
      M0 (register A only), R1 (AB-embedded, one auxiliary) and R-ABC
      (ABC-embedded-hierarchical, two auxiliaries);
  (2) per-seed revival detection on each curve (local maxima past the main lobe);
  (3) autocorrelation of ALL register observables -> effective timescale tau_A, tau_B,
      tau_C per seed (median over observables), and paired tests of tau_A<tau_B<tau_C.

Decision `two_separated_scales` and its p-value are DATA-DRIVEN. If the scales do not
separate, this reports the negative — the hypothesis is not forced.

Model map (task taxonomy -> real architecture):
  M0    -> EmbeddedModelGPU("M0-embedded")     (single register A, non-Markovian channel)
  R1    -> "AB-embedded"                        (one auxiliary; Eq.8-style)
  R-ABC -> "ABC-embedded-hierarchical"          (omega_b=0.5, omega_c=0.1, eta_bc<eta_ab)

Writes ONLY to results_review/. Run from repo root:
  python3 experiments_review/phase3a_memory_scales.py
"""
from __future__ import annotations
import sys, json, time, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import qrc_pipeline as p  # renamed module in this mirror  # noqa: E402

OUT = ROOT / "results_review"
OUT.mkdir(exist_ok=True)

TAU_MAX = 40
WASHOUT, TRAIN, TEST = 300, 900, 600
AC_LEN, AC_BURN = 700, 150          # autocorrelation drive length / burn-in
SEEDS = list(p.CFG.eval_seeds)      # 20
ALPHA = 1e-6
INV_E = math.exp(-1)


def build_model(tag):
    if tag == "M0":
        return p.EmbeddedModelGPU(p.CFG.n_a, "M0-embedded").reset()
    if tag == "R1":
        return p.make_embedded_model("AB-embedded", p.best_params("AB-embedded")).reset()
    if tag == "R-ABC":
        return p.make_embedded_model("ABC-embedded-hierarchical",
                                     p.best_params("ABC-embedded-hierarchical")).reset()
    raise ValueError(tag)


def stm_curve(model, grid, seq, slices):
    feats = p.drive_features(model, seq, grid, register="A")
    caps = []
    for tau in range(TAU_MAX + 1):
        y = p.stm_target(seq, tau)
        c, _ = p.evaluate_capacity_from_features(feats, seq, y, slices, alpha=ALPHA)
        caps.append(max(0.0, float(c)))
    return np.array(caps)


def detect_revivals(caps, min_prom=0.015, min_height=0.02):
    """Local maxima past the initial main lobe (tau>=2), with prominence over the
    preceding local minimum. Returns list of (tau, height)."""
    revivals = []
    for t in range(2, len(caps) - 1):
        if caps[t] > caps[t - 1] and caps[t] >= caps[t + 1] and caps[t] >= min_height:
            prev_min = np.min(caps[max(0, t - 8):t]) if t > 0 else caps[t]
            if caps[t] - prev_min >= min_prom:
                revivals.append((int(t), float(caps[t])))
    return revivals


def autocorr_tau(series):
    """1/e crossing lag of a scalar time series; np.nan if variance negligible."""
    x = np.asarray(series, dtype=float)
    if x.std() < 1e-9:
        return np.nan
    x = x - x.mean()
    ac = np.correlate(x, x, mode="full")
    ac = ac[len(ac) // 2:]
    ac = ac / (ac[0] + 1e-12)
    below = np.where(ac < INV_E)[0]
    return float(below[0]) if len(below) else float(len(ac) - 1)


def layer_timescales(seed, tag, registers):
    """Median 1/e autocorrelation timescale over ALL observables of each register."""
    grid = p.build_channel_grid_gpu(seed, p.CFG.n_a)
    model = build_model(tag)
    seq = p.iid_inputs(seed + 777, AC_LEN)
    buffers = {r: [] for r in registers}
    for s in seq:
        model.step(float(s), grid)
        for r in registers:
            buffers[r].append(model.features_t(r).double().cpu().numpy())
    out = {}
    for r in registers:
        arr = np.asarray(buffers[r])[AC_BURN:]          # (T, nfeat)
        taus = [autocorr_tau(arr[:, j]) for j in range(arr.shape[1])]
        taus = [t for t in taus if not np.isnan(t)]
        out[r] = float(np.median(taus)) if taus else np.nan
        out[r + "_n_obs"] = len(taus)
    return out


def main():
    t_start = time.time()
    slices = p.split_slices(WASHOUT, TRAIN, TEST)
    seq_len = WASHOUT + TRAIN + TEST
    tags = ["M0", "R1", "R-ABC"]

    # ---- timing probe on the first seed (print estimate before the full sweep) ----
    print(f"[probe] GPU sweep: {len(SEEDS)} seeds x {tags} ; "
          f"STM len={seq_len}, taus=0..{TAU_MAX}; autocorr len={AC_LEN}")
    s0 = SEEDS[0]
    grid0 = p.build_channel_grid_gpu(s0, p.CFG.n_a)
    seq0 = p.iid_inputs(s0, seq_len)
    tp = time.time()
    _ = stm_curve(build_model("R-ABC"), grid0, seq0, slices)   # heaviest model
    dt_abc = time.time() - tp
    est_stm = dt_abc * len(SEEDS) * 1.6            # M0+R1 together ~0.6x of ABC
    est_ac = dt_abc * 0.5 * len(SEEDS) * 1.5       # autocorr drive (ABC+R1), shorter
    print(f"[probe] one R-ABC STM curve = {dt_abc:.1f}s -> estimated total "
          f"~{(est_stm + est_ac)/60:.1f} min on this GPU. Proceeding.")

    # ---- (1)+(2) STM curves and revivals ----
    stm_rows, rev_rows = [], []
    for seed in SEEDS:
        grid = p.build_channel_grid_gpu(seed, p.CFG.n_a)
        seq = p.iid_inputs(seed, seq_len)
        for tag in tags:
            caps = stm_curve(build_model(tag), grid, seq, slices)
            for tau, c in enumerate(caps):
                stm_rows.append({"seed": seed, "model": tag, "tau": tau, "capacity": c})
            revs = detect_revivals(caps)
            rev_rows.append({"seed": seed, "model": tag,
                             "n_revivals": len(revs),
                             "revival_taus": ";".join(str(t) for t, _ in revs) or "-",
                             "tau_at_global_peak": int(np.argmax(caps)),
                             "total_stm_capacity": float(caps.sum())})
        print(f"[stm] seed {seed} done (+{time.time()-t_start:.0f}s)")
    pd.DataFrame(stm_rows).to_csv(OUT / "memory_scales.csv", index=False)
    rev = pd.DataFrame(rev_rows)
    rev.to_csv(OUT / "memory_scales_revivals.csv", index=False)

    # ---- (3) layer timescales ----
    lt_rows = []
    for seed in SEEDS:
        abc = layer_timescales(seed, "R-ABC", ["A", "B", "C"])
        ab = layer_timescales(seed, "R1", ["A", "B"])
        lt_rows.append({"seed": seed,
                        "tau_A": abc["A"], "tau_B": abc["B"], "tau_C": abc["C"],
                        "tau_A_R1": ab["A"], "tau_B_R1": ab["B"],
                        "n_obs_A": abc["A_n_obs"], "n_obs_B": abc["B_n_obs"], "n_obs_C": abc["C_n_obs"]})
        print(f"[autocorr] seed {seed} tau_A/B/C = "
              f"{abc['A']:.1f}/{abc['B']:.1f}/{abc['C']:.1f} (+{time.time()-t_start:.0f}s)")
    lt = pd.DataFrame(lt_rows)
    lt.to_csv(OUT / "memory_scales_layer_timescales.csv", index=False)

    # ---- tests ----
    def one_sided_greater(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        m = ~(np.isnan(x) | np.isnan(y))
        x, y = x[m], y[m]
        if len(x) < 2 or np.all(x == y):
            return np.nan, int(len(x))
        try:
            return float(stats.wilcoxon(x, y, alternative="greater").pvalue), int(len(x))
        except Exception:
            return np.nan, int(len(x))

    p_BC, n_BC = one_sided_greater(lt.tau_C, lt.tau_B)     # tau_C > tau_B ?
    p_AB, n_AB = one_sided_greater(lt.tau_B, lt.tau_A)     # tau_B > tau_A ?
    holm_layer = p.holm([p_BC, p_AB])
    p_BC_h, p_AB_h = holm_layer[0], holm_layer[1]

    r_abc = rev[rev.model == "R-ABC"].set_index("seed").n_revivals
    r_r1 = rev[rev.model == "R1"].set_index("seed").n_revivals
    common = r_abc.index.intersection(r_r1.index)
    p_rev, n_rev = one_sided_greater(r_abc.loc[common].values, r_r1.loc[common].values)

    med = {k: float(np.nanmedian(lt[k])) for k in ["tau_A", "tau_B", "tau_C"]}
    ordered = med["tau_C"] > med["tau_B"] > med["tau_A"]
    layer_sep_sig = (p_BC_h < 0.05) and (p_AB_h < 0.05)
    two_separated_scales = bool(ordered and layer_sep_sig)
    primary_p = float(np.nanmax([p_BC_h, p_AB_h]))

    summary = {
        "two_separated_scales": two_separated_scales,
        "primary_p_value": primary_p,
        "decision_basis": "tau_A < tau_B < tau_C with both consecutive gaps significant "
                          "(paired one-sided Wilcoxon, Holm-corrected) across "
                          f"{len(lt)} seeds",
        "median_tau_A": med["tau_A"], "median_tau_B": med["tau_B"], "median_tau_C": med["tau_C"],
        "p_tauC_gt_tauB": p_BC, "p_tauC_gt_tauB_holm": p_BC_h, "n_BC": n_BC,
        "p_tauB_gt_tauA": p_AB, "p_tauB_gt_tauA_holm": p_AB_h, "n_AB": n_AB,
        "median_revivals_R_ABC": float(r_abc.median()),
        "median_revivals_R1": float(r_r1.median()),
        "median_revivals_M0": float(rev[rev.model == "M0"].n_revivals.median()),
        "p_revivals_RABC_gt_R1": p_rev, "n_rev": n_rev,
        "note_revivals": "STM-revival corroboration; primary decision is the "
                         "autocorrelation timescale separation above.",
        "config": {"tau_max": TAU_MAX, "stm_len": seq_len, "ac_len": AC_LEN,
                   "seeds": SEEDS, "wall_seconds": round(time.time() - t_start, 1)},
    }
    (OUT / "memory_scales_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("\n================ MEMORY SCALES (item 6) ================")
    print(f"median tau_A / tau_B / tau_C = {med['tau_A']:.1f} / {med['tau_B']:.1f} / {med['tau_C']:.1f}")
    print(f"tau_C > tau_B : p={p_BC:.4g} (Holm {p_BC_h:.4g});  "
          f"tau_B > tau_A : p={p_AB:.4g} (Holm {p_AB_h:.4g})")
    print(f"median revivals  M0/R1/R-ABC = "
          f"{summary['median_revivals_M0']:.0f}/{summary['median_revivals_R1']:.0f}/"
          f"{summary['median_revivals_R_ABC']:.0f};  R-ABC>R1 p={p_rev}")
    print(f">>> two_separated_scales = {two_separated_scales}  (primary p = {primary_p:.4g})")
    print(f"wall time {(time.time()-t_start)/60:.1f} min")
    print(f"wrote {OUT/'memory_scales.csv'} + revivals + layer_timescales + summary.json")


if __name__ == "__main__":
    main()
