#!/usr/bin/env python3
"""Phase 3B (items 2 + 8) — rollout-validity gate for Mackey-Glass forecast (T3).

No GPU: the production v2 pipeline already saves, per seed and model, the two flags the
critique asks for — `teacher_forced_ok` (the supervised readout reached R^2 >= 0.99
BEFORE the autonomous rollout) and `out_of_range_fraction` / `grid_clamps` (how much the
autonomous trajectory left the channel-grid support and had to be clamped). This script
recomputes the head-to-head comparisons under a strict validity gate:

  a seed is admissible for a comparison iff BOTH compared models, on that seed, have
  teacher_forced_ok == True AND out_of_range_fraction <= OOR_MAX (0.5).

Metrics (VPT, NRMSE_150, ...) are compared on the inner join of admissible seeds. A
comparison with fewer than CFG.min_decision_seeds (20) admissible seeds is marked
`inconclusive`. We also emit the metric stratified by out_of_range_fraction bin, so a
"win" that is really grid saturation rather than dynamics is visible.

Reads immutable mackey_glass_{standard,two_delay}.csv. Writes only to results_review/.
Run from repo root:  python3 experiments_review/phase3b_mg_rollout_gate.py
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import qrc_pipeline as p  # renamed module in this mirror  # noqa: E402

SRC = ROOT / "results_abc_comparison_v2"
OUT = ROOT / "results_review"
OUT.mkdir(exist_ok=True)
MIN = p.CFG.min_decision_seeds
OOR_MAX = 0.5  # matches the B2 stratification threshold


def admissible(df: pd.DataFrame, model: str) -> set:
    g = df[df.model == model]
    ok = g[(g.teacher_forced_ok.astype(bool)) & (g.out_of_range_fraction <= OOR_MAX)]
    return set(int(s) for s in ok.seed)


def metric_by_seed(df, model, metric):
    g = df[df.model == model]
    return g.groupby("seed")[metric].mean()


def run_series(fname, series_name, rows, strat_rows):
    mg = p.load_csv(SRC / fname)
    if mg.empty:
        return
    for b, a in p.MG_COMPARISONS:  # a is the '2nd' element, reported as a - b
        if not ((mg.model == a).any() and (mg.model == b).any()):
            continue
        adm_a, adm_b = admissible(mg, a), admissible(mg, b)
        both = sorted(adm_a & adm_b)
        present = sorted(set(mg[mg.model == a].seed) & set(mg[mg.model == b].seed))
        for metric, larger in [("valid_prediction_time", True), ("nrmse_150", False),
                               ("mse_150", False), ("r2_150", True)]:
            sa, sb = metric_by_seed(mg, a, metric), metric_by_seed(mg, b, metric)
            row = {"series": series_name, "comparison": f"{a} vs {b}", "metric": metric,
                   "n_present": len(present), "n_a_valid": len(adm_a), "n_b_valid": len(adm_b),
                   "n_gated_both_valid": len(both),
                   "inconclusive_lt_min": len(both) < MIN,
                   "gate": f"teacher_forced_ok & out_of_range<= {OOR_MAX}"}
            if both:
                av = sa.loc[both].values
                bv = sb.loc[both].values
                st = p.paired_stats(av, bv, larger_better=larger)
                eff = p.orient_effect(st, larger, report="a_minus_b")
                row.update({"mean_a": st["mean_a"], "mean_b": st["mean_b"],
                            "median_a": st["median_a"], "median_b": st["median_b"],
                            "mean_diff_a_minus_b": eff["mean_diff"],
                            "ci95_lo": eff["ci95_lo"], "ci95_hi": eff["ci95_hi"],
                            "cohen_dz": eff["cohen_dz"], "wins": eff["wins"],
                            "losses": eff["losses"], "p_wilcoxon": st["p_wilcoxon"]})
            else:
                # what the ungated (all present seeds) comparison would have said
                common = sorted(set(sa.index) & set(sb.index))
                if common:
                    st = p.paired_stats(sa.loc[common].values, sb.loc[common].values, larger_better=larger)
                    row.update({"mean_a": st["mean_a"], "mean_b": st["mean_b"],
                                "median_a": st["median_a"], "median_b": st["median_b"],
                                "ungated_note": "n_gated=0; means shown are ungated (all seeds)"})
            rows.append(row)
        # OOR stratification per model (both arms of the comparison)
        for model in (a, b):
            g = mg[mg.model == model]
            for lo, hi, lbl in [(-0.1, OOR_MAX, f"OOR<={OOR_MAX}"), (OOR_MAX, 1.1, f"OOR>{OOR_MAX}")]:
                sub = g[(g.out_of_range_fraction > lo) & (g.out_of_range_fraction <= hi)]
                if len(sub):
                    strat_rows.append({
                        "series": series_name, "model": model, "oor_bin": lbl,
                        "n": len(sub), "n_tf_ok": int(sub.teacher_forced_ok.astype(bool).sum()),
                        "median_nrmse_150": float(sub.nrmse_150.median()),
                        "median_vpt": float(sub.valid_prediction_time.median()),
                        "median_grid_clamps": float(sub.grid_clamps.median()),
                        "max_grid_clamps": int(sub.grid_clamps.max())})


def main():
    rows, strat_rows = [], []
    run_series("mackey_glass_standard.csv", "standard", rows, strat_rows)
    run_series("mackey_glass_two_delay.csv", "two_delay", rows, strat_rows)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "mg_rollout_gate.csv", index=False)
    strat = pd.DataFrame(strat_rows).drop_duplicates()
    strat.to_csv(OUT / "mg_rollout_gate_stratified.csv", index=False)

    prim = df[df.metric.isin(["valid_prediction_time", "nrmse_150"])]
    decidable = prim[~prim.inconclusive_lt_min]
    summary = {
        "gate": f"teacher_forced_ok AND out_of_range_fraction <= {OOR_MAX}, inner join over both arms",
        "min_decision_seeds": MIN,
        "n_comparisons_metric_rows": int(len(df)),
        "n_primary_rows": int(len(prim)),
        "n_primary_inconclusive": int(prim.inconclusive_lt_min.sum()),
        "n_primary_decidable": int(len(decidable)),
        "decidable_comparisons": sorted(set(
            f"{r.series}:{r.comparison}" for _, r in decidable.iterrows())),
        "inconclusive_comparisons": sorted(set(
            f"{r.series}:{r.comparison} (n_gated={r.n_gated_both_valid})"
            for _, r in prim[prim.inconclusive_lt_min].iterrows())),
    }
    (OUT / "mg_rollout_gate_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"metric-rows: {len(df)}  (primary VPT/NRMSE rows: {len(prim)})")
    print(f"primary inconclusive (n_gated<{MIN}): {int(prim.inconclusive_lt_min.sum())} / {len(prim)}")
    print("\nPer comparison (VPT), gated n and verdict:")
    for _, r in df[df.metric == "valid_prediction_time"].iterrows():
        verdict = "INCONCLUSIVE" if r.inconclusive_lt_min else "decidable"
        print(f"  [{r.series}] {r.comparison}: n_gated={r.n_gated_both_valid} "
              f"(a_valid={r.n_a_valid}, b_valid={r.n_b_valid}) -> {verdict}")
    print(f"\nwrote {OUT/'mg_rollout_gate.csv'} and _stratified.csv")


if __name__ == "__main__":
    main()
