#!/usr/bin/env python3
"""Phase 2 (item 3) — seed-keyed paired statistics, no GPU.

The production `run_statistics` in embedded_effective_qrc_pipeline_v2.py builds each
paired comparison by `.sort_values("seed").values` and then `paired_stats` truncates
with `a[:n], b[:n]` (min length). When a seed is missing in one arm this pairs
mismatched seeds by *position*. This script recomputes every paired comparison with an
explicit merge on the `seed` key (intersection of seeds present in BOTH arms), records
the seeds actually used, and marks a comparison `inconclusive` when the common set has
< CFG.min_decision_seeds (20) seeds. It also reports, for each comparison, whether the
positional method would have used a different n or reached a different significance
verdict, to quantify item 3's impact.

Reads the immutable per-seed CSVs in results_abc_comparison_v2/. Writes ONLY to
results_review/. Does not import torch paths that need a GPU.

Run from the repository root:  python3 experiments_review/phase2_paired_stats_seed_merged.py
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


def _series_by_seed(df: pd.DataFrame, model: str, value: str) -> pd.Series:
    """Collapse to one value per seed (mean over any duplicate rows), indexed by seed."""
    g = df[df.model == model]
    if g.empty:
        return pd.Series(dtype=float)
    return g.groupby("seed")[value].mean().sort_index()


def _positional(a: pd.Series, b: pd.Series, larger_better: bool):
    """Reproduce the production behaviour: sort by seed, then truncate to min length."""
    av = a.sort_index().values
    bv = b.sort_index().values
    n = min(len(av), len(bv))
    st = p.paired_stats(av[:n], bv[:n], larger_better=larger_better)
    return st, n


def _seed_merged(a: pd.Series, b: pd.Series, larger_better: bool):
    common = a.index.intersection(b.index)
    st = p.paired_stats(a.loc[common].values, b.loc[common].values, larger_better=larger_better)
    return st, sorted(int(s) for s in common)


def compare(df, model_a, model_b, value, larger_better, family, metric, task):
    """model_a is the '2nd' element (reported as a-b). Mirrors run_statistics labelling."""
    a = _series_by_seed(df, model_a, value)
    b = _series_by_seed(df, model_b, value)
    if a.empty or b.empty:
        return None
    st_new, common = _seed_merged(a, b, larger_better)
    st_old, n_old = _positional(a, b, larger_better)
    eff = p.orient_effect(st_new, larger_better, report="a_minus_b")
    n_common = len(common)
    a_only = sorted(int(s) for s in a.index.difference(b.index))
    b_only = sorted(int(s) for s in b.index.difference(a.index))
    row = {
        "family": family, "metric": metric, "task": task,
        "comparison": f"{model_a} vs {model_b}",
        "n_common": n_common, "n_a": len(a), "n_b": len(b),
        "n_positional": n_old, "misaligned_seeds": int(len(a_only) + len(b_only)),
        "seeds_a_only": ",".join(map(str, a_only)) or "-",
        "seeds_b_only": ",".join(map(str, b_only)) or "-",
        "mean_a": st_new.get("mean_a"), "mean_b": st_new.get("mean_b"),
        "mean_diff_a_minus_b": eff["mean_diff"],
        "ci95_lo": eff["ci95_lo"], "ci95_hi": eff["ci95_hi"], "cohen_dz": eff["cohen_dz"],
        "wins": eff["wins"], "losses": eff["losses"],
        "p_wilcoxon": st_new.get("p_wilcoxon"), "p_ttest": st_new.get("p_ttest"),
        "inconclusive_lt_min": n_common < MIN,
        # positional counterparts, to quantify item-3 impact
        "p_wilcoxon_positional": st_old.get("p_wilcoxon"),
        "mean_diff_positional_internal": st_old.get("mean_diff"),
        "seeds_common": ",".join(map(str, common)),
    }
    return row


def main():
    rows = []
    cap = p.load_csv(SRC / "multiscale_capacities.csv")
    if not cap.empty:
        cap = cap[~cap.task.astype(str).str.startswith("delay_pair")]
        for task in sorted(cap.task.dropna().unique()):
            for b, a in p.CAPACITY_COMPARISONS:
                r = compare(cap[cap.task == task], a, b, "capacity", True, "capacity", "capacity", task)
                if r:
                    rows.append(r)
    for fname, fam in [("mackey_glass_standard.csv", "MG_standard"),
                       ("mackey_glass_two_delay.csv", "MG_two_delay")]:
        mg = p.load_csv(SRC / fname)
        if mg.empty:
            continue
        for metric, larger in [("mse_150", False), ("nrmse_150", False),
                               ("r2_150", True), ("valid_prediction_time", True)]:
            for b, a in p.MG_COMPARISONS:
                r = compare(mg, a, b, metric, larger, fam, metric, fam)
                if r:
                    rows.append(r)
    paper = p.load_csv(SRC / "paper_replication_mackey_glass.csv")
    if not paper.empty:
        paper = paper.copy()
        paper["model"] = "omega_" + paper.omega.astype(str)
        for metric, larger in [("mse_150", False), ("nrmse_150", False),
                               ("r2_150", True), ("valid_prediction_time", True)]:
            for a_lbl, b_lbl in [("omega_0.5", "omega_1.0"), ("omega_0.5", "omega_0.0")]:
                r = compare(paper, a_lbl, b_lbl, metric, larger, "paper_MG", metric, "paper_MG")
                if r:
                    r["comparison"] = r["comparison"].replace("omega_", "Omega")
                    rows.append(r)
    nm = p.load_csv(SRC / "paper_replication_nonmarkovianity.csv")
    if not nm.empty:
        nm = nm.copy()
        nm["model"] = "omega_" + nm.omega.astype(str)
        r = compare(nm, "omega_0.5", "omega_1.0", "nonmarkovianity", True,
                    "paper_nonmark", "nonmarkovianity(backflow proxy)", "paper_nonmark")
        if r:
            r["comparison"] = "Omega0.5 vs Omega1.0"
            rows.append(r)
    stm = p.load_csv(SRC / "paper_replication_stm.csv")
    if not stm.empty:
        tail = stm[stm.tau >= 10]
        agg = tail.groupby(["omega", "seed"])["capacity_ols"].mean().reset_index()
        agg["model"] = "omega_" + agg.omega.astype(str)
        r = compare(agg, "omega_0.5", "omega_1.0", "capacity_ols", True,
                    "paper_STM", "stm_capacity_tau_ge_10", "paper_STM")
        if r:
            r["comparison"] = "Omega0.5 vs Omega1.0"
            rows.append(r)

    df = pd.DataFrame(rows)
    # Holm across the decidable set only (n_common >= MIN); inconclusive rows excluded.
    df["p_wilcoxon_holm"] = np.nan
    dec = df.index[~df.inconclusive_lt_min]
    if len(dec):
        df.loc[dec, "p_wilcoxon_holm"] = p.holm(df.loc[dec, "p_wilcoxon"].values)
    df["ci_excludes_zero"] = (df.ci95_lo > 0) | (df.ci95_hi < 0)
    df["significant"] = (df.p_wilcoxon_holm < 0.05) & df.ci_excludes_zero & (~df.inconclusive_lt_min)

    # Item-3 impact: did positional pairing ever use a different n than the seed-keyed set?
    df["n_changed_vs_positional"] = df.n_common != df.n_positional

    cols = ["family", "metric", "task", "comparison", "n_common", "n_positional",
            "n_changed_vs_positional", "misaligned_seeds", "inconclusive_lt_min",
            "mean_diff_a_minus_b", "ci95_lo", "ci95_hi", "cohen_dz", "wins", "losses",
            "p_wilcoxon", "p_wilcoxon_holm", "ci_excludes_zero", "significant",
            "mean_a", "mean_b", "n_a", "n_b", "seeds_a_only", "seeds_b_only", "seeds_common"]
    df = df[cols]
    df.to_csv(OUT / "paired_statistics_seed_merged.csv", index=False)

    summary = {
        "min_decision_seeds": MIN,
        "n_comparisons": int(len(df)),
        "n_inconclusive": int(df.inconclusive_lt_min.sum()),
        "n_significant": int(df.significant.sum()),
        "n_where_positional_used_different_n": int(df.n_changed_vs_positional.sum()),
        "comparisons_with_misaligned_seeds": df[df.misaligned_seeds > 0][
            ["family", "metric", "comparison", "n_common", "n_positional",
             "seeds_a_only", "seeds_b_only"]].to_dict("records"),
    }
    (OUT / "paired_statistics_seed_merged_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"comparisons: {len(df)}")
    print(f"inconclusive (n_common < {MIN}): {int(df.inconclusive_lt_min.sum())}")
    print(f"significant (Holm<0.05 & CI excl 0 & n>={MIN}): {int(df.significant.sum())}")
    print(f"comparisons where positional pairing used a DIFFERENT n: "
          f"{int(df.n_changed_vs_positional.sum())}")
    mis = df[df.misaligned_seeds > 0]
    if len(mis):
        print("\nComparisons with seed misalignment (item-3 exposure):")
        for _, r in mis.iterrows():
            print(f"  [{r.family}/{r.metric}] {r.comparison}: "
                  f"n_common={r.n_common} vs n_positional={r.n_positional} "
                  f"(a_only={r.seeds_a_only} b_only={r.seeds_b_only})")
    else:
        print("\nNo seed misalignment: all compared arms share identical seed sets "
              "in the v2 CSVs (positional == seed-keyed here). Guard added for provenance.")
    print(f"\nwrote {OUT/'paired_statistics_seed_merged.csv'}")


if __name__ == "__main__":
    main()
