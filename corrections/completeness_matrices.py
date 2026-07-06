"""B4 (G2/M2): run validate_run retroactively on v2, v3, v4, v5.

Reads the immutable results_* dirs, writes 4 completeness matrices + a listing
to results_corrections_v6/. Does NOT touch the originals.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import qrc_pipeline as v2  # noqa: E402
import qrc_experiments_robustness as v4  # noqa: E402
import qrc_experiments_scaling as v5  # noqa: E402

V6 = os.path.join(ROOT, "results_corrections_v6")

# ---- expected specs -------------------------------------------------------
EXPECTED_V2 = {"tables": {
    "paper_replication_mackey_glass.csv": {
        "cell_combos": [{"model": "AB-embedded", "omega": w} for w in (0.0, 0.5, 1.0)],
        "seed_col": "seed", "seeds": list(range(100)),
        "value_cols": ["mse_150", "nrmse_150", "r2_150"]},
    "multiscale_capacities.csv": {  # 15 models over the 20 eval seeds
        "cell_combos": None,  # filled from data models below
        "seed_col": "seed", "seeds": list(range(20)), "value_cols": ["capacity"]},
}}
EXPECTED_V3 = {"tables": {
    "aux_dimension_sweep.csv": {
        "cell_combos": [{"d_B": d} for d in (2, 4, 8, 16, 32, 64)],
        "seed_col": "seed", "seeds": list(range(20)), "value_cols": ["capacity"]},
}}


def _fill_multiscale_models():
    df = pd.read_csv(os.path.join(ROOT, "results_abc_comparison_v2", "multiscale_capacities.csv"))
    models = sorted(df.model.dropna().unique().tolist())
    EXPECTED_V2["tables"]["multiscale_capacities.csv"]["cell_combos"] = [{"model": m} for m in models]


def run_one(tag, results_dir, expected):
    rep = v2.validate_run(results_dir, expected,
                          out_name=f"{tag}_completeness_matrix.csv", out_dir=V6)
    print(f"\n=== {tag}: status={rep['status']} missing={rep['n_missing']} nonfinite={rep['n_nonfinite']} ===")
    for tname, t in rep["tables"].items():
        print(f"  {tname}: effective={t['n_present']} missing={t['n_missing']} nonfinite={t['n_nonfinite']} -> {t['status']}")
    return rep


def main():
    _fill_multiscale_models()
    reps = {}
    reps["v2"] = run_one("v2", os.path.join(ROOT, "results_abc_comparison_v2"), EXPECTED_V2)
    reps["v3"] = run_one("v3", os.path.join(ROOT, "results_extra_v3"), EXPECTED_V3)
    reps["v4"] = run_one("v4", v4.V4_DIR, v4.expected_tables())
    reps["v5"] = run_one("v5", v5.V5_DIR, v5.expected_tables(range(20)))

    # Consolidated listing of exactly what is missing / non-finite.
    lines = ["# B4 — Consolidated completeness (validate_run retroactive on v2-v5)", ""]
    for tag, rep in reps.items():
        lines.append(f"## {tag}: **{rep['status']}** (missing={rep['n_missing']}, non-finite={rep['n_nonfinite']})")
        m = pd.read_csv(os.path.join(V6, f"{tag}_completeness_matrix.csv"))
        for status in ("missing", "nonfinite"):
            sub = m[m.status == status]
            if len(sub) == 0:
                continue
            lines.append(f"\n**{status} ({len(sub)} cells):**\n")
            for table in sorted(sub.table.unique()):
                st = sub[sub.table == table]
                keycols = [c for c in st.columns if c not in ("status", "n_rows", "seed", "table")
                           and st[c].notna().any()]
                grp = st.groupby(keycols, dropna=False).seed.apply(
                    lambda s: ",".join(map(str, sorted(s.unique())))).reset_index() if keycols \
                    else pd.DataFrame([{"seed": ",".join(map(str, sorted(st.seed.unique())))}])
                for _, r in grp.iterrows():
                    desc = ", ".join(f"{c}={r[c]}" for c in keycols)
                    lines.append(f"- {table} [{desc}]: seeds [{r['seed']}]")
        lines.append("")
    with open(os.path.join(V6, "b4_completeness_listing.md"), "w") as fh:
        fh.write("\n".join(lines))
    print("\nWrote b4_completeness_listing.md + 4 matrices to results_corrections_v6/")


if __name__ == "__main__":
    main()
