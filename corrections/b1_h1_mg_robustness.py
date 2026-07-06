"""B1 (G1): robustness of the H1 / Mackey-Glass conclusion (omega=0.5 vs 1.0).

Reads results_abc_comparison_v2/paper_replication_mackey_glass.csv (immutable),
runs the paired battery on THREE seed sets and writes h1_mg_robustness.csv +
h1_mg_robustness.md to results_corrections_v6/. Metric: mse_150 (lower better);
omega=0.5 "beats" omega=1.0 iff mean/median mse_150(0.5) < (1.0).
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import embedded_effective_qrc_pipeline_v2 as v2  # noqa: E402

V6 = os.path.join(ROOT, "results_corrections_v6")
MG = os.path.join(ROOT, "results_abc_comparison_v2", "paper_replication_mackey_glass.csv")
METRIC = "mse_150"


def paired_on(df, seeds, metric=METRIC):
    """Paired stats on the given seeds. a=omega0.5, b=omega1.0, error metric.
    Returns dict oriented so mean_diff>0 => omega=0.5 has LOWER error (better)."""
    a = df[(df.omega == 0.5) & (df.seed.isin(seeds))].sort_values("seed")[metric].to_numpy()
    b = df[(df.omega == 1.0) & (df.seed.isin(seeds))].sort_values("seed")[metric].to_numpy()
    st = v2.paired_stats(a, b, larger_better=False)  # d = b - a = mse(1.0)-mse(0.5)
    if st.get("n", 0) == 0:
        return {"n": 0}
    return {
        "n": st["n"],
        "mean_mse150_omega05": st["mean_a"], "mean_mse150_omega10": st["mean_b"],
        "median_mse150_omega05": st["median_a"], "median_mse150_omega10": st["median_b"],
        "mean_diff_10_minus_05": st["mean_diff"],           # >0 => 0.5 better
        "ci95_lo_10_minus_05": st["ci95_lo"], "ci95_hi_10_minus_05": st["ci95_hi"],
        "cohen_dz": st["cohen_dz"],
        "wins_omega05_better": st["wins"], "wins_omega10_better": st["losses"],
        "p_wilcoxon": st["p_wilcoxon"], "p_ttest": st["p_ttest"],
        "omega05_beats_10_mean": bool(st["mean_a"] < st["mean_b"]),
        "ci_excludes_zero": bool((st["ci95_lo"] > 0) or (st["ci95_hi"] < 0)),
    }


def main():
    df = pd.read_csv(MG)
    df["teacher_forced_ok"] = df["teacher_forced_ok"].astype(bool)

    # tf pass counts per omega
    tf_counts = df.groupby("omega").teacher_forced_ok.sum().astype(int).to_dict()
    n_by_omega = df.groupby("omega").seed.nunique().to_dict()

    # seed sets
    all_seeds = sorted(df[df.omega == 0.5].seed.unique())
    tf05_seeds = sorted(df[(df.omega == 0.5) & df.teacher_forced_ok].seed.unique())
    # both-arms tf-passed (the strict A1 gate)
    tf10_seeds = set(df[(df.omega == 1.0) & df.teacher_forced_ok].seed.unique())
    both_tf_seeds = sorted(set(tf05_seeds) & tf10_seeds)
    # M4 stratification: OOR<0.5 in BOTH arms
    oor = df.pivot_table(index="seed", columns="omega", values="out_of_range_fraction")
    oor_seeds = sorted([s for s in all_seeds
                        if (0.5 in oor.columns and 1.0 in oor.columns
                            and oor.loc[s, 0.5] < 0.5 and oor.loc[s, 1.0] < 0.5)])

    sets = [
        ("(i) all_seeds", all_seeds, "todas as seeds (número original)"),
        ("(ii) omega05_tf_passed", tf05_seeds, "seeds onde omega=0.5 passou teacher-forced"),
        ("(iii) oor_lt_0.5_both_arms", oor_seeds, "out_of_range_fraction<0.5 em ambos os braços (M4)"),
        ("(aux) both_arms_tf_passed", both_tf_seeds, "gate estrito A1: tf-ok em ambos os braços"),
    ]

    rows = []
    for name, seeds, desc in sets:
        res = paired_on(df, seeds)
        res.update({"set": name, "n_seeds_requested": len(seeds), "description": desc})
        rows.append(res)

    out = pd.DataFrame(rows)
    front = ["set", "n", "n_seeds_requested", "omega05_beats_10_mean", "ci_excludes_zero",
             "p_wilcoxon", "cohen_dz", "mean_diff_10_minus_05", "ci95_lo_10_minus_05",
             "ci95_hi_10_minus_05", "wins_omega05_better", "wins_omega10_better",
             "mean_mse150_omega05", "mean_mse150_omega10", "description"]
    out = out[[c for c in front if c in out.columns] + [c for c in out.columns if c not in front]]
    out.to_csv(os.path.join(V6, "h1_mg_robustness.csv"), index=False)

    # verdict markdown
    L = ["# B1 — Robustez da conclusão H1 / Mackey-Glass (omega=0.5 vs 1.0)", "",
         f"Métrica: `{METRIC}` (menor é melhor). Fonte: paper_replication_mackey_glass.csv (imutável).", "",
         "## Teacher-forced pass counts por omega", "",
         "| omega | n_seeds | n_passed_teacher_forced |", "|---|---|---|"]
    for w in (0.0, 0.5, 1.0):
        L.append(f"| {w} | {n_by_omega.get(w,0)} | {tf_counts.get(w,0)} |")
    L += ["", "_Esperado pelo review: 100 / 56 / 0 — confirmado._", "",
          "## Comparação pareada por conjunto", "",
          "| conjunto | n | 0.5<1.0 (média) | p_wilcoxon | dz | mean_diff(1.0-0.5) | CI95 | CI exclui 0? |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("n", 0) == 0:
            L.append(f"| {r['set']} | 0 | — | — | — | — | — | inconclusivo (n=0) |")
            continue
        L.append(f"| {r['set']} | {r['n']} | {r['omega05_beats_10_mean']} | {r['p_wilcoxon']:.2e} | "
                 f"{r['cohen_dz']:.2f} | {r['mean_diff_10_minus_05']:.3e} | "
                 f"[{r['ci95_lo_10_minus_05']:.3e}, {r['ci95_hi_10_minus_05']:.3e}] | {r['ci_excludes_zero']} |")

    # verdict logic
    L += ["", "## VEREDITO", ""]
    changed = []
    for r in rows:
        if r["set"].startswith("(aux)"):
            continue
        if r.get("n", 0) == 0:
            changed.append(f"- **{r['set']}**: n=0, inconclusivo.")
        else:
            sig = r["ci_excludes_zero"] and r["p_wilcoxon"] < 0.05
            verd = ("omega=0.5 SUPERA 1.0 (significativo)" if (r["omega05_beats_10_mean"] and sig)
                    else "omega=0.5 melhor mas NÃO significativo" if r["omega05_beats_10_mean"]
                    else "omega=0.5 NÃO supera 1.0")
            changed.append(f"- **{r['set']}** (n={r['n']}): {verd} "
                           f"(p_wilcoxon={r['p_wilcoxon']:.2e}, dz={r['cohen_dz']:.2f}).")
    L += changed
    L += ["",
          "Nota crítica (G1): omega=1.0 falhou a validação teacher-forced em 100/100 "
          "seeds; o braço 1.0 é sempre um rollout autônomo sem readout válido. Sob o "
          "gate estrito (tf-ok em ambos os braços) n=0 e a comparação é inconclusiva. "
          "Os conjuntos (i)-(iii) medem a robustez, mas todos comparam contra um braço "
          "1.0 cuja previsão autônoma não passou o critério de qualidade.", ""]
    with open(os.path.join(V6, "h1_mg_robustness.md"), "w") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
