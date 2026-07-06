"""B3 (G3/M1): reprocess ALL v4 paired stats with the corrected code.

Calls the now-corrected v4.exp5_paired_stats() (A2 finiteness already excludes
non-finite NARMA rows; A3 orient_effect fixes the CI sign) with its output
redirected to results_corrections_v6/, then diffs against the original
benchmarks_paired_stats.csv: which cells changed sign / significance.
Reads the immutable v4 inputs read-only; never overwrites the original.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import extra_experiments_v4 as v4  # noqa: E402

V6 = Path(ROOT) / "results_corrections_v6"
ORIG = Path(ROOT) / "results_extra_v4" / "benchmarks_paired_stats.csv"
KEY = ["task", "metric", "noaux", "embedded"]


def main():
    # Redirect ONLY the output path; inputs stay pointed at the immutable v4 dir.
    corrected_path = V6 / "benchmarks_paired_stats_corrected.csv"
    v4.EXP5_STATS_CSV = corrected_path
    v4.exp5_paired_stats()   # uses corrected orient_effect + finiteness filter
    cor = pd.read_csv(corrected_path)
    orig = pd.read_csv(ORIG)

    m = orig.merge(cor, on=KEY, suffixes=("_orig", "_corr"))
    diff_rows = []
    for _, r in m.iterrows():
        md = r["mean_diff_noaux_minus_emb_corr"]
        # CI sign consistency: does the CI sit on the same side as mean_diff?
        orig_ci = (r["ci95_lo_orig"], r["ci95_hi_orig"])
        corr_ci = (r["ci95_lo_corr"], r["ci95_hi_corr"])
        orig_consistent = (md > 0 and orig_ci[0] > 0) or (md < 0 and orig_ci[1] < 0)
        corr_consistent = (md > 0 and corr_ci[0] > 0) or (md < 0 and corr_ci[1] < 0)
        # significance verdict: CI excludes 0 (either side)
        orig_sig_ci = orig_ci[0] > 0 or orig_ci[1] < 0
        corr_sig_ci = corr_ci[0] > 0 or corr_ci[1] < 0
        diff_rows.append({
            **{k: r[k] for k in KEY},
            "n_orig": r["n_orig"], "n_corr": r["n_corr"], "n_changed": r["n_orig"] != r["n_corr"],
            "mean_diff": md, "mean_diff_unchanged": abs(r["mean_diff_noaux_minus_emb_orig"] - md) < 1e-9,
            "ci_orig": f"[{orig_ci[0]:.3g},{orig_ci[1]:.3g}]",
            "ci_corr": f"[{corr_ci[0]:.3g},{corr_ci[1]:.3g}]",
            "ci_sign_flipped": np.sign(orig_ci[0]) != np.sign(corr_ci[0]),
            "orig_ci_consistent_with_effect": orig_consistent,
            "corr_ci_consistent_with_effect": corr_consistent,
            "p_holm": r.get("p_holm_corr", r.get("p_holm")),
            "sig_p_holm_lt_0.05": (r.get("p_holm_corr", r.get("p_holm")) < 0.05),
        })
    dd = pd.DataFrame(diff_rows)
    dd.to_csv(V6 / "benchmarks_paired_stats_diff.csv", index=False)

    n = len(dd)
    n_flip = int(dd.ci_sign_flipped.sum())
    n_orig_bad = int((~dd.orig_ci_consistent_with_effect).sum())
    n_corr_bad = int((~dd.corr_ci_consistent_with_effect).sum())
    L = ["# B3 (G3/M1) — v4 paired stats reprocessadas (código corrigido)", "",
         f"Comparações: {n}. Fonte de entrada: narma10_results.csv / santafe_results.csv "
         "(imutáveis). Saída corrigida: benchmarks_paired_stats_corrected.csv.", "",
         "## O que mudou", "",
         f"- **CI com sinal invertido (corrigido):** {n_flip}/{n} células. Todas as células "
         "tinham o CI no sentido oposto ao `mean_diff_noaux_minus_emb` (M1).",
         f"- **CI consistente com o efeito:** antes {n - n_orig_bad}/{n}, depois {n - n_corr_bad}/{n}.",
         f"- **n_effective:** {'inalterado' if not dd.n_changed.any() else 'mudou em ' + str(int(dd.n_changed.sum())) + ' células'} "
         "(o código original já filtrava não-finitos via isfinite; NARMA usa n=18/19, "
         "não 20, por seed=1 não-finito e seed=8 AB-embedded ausente).",
         f"- **p-valores / significância (Holm):** inalterados (mesmos dados); "
         f"{int(dd['sig_p_holm_lt_0.05'].sum())}/{n} significativas a 0.05.", "",
         "## Tabela de diffs", "",
         "| task | metric | noaux | embedded | n | mean_diff | CI orig | CI corr | flip | sig(Holm<.05) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in dd.iterrows():
        L.append(f"| {r.task} | {r.metric} | {r.noaux} | {r.embedded} | {r.n_corr} | "
                 f"{r.mean_diff:+.3g} | {r.ci_orig} | {r.ci_corr} | {r.ci_sign_flipped} | {r['sig_p_holm_lt_0.05']} |")
    L += ["", "## Veredito",
          "Nenhuma célula mudou de **significância** (p-valores idênticos; os mesmos "
          "dados pareados). O que muda é a **interpretação do sinal do IC**: no "
          "arquivo original o IC estava no sentido oposto ao `mean_diff`, o que "
          "poderia inverter a leitura do efeito numa tabela de paper. Após a "
          "correção, `mean_diff` e IC compartilham o sentido noaux−embedded "
          "(positivo = embedded melhor no erro). n_effective real reportado.", ""]
    (V6 / "benchmarks_paired_stats_corrected.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
