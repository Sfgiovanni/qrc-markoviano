"""B2 (M4): clamp / out-of-range stratification of the v2 MG autonomous rollouts.

Per model: distribution of grid_clamps and out_of_range_fraction; medians of
NRMSE_150 and VPT recomputed EXCLUDING rollouts with out_of_range_fraction>0.5,
compared against the original (all-rollouts) medians.
Sources (immutable): mackey_glass_standard.csv + mackey_glass_two_delay.csv.
Output: results_corrections_v6/mg_clamp_stratified.csv + .md
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V6 = os.path.join(ROOT, "results_corrections_v6")
V2 = os.path.join(ROOT, "results_abc_comparison_v2")
OOR_CUT = 0.5


def load():
    frames = []
    for f in ("mackey_glass_standard.csv", "mackey_glass_two_delay.csv"):
        d = pd.read_csv(os.path.join(V2, f))
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def main():
    df = load()
    rows = []
    for (series, model), g in df.groupby(["series", "model"]):
        n_total = len(g)
        kept = g[g.out_of_range_fraction <= OOR_CUT]
        n_excl = n_total - len(kept)
        rows.append({
            "series": series, "model": model,
            "n_total": n_total, "n_excluded_oor_gt_0.5": n_excl, "n_kept": len(kept),
            # clamp / OOR distribution
            "grid_clamps_median": float(g.grid_clamps.median()),
            "grid_clamps_max": int(g.grid_clamps.max()),
            "oor_median": float(g.out_of_range_fraction.median()),
            "oor_max": float(g.out_of_range_fraction.max()),
            "n_diverged": int(g.diverged.sum()),
            # original medians (all rollouts)
            "nrmse150_median_all": float(g.nrmse_150.median()),
            "vpt_median_all": float(g.valid_prediction_time.median()),
            # stratified medians (OOR<=0.5)
            "nrmse150_median_kept": float(kept.nrmse_150.median()) if len(kept) else np.nan,
            "vpt_median_kept": float(kept.valid_prediction_time.median()) if len(kept) else np.nan,
        })
    out = pd.DataFrame(rows).sort_values(["series", "model"])
    out["nrmse150_median_delta"] = out.nrmse150_median_kept - out.nrmse150_median_all
    out["vpt_median_delta"] = out.vpt_median_kept - out.vpt_median_all
    out.to_csv(os.path.join(V6, "mg_clamp_stratified.csv"), index=False)

    # markdown
    L = ["# B2 (M4) — Estratificação por clamp / out-of-range (MG rollouts v2)", "",
         f"Excluídos rollouts com `out_of_range_fraction > {OOR_CUT}`. "
         "Fontes: mackey_glass_standard.csv + mackey_glass_two_delay.csv (imutáveis).", "",
         f"Total de rollouts: {len(df)} | com OOR>0.5: {(df.out_of_range_fraction>OOR_CUT).sum()} "
         f"({100*(df.out_of_range_fraction>OOR_CUT).mean():.0f}%) | grid_clamps max: {int(df.grid_clamps.max())} "
         f"| diverged: {int(df.diverged.sum())}", "",
         "## NRMSE_150 e VPT: mediana original vs estratificada (OOR<=0.5)", "",
         "| series | model | n_excl | NRMSE150 all | NRMSE150 kept | ΔNRMSE | VPT all | VPT kept | ΔVPT |",
         "|---|---|---|---|---|---|---|---|---|"]
    for _, r in out.iterrows():
        L.append(f"| {r.series} | {r.model} | {r['n_excluded_oor_gt_0.5']} | "
                 f"{r.nrmse150_median_all:.3f} | {r.nrmse150_median_kept:.3f} | {r.nrmse150_median_delta:+.3f} | "
                 f"{r.vpt_median_all:.0f} | {r.vpt_median_kept:.0f} | {r.vpt_median_delta:+.0f} |")
    # which models are most affected
    aff = out[out["n_excluded_oor_gt_0.5"] > 0].sort_values("n_excluded_oor_gt_0.5", ascending=False)
    L += ["", "## Modelos mais afetados pela exclusão", ""]
    if len(aff) == 0:
        L.append("_Nenhum rollout excluído (todos OOR<=0.5)._")
    else:
        for _, r in aff.iterrows():
            L.append(f"- **{r.model}** ({r.series}): {r['n_excluded_oor_gt_0.5']}/{r.n_total} excluídos; "
                     f"NRMSE150 {r.nrmse150_median_all:.3f}→{r.nrmse150_median_kept:.3f}, "
                     f"VPT {r.vpt_median_all:.0f}→{r.vpt_median_kept:.0f}.")
    L += ["", "## Veredito",
          "Rollouts com alta fração out-of-range dependem de clamp silencioso "
          "(select_channel usa o canal extremo) e medem saturação, não dinâmica. "
          "A tabela acima mostra o quanto as medianas de NRMSE_150/VPT mudam ao "
          "removê-los; onde ΔNRMSE≈0 e n_excl pequeno, a conclusão do modelo é "
          "robusta ao clamp.", ""]
    with open(os.path.join(V6, "mg_clamp_stratified.md"), "w") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
