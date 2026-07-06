"""C1 (G2) fits: refit the scaling law with the full 4 gamma points.

Combines v5's existing STM curves (gamma 0.02/0.05/0.1 + all M0) with the newly
computed gamma=0.2 AB-embedded curves (c1_gamma02), then calls v5.build_fits()
to regenerate the fits with 4 points + bootstrap CIs. Writes
scaling_fits_corrected.json + scaling_law_final.md to results_corrections_v6/.
Reads originals read-only.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import qrc_pipeline as v2  # noqa: E402
import qrc_experiments_scaling as v5  # noqa: E402

V6 = Path(ROOT) / "results_corrections_v6"
C1 = V6 / "c1_gamma02"


def main():
    v5_stm = pd.read_csv(Path(ROOT) / "results_extra_v5" / "dynamical_sweep_stm.csv")
    c1_stm = pd.read_csv(C1 / "dynamical_sweep_stm.csv")
    new_ab = c1_stm[(c1_stm.config == "g0.2_epi4") & (c1_stm.model == "AB-embedded")]
    n_new = new_ab.seed.nunique()
    frames = [v5_stm, new_ab]
    # Fold in the C2-recovered gamma=0.02 seed15 so that gamma point uses 20 seeds.
    seed15_path = V6 / "c2_recovery" / "v5_g0.02_seed15_stm.csv"
    if seed15_path.exists():
        s15 = pd.read_csv(seed15_path)
        frames.append(s15)
        print(f"folding in C2 g0.02 seed15 ({s15.seed.nunique()} seed)")
    print(f"combining: v5 rows={len(v5_stm)}, new g0.2_epi4 AB seeds={n_new}")
    combined = pd.concat(frames, ignore_index=True)
    combined_path = V6 / "scaling_stm_combined.csv"
    combined.to_csv(combined_path, index=False)

    # Point v5 fit machinery at the combined data + corrected outputs.
    v5.STM_CSV = combined_path
    v5.FITS_JSON = V6 / "scaling_fits_corrected.json"
    fits = v5.build_fits()
    fits = json.loads((V6 / "scaling_fits_corrected.json").read_text())

    gs = fits["gamma_sweep"]["gammas"]
    tau = fits["gamma_sweep"]["tau_mem"]
    fm = fits["gamma_sweep"]["tau_FM"]
    exc = fits["gamma_sweep"]["excess"]
    fa = fits["fit_a_tau_vs_inv_gamma"]
    fb = fits["fit_b_excess_vs_inv_gamma"]
    cons = fits["consistency_v3_v4"]

    L = ["# C1 (G2) — Lei de escala do horizonte de memória (4 pontos, corrigida)", "",
         f"Gerado combinando as curvas STM de gamma=0.02/0.05/0.1 (v5, imutável) com "
         f"gamma=0.2 AB-embedded recomputado ({n_new}/20 seeds). Threshold C>0.1.", "",
         "## Pontos (AB-embedded, eta=pi/4)", "",
         "| gamma | 1/gamma | tau_mem | tau_FM (M0) | excess |", "|---|---|---|---|---|"]
    for g, t, f, e in zip(gs, tau, fm, exc):
        L.append(f"| {g} | {1.0/g:.1f} | {t:.0f} | {f:.0f} | {e:.0f} |")
    def fmt(x, nd=3):
        return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "n/a"

    L += ["", "## Ajustes (log-log)", "",
          f"- **fit_a: tau_mem ~ (1/gamma)^p** — expoente p = **{fmt(fa['exponent'])}** "
          f"(IC95 boot [{fmt(fa.get('exp_ci95_lo'))}, {fmt(fa.get('exp_ci95_hi'))}]), "
          f"intercepto={fmt(fa['intercept'])}, R2={fmt(fa['r2'])}, n={fa['n']}.",
          f"- **fit_b: (tau_mem - tau_FM) ~ (1/gamma)^p** — expoente p = **{fmt(fb['exponent'])}** "
          f"(n={fb['n']}). Excesso = {[int(e) for e in exc]}: tau_mem ~ tau_FM (M0), "
          "quase sem excesso positivo, logo o ajuste log-log é degenerado (só "
          "pontos com excesso>0 entram) — o ganho de horizonte do embedded sobre "
          "o M0 é marginal nestes gammas.",
          "",
          "## Consistência com dimensão (v3) e topologia (v4) em gamma=0.1", ""]
    ref = cons.get("reference_tau_mem_gamma01")
    L.append(f"Referência tau_mem(gamma=0.1) = **{ref}**.")
    L.append(f"- v3 (d_B): {cons.get('v3_dimension_memory_range')}")
    L.append(f"- v4 (topologia): {cons.get('v4_topology_memory_range')}")
    allpts = cons.get("all_gamma01_points")
    L.append(f"- pontos gamma=0.1 (v3+v4): {allpts}")
    L.append(f"- **consistentes com a referência (|Δ|<=5): {cons.get('consistent')}**")
    L += ["",
          "Os pontos de v3 (dimensão d_B) e v4 (topologia) são medidos a gamma=0.1 "
          "fixo, portanto 'caem na curva' se forem consistentes com o ponto gamma=0.1 "
          f"da lei de escala (tau_mem~{ref}). O teste acima confirma/refuta isso.", "",
          "## Comparação com o fit original (3 pontos)", ""]
    try:
        orig = json.loads((Path(ROOT) / "results_extra_v5" / "scaling_fits.json").read_text())
        oa = orig["fit_a_tau_vs_inv_gamma"]
        L.append(f"- original (3 pts): p={oa['exponent']:.3f}, R2={oa['r2']:.3f}, n={oa['n']}")
        L.append(f"- corrigido (4 pts): p={fa['exponent']:.3f}, R2={fa['r2']:.3f}, n={fa['n']}")
    except Exception as exc:  # noqa: BLE001
        L.append(f"_original fit indisponível: {exc}_")
    (V6 / "scaling_law_final.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
