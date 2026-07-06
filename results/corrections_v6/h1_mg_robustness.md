# B1 — Robustez da conclusão H1 / Mackey-Glass (omega=0.5 vs 1.0)

Métrica: `mse_150` (menor é melhor). Fonte: paper_replication_mackey_glass.csv (imutável).

## Teacher-forced pass counts por omega

| omega | n_seeds | n_passed_teacher_forced |
|---|---|---|
| 0.0 | 100 | 100 |
| 0.5 | 100 | 56 |
| 1.0 | 100 | 0 |

_Esperado pelo review: 100 / 56 / 0 — confirmado._

## Comparação pareada por conjunto

| conjunto | n | 0.5<1.0 (média) | p_wilcoxon | dz | mean_diff(1.0-0.5) | CI95 | CI exclui 0? |
|---|---|---|---|---|---|---|---|
| (i) all_seeds | 100 | True | 3.54e-17 | 1.90 | 2.819e-02 | [2.522e-02, 3.107e-02] | True |
| (ii) omega05_tf_passed | 56 | True | 8.41e-11 | 2.30 | 2.876e-02 | [2.549e-02, 3.201e-02] | True |
| (iii) oor_lt_0.5_both_arms | 100 | True | 3.54e-17 | 1.90 | 2.819e-02 | [2.522e-02, 3.107e-02] | True |
| (aux) both_arms_tf_passed | 0 | — | — | — | — | — | inconclusivo (n=0) |

## VEREDITO

- **(i) all_seeds** (n=100): omega=0.5 SUPERA 1.0 (significativo) (p_wilcoxon=3.54e-17, dz=1.90).
- **(ii) omega05_tf_passed** (n=56): omega=0.5 SUPERA 1.0 (significativo) (p_wilcoxon=8.41e-11, dz=2.30).
- **(iii) oor_lt_0.5_both_arms** (n=100): omega=0.5 SUPERA 1.0 (significativo) (p_wilcoxon=3.54e-17, dz=1.90).

Nota crítica (G1): omega=1.0 falhou a validação teacher-forced em 100/100 seeds; o braço 1.0 é sempre um rollout autônomo sem readout válido. Sob o gate estrito (tf-ok em ambos os braços) n=0 e a comparação é inconclusiva. Os conjuntos (i)-(iii) medem a robustez, mas todos comparam contra um braço 1.0 cuja previsão autônoma não passou o critério de qualidade.
