# B2 (M4) — Estratificação por clamp / out-of-range (MG rollouts v2)

Excluídos rollouts com `out_of_range_fraction > 0.5`. Fontes: mackey_glass_standard.csv + mackey_glass_two_delay.csv (imutáveis).

Total de rollouts: 520 | com OOR>0.5: 93 (18%) | grid_clamps max: 955 | diverged: 43

## NRMSE_150 e VPT: mediana original vs estratificada (OOR<=0.5)

| series | model | n_excl | NRMSE150 all | NRMSE150 kept | ΔNRMSE | VPT all | VPT kept | ΔVPT |
|---|---|---|---|---|---|---|---|---|
| MG_standard | AB-Markov | 0 | 0.981 | 0.981 | +0.000 | 2 | 2 | +0 |
| MG_standard | AB-embedded | 1 | 0.844 | 0.786 | -0.058 | 42 | 42 | +0 |
| MG_standard | AB-noaux-kraus | 20 | 2.427 | nan | +nan | 4 | nan | +nan |
| MG_standard | ABC-Markov | 0 | 0.981 | 0.981 | +0.000 | 2 | 2 | +0 |
| MG_standard | ABC-embedded-C-off | 0 | 0.720 | 0.720 | +0.000 | 2 | 2 | +0 |
| MG_standard | ABC-embedded-hierarchical | 1 | 1.267 | 1.264 | -0.004 | 28 | 28 | +0 |
| MG_standard | ABC-embedded-parallel | 0 | 1.242 | 1.242 | +0.000 | 28 | 28 | +0 |
| MG_standard | ABC-embedded-tied | 2 | 1.058 | 1.044 | -0.014 | 36 | 36 | +0 |
| MG_standard | ABC-noaux-hierarchical | 10 | 0.835 | 0.780 | -0.055 | 1 | 1 | +0 |
| MG_standard | ABC-noaux-kraus | 10 | 0.835 | 0.780 | -0.055 | 1 | 1 | +0 |
| MG_standard | ABC-noaux-shuffled-history | 0 | 1.325 | 1.325 | +0.000 | 3 | 3 | +0 |
| MG_standard | ABC-noaux-tied | 0 | 0.662 | 0.662 | +0.000 | 42 | 42 | +0 |
| MG_standard | M0-noaux | 19 | 5.018 | 1.512 | -3.505 | 12 | 15 | +3 |
| MG_two_delay | AB-Markov | 0 | 0.380 | 0.380 | +0.000 | 3 | 3 | +0 |
| MG_two_delay | AB-embedded | 0 | 0.006 | 0.006 | +0.000 | 1000 | 1000 | +0 |
| MG_two_delay | AB-noaux-kraus | 1 | 0.042 | 0.041 | -0.001 | 634 | 656 | +22 |
| MG_two_delay | ABC-Markov | 0 | 0.380 | 0.380 | +0.000 | 3 | 3 | +0 |
| MG_two_delay | ABC-embedded-C-off | 0 | 0.027 | 0.027 | +0.000 | 1000 | 1000 | +0 |
| MG_two_delay | ABC-embedded-hierarchical | 0 | 0.016 | 0.016 | +0.000 | 1000 | 1000 | +0 |
| MG_two_delay | ABC-embedded-parallel | 0 | 0.018 | 0.018 | +0.000 | 1000 | 1000 | +0 |
| MG_two_delay | ABC-embedded-tied | 8 | 0.196 | 0.095 | -0.101 | 86 | 182 | +96 |
| MG_two_delay | ABC-noaux-hierarchical | 0 | 0.017 | 0.017 | +0.000 | 1000 | 1000 | +0 |
| MG_two_delay | ABC-noaux-kraus | 0 | 0.017 | 0.017 | +0.000 | 1000 | 1000 | +0 |
| MG_two_delay | ABC-noaux-shuffled-history | 0 | 1.345 | 1.345 | +0.000 | 1 | 1 | +0 |
| MG_two_delay | ABC-noaux-tied | 20 | 2.955 | nan | +nan | 1 | nan | +nan |
| MG_two_delay | M0-noaux | 1 | 0.014 | 0.013 | -0.001 | 551 | 704 | +153 |

## Modelos mais afetados pela exclusão

- **AB-noaux-kraus** (MG_standard): 20/20 excluídos; NRMSE150 2.427→nan, VPT 4→nan.
- **ABC-noaux-tied** (MG_two_delay): 20/20 excluídos; NRMSE150 2.955→nan, VPT 1→nan.
- **M0-noaux** (MG_standard): 19/20 excluídos; NRMSE150 5.018→1.512, VPT 12→15.
- **ABC-noaux-hierarchical** (MG_standard): 10/20 excluídos; NRMSE150 0.835→0.780, VPT 1→1.
- **ABC-noaux-kraus** (MG_standard): 10/20 excluídos; NRMSE150 0.835→0.780, VPT 1→1.
- **ABC-embedded-tied** (MG_two_delay): 8/20 excluídos; NRMSE150 0.196→0.095, VPT 86→182.
- **ABC-embedded-tied** (MG_standard): 2/20 excluídos; NRMSE150 1.058→1.044, VPT 36→36.
- **AB-embedded** (MG_standard): 1/20 excluídos; NRMSE150 0.844→0.786, VPT 42→42.
- **ABC-embedded-hierarchical** (MG_standard): 1/20 excluídos; NRMSE150 1.267→1.264, VPT 28→28.
- **AB-noaux-kraus** (MG_two_delay): 1/20 excluídos; NRMSE150 0.042→0.041, VPT 634→656.
- **M0-noaux** (MG_two_delay): 1/20 excluídos; NRMSE150 0.014→0.013, VPT 551→704.

## Veredito
Rollouts com alta fração out-of-range dependem de clamp silencioso (select_channel usa o canal extremo) e medem saturação, não dinâmica. A tabela acima mostra o quanto as medianas de NRMSE_150/VPT mudam ao removê-los; onde ΔNRMSE≈0 e n_excl pequeno, a conclusão do modelo é robusta ao clamp.
