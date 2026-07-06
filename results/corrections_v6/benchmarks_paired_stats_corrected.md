# B3 (G3/M1) — v4 paired stats reprocessadas (código corrigido)

Comparações: 16. Fonte de entrada: narma10_results.csv / santafe_results.csv (imutáveis). Saída corrigida: benchmarks_paired_stats_corrected.csv.

## O que mudou

- **CI com sinal invertido (corrigido):** 14/16 células. Todas as células tinham o CI no sentido oposto ao `mean_diff_noaux_minus_emb` (M1).
- **CI consistente com o efeito:** antes 0/16, depois 14/16.
- **n_effective:** inalterado (o código original já filtrava não-finitos via isfinite; NARMA usa n=18/19, não 20, por seed=1 não-finito e seed=8 AB-embedded ausente).
- **p-valores / significância (Holm):** inalterados (mesmos dados); 11/16 significativas a 0.05.

## Tabela de diffs

| task | metric | noaux | embedded | n | mean_diff | CI orig | CI corr | flip | sig(Holm<.05) |
|---|---|---|---|---|---|---|---|---|---|
| NARMA10 | nmse | ABC-noaux-kraus | AB-embedded | 18 | +0.135 | [-0.151,-0.121] | [0.121,0.151] | True | True |
| NARMA10 | nmse | ABC-noaux-kraus | ABC-embedded-hierarchical | 19 | +0.0927 | [-0.112,-0.0755] | [0.0755,0.112] | True | True |
| NARMA10 | nmse | ABC-noaux-tied | AB-embedded | 18 | +0.00236 | [-0.026,0.0215] | [-0.0215,0.026] | False | False |
| NARMA10 | nmse | ABC-noaux-tied | ABC-embedded-hierarchical | 19 | -0.0413 | [0.0209,0.0613] | [-0.0613,-0.0209] | True | True |
| NARMA10 | nrmse | ABC-noaux-kraus | AB-embedded | 18 | +0.131 | [-0.147,-0.116] | [0.116,0.147] | True | True |
| NARMA10 | nrmse | ABC-noaux-kraus | ABC-embedded-hierarchical | 19 | +0.0861 | [-0.105,-0.0695] | [0.0695,0.105] | True | True |
| NARMA10 | nrmse | ABC-noaux-tied | AB-embedded | 18 | +0.00302 | [-0.0289,0.0227] | [-0.0227,0.0289] | False | False |
| NARMA10 | nrmse | ABC-noaux-tied | ABC-embedded-hierarchical | 19 | -0.0426 | [0.0218,0.0626] | [-0.0626,-0.0218] | True | True |
| SantaFe_teacher_forced | nrmse_tf | ABC-noaux-kraus | AB-embedded | 20 | +0.143 | [-0.149,-0.137] | [0.137,0.149] | True | True |
| SantaFe_teacher_forced | nrmse_tf | ABC-noaux-kraus | ABC-embedded-hierarchical | 20 | +0.15 | [-0.156,-0.145] | [0.145,0.156] | True | True |
| SantaFe_teacher_forced | nrmse_tf | ABC-noaux-tied | AB-embedded | 20 | -0.0109 | [0.00748,0.0148] | [-0.0148,-0.00748] | True | True |
| SantaFe_teacher_forced | nrmse_tf | ABC-noaux-tied | ABC-embedded-hierarchical | 20 | -0.00377 | [0.000163,0.00754] | [-0.00754,-0.000163] | True | False |
| SantaFe_rollout | nrmse_rollout | ABC-noaux-kraus | AB-embedded | 20 | +6.97 | [-9.12,-4.89] | [4.89,9.12] | True | True |
| SantaFe_rollout | nrmse_rollout | ABC-noaux-kraus | ABC-embedded-hierarchical | 20 | +7.05 | [-8.86,-5.16] | [5.16,8.86] | True | True |
| SantaFe_rollout | nrmse_rollout | ABC-noaux-tied | AB-embedded | 20 | +1.98 | [-4.09,-0.0454] | [0.0454,4.09] | True | False |
| SantaFe_rollout | nrmse_rollout | ABC-noaux-tied | ABC-embedded-hierarchical | 20 | +2.06 | [-3.7,-0.461] | [0.461,3.7] | True | False |

## Veredito
Nenhuma célula mudou de **significância** (p-valores idênticos; os mesmos dados pareados). O que muda é a **interpretação do sinal do IC**: no arquivo original o IC estava no sentido oposto ao `mean_diff`, o que poderia inverter a leitura do efeito numa tabela de paper. Após a correção, `mean_diff` e IC compartilham o sentido noaux−embedded (positivo = embedded melhor no erro). n_effective real reportado.
