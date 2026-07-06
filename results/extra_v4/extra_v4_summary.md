# Extra experiments v4 — summary

_Generated 2026-07-06T08:08:15.500254_

Run config: `{"exp4_n_shots": [100.0, 1000.0, 10000.0, 100000.0, null], "exp6_eval_seeds": 20, "santafe_rollout": true}` | measured budget: 4.068 h

## Verdict 1 — robustness to finite-sampling (shot) noise (Exp 4)

**noaux advantage does NOT retain 80% at any finite N_shots tested**

Mean multiscale capacity (STM tau in {5,10,15,20,30} + product s10*s20) by model vs N_shots:

| model | 1e2 | 1e3 | 1e4 | 1e5 | inf |
|---|---|---|---|---|---|
| ABC-noaux-kraus | 0.001705 | 0.02468 | 0.1481 | 0.3137 | 0.6413 |
| ABC-noaux-tied | 0.001096 | 0.005009 | 0.03538 | 0.1081 | 0.298 |
| AB-embedded | 0.001561 | 0.01173 | 0.05429 | 0.1401 | 0.3776 |
| ABC-embedded-hierarchical | 0.001078 | 0.003876 | 0.02564 | 0.08 | 0.2766 |

noaux−embedded gap (ABC-noaux-kraus − ABC-embedded-hierarchical) by N_shots: 1e2=0.0006269, 1e3=0.0208, 1e4=0.1224, 1e5=0.2337, inf=0.3646 (exact=0.3646); the advantage erodes as shots fall.

Min N_shots retaining >=80% of exact multiscale capacity, by model:

| model | min N_shots (>=80%) |
|---|---|
| ABC-noaux-kraus | None |
| ABC-noaux-tied | None |
| AB-embedded | None |
| ABC-embedded-hierarchical | None |

## Verdict 2 — position on standard benchmarks (Exp 5)

NARMA-10 mean NMSE / NRMSE by model (20 seeds, no retuning):

| model | NMSE | NRMSE |
|---|---|---|
| ABC-noaux-kraus | 0.3464 | 0.5881 |
| ABC-noaux-tied | 0.2123 | 0.4594 |
| AB-embedded | 0.2098 | 0.456 |
| ABC-embedded-hierarchical | 0.2537 | 0.502 |
| M0-noaux | 0.4579 | 0.6731 |
| AB-Markov | 0.6827 | 0.8261 |

Santa Fe laser A, mean NRMSE by model:

| model | NRMSE (teacher-forced) | NRMSE (100-step rollout) | VPT |
|---|---|---|---|
| ABC-noaux-kraus | 0.2404 | 19.21 | 1.7 |
| ABC-noaux-tied | 0.08643 | 14.23 | 4.2 |
| AB-embedded | 0.09731 | 12.24 | 3.9 |
| ABC-embedded-hierarchical | 0.09021 | 12.16 | 3.3 |
| M0-noaux | 0.1951 | 6.042 | 1.2 |
| AB-Markov | 0.2228 | 7.486 | 0.2 |

Paired noaux-vs-embedded (Wilcoxon + Holm, smaller error better):

| task | metric | noaux | embedded | mean_diff (noaux−emb) | p_holm |
|---|---|---|---|---|---|
| NARMA10 | nmse | ABC-noaux-kraus | AB-embedded | 0.1354 | 8.392e-05 |
| NARMA10 | nmse | ABC-noaux-kraus | ABC-embedded-hierarchical | 0.09274 | 5.341e-05 |
| NARMA10 | nmse | ABC-noaux-tied | AB-embedded | 0.002361 | 1 |
| NARMA10 | nmse | ABC-noaux-tied | ABC-embedded-hierarchical | -0.04132 | 0.008469 |
| NARMA10 | nrmse | ABC-noaux-kraus | AB-embedded | 0.131 | 8.392e-05 |
| NARMA10 | nrmse | ABC-noaux-kraus | ABC-embedded-hierarchical | 0.0861 | 5.341e-05 |
| NARMA10 | nrmse | ABC-noaux-tied | AB-embedded | 0.003022 | 1 |
| NARMA10 | nrmse | ABC-noaux-tied | ABC-embedded-hierarchical | -0.04259 | 0.008198 |
| SantaFe_teacher_forced | nrmse_tf | ABC-noaux-kraus | AB-embedded | 0.1431 | 3.052e-05 |
| SantaFe_teacher_forced | nrmse_tf | ABC-noaux-kraus | ABC-embedded-hierarchical | 0.1502 | 3.052e-05 |
| SantaFe_teacher_forced | nrmse_tf | ABC-noaux-tied | AB-embedded | -0.01088 | 6.866e-05 |
| SantaFe_teacher_forced | nrmse_tf | ABC-noaux-tied | ABC-embedded-hierarchical | -0.003774 | 0.4216 |
| SantaFe_rollout | nrmse_rollout | ABC-noaux-kraus | AB-embedded | 6.967 | 8.583e-05 |
| SantaFe_rollout | nrmse_rollout | ABC-noaux-kraus | ABC-embedded-hierarchical | 7.05 | 8.583e-05 |
| SantaFe_rollout | nrmse_rollout | ABC-noaux-tied | AB-embedded | 1.981 | 0.6073 |
| SantaFe_rollout | nrmse_rollout | ABC-noaux-tied | ABC-embedded-hierarchical | 2.064 | 0.1638 |

## Verdict 3 — topology control (Exp 6)

**topology does NOT rescue memory range: parallel n_B=6 no better than d_B=16**

Memory range (max tau, C>0.1): parallel d_B=64 = 20, chain d_B=64 = 17, chain d_B=16 = 21.

Paired STM total, parallel vs chain d_B=16: mean_diff=0.06447, 95% CI [-0.1467, 0.2626], Wilcoxon p=0.5678, n=19.
Paired STM total, parallel vs chain d_B=64: mean_diff=1.38, 95% CI [1.246, 1.505], Wilcoxon p=3.815e-06, n=19.

## Anomalies (failed_runs.csv + decisions_log.md)

| timestamp | context | reason |
|---|---|---|
| 2026-07-05T22:29:48.423140 | exp4/AB-embedded/seed0 | watchdog_step_timeout |
| 2026-07-06T00:17:35.959662 | narma/AB-embedded/seed8 | watchdog_step_timeout |
| 2026-07-06T02:27:44.738990 | exp6/seed12 | watchdog_step_timeout |

**decisions_log.md:**

- [2026-07-05 22:17:53] **MG shot-noise scope** — autonomous rollout capped at 150 steps with 10 noise reps (NRMSE_150 + VPT); design choice to bound budget, not a degradation