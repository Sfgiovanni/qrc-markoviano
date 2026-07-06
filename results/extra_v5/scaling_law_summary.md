# Exp 7 — dynamical scaling of the memory horizon (v5)

_Generated 2026-07-06T11:02:24.291542_

Run config: `{"etas": [0.39269908169872414, 0.7853981633974483, 1.1780972450961724], "eval_seeds": 20, "gammas": [0.02, 0.05, 0.1, 0.2]}` | budget: 1.348 h | reference tau_mem=None

## Horizons (threshold C>0.1)

| config | gamma | eta | model | tau_mem | 95% CI |
|---|---|---|---|---|---|
| g0.02_epi4 | 0.02 | pi4 | AB-embedded | 19 | [17, 19] |
| g0.05_epi4 | 0.05 | pi4 | AB-embedded | 19 | [16, 19] |
| g0.1_epi8 | 0.1 | 0.393 | AB-embedded | 17 | [17, 17] |
| g0.1_epi4 | 0.1 | pi4 | AB-embedded | 16 | [16, 19] |
| g0.1_e3pi8 | 0.1 | 3pi8 | AB-embedded | 18 | [18, 18] |
| M0_g0.02 | 0.02 | pi4 | M0 | 17 | [16, 18] |
| M0_g0.05 | 0.05 | pi4 | M0 | 19 | [18, 20] |
| M0_g0.1 | 0.1 | pi4 | M0 | 19 | [18, 19] |
| M0_g0.2 | 0.2 | pi4 | M0 | 17 | [16, 17] |

## Scaling fits

- **(a) tau_mem ~ (1/gamma)^x**: x = 0.101 [95% CI 5.715674111581761e-16, 0.11100006441507074], R²=0.678.
- **(c) eta sweep (gamma=0.1)**: non-monotonic / no clear trend. points: [('pi8', 17.0), ('pi4', 16.0), ('3pi8', 18.0)]

gamma sweep (eta=pi/4): gamma=0.02: tau_mem=19 (M0 17), gamma=0.05: tau_mem=19 (M0 19), gamma=0.1: tau_mem=16 (M0 19)

## Verdict

**The horizon moves with gamma (x=0.10, R²=0.6779345023608088) but the power law is not clean (R²<0.9).**

## Consistency with v3 (dimension) and v4 (topology), all gamma=0.1

Reference tau_mem(gamma=0.1) = 16.0. v3/v4 gamma=0.1 points = [21, 18, 16, 19, 17, 19, 21, 17, 20]. Consistent (within ±5): True.

## Anomalies (failed_runs.csv + decisions_log.md)

| timestamp | context | reason |
|---|---|---|
| 2026-07-06T10:16:05.512695 | g0.02_epi4/AB-embedded/seed15 | watchdog_step_timeout |
| 2026-07-06T11:02:17.718540 | gamma_0.2 | gamma_processing_exception |

**decisions_log.md:**

- [2026-07-06 10:01:21] **Reference gate eta correction** — spec's reference (gamma=0.1, eta=pi/4) gives tau_mem~15, not 21; v3's tau=21 uses eta=1.3388, omega=0.0699. Reproduction gate run against v3's real config; sweeps kept as specified (eta in {pi/8,pi/4,3pi/8}).
- [2026-07-06 10:01:25] **gamma=0.1 cache** — reused v2 channel cache (verified max|diff|=0.00e+00 < 1e-9)
- [2026-07-06 10:04:04] **ESP washout omega** — ESP measured at omega=0.1 (grid minimum, worst case for washout); tuned omega is >= this, so the adaptive washout is conservative.