# Extra experiments v3 — summary

_Generated 2026-07-05T20:29:26.122589_

Run config: `{"dB64_eval_seeds": 20, "exp1_trials": 32, "exp3_readouts": ["A_only", "B_only", "C_only", "full"]}` | measured budget: 6.748 h

## Verdict 1 — memory range vs auxiliary dimension (Exp 1)

**memory range does NOT scale up with d_B**

| d_B | n_B qubits | memory range (max tau, C>0.1) |
|---|---|---|
| 2 | 1 | 18 |
| 4 | 2 | 19 |
| 8 | 3 | 19 |
| 16 | 4 | 21 |
| 32 | 5 | 16 |
| 64 | 6 | 17 |

Paired STM total, d_B=64 vs d_B=16: mean_diff=-1.2545, 95% CI [-1.4480, -1.0651], Wilcoxon p=1.907e-06, n=20.

## Verdict 2 — subtuning defense (Exp 2)

**subtuning WAS a factor: 64x8 retuning significantly improves the retuned model**

STM total (retuned vs v2): mean_diff=0.0152, 95% CI [0.0055, 0.0265], Wilcoxon p=0.009436, n=20.

## Verdict 3 — mechanism: storage vs backflow (Exp 3)

**AB-embedded: STORAGE failure (tau~30 absent everywhere); ABC-embedded-hierarchical: STORAGE failure (tau~30 absent everywhere)**

Mean STM capacity in the tau in [25,35] band by readout:

```json
{
  "AB-embedded": {
    "A_only": 0.0010747973316150273,
    "B_only": 0.0011368871174558663,
    "full": 0.0016266572499576317
  },
  "ABC-embedded-hierarchical": {
    "A_only": 0.0009580761845294612,
    "B_only": 0.0009692615244983642,
    "C_only": 0.00092400120900304,
    "full": 0.0009653676268320401
  }
}
```

## Anomalies (failed_runs.csv)

None recorded.