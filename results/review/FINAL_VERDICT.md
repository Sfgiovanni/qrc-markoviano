# FINAL VERDICT — QRC non-Markovian ABC study (single canonical verdict)

**Date:** 2026-07-07 · **Status:** canonical. This document supersedes every earlier
summary as the authoritative set of conclusions. All numbers below are **recalculated
from the CSVs** (paths given), not asserted from memory.

> **Superseded (kept as historical record, do not cite as conclusions):**
> `results_abc_comparison_v2/final_summary.md`, `results_abc_comparison_v2/preliminary_analysis_notes.md`,
> `results_corrections_v6/corrections_summary.md`, `results_corrections_v6/scaling_law_final.md`,
> `results_extra_v3/extra_experiments_summary.md`, `results_extra_v4/extra_v4_summary.md`,
> `results_extra_v5/scaling_law_summary.md`. Where any of these disagree with this file,
> **this file wins.**

Provenance & environment: `results_review/requirements.lock` (original run env vs.
re-analysis env — they differ; see item 10), `results_review/csv_hashes.txt` (sha256 of
every CSV feeding a figure/verdict). Review scripts: `experiments_review/`.

---

## 1. Headline conclusions

1. **A single non-Markovian auxiliary (R1 = AB-embedded) helps; the paper's ω-effect
   reproduces.** ω=0.5 beats ω=1.0 on Mackey-Glass MSE₁₅₀ and on STM(τ≥10)
   (`paper_replication_*`, 100 seeds, Wilcoxon-Holm p≪1e-10). This is the robust
   positive result of the study.
2. **A second auxiliary (R-ABC) does NOT add a second, longer memory scale at the
   readout — item 6 is a NEGATIVE result.** Over 20 seeds with a fine STM sweep,
   R-ABC has *lower* total STM capacity and a *shorter* memory horizon than R1
   (details §2). `two_separated_scales = false`.
3. **The "quantum-embedded" and the "classical-effective" (no-aux) models are, on the
   forecasting task, statistically indistinguishable where both are validly comparable**
   (ABC-embedded vs ABC-noaux-hierarchical, two-delay MG, gated: p≈0.67 VPT / 0.70
   NRMSE). The no-aux model is an **effective model with classical memory**, never a
   physical equivalence (item 5).
4. **The autonomous-rollout comparisons are mostly inconclusive under a strict validity
   gate (item 2+8):** 22/28 primary head-to-heads fail because one arm never passes
   teacher-forcing. The surviving comparisons do not favour the deeper hierarchy.
5. **The γ-scaling of memory is weak but non-zero** (τ_mem ~ (1/γ)^p, p≈0.089, R²≈0.77,
   n=4) — reported as a minor, not central, claim (item 7).

Net: the study supports **one** useful non-Markovian memory scale, not a demonstrated
multiscale hierarchy, and shows **no** advantage of the embedded quantum model over its
classical effective counterpart on the tested tasks.

---

## 2. Item-by-item resolution (the 10-point critique)

### Item 1 — Contradictory final artifacts → RESOLVED (this file)
This is now the single canonical verdict; all prior summaries are marked superseded
above. Conclusions are recomputed from CSVs with hashes recorded.

### Item 2 — Teacher-forced gate → RESOLVED (code) + re-gated here
Production gate (`run_paper_replication`, `gate_diagnostics`) already restricts the
primary ω-comparison to seeds with `teacher_forced_ok` in both arms and reports
`inconclusive` when n<20. The general MG benchmark comparisons are re-gated in
`experiments_review/phase3b_mg_rollout_gate.py` (see item 8).

### Item 3 — Pairing by position vs. seed → RESOLVED (recompute, no GPU)
`experiments_review/phase2_paired_stats_seed_merged.py` recomputes every paired
comparison with an explicit `seed`-key merge (intersection of both arms) instead of the
production `a[:n], b[:n]` positional truncation. Result
(`results_review/paired_statistics_seed_merged.csv`):
- **198 comparisons; 0 with seed misalignment; 0 inconclusive; 145 significant**
  (Holm<0.05 & CI excludes 0 & n≥20).
- In the v2 tables **every compared arm carries all 20 seeds**, so positional == seed-keyed
  *here* — the v2 `paired_statistics.csv` conclusions are **unaffected**. The bug was
  **latent** in v2; the real exposure was v4 (missing/non-finite seeds), already fixed in
  v6 (`benchmarks_paired_stats_corrected/recovered`). The guard is now explicit and logged.

### Item 4 — "Hierarchical ≡ Kraus / not structurally distinct" → RESOLVED (report null)
The variants ARE structurally distinct in code (`ABC-*-kraus/tied/hierarchical`) and are
compared. The honest finding is a **null**: ABC-embedded-hierarchical vs ABC-embedded-tied
is not robustly separable on the forecast task under the validity gate
(`mg_rollout_gate.csv`: two-delay n_gated=12 < 20 → inconclusive; standard n_gated=17 →
inconclusive). We therefore do **not** claim the hierarchical structure buys a distinct
functional behaviour on these tasks.

### Item 5 — Effective model ≠ physical equivalence → RESOLVED (narrative)
The no-aux models (`M0/AB/ABC-noaux-*`, incl. the EMA classical-memory buffer) are
described throughout this verdict as **"effective models with classical memory"**. No
claim of physical/dynamical equivalence is made. Empirically they even **match** the
embedded model where validly comparable (§1.3), which is a statement about *task
performance*, not about the underlying physics. `hypothesis_decisions.json` H5 already
records "practical equivalence not demonstrated"; this file reinforces the wording.

### Item 6 — Two memory scales → RESOLVED as a NEGATIVE (GPU re-sim, 20 seeds)
`experiments_review/phase3a_memory_scales.py`, `results_review/memory_scales*.csv`,
`memory_scales_summary.json`. Fine STM sweep (τ=0..40) + per-seed revival detection +
autocorrelation of ALL register observables, all 20 eval seeds.

| quantity | M0 | R1 (AB) | R-ABC |
|---|---|---|---|
| total STM capacity (Σ_τ, mean over seeds) | 8.71 | **12.07** | 10.38 |
| memory horizon τ where C(τ)≥0.1 | 18 | 17 | **14** |
| C(τ=10) | 0.334 | **0.716** | 0.481 |
| median STM revivals | 1.25 | 1.55 | **0.05** |
| median layer autocorr τ_A / τ_B / τ_C | — | — | **1 / 1 / 2** |

- **No second long-τ revival in R-ABC.** All three curves are single decaying lobes.
- Adding the C register **reduces** linear STM capacity and horizon vs R1; it does not
  create a second, longer scale at the A-readout.
- Layer autocorr shows τ_C (≈2) > τ_B (≈1) — significant (p=6.5e-6) — but τ_B is **not**
  > τ_A (both ≈1; p=nan/1). So there is at most a marginal internal C-timescale that
  never manifests as usable readout memory.
- **`two_separated_scales = false`, primary p = 1.** The revival comparison
  (R-ABC > R1) is rejected (p≈0.9998). The multiscale-memory hypothesis is **not forced**
  and is reported negative.

### Item 7 — 4-point γ scaling law → RESOLVED (weak claim, honestly scoped)
`results_corrections_v6/scaling_fits_corrected.json`: τ_mem ~ (1/γ)^p with
**p≈0.089, R²≈0.773, n=4** (γ=0.02/0.05/0.1/0.2), bootstrap CI excludes 0 → weak but
non-zero. The excess τ_mem−τ_FM is degenerate (≈0), i.e. little horizon gain over the M0
baseline in this γ range. Treated as a **minor** result. An 8–12-point log-spaced sweep
is prepared (`experiments_review/phase3c_scaling_sweep.py`, NOT run — see §4).

### Item 8 — Clamping in rollouts → RESOLVED (recompute + re-gate)
`results_review/mg_rollout_gate_stratified.csv` + v6 `mg_clamp_stratified.md`. Heavy
grid-clamping (up to ~939 clamps) affects only **1–2 seeds** of the embedded models; the
strongly-clamped rollouts are the **no-aux/Markov baselines** (their NRMSE reflects grid
saturation, not dynamics). Under the strict gate (`teacher_forced_ok ∧ out_of_range≤0.5`,
inner join over both arms) **22/28 primary comparisons are inconclusive**; the few
decidable ones (two-delay) do not favour the deeper hierarchy — e.g. ABC-embedded-hier vs
AB-embedded VPT is *lower* (mean Δ≈−90, p=0.018). Embedded medians are stable across OOR
strata → embedded conclusions are robust to clamping; baseline "wins" were saturation.

### Item 9 — Non-Markovianity is a proxy → RESOLVED (narrative)
`paper_nonmark_for_seed` computes the sum of positive increments of the **trace distance**
between two fixed initial states of register A over one sampled input sequence — a
2-state BLP backflow *witness*, not the optimised non-Markovianity measure. Everywhere in
this verdict the quantity is named **"sampled backflow proxy (trace distance)"**. The
CSV column keeps its name for continuity; only the interpretation is corrected. The
reported ω=0.5>ω=1.0 backflow effect (dz≈0.52, p≈4e-4) stands **as a proxy signal only**.

### Item 10 — Reproducibility → RESOLVED
`results_review/requirements.lock` pins the environment and **flags that the original v2
run (py3.12 / torch2.5.1 / numpy2.5.0) differs from the re-analysis env
(py3.8 / torch2.4.1 / numpy1.24.4)** — exact bit-reproduction needs the original env; the
qualitative results are robust to the drift. `results_review/csv_hashes.txt` records the
sha256 of all 70 CSVs feeding figures/verdict. Seeds are fixed by `CFG` and deterministic.

---

## 3. What was re-simulated vs. recomputed vs. registered

| | method | GPU | artifact |
|---|---|---|---|
| Item 3 | recompute (seed-key merge) | no | `paired_statistics_seed_merged.csv` |
| Item 6 | **re-simulation** (20 seeds, fine τ) | **yes (~25 min)** | `memory_scales*.csv`, `_summary.json` |
| Item 2+8 | recompute (re-gate saved rollouts) | no | `mg_rollout_gate*.csv`, `_summary.json` |
| Items 1,4,5,9 | narrative / null-reporting | no | this file |
| Item 7 | registered from v6 | (v6 GPU) | `scaling_fits_corrected.json` |
| Item 10 | env lock + hashes | no | `requirements.lock`, `csv_hashes.txt` |

## 4. What remains inconclusive / not run

- **Rollout head-to-heads (item 2+8):** 22/28 primary comparisons inconclusive under the
  strict validity gate (baselines never pass teacher-forcing). This is a *limitation of
  the comparison*, honestly reported, not a hidden win.
- **γ-scaling with 8–12 points (item 7, Phase 3C):** `experiments_review/phase3c_scaling_sweep.py`
  is prepared but **NOT executed** — it needs explicit go-ahead (it is not a central claim).
- **Hierarchical vs tied distinction (item 4):** inconclusive under the gate (n_gated<20).

## 5. One-line bottom line

A single non-Markovian memory scale is real and useful; a **second** scale from the C
register is **not demonstrated** at the readout; and the embedded quantum model shows **no
performance advantage** over its classical effective counterpart on the tested tasks.
