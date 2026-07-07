# Embedded & Effective Non-Markovian Quantum Reservoir Computing

Numerical study comparing **embedded** (auxiliary-mode) and **effective**
(no-auxiliary) hierarchical ABC architectures for non-Markovian **Quantum
Reservoir Computing (QRC)**. The pipeline replicates and extends the results of
[arXiv:2505.02491](https://arxiv.org/abs/2505.02491), evolving embedded density
matrices on the GPU (`complex64`, local reshape/permute/matmul — never a global
superoperator over the full register) with `complex128` CPU validation.

The study evaluates linear/quadratic memory capacity, short-term memory (STM),
information processing capacity (IPC), and autonomous Mackey–Glass prediction,
with paired-seed hypothesis testing (n ≥ 20), Optuna tuning, washout-convergence
diagnostics, and a scaling-law analysis of memory horizon vs. dynamical scales.

## Repository layout

```
.
├── qrc_pipeline.py                    # main GPU pipeline (embedded vs effective ABC comparison)
├── qrc_pipeline_reference.py          # CPU reference implementation / scientific specification
├── qrc_experiments_architecture.py    # d_B sweep, ABC retuning, readout location A/B/C
├── qrc_experiments_robustness.py      # shot noise, NARMA/Santa Fe benchmarks, topology (parallel vs chain)
├── qrc_experiments_scaling.py         # scaling law: memory horizon vs gamma/eta
├── corrections/                       # post-review reanalysis / recovery scripts (see below)
├── review/                            # methodological-critique round (10-item), analysis scripts
├── tests/                             # regression tests (NARMA finiteness, paired-sign)
├── notebooks/
│   ├── qrc_paper.ipynb                # paper replication notebook (main)
│   └── qrc_paper_reference.ipynb      # paper replication notebook (CPU reference)
├── figures/                           # publication figures (PNG + PDF)
├── results/                           # curated, slim run outputs (see "Results" below)
│   ├── v2_abc_comparison/             # summaries, configs, hypothesis decisions, small CSVs
│   ├── extra_v3/  extra_v4/  extra_v5/  # per-round summaries and verdicts
│   ├── corrections_v6/                # corrections_summary.md, scaling law, completeness matrices
│   └── review/                        # FINAL_VERDICT.md (canonical), memory-scales, gated stats, hashes
├── docs/
│   └── code_review_v2_v5.md           # internal code review report
├── requirements.txt
└── LICENSE                            # MIT
```

The five top-level `qrc_*.py` modules must stay in the same directory: the
experiment, correction and test scripts all `import qrc_pipeline` (and, where
relevant, the experiment modules) as top-level modules.

### `corrections/`

Scripts from the post-review round that reprocess results and recover lost runs.
They correspond to the workstreams tracked in `docs/code_review_v2_v5.md`:

| Script                          | Purpose                                                        |
|---------------------------------|----------------------------------------------------------------|
| `mg_robustness.py`              | Mackey–Glass H1 robustness across three seed sets              |
| `clamp_stratified.py`           | Clamp / out-of-range stratification of Mackey–Glass rollouts   |
| `robustness_paired_stats.py`    | Recompute robustness-experiment paired statistics              |
| `completeness_matrices.py`      | Per-round run-completeness matrices                            |
| `scaling_run.py` / `scaling_fits.py` / `scaling_gate.py` | Four-point scaling-law refit and its acceptance gate |
| `recovery.py` / `recovery_finish.py` | Recover watchdog/NaN-lost seed units                      |

### `review/`

Scripts from a second, **methodological-critique** round that applied a 10-point
critique to the study with minimal GPU re-simulation. Outputs are in
`results/review/`; the single authoritative conclusion set is
**`results/review/FINAL_VERDICT.md`**, which supersedes all earlier summaries.

| Script                              | Purpose                                                             |
|-------------------------------------|---------------------------------------------------------------------|
| `phase2_paired_stats_seed_merged.py`| Recompute paired stats by explicit `seed` key (vs. positional truncation) |
| `phase3b_mg_rollout_gate.py`        | Re-gate Mackey–Glass rollouts by `teacher_forced_ok ∧ out_of_range` |
| `phase3a_memory_scales.py`          | GPU re-sim: do two separated memory scales exist? (result: **no**)  |
| `phase3c_scaling_sweep.py`          | Optional 8–12-point γ scaling law — **guarded, not run** (needs `--run`) |
| `make_csv_hashes.py`                | sha256 manifest of CSVs feeding figures/verdict                     |

These scripts are the canonical analysis code; they are executed against the full
source tree (which carries the large per-seed input CSVs and channel cache not
shipped in this slim mirror). `results/review/` ships their outputs.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA-capable GPU is used by the embedded evolution when available; the code
falls back to CPU otherwise. Developed and run with Python 3.8+.

## Running

Run every script **from the repository root** (output paths are relative to the
working directory):

```bash
python qrc_pipeline.py                   # main ABC comparison
python qrc_experiments_architecture.py   # supplementary sweeps (d_B, retuning, readout)
python qrc_experiments_robustness.py     # shot noise, NARMA/Santa Fe, topology
python qrc_experiments_scaling.py        # scaling-law experiment
pytest tests/                            # regression tests
```

Each run (re)creates its own working directory at the repo root
(`results_abc_comparison_v2/`, `results_extra_v3/`, …) plus figure and cache
folders. These are **git-ignored** — see the note below.

## Results

`results/` holds a **curated, slim snapshot**: Markdown summaries, JSON verdicts,
configs, hypothesis decisions, completeness matrices, and small CSVs. The large
intermediate dumps (per-seed capacity tables, STM traces, shot-noise sweeps),
Optuna SQLite stores, channel caches, and console logs are **not** committed —
they are fully regenerated by re-running the scripts above. This keeps the
repository small (~3 MB) while preserving every headline result and its
provenance.

Start here:
- **`results/review/FINAL_VERDICT.md`** — single canonical verdict (supersedes all summaries below)
- `results/corrections_v6/corrections_summary.md` — executive scientific verdict (post-review round)
- `results/v2_abc_comparison/final_summary.md` — main ABC comparison results
- `results/extra_v{3,4,5}/*summary*.md` — supplementary experiment summaries
- `docs/code_review_v2_v5.md` — internal code review

## License

MIT — see [LICENSE](LICENSE).
