# B4 — Consolidated completeness (validate_run retroactive on v2-v5)

## v2: **complete** (missing=0, non-finite=0)

## v3: **complete** (missing=0, non-finite=0)

## v4: **partial** (missing=12, non-finite=6)

**missing (12 cells):**

- narma10_results.csv [model=AB-embedded]: seeds [8]
- shot_noise_capacities.csv [model=AB-embedded, n_shots=100.0]: seeds [0]
- shot_noise_capacities.csv [model=AB-embedded, n_shots=1000.0]: seeds [0]
- shot_noise_capacities.csv [model=AB-embedded, n_shots=10000.0]: seeds [0]
- shot_noise_capacities.csv [model=AB-embedded, n_shots=100000.0]: seeds [0]
- shot_noise_capacities.csv [model=AB-embedded, n_shots=inf]: seeds [0]
- shot_noise_mackey.csv [model=AB-embedded, n_shots=100.0]: seeds [0]
- shot_noise_mackey.csv [model=AB-embedded, n_shots=1000.0]: seeds [0]
- shot_noise_mackey.csv [model=AB-embedded, n_shots=10000.0]: seeds [0]
- shot_noise_mackey.csv [model=AB-embedded, n_shots=100000.0]: seeds [0]
- shot_noise_mackey.csv [model=AB-embedded, n_shots=inf]: seeds [0]
- topology_control.csv [topology=parallel]: seeds [12]

**nonfinite (6 cells):**

- narma10_results.csv [model=AB-Markov]: seeds [1]
- narma10_results.csv [model=AB-embedded]: seeds [1]
- narma10_results.csv [model=ABC-embedded-hierarchical]: seeds [1]
- narma10_results.csv [model=ABC-noaux-kraus]: seeds [1]
- narma10_results.csv [model=ABC-noaux-tied]: seeds [1]
- narma10_results.csv [model=M0-noaux]: seeds [1]

## v5: **partial** (missing=21, non-finite=0)

**missing (21 cells):**

- dynamical_sweep_stm.csv [config=g0.02_epi4, model=AB-embedded]: seeds [15]
- dynamical_sweep_stm.csv [config=g0.2_epi4, model=AB-embedded]: seeds [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
