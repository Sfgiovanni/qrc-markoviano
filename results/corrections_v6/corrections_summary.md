# Corrections summary — rodada v6 (review v2–v5)

Data: 2026-07-06. Correções de código nos scripts (commit por item, ID do review
na mensagem); reanálises e reexecuções em `results_corrections_v6/`. Diretórios
`results_*` originais preservados como registro histórico imutável.

Ambiente: GPU RTX 3080 Ti compartilhada (≈10 GB ocupados por job de terceiros
durante todo o run — ver decisions_log.md). Testes A2/A3 verdes; gate da Fase 0
satisfeito (validate_run reproduz as lacunas conhecidas; gate C1 ref_tau=21∈[19,23]).

---

## Veredito científico (resumo executivo)

1. **H1-MG sobrevive aos três conjuntos do B1?** Numericamente **sim** em todos:
   ω=0.5 tem menor `mse_150` que ω=1.0 em (i) 100 seeds, (ii) 56 tf-ok, (iii)
   OOR<0.5 — sempre p_wilcoxon<1e-10, dz≈1.9–2.3, IC exclui 0. **Porém** a
   conclusão como afirmação de *qualidade de reservatório* **não se sustenta**:
   ω=1.0 falha o teacher-forced em 100/100 seeds, então sob o gate estrito
   (tf-ok em ambos os braços) n=0 → **inconclusivo**. A comparação mede "ω=1.0
   não faz rollout autônomo de MG", não "ω=0.5 é um reservatório melhor".

2. **Alguma célula do v4 mudou de significância?** **Não.** Os p-valores são
   idênticos (mesmos dados pareados). O que mudou: 14/16 células tinham o IC no
   **sinal oposto** ao `mean_diff` (M1); corrigido para o sentido noaux−embedded.

3. **Lei de escala com 4 pontos:** τ_mem por γ (thr 0.1, AB-embedded, η=π/4):
   γ=0.02→19, 0.05→19, 0.1→16, 0.2→**16** (novo, 20 seeds). Ajuste
   **τ_mem ~ (1/γ)^p: p=0.089, R²=0.773, n=4, IC95 bootstrap [0.025, 0.089]**
   (vs 3 pontos original p=0.101, R²=0.678) — dependência **fraca mas não-nula**
   (IC exclui 0). O ajuste do **excesso** (τ_mem − τ_FM) é **degenerado**:
   excesso=[2,0,−3,−1] (τ_mem ≈ τ_FM do M0), ou seja o ganho de horizonte do
   embedded sobre o baseline M0 é **marginal** nesses γ. Os pontos de **dimensão
   (v3)** e **topologia (v4)**, todos medidos a γ=0.1, **caem na curva**: são
   consistentes com a referência τ_mem(γ=0.1)=16 dentro de |Δ|≤5.

**Completude consolidada:** v2 completo, v3 completo, v4 parcial (12 missing / 6
não-finitos), v5 parcial (21 missing). Detalhe em `b4_completeness_listing.md`.

---

## Parte A — Correções de código (commits por ID)

### G1 — v2 aceitava MG sem teacher-forced válido  → **A1**
`embedded_effective_qrc_pipeline_v2.py`: `run_paper_replication`/`paper_mg_for_seed`.
Todas as linhas continuam salvas; adicionada coluna `included_in_primary =
teacher_forced_ok`. O gate agora: reporta `n_passed_teacher_forced` por ω; a
comparação primária ω=0.5 vs 1.0 usa só seeds tf-ok em **ambos** os braços; se
n_pareado < `min_decision_seeds`(20) → `primary_verdict = inconclusive`. A stat
legada com todas as seeds fica separada e rotulada.

### G3 — NARMA-10 com NaN/overflow entrando no resumo → **A2**
`extra_experiments_v4.py`: `narma10_target` agora propaga overflow como inf e
**assere finitude**; `narma10_target_for_seed` refaz com seed+10000 (≤3×,
registra o remapeamento) ou levanta erro. `v2.all_finite()` é o **validador
único** usado antes de gravar qualquer linha; `run_narma`/`run_santafe` pulam +
`record_failure` em vez de gravar métricas não-finitas. Teste unitário confirma
que **seed=1** (as 6 linhas NaN históricas) diverge e remapeia para 10001.

### M1 — IC com sinal inconsistente no v4 → **A3**
`v2.orient_effect()` orienta mean_diff, ci95, dz, wins ao mesmo sentido;
`exp5_paired_stats` reporta tudo como noaux−embedded. Teste cobre os dois modos
`larger_better`.

### M3 — cache GPU sem gamma na chave → **A4**
`_GRID_GPU_CACHE` agora inclui `(seed,n,g,gamma,dt,s_min,s_max)`; elimina reuso
silencioso de canais entre gammas.

### M2/G2 — fases marcadas completas com dados faltando → **A5**
`v2.validate_run(results_dir, expected)` gera `completeness_matrix.csv`
(config×modelo×seed: presente/faltante/não-finito) e retorna completo/parcial.
`v2.write_validated_completion()` só escreve `*_complete.json` se completo, senão
`*_partial.json`. v4/v5 `main` passam a gatear o summary por ela. Reproduz
exatamente as lacunas do review.

### P1/P2/P3 → **A6**
P1: v5 summary lia `reference_tau_mem` (sempre None) → corrigido para
`infra_repro_tau_mem`. P2: `load_santafe` tenta todas as URLs (A.dat+A.cont) em
vez de abortar na primeira falha. P3: `v2.completeness_markdown()` reporta
n_effective/n_missing/n_nonfinite por tabela; v4/v5 summaries o incluem.

---

## Parte B — Reanálises (dados existentes → v6)

### B1 (G1) — `h1_mg_robustness.{csv,md}`
tf por ω: **100/56/0** (confirma review). ω=0.5 bate ω=1.0 em todos os 3
conjuntos (p<1e-10, dz≈1.9–2.3). Conjunto (iii)≡(i) porque os rollouts ω=0.5 e
1.0 têm OOR=0 (sem clamp). Gate estrito (tf ambos) → n=0, inconclusivo como
alegação de qualidade. Veredito completo acima.

### B2 (M4) — `mg_clamp_stratified.{csv,md}`
520 rollouts (standard+two_delay); 18% com OOR>0.5; clamps até 955. Excluindo
OOR>0.5, os **fortemente clampados são as baselines noaux** (AB-noaux-kraus
20/20, ABC-noaux-tied 20/20, M0-noaux 19/20 → seus NRMSE eram saturação, não
dinâmica). Os **embedded quase não são clampados** (n_excl 0–2), medianas
NRMSE150/VPT estáveis sob estratificação → conclusões embedded robustas ao clamp.

### B3 (G3/M1) — `benchmarks_paired_stats_corrected.{csv,md}` + `_diff.csv`
16 comparações. **14/16** tinham IC com sinal oposto ao efeito (corrigido);
consistência IC↔efeito 0/16 → 14/16 (as 2 restantes cruzam 0 = n.s.).
**Nenhuma mudança de significância** (p idênticos; 11/16 sig. a Holm<0.05).
n_effective real: NARMA usa n=18/19 (não 20) por seed=1 não-finito + seed=8
AB-embedded ausente.

### B4 (G2/M2) — `{v2,v3,v4,v5}_completeness_matrix.csv` + `b4_completeness_listing.md`
- v2 **completo** (MG paper 300/300; multiscale 300/300).
- v3 **completo** (aux_dimension 120/120).
- v4 **parcial**: narma seed8/AB-embedded ausente + seed1 não-finito (6 modelos);
  shot_noise seed0/AB-embedded (2 tabelas × 5 níveis de shots); topologia seed12.
- v5 **parcial**: g0.2_epi4/AB-embedded (20 seeds) + g0.02_epi4 seed15.

---

## Parte C — Reexecuções mínimas (GPU)

### C1 (G2) — gamma=0.2 + refit da lei de escala → `scaling_fits_corrected.json`, `scaling_law_final.md`
Gate interno **PASSOU**: reprodução do horizonte de referência γ=0.1 com o código
corrigido → `ref_tau=21 ∈ [19,23]` (`c1_gate_result.txt`); τ_FM(M0,γ=0.2)=17 bate
com o v5. Ponto γ=0.2 reexecutado (tuning 9ω×4 seeds → ω*=0.1; eval **20/20
seeds**; STM τ=0..80) → τ_mem=16. Refit dos 4 pontos + IC bootstrap acima
(veredito 3). Execução isolada em `c1_gamma02/`; originais não tocados.

### C2 (M2) — recuperação de seeds perdidas → `c2_recovery/`
**Todas as 5 unidades recuperadas**, integradas só ao v6 (`c2_status.md`):
- v5 g0.02 AB-embedded seed15 (STM, 81 linhas) — dobrada no fit (γ=0.02→20 seeds).
- v4 NARMA seed1 (6 modelos, regen A2 → seed 10001) + seed8 → tabela **20 seeds,
  120 linhas, todas finitas**.
- v4 shot-noise seed0 (246 cap + 41 MG).
- v4 topologia parallel seed12 (52 linhas).
Bônus: `benchmarks_paired_stats_recovered.csv` refaz o pareado do v4 com NARMA
completo (**n=20** em todas as comparações, antes 18/19) — sinais consistentes e
**significância inalterada** (confirma B3).

**Incidente corrigido:** um typo de redirecionamento fez o `exp4_mackey` apender
seed0 no `shot_noise_mackey.csv` imutável (3239→3280); detectado e **restaurado**
do baseline (git), além de uma linha de log espúria em `v4/run.log`. Registro em
`decisions_log.md`. Nenhum outro original foi alterado.

---

## Gate da Fase 0 (go/no-go) — **PASSOU**
- Testes unitários A2/A3: **verdes** (4+4).
- `validate_run` reproduz as lacunas conhecidas: v4 (narma seed8 + seed1
  não-finito, shot_noise seed0, topo seed12) e v5 (g0.2 20 seeds, g0.02 seed15). ✓
- Gate C1 (referência γ=0.1): `ref_tau=21 ∈ [19,23]`. ✓
- Orçamento medido < 8 h. ✓

## O que permanece parcial (nos ORIGINAIS; recuperado no v6)
- v4: shot_noise seed0, narma seed1/seed8, topologia seed12 — recuperados em
  `c2_recovery/` (originais permanecem como registro histórico).
- v5: γ=0.2 (C1) e g0.02 seed15 (C2) — recuperados no v6.
- Registro completo em `failed_runs.csv` e `decisions_log.md`.
