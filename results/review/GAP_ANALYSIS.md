# GAP_ANALYSIS — crítica metodológica (10 itens) vs. estado atual do repositório

Data: 2026-07-07. **Fase 0 — apenas leitura. Nada foi alterado nem re-executado.**

Este documento classifica cada um dos 10 itens da crítica em **uma** de quatro
categorias, com evidência (arquivo + célula/função/variável), e propõe a ação mínima.

---

## 0. Discrepâncias entre a spec da tarefa e o repositório real (ler primeiro)

A tarefa assume nomes de arquivos, símbolos e config que **não existem** aqui. O
mapeamento real (verificado por `grep`/leitura) é:

| A tarefa diz | Realidade no repo |
|---|---|
| `qrc_abc_residual.ipynb`, `qrc_comparison.ipynb` | **um** notebook: `embedded_and_effective_hierarchical_abc_qrc_paper_v2.ipynb`; lógica em `embedded_effective_qrc_pipeline_v2.py` (+ `extra_experiments_v3/v4/v5.py`) |
| `results_abc/`, `results/` | `results_abc_comparison_v2/`, `results_extra_v3/`, `results_extra_v4/`, `results_extra_v5/` (imutáveis) e `results_corrections_v6/` (rodada de correção já feita) |
| `ResidualReservoir`, `make_M0/make_R1/make_RABC` | `EmbeddedModelGPU`/`EmbeddedModelNP`/`NoAuxModel`; `make_model` / `make_embedded_model` / `make_noaux_model` |
| `add_ema_features` | memória clássica efetiva embutida em `NoAuxModel` (EMA sobre features) |
| `OBS_STM`, `stm_capacity_curve` | `stm_target` + `evaluate_capacity_from_features` |
| `run_forecast_task`, `ForecastModel` | `run_mg_model` + `autonomous_rollout` |
| `generate_mackey_glass` | `mackey_glass` |
| `cfg` (N=3, τ_E=10, τ_B=15, τ_C=30, eval_seeds=range(40), n_seeds_forecast=14) | `CFG` (`n_a=4`, `eval_seeds=range(20)`, `paper_eval_seeds=range(20)`, `min_decision_seeds=20`) — **eval_seeds é 20, não 40** (orçamento GPU reduziu de `planned_eval_seeds=100`) |
| Modelos M0/R1/R-ABC/C-ABC | `EMBEDDED_ARCHES` (ex. `AB-embedded`, `ABC-embedded-hierarchical`) vs. `NOAUX_ARCHES` (ex. `M0-noaux`, `AB-noaux-kraus`, `ABC-noaux-{kraus,tied,hierarchical,...}`) |

**Consequência prática:** os scripts standalone da Fase 3 devem importar os símbolos
**reais** de `embedded_effective_qrc_pipeline_v2.py` (`make_model`, `drive_features`,
`stm_target`, `evaluate_capacity_from_features`, `build_channel_grid_gpu`,
`iid_inputs`, `run_mg_model`, `autonomous_rollout`, `paper_nonmark_for_seed`, `CFG`).
Onde a tarefa diz "40 eval_seeds", o real disponível é **20** (uso todas as 20).

**Contexto importante:** já existe uma rodada de correção **v6**
(`results_corrections_v6/`, commits `A1–A6` de código + `B1–B4` recompute +
`C1–C2` GPU) que ataca vários destes itens. `corrections_summary.md` e
`decisions_log.md` documentam. Isso muda muita coisa de "fazer" para "registrar".

---

## 1. Tabela de classificação

| # | Item da crítica | Categoria | Evidência (arquivo · função/variável) | Ação mínima |
|---|---|---|---|---|
| 1 | Artefatos finais contraditórios | **SÓ DOC/NARRATIVA** | Múltiplos resumos coexistem sem veredito único: `results_abc_comparison_v2/final_summary.md`, `results_corrections_v6/corrections_summary.md`, `results_extra_v*/…_summary.md`, `scaling_law_final.md` | Fase 1: criar `results_review/FINAL_VERDICT.md` recalculado dos CSVs; marcar os antigos como *superseded*. Sem compute. |
| 2 | Gate teacher-forced global | **JÁ RESOLVIDO** (primária) + resim mínima em 3B | `pipeline_v2.py`: `run_paper_replication` (l.1678–1759) grava `included_in_primary = teacher_forced_ok`, `gate_diagnostics` (l.1646), gate inconclusivo se `n_paired_primary < min_decision_seeds` (l.1740). Reanálise `results_corrections_v6/h1_mg_robustness.md` (tf-ok por ω = 100/56/0 → primária inconclusiva) | Registrar no veredito. O gate conjunto tf+OOR **no forecast T3** é a parte 3B (ver #8). |
| 3 | Pareamento por posição vs. seed | **RECOMPUTE BARATO** | `paired_stats` trunca por posição: `n=min(len(a),len(b)); a,b=a[:n],b[:n]` (l.2041–2042). Chamadores passam `.sort_values("seed").values` (l.2241–2277) → **posicional**. O bloco de equivalência (l.2292–2295) já faz `groupby("seed")`+`index.intersection` (correto). v6 `B3` corrigiu só as stats do v4, **não** o `paired_statistics.csv` do v2. | Fase 2: refazer `paired_statistics.csv` com merge por (seed, task, modelo) a partir dos CSVs por-seed; registrar seeds usadas; marcar `inconclusive` se comum < 20. Sem GPU. |
| 4 | Hierárquico ≡ Kraus (não estruturalmente distinto) | **SÓ DOC/NARRATIVA** (+ recompute via #3) | Variantes existem e são comparadas: `NOAUX_ARCHES` inclui `ABC-noaux-{kraus,tied,hierarchical}`; figura `paired_differences_hier_minus_tied.*`; `equivalence_tests.csv`. A comparação está feita; o achado honesto é **indistinguibilidade estatística** | Fase 1/2: reportar como resultado **nulo** (não como distinção estrutural), usando as stats seed-merged do item 3. |
| 5 | Modelo efetivo ≠ equivalência física | **SÓ DOC/NARRATIVA** | `NoAuxModel` (memória clássica via EMA) é o "efetivo" (M4/C-ABC). Verificar wording em `final_summary.md`/notebook | Fase 1: descrever noaux como "modelo efetivo com memória clássica", nunca "equivalência física". Se o texto já faz isso, só registrar. |
| 6 | Duas escalas de memória não demonstradas | **RE-SIMULAÇÃO NECESSÁRIA** (gap central) | `run_multiscale_and_ipc` (l.1901): STM só em 7 τ grossos `[0,5,10,20,30,40,50]` (l.1929); detecção de pico só na curva **média** (l.1953–1963, sem teste por-seed); autocorrelação de camadas A/B/C só em `eval_seeds[:2]` (**2 seeds**, l.1965). Sem teste de "dois revivals separados" | Fase 3A: τ fino, detecção de revival **por seed** + teste estatístico de duas escalas separadas, autocorr de todos observáveis nas 20 seeds. GPU (N=4). Reportar negativo se não separar. |
| 7 | Lei de escala com 4 pontos de γ | **JÁ RESOLVIDO** (parcial) + **RE-SIM OPCIONAL** (3C) | `results_corrections_v6/scaling_fits_corrected.json`, `scaling_law_final.md`: refit 4 pts (γ=0.02/0.05/0.1/0.2), `p=0.089, R²=0.773, n=4`, IC bootstrap exclui 0 (fraco mas não-nulo); excesso τ_mem−τ_FM degenerado | Registrar no veredito. 8–12 pts em log = Fase 3C, **NÃO rodar sem OK** (não é alegação central). |
| 8 | Clamping em rollouts | **JÁ RESOLVIDO / RECOMPUTE** (B2) + resim mínima em 3B | `run_mg_model` (l.1150) retorna nº de clamps; `results_corrections_v6/mg_clamp_stratified.md`: embedded quase não clampa, baselines noaux saturam (clamp = saturação, não dinâmica). Estratificação já feita sobre dados existentes | Registrar. Gate conjunto `teacher_forced_ok ∧ out_of_range_fraction` por seed no forecast T3 = Fase 3B (itens 2+8 juntos). |
| 9 | Não-markovianidade é proxy | **SÓ DOC/NARRATIVA** | `paper_nonmark_for_seed` (l.1593–1617): soma dos incrementos positivos da **distância de traço** entre 2 estados iniciais no registro A, sobre **uma** sequência amostrada (`nonmarkovianity = positive_sum`). É testemunha BLP de 2 estados, não a medida otimizada | Fase 1: renomear na narrativa para "proxy de backflow amostrado (distância de traço)". Código/CSV mantêm a coluna; só o texto muda. |
| 10 | Reprodutibilidade | **RECOMPUTE BARATO / DOC** | Existe `environment.json` por run, `config.json`, seeds fixas, determinismo. **Falta**: `requirements.lock` (versões fixadas) + hash de cada CSV que alimenta as figuras finais | Fase 1: gerar `requirements.lock` do ambiente atual + `csv_hashes.txt` (sha256). Sem GPU. |

---

## 2. Resumo por categoria

- **JÁ RESOLVIDO (registrar, não refazer):** #2 (primária), #7 (4 pts), #8 (estratificação) — via rodada v6.
- **SÓ DOC/NARRATIVA:** #1, #4, #5, #9 — Fase 1, sem compute.
- **RECOMPUTE BARATO (sem GPU):** #3 (seed-merge do `paired_statistics.csv`), #10 (lock + hashes) — Fase 1/2.
- **RE-SIMULAÇÃO NECESSÁRIA (GPU):** #6 (Fase 3A, escalas de memória — gap central); #7 opcional (Fase 3C, 8–12 γ, só com OK); #2+#8 gate conjunto no forecast T3 (Fase 3B, mínimo).

## 3. Plano de execução proposto (após aprovação)

- **Fase 1 (sem compute):** `FINAL_VERDICT.md` (recalc dos CSVs; antigos → *superseded*); reescrita narrativa #5/#9 (registrar se já correto); `requirements.lock` + `csv_hashes.txt` (#10).
- **Fase 2 (recompute, sem GPU):** merge por (seed, tarefa, modelo) → novo `paired_statistics` em `results_review/`; log das seeds efetivas; `inconclusive` se comum < 20; recalcular #3, #4.
- **Fase 3 (GPU, isolada em `experiments_review/`, importando o pipeline real):**
  - **3A** escalas de memória (20 seeds, τ fino, teste de dois revivals) → `results_review/memory_scales.csv` + `two_separated_scales: true/false` com p-valor.
  - **3B** gate de validade do rollout T3 (tf-ok + out_of_range_fraction + clamps por seed; inner-join ≥20 senão inconclusiva).
  - **3C** OPCIONAL 8–12 γ — só com seu OK.

Antes de cada re-simulação: imprimir estimativa de tempo/GPU e nº de seeds; preservar
seeds da Config; **não** tocar em `results_abc_comparison_v2/` nem `results_extra_v*/`.

---

**Fase 0 aprovada; Fases 1–3A/3B executadas.** Veredito canônico consolidado em
`FINAL_VERDICT.md`. Item 6 resultou **negativo** (`two_separated_scales=false`). Fase 3C
(8–12 γ) permanece **preparada mas não executada** (`experiments_review/phase3c_scaling_sweep.py`,
requer OK explícito).
