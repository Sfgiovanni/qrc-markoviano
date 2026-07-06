# Notas preliminares (dados finais já disponíveis) — base do report_final.md

Fonte: dados v2 GPU já completos (tuning, paper 100 seeds, multiscale 20 seeds).
Mackey (autônomo) e statistics pareado formal ainda rodando na retomada.
Todos os números abaixo são de `multiscale_capacities.csv`, `ipc_by_component.csv`,
`effective_memory_scales.csv`, `noaux_best_parameters.csv`, `paper_replication_gate.json`.
Readout: A_only_66 (Pauli só em A), N=20 eval seeds pareadas.

## (iii) Vantagem de Omega intermediário — REPRODUZIDA (aceita)
`paper_replication_gate.json`, 100 seeds pareadas, protocolo 1000/1000/1000:
- MSE150 médio: Ω=0.0 → 0.2020 ; **Ω=0.5 → 0.0297** ; Ω=1.0 → 0.0579.
- Ω=0.5 vence Ω=1.0: mean_diff=0.0282, IC95 [0.0252, 0.0311], **Cohen dz=1.90**,
  p_wilcoxon=3.5e-17, p_ttest=7.9e-35, 94 vitórias / 6 derrotas em 100.
- STM(τ≥10) médio: Ω=0.0 0.121 > Ω=0.5 0.0146 > Ω=1.0 0.0011.
→ A não-Markovianidade intermediária melhora previsão de MG; reproduz o paper. ACEITA.

## (i) ABC embedded N=4 supera AB embedded? — NÃO (preliminarmente rejeitada)
STM total (soma degree1 sobre τ), média±dp por seed (n=20):
- AB-embedded              2.694 ± 0.045
- ABC-embedded-tied        2.701 ± 0.057   (≈ empata com AB)
- ABC-embedded-parallel    2.634 ± 0.073
- **ABC-embedded-hierarchical 2.407 ± 0.073** (MENOR que AB)
- ABC-embedded-C-off       1.763 ; ABC-Markov 1.119
Degree2 (cross-memory) total média/seed:
- AB-embedded 3.287 ; ABC-tied 3.233 ; **ABC-hierarchical 2.369 (menor)**.
→ O embedding hierárquico N=4 NÃO supera o AB embedded em capacidade de memória
  (linear nem não-linear). Tied/parallel apenas empatam. Aguardando Wilcoxon+Holm
  formal (run_statistics) + tarefas MG autônomas para fechar. Tendência: REJEITADA.

## (ii) Múltiplos revivals em τ_B e τ_C (N=4)? — NÃO observados (rejeitada)
Curva STM(τ) média (degree1), readout A_only:
  τ:        0      5      10     20     30     40
  AB-emb    .998   .957   .718   .018   .000   .001
  ABC-hier  .999   .926   .477   .002   .001   .001
  ABC-tied  .999   .912   .757   .030   .001   .002
- Decaimento monótono a partir de τ=0; SEM pico secundário (revival) em τ maiores.
- τ_B, τ_C do construto noaux residual são curtos (tuned tau_b≈9, tau_c≈16–33);
  `effective_memory_scales.csv` reporta scale_rank=1 num único pico (τ=0 p/ embedded).
- Diagnóstico N=4 (A/B/C_diag_N4, seeds 0–1) mostra autocorrelação τ_peak≈1–2,
  C_diag levemente maior (0.28–0.31) — timescale curto, não um segundo revival de STM.
→ A previsão de duas escalas com revivals múltiplos NÃO se sustenta nos dados N=4. REJEITADA.

## (iv) Re-tuning noaux muda conclusões? — NÃO (colapso é estrutural)
- ABC-noaux-hierarchical e ABC-noaux-kraus: STM total idêntico por seed
  (max|dif|=0.0, allclose=True; ambos 5.124 ± 0.094).
- `noaux_best_parameters.csv`: params byte-idênticos nas 3 tarefas
  (ex.: tau_b=9, lambda_b=0.848660, p_b=0.258903, tau_c=16, lambda_c=0.062088,
   p_c=0.928170, objective=0.988832 — iguais em hier e kraus).
- CAUSA (código, linha 876 do pipeline): `name in ("ABC-noaux-residual",
  "ABC-noaux-kraus","ABC-noaux-hierarchical")` → MESMA construção NoAuxModel.
  Objetivo Optuna idêntico (seed 42) → mesmos trials → mesmo ótimo.
→ Re-tuning (v2: 64 trials × 12 seeds, ≥ exigido) NÃO separa os dois: são o mesmo
  modelo por construção. A conclusão anterior (colapso) mantém-se; NÃO é artefato de
  poucas seeds/trials. LIMITAÇÃO de modelagem a declarar (o "hierarchical" noaux não
  está implementado como distinto do kraus) — não corrigir a física sem pedido.

## Observações de qualidade
- Mackey autônomo: quase todo run marca teacher_forced_r2<0.99 (limiar estrito);
  linhas ainda salvas com métricas — MG é tarefa dura/divergente (avaliar se é
  discriminativa; muitos rollouts divergem). Não invalida capacidade de memória.
- noaux 16×16 roda em CPU exato complex128 (device_policy) — deliberado, não viola
  a regra de GPU (que visa a densidade 4096×4096 embedded). Declarar no relatório.
