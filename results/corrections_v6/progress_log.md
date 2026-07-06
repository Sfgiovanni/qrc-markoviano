# Progress log — rodada de correção v6

Heartbeat alvo: 15 min. Unidade de checkpoint por item (A1..C2).

- [2026-07-06] Início. Review lido (relatorio_code_review_v2_v5.md). Estrutura
  mapeada. git init + results_corrections_v6/ criados. GPU ocupada por terceiros.
  Começando Parte A (correções de código, sem GPU).
- [2026-07-06 13:37] Parte A completa (A1-A6): 6 commits, testes A2/A3 verdes. validate_run reproduz lacunas conhecidas v4 (12 missing/6 nonfinite) e v5 (21 missing). Iniciando Parte B (reanálises CPU).
- [2026-07-06 13:45] Parte B completa (B1-B4). Verificando GPU para Parte C.
- [2026-07-06 14:42] Parte C completa. C1: γ=0.2 20/20 seeds, τ_mem=16; lei de escala 4 pts p=0.089 R²=0.773. C2: 5/5 unidades recuperadas (v6). Incidente de escrita em imutável detectado e restaurado. corrections_summary.md finalizado. RUN COMPLETO.
