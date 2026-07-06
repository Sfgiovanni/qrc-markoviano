# Decisions log — rodada de correção v6

Formato: `[timestamp] (ID) decisão — justificativa`.
Regra: decisão não coberta pela spec => opção conservadora + registro aqui.

## Setup

- [2026-07-06] Projeto não era repositório git. `git init` executado para permitir
  commit por item (exigência da task). `.gitignore` exclui caches/zips/sqlite/npz
  do versionamento (registro histórico volumoso; scripts e CSVs pequenos entram).
- [2026-07-06] GPU (RTX 3080 Ti, 12 GB) está ~10.5 GB ocupada por jobs de OUTROS
  usuários (mahlow/miziara, EQ-GAN). Só ~1.7 GB livre. Parte C (GPU) fica gated
  por memória; A e B (CPU) prosseguem. Reavaliar C após A/B; reservatório é N=4,
  cabe em <2 GB, mas pode exigir esperar liberação. Decisão conservadora: não
  matar processos de terceiros; usар PYTORCH/limitar memória e, se não couber,
  registrar em ABORTED/partial em vez de forçar.

## Incidente (corrigido) — C2 escreveu em arquivo imutável

- [2026-07-06 14:31] BUG: em `recover_shot_noise_seed0`, o global do CSV de MG do
  shot-noise foi redirecionado com o nome errado (`EXP4_MACKEY_CSV`, inexistente)
  em vez de `EXP4_MG_CSV`. Como AB-embedded seed0 estava ausente no original, o
  `exp4_mackey` APENDOU as linhas de seed0 em
  `results_extra_v4/shot_noise_mackey.csv` (imutável), levando-o de 3239→3280
  linhas. CORREÇÃO: `git checkout 27864ef -- results_extra_v4/shot_noise_mackey.csv`
  restaurou o original (3239 linhas, seed0 ausente). Verificado que NENHUM outro
  arquivo original foi tocado (find -newermt). Redirecionamento corrigido para
  `EXP4_MG_CSV`; shot_noise_mackey seed0 refeito em v6.
