# Progress log — QRC ABC v2 (GPU) — RETOMADA

Paper base: arXiv:2505.02491. Diretório de computação: `results_abc_comparison_v2/`.
Entregáveis finais serão espelhados em `results_abc_gpu/`.

## Auditoria de ambiente (2026-07-04 ~15:03)

- **GPU: NVIDIA GeForce RTX 3080 Ti, 12.63 GB** (só ~414 MB em uso por outro processo). `torch.cuda.is_available()=True`.
- torch do venv de execução (`~/venvs/qrc_v2`): **2.5.1+cu121**; driver 535.309.01 / CUDA 12.2.
- **Requisito de GPU: ATENDIDO.** Modelos embedded (256×256 AB, 4096×4096 ABC N=4) rodam em `cuda`; modelos noaux 16×16 rodam em CPU exato `complex128` por decisão de precisão registrada em `config.json:device_policy` (GPU não acelera 16×16). Sem fallback silencioso para CPU nos experimentos caros.
- **Micro-benchmark (do `benchmark_v2.json`, medido em GPU):** AB-embedded 3.46 ms/passo; **ABC-embedded-hierarchical N=4 19.85 ms/passo** (559.7 MB VRAM). Extrapolação do protocolo completo → orçamento de ~25 h (registrado em `budget_v2.json`), com redução documentada de ABC tuning para 32 trials × 4 seeds (em `failed_runs.csv`).

## Descoberta: execução v2 já estava ~90% pronta

Um run v2 em GPU foi lançado em 2026-07-03 16:10 e rodou ~22.6 h. **Parou limpo (sem crash/OOM/traceback)** na fase Mackey — encerramento externo. O pipeline é checkpointed por fase (marker) e por (seed, arch) dentro da fase Mackey.

| Fase | Marker | Status na retomada |
|------|--------|--------------------|
| sanity | ✅ | completa (29 checks) |
| smoke | ✅ | completa |
| washout | ✅ | completa |
| tuning | ✅ | AB+ABC embedded N=4 (32 trials×4 seeds) + noaux (64 trials) |
| paper (AB embedded, 100 seeds, 1000/1000/1000) | ✅ | gate PASSOU: Ω=0.5 vence Ω=1.0 (dz=1.90, p_wilcoxon=3.5e-17, 94/100) |
| multiscale (ABC embedded N=4, 15 arqs × 20 seeds) | ✅ | `multiscale_capacities.csv` = 7200 linhas |
| **mackey** | ❌ | **retomando na seed 5 de 20** (dados 0–4 salvos) |
| statistics | ❌ | pendente |
| figures | ❌ | pendente (`figures_abc_comparison_v2/` vazio) |

## Decisão

Retomar o run checkpointed in-place (aproveita ~22.6 h já computadas) em vez de recomeçar (~25 h redundantes). Comando: `~/venvs/qrc_v2/bin/python -u qrc_pipeline.py` (resume automático). Nenhuma redução adicional de protocolo introduzida na retomada.

---

## Entradas de progresso

### 2026-07-04 15:19:00
- Fase atual: **mackey (seed 5/20)** | processo vivo: **sim**
- Concluído desde o último: seeds mackey completas = 5/20 (75 linhas em mackey_glass_standard.csv)
- **Global concluído: 83.7% | restante: 16.3%** (base: custo estimado das fases, não contagem)
- ETA: ~5.2 h (15 seeds mackey restantes @18.7min) + ~0.5 h pós
- GPU (util%,mem_used,mem_total,temp): 100, 1443, 12288, 86 | compute-apps: 689244, /home/miziara/dementia-boost/.venv/bin/python3, 384 MiB;1348329, /home/franco/venvs/qrc_v2/bin/python, 1026 MiB;

### 2026-07-04 15:32:00
- Fase atual: **mackey (seed 6/20)** | processo vivo: **sim**
- Concluído desde o último: seeds mackey completas = 6/20 (87 linhas em mackey_glass_standard.csv)
- **Global concluído: 84.7% | restante: 15.3%** (base: custo estimado das fases, não contagem)
- ETA: ~4.9 h (14 seeds mackey restantes @18.7min) + ~0.5 h pós
- GPU (util%,mem_used,mem_total,temp): 100, 1443, 12288, 86 | compute-apps: 689244, /home/miziara/dementia-boost/.venv/bin/python3, 384 MiB;1348329, /home/franco/venvs/qrc_v2/bin/python, 1026 MiB;

### 2026-07-04 15:45:00
- Fase atual: **mackey (seed 6/20)** | processo vivo: **sim**
- Concluído desde o último: seeds mackey completas = 6/20 (91 linhas em mackey_glass_standard.csv)
- **Global concluído: 84.7% | restante: 15.3%** (base: custo estimado das fases, não contagem)
- ETA: ~4.9 h (14 seeds mackey restantes @18.7min) + ~0.5 h pós
- GPU (util%,mem_used,mem_total,temp): 100, 1443, 12288, 84 | compute-apps: 689244, /home/miziara/dementia-boost/.venv/bin/python3, 384 MiB;1348329, /home/franco/venvs/qrc_v2/bin/python, 1026 MiB;

### 2026-07-04 15:58:00
- Fase atual: **mackey (seed 7/20)** | processo vivo: **sim**
- Concluído desde o último: seeds mackey completas = 7/20 (102 linhas em mackey_glass_standard.csv)
- **Global concluído: 85.7% | restante: 14.3%** (base: custo estimado das fases, não contagem)
- ETA: ~4.6 h (13 seeds mackey restantes @18.7min) + ~0.5 h pós
- GPU (util%,mem_used,mem_total,temp): 100, 1571, 12288, 83 | compute-apps: 689244, /home/miziara/dementia-boost/.venv/bin/python3, 384 MiB;1348329, /home/franco/venvs/qrc_v2/bin/python, 1154 MiB;

### 2026-07-04 16:11:00
- Fase atual: **mackey (seed 8/20)** | processo vivo: **sim**
- Concluído desde o último: seeds mackey completas = 8/20 (114 linhas em mackey_glass_standard.csv)
- **Global concluído: 86.6% | restante: 13.4%** (base: custo estimado das fases, não contagem)
- ETA: ~4.2 h (12 seeds mackey restantes @18.7min) + ~0.5 h pós
- GPU (util%,mem_used,mem_total,temp): 100, 1571, 12288, 87 | compute-apps: 689244, /home/miziara/dementia-boost/.venv/bin/python3, 384 MiB;1348329, /home/franco/venvs/qrc_v2/bin/python, 1154 MiB;

### 2026-07-04 16:24:00
- Fase atual: **mackey (seed 9/20)** | processo vivo: **sim**
- Concluído desde o último: seeds mackey completas = 9/20 (126 linhas em mackey_glass_standard.csv)
- **Global concluído: 87.6% | restante: 12.4%** (base: custo estimado das fases, não contagem)
- ETA: ~3.9 h (11 seeds mackey restantes @18.7min) + ~0.5 h pós
- GPU (util%,mem_used,mem_total,temp): 100, 1571, 12288, 87 | compute-apps: 689244, /home/miziara/dementia-boost/.venv/bin/python3, 384 MiB;1348329, /home/franco/venvs/qrc_v2/bin/python, 1154 MiB;

### 2026-07-04 16:37:00
- Fase atual: **mackey (seed 10/20)** | processo vivo: **sim**
- Concluído desde o último: seeds mackey completas = 10/20 (130 linhas em mackey_glass_standard.csv)
- **Global concluído: 88.6% | restante: 11.4%** (base: custo estimado das fases, não contagem)
- ETA: ~3.6 h (10 seeds mackey restantes @18.7min) + ~0.5 h pós
- GPU (util%,mem_used,mem_total,temp): 7, 1571, 12288, 74 | compute-apps: 689244, /home/miziara/dementia-boost/.venv/bin/python3, 384 MiB;1348329, /home/franco/venvs/qrc_v2/bin/python, 1154 MiB;

### 2026-07-04 16:50:00
- Fase atual: **mackey (seed 10/20)** | processo vivo: **sim**
- Concluído desde o último: seeds mackey completas = 10/20 (142 linhas em mackey_glass_standard.csv)
- **Global concluído: 88.6% | restante: 11.4%** (base: custo estimado das fases, não contagem)
- ETA: ~3.6 h (10 seeds mackey restantes @18.7min) + ~0.5 h pós
- GPU (util%,mem_used,mem_total,temp): 100, 1571, 12288, 83 | compute-apps: 689244, /home/miziara/dementia-boost/.venv/bin/python3, 384 MiB;1348329, /home/franco/venvs/qrc_v2/bin/python, 1154 MiB;

### 2026-07-04 17:03:00
- Fase atual: **mackey (seed 11/20)** | processo vivo: **sim**
- Concluído desde o último: seeds mackey completas = 11/20 (153 linhas em mackey_glass_standard.csv)
- **Global concluído: 89.6% | restante: 10.4%** (base: custo estimado das fases, não contagem)
- ETA: ~3.3 h (9 seeds mackey restantes @18.7min) + ~0.5 h pós
- GPU (util%,mem_used,mem_total,temp): 100, 1571, 12288, 87 | compute-apps: 689244, /home/miziara/dementia-boost/.venv/bin/python3, 384 MiB;1348329, /home/franco/venvs/qrc_v2/bin/python, 1154 MiB;

### 2026-07-04 17:16:00
- Fase atual: **mackey (seed 12/20)** | processo vivo: **sim**
- Concluído desde o último: seeds mackey completas = 12/20 (165 linhas em mackey_glass_standard.csv)
- **Global concluído: 90.6% | restante: 9.4%** (base: custo estimado das fases, não contagem)
- ETA: ~3.0 h (8 seeds mackey restantes @18.7min) + ~0.5 h pós
- GPU (util%,mem_used,mem_total,temp): 100, 1571, 12288, 87 | compute-apps: 689244, /home/miziara/dementia-boost/.venv/bin/python3, 384 MiB;1348329, /home/franco/venvs/qrc_v2/bin/python, 1154 MiB;

### 2026-07-04 17:29:01
- Fase atual: **mackey (seed 13/20)** | processo vivo: **sim**
- Concluído desde o último: seeds mackey completas = 13/20 (175 linhas em mackey_glass_standard.csv)
- **Global concluído: 91.6% | restante: 8.4%** (base: custo estimado das fases, não contagem)
- ETA: ~2.7 h (7 seeds mackey restantes @18.7min) + ~0.5 h pós
- GPU (util%,mem_used,mem_total,temp): 5, 1184, 12288, 52 | compute-apps: 1348329, /home/franco/venvs/qrc_v2/bin/python, 1154 MiB;

### 2026-07-04 17:42:01
- Fase atual: **mackey (seed 13/20)** | processo vivo: **sim**
- Concluído desde o último: seeds mackey completas = 13/20 (182 linhas em mackey_glass_standard.csv)
- **Global concluído: 91.6% | restante: 8.4%** (base: custo estimado das fases, não contagem)
- ETA: ~2.7 h (7 seeds mackey restantes @18.7min) + ~0.5 h pós
- GPU (util%,mem_used,mem_total,temp): 100, 1184, 12288, 83 | compute-apps: 1348329, /home/franco/venvs/qrc_v2/bin/python, 1154 MiB;

### 2026-07-04 17:55:01
- Fase atual: **mackey (seed 14/20)** | processo vivo: **sim**
- Concluído desde o último: seeds mackey completas = 14/20 (194 linhas em mackey_glass_standard.csv)
- **Global concluído: 92.5% | restante: 7.5%** (base: custo estimado das fases, não contagem)
- ETA: ~2.4 h (6 seeds mackey restantes @18.7min) + ~0.5 h pós
- GPU (util%,mem_used,mem_total,temp): 100, 1184, 12288, 83 | compute-apps: 1348329, /home/franco/venvs/qrc_v2/bin/python, 1154 MiB;

### 2026-07-04 18:08:01
- Fase atual: **mackey (seed 15/20)** | processo vivo: **sim**
- Concluído desde o último: seeds mackey completas = 15/20 (206 linhas em mackey_glass_standard.csv)
- **Global concluído: 93.5% | restante: 6.5%** (base: custo estimado das fases, não contagem)
- ETA: ~2.1 h (5 seeds mackey restantes @18.7min) + ~0.5 h pós
- GPU (util%,mem_used,mem_total,temp): 100, 1184, 12288, 86 | compute-apps: 1348329, /home/franco/venvs/qrc_v2/bin/python, 1154 MiB;

### 2026-07-04 18:21:01
- Fase atual: **mackey (seed 16/20)** | processo vivo: **sim**
- Concluído desde o último: seeds mackey completas = 16/20 (218 linhas em mackey_glass_standard.csv)
- **Global concluído: 94.5% | restante: 5.5%** (base: custo estimado das fases, não contagem)
- ETA: ~1.7 h (4 seeds mackey restantes @18.7min) + ~0.5 h pós
- GPU (util%,mem_used,mem_total,temp): 98, 1184, 12288, 86 | compute-apps: 1348329, /home/franco/venvs/qrc_v2/bin/python, 1154 MiB;

### 2026-07-04 18:34:01
- Fase atual: **mackey (seed 17/20)** | processo vivo: **sim**
- Concluído desde o último: seeds mackey completas = 17/20 (230 linhas em mackey_glass_standard.csv)
- **Global concluído: 95.5% | restante: 4.5%** (base: custo estimado das fases, não contagem)
- ETA: ~1.4 h (3 seeds mackey restantes @18.7min) + ~0.5 h pós
- GPU (util%,mem_used,mem_total,temp): 100, 1184, 12288, 86 | compute-apps: 1348329, /home/franco/venvs/qrc_v2/bin/python, 1154 MiB;

### 2026-07-04 18:47:01
- Fase atual: **mackey (seed 18/20)** | processo vivo: **sim**
- Concluído desde o último: seeds mackey completas = 18/20 (242 linhas em mackey_glass_standard.csv)
- **Global concluído: 96.5% | restante: 3.5%** (base: custo estimado das fases, não contagem)
- ETA: ~1.1 h (2 seeds mackey restantes @18.7min) + ~0.5 h pós
- GPU (util%,mem_used,mem_total,temp): 98, 1184, 12288, 87 | compute-apps: 1348329, /home/franco/venvs/qrc_v2/bin/python, 1154 MiB;

### 2026-07-04 19:00:01
- Fase atual: **mackey (seed 19/20)** | processo vivo: **sim**
- Concluído desde o último: seeds mackey completas = 19/20 (251 linhas em mackey_glass_standard.csv)
- **Global concluído: 97.4% | restante: 2.6%** (base: custo estimado das fases, não contagem)
- ETA: ~0.8 h (1 seeds mackey restantes @18.7min) + ~0.5 h pós
- GPU (util%,mem_used,mem_total,temp): 0, 1184, 12288, 50 | compute-apps: 1348329, /home/franco/venvs/qrc_v2/bin/python, 1154 MiB;

### 2026-07-04 19:13:02
- Fase atual: **mackey (seed 19/20)** | processo vivo: **sim**
- Concluído desde o último: seeds mackey completas = 19/20 (259 linhas em mackey_glass_standard.csv)
- **Global concluído: 97.4% | restante: 2.6%** (base: custo estimado das fases, não contagem)
- ETA: ~0.8 h (1 seeds mackey restantes @18.7min) + ~0.5 h pós
- GPU (util%,mem_used,mem_total,temp): 98, 1184, 12288, 83 | compute-apps: 1348329, /home/franco/venvs/qrc_v2/bin/python, 1154 MiB;

### 2026-07-04 19:26:02
- Fase atual: **CONCLUÍDO (full)** | processo vivo: **sim**
- Concluído desde o último: seeds mackey completas = 20/20 (260 linhas em mackey_glass_standard.csv)
- **Global concluído: 100.0% | restante: 0.0%** (base: custo estimado das fases, não contagem)
- ETA: ~20-30 min (statistics + figuras + notebook)
- GPU (util%,mem_used,mem_total,temp): 0, 27, 12288, 34 | compute-apps: nenhum

### 2026-07-04 19:26:02 — pipeline CONCLUÍDO (full_complete.json presente). Heartbeat encerrado.
