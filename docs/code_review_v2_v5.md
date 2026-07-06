# Code review - V2, V3, V4 e V5

Data do review: 2026-07-06

Escopo revisado:

- `embedded_effective_qrc_pipeline_v2.py`
- `extra_experiments_v3.py`
- `extra_experiments_v4.py`
- `extra_experiments_v5.py`
- logs `v2_*`, `v3_console.log`, `v4_console.log`, `v5_console.log`
- artefatos principais em `results_abc_comparison_v2/`, `results_extra_v3/`, `results_extra_v4/` e `results_extra_v5/`

Validação executada:

- `python3 -m py_compile embedded_effective_qrc_pipeline_v2.py extra_experiments_v3.py extra_experiments_v4.py extra_experiments_v5.py`
- Checagem de logs por `Traceback`, `RuntimeWarning`, `NaN`, `inf`, `WatchdogError` e falhas registradas.
- Checagem dos CSVs finais por contagem de seeds, valores `NaN`/`inf` e combinações ausentes.

Resultado geral:

- Nao ha erro de sintaxe nos scripts principais.
- V3 esta relativamente consistente nos artefatos principais.
- V2, V4 e V5 tem problemas metodologicos que podem afetar conclusoes do estudo.
- Os problemas mais graves sao: aceitacao de resultados Mackey-Glass sem validacao teacher-forced suficiente, V5 marcado como completo com `gamma=0.2` ausente, e V4/NARMA com `NaN`/overflow entrando nos resultados.

## Achados graves

### G1 - V2 aceita resultados Mackey-Glass mesmo quando a validacao teacher-forced falha

Arquivo:

- `embedded_effective_qrc_pipeline_v2.py:1156`
- `embedded_effective_qrc_pipeline_v2.py:1662`
- `embedded_effective_qrc_pipeline_v2.py:1924`

O problema:

`run_mg_model()` calcula `teacher_forced_ok`, mas `run_paper_replication()` grava os resultados de Mackey-Glass sem barrar ou separar os casos que falharam nessa validacao. Isso afeta diretamente qualquer conclusao sobre previsao autonoma, porque o rollout passa a ser analisado mesmo quando o readout one-step teacher-forced nao atingiu o criterio minimo.

Evidencia nos resultados:

- `results_abc_comparison_v2/paper_replication_mackey_glass.csv`: 144 falhas de `teacher_forced_ok` em 300 linhas.
- Por omega:
  - `omega=0.0`: 100/100 ok.
  - `omega=0.5`: 56/100 ok.
  - `omega=1.0`: 0/100 ok.
- `results_abc_comparison_v2/failed_runs.csv`: 171 registros `teacher_forced_r2_below_threshold`.

Impacto:

Conclusoes do tipo "Omega=0.5 superou Omega=1.0 em MG" ficam frágeis, porque `omega=1.0` falhou em 100% das validacoes teacher-forced. A comparacao passa a misturar qualidade de modelo com falha de protocolo.

Recomendacao:

- Tratar `teacher_forced_ok == False` como exclusao do rollout principal ou como categoria separada.
- Adicionar um gate: se menos de `min_decision_seeds` passarem na validacao, a comparacao deve ser marcada como inconclusiva.
- No resumo final, reportar explicitamente `n_passed_teacher_forced` por modelo/omega.

### G2 - V5 foi marcado como completo com `gamma=0.2` faltando

Arquivo:

- `extra_experiments_v5.py:939`
- `extra_experiments_v5.py:944`
- `extra_experiments_v5.py:945`

O problema:

No `main()`, uma excecao durante `process_gamma()` e capturada, registrada e o script continua ate `write_summary()` e `write_marker("summary")`. No run atual, `gamma=0.2` falhou por watchdog antes da avaliacao `AB-embedded`, mas a execucao escreveu `summary_complete.json`.

Evidencia nos resultados:

- `v5_console.log`: `ERROR processing gamma=0.2: WatchdogError(...)`.
- `results_extra_v5/failed_runs.csv`: `gamma_0.2,gamma_processing_exception`.
- `results_extra_v5/dynamical_sweep_stm.csv`: existe `M0_g0.2`, mas nao existe `g0.2_epi4` para `AB-embedded`.
- `results_extra_v5/scaling_law_summary.md`: o run config ainda lista `gammas: [0.02, 0.05, 0.1, 0.2]`.

Impacto:

O ajuste de escala `tau_mem ~ 1/gamma^x` foi feito com apenas 3 pontos reais para `AB-embedded` (`0.02`, `0.05`, `0.1`) em vez dos 4 planejados. Isso muda a interpretacao do Exp 7 e pode invalidar a afirmacao de sweep completo.

Recomendacao:

- Nao escrever `summary_complete.json` se qualquer gamma planejado nao tiver todos os modelos/seeds minimos.
- Adicionar uma tabela de completude por `gamma`, `eta`, modelo e seed.
- Marcar o resumo como `partial` quando houver falha tolerada.

### G3 - V4/NARMA-10 contem `NaN` por overflow e esses dados entram no resumo

Arquivo:

- `extra_experiments_v4.py:724`
- `extra_experiments_v4.py:757`
- `embedded_effective_qrc_pipeline_v2.py:1002`

O problema:

`narma10_target()` pode explodir numericamente. O log mostra `RuntimeWarning: overflow encountered in scalar multiply`, seguido de warnings em `mse_metrics()` e `capacity_score()`. O codigo nao valida que `target`, `pred`, `mse`, `nrmse` e `r2` sao finitos antes de gravar os resultados.

Evidencia nos resultados:

- `results_extra_v4/narma10_results.csv` tem 6 linhas com `NaN` em `nmse`, `nrmse` e `r2`.
- Todas as 6 linhas invalidas sao `seed=1`, uma para cada modelo.
- `AB-embedded` tambem perdeu `seed=8` por watchdog.
- O resumo `results_extra_v4/extra_v4_summary.md` diz "20 seeds", mas NARMA usa efetivamente 18 ou 19 dependendo da comparacao.

Impacto:

Medias e testes pareados do Exp 5/NARMA podem estar incorretos ou baseados em numero de seeds menor que o declarado. Isso afeta a conclusao sobre benchmarks padrao.

Recomendacao:

- Validar `np.isfinite(target).all()` logo apos gerar NARMA.
- Se o target divergir, registrar falha e gerar outra seed, ou redefinir a parametrizacao NARMA para uma versao numericamente estavel.
- No resumo, usar `n_finite` real em vez de texto fixo "20 seeds".

## Achados medios

### M1 - Intervalo de confianca em V4 tem sinal inconsistente

Arquivo:

- `extra_experiments_v4.py:860`
- `extra_experiments_v4.py:865`

O problema:

`exp5_paired_stats()` chama `paired_stats(..., larger_better=False)`. Nesse modo, a diferenca interna e calculada como `embedded - noaux` para metricas de erro. Depois o codigo grava `mean_diff_noaux_minus_emb` como `mean_noaux - mean_embedded`, mas reaproveita o IC de `paired_stats`, que esta no sinal oposto.

Evidencia:

Em `results_extra_v4/benchmarks_paired_stats.csv`, ha linhas com `mean_diff_noaux_minus_emb` positivo e `ci95_lo/ci95_hi` negativos.

Impacto:

O sinal do efeito pode ser interpretado invertido. Isso e especialmente perigoso em tabelas de paper.

Recomendacao:

- Ou gravar a diferenca no mesmo sentido de `paired_stats`.
- Ou inverter `ci95_lo`/`ci95_hi` quando publicar `noaux - embedded`.

### M2 - Watchdog remove dados, mas fases ainda sao marcadas como completas

Arquivos:

- `extra_experiments_v4.py:621`
- `extra_experiments_v4.py:884`
- `extra_experiments_v4.py:1010`
- `extra_experiments_v5.py:290`
- `extra_experiments_v5.py:913`

O problema:

Falhas por seed/config sao registradas, mas as fases continuam e escrevem marcadores `*_complete.json`.

Evidencia nos resultados:

- V4:
  - `shot_noise_capacities.csv`: `AB-embedded` tem 19 seeds; falta seed 0.
  - `shot_noise_mackey.csv`: `AB-embedded` tem 19 seeds; falta seed 0.
  - `narma10_results.csv`: `AB-embedded` tem 19 seeds; falta seed 8, alem de NaNs em seed 1.
  - `topology_control.csv`: topologia `parallel` tem 19 seeds; falta seed 12.
- V5:
  - `g0.02_epi4/AB-embedded`: 19 seeds; falta seed 15.
  - `gamma=0.2`: sem avaliacao `AB-embedded`.

Impacto:

Os resumos podem declarar conclusoes como se o protocolo completo tivesse sido executado.

Recomendacao:

- Definir minimo por fase, por exemplo `n_seeds >= 20` ou `n_seeds >= min_decision_seeds`.
- Se nao cumprir, marcar fase como `partial_complete` ou abortar antes do resumo final.
- Incluir matriz de completude nos summaries.

### M3 - Cache GPU de canais nao inclui `gamma` na chave

Arquivo:

- `embedded_effective_qrc_pipeline_v2.py:310`
- `embedded_effective_qrc_pipeline_v2.py:332`
- `embedded_effective_qrc_pipeline_v2.py:335`

O problema:

O cache em disco inclui `CFG.gamma` no nome, mas o cache GPU em memoria usa apenas `(seed, n, grid_size)`. V5 limpa manualmente o cache em `set_gamma()`, mas o helper base continua perigoso para qualquer outro sweep que altere `CFG.gamma`.

Impacto:

Pode haver uso silencioso de canais de um gamma anterior, produzindo resultados cientificamente invalidos sem erro visivel.

Recomendacao:

Adicionar `CFG.gamma`, `CFG.dt`, `CFG.grid_s_min` e `CFG.grid_s_max` na chave de `_GRID_GPU_CACHE`.

### M4 - Rollouts autonomos usam input fora da faixa e dependem de clamp silencioso

Arquivos:

- `embedded_effective_qrc_pipeline_v2.py:1116`
- `embedded_effective_qrc_pipeline_v2.py:1120`
- `embedded_effective_qrc_pipeline_v2.py:705`

O problema:

O valor previsto `y` e realimentado diretamente em `model.step(y, grid)`. Se `y` sai da faixa da grade, `select_channel_gpu()` usa o primeiro/ultimo canal e `last_clamped` apenas contabiliza o evento.

Evidencia nos resultados:

- Em V2, ha rollouts com ate 950 clamps.
- `out_of_range_fraction` chega a 0.957.
- `diverged` aparece em 14 a 29 linhas dependendo da tabela Mackey-Glass.

Impacto:

Parte das previsoes autonomas pode estar medindo comportamento saturado por clamp, nao dinamica real do modelo.

Recomendacao:

- Reportar `grid_clamps` como criterio central.
- Excluir ou separar rollouts com alta fracao out-of-range.
- Considerar clipping explicito e documentado ou treinar readout com restricao de faixa.

## Achados pequenos

### P1 - V5 imprime `reference tau_mem=None` no resumo

Arquivo:

- `extra_experiments_v5.py:844`

O problema:

O resumo le `reference_tau_mem`, mas o JSON usa `infra_repro_tau_mem`.

Impacto:

Erro de relatorio/reprodutibilidade. Pequeno, mas confunde o leitor.

Recomendacao:

Trocar para `b.get("infra_repro_tau_mem")`.

### P2 - Fallback de download do Santa Fe para cedo demais

Arquivo:

- `extra_experiments_v4.py:740`
- `extra_experiments_v4.py:744`
- `extra_experiments_v4.py:746`

O problema:

`load_santafe()` retorna `None` no primeiro URL que falha, em vez de tentar o proximo URL.

Impacto:

Fragilidade operacional. No run atual havia arquivo local, entao nao contaminou o resultado atual.

Recomendacao:

Registrar a falha e continuar tentando as demais URLs.

### P3 - Texto dos resumos usa numero planejado de seeds em vez do numero efetivo

Arquivos:

- `extra_experiments_v4.py:1091`
- `extra_experiments_v5.py:842`

O problema:

Os summaries exibem configuracao planejada, mas nem sempre deixam claro quantos seeds efetivos entraram nas medias apos falhas/NaNs.

Impacto:

Pode induzir interpretacao errada da robustez estatistica.

Recomendacao:

Adicionar `n_effective`, `n_missing`, `n_nonfinite` por tabela/metric/model.

## Avaliacao por versao

### V2

Status: executa e gerou artefatos completos, mas ha risco cientifico importante em Mackey-Glass.

Pontos fortes:

- 100 seeds na replicacao do paper.
- 20 seeds nas avaliacoes pareadas.
- Validador GPU/CPU e sanity checks existem.

Pontos problematicos:

- Resultados de rollout sao aceitos mesmo quando teacher-forced falha.
- Muitos clamps/out-of-range em previsao autonoma.
- Cache GPU e perigoso se `gamma` mudar sem limpeza manual.

### V3

Status: melhor estado entre as versoes revisadas.

Evidencia:

- `aux_dimension_sweep.csv`: 6240 linhas.
- 20 seeds para cada `d_B`.
- Sem `NaN`/`inf` critico nas tabelas principais revisadas.

Risco:

- O mesmo padrao geral de tolerar excecoes e marcar fase completa existe, mas nao apareceu como contaminacao nos resultados atuais.

### V4

Status: resultados parcialmente contaminados.

Problemas principais:

- NARMA com `NaN` por overflow.
- Seeds ausentes por watchdog.
- IC com sinal inconsistente nas estatisticas pareadas.
- Summary declara 20 seeds em NARMA, mas o n efetivo e menor.

### V5

Status: run parcial marcado como completo.

Problemas principais:

- `gamma=0.2` sem curva `AB-embedded`.
- Fit de escala feito com 3 pontos, nao 4.
- Summary mostra run config completo, mas dados reais incompletos.
- Pequeno erro de chave: `reference tau_mem=None`.

## Recomendacoes prioritarias

1. Corrigir V2 Mackey-Glass para separar/excluir rollouts sem `teacher_forced_ok`.
2. Corrigir V5 para nao marcar completo quando configs planejadas faltarem.
3. Corrigir V4 NARMA para rejeitar targets/metrica nao finitos.
4. Corrigir sinal dos ICs em `benchmarks_paired_stats.csv`.
5. Adicionar uma rotina unica de validacao pos-run:
   - contagem esperada vs real por tabela;
   - `NaN`/`inf`;
   - seeds faltantes;
   - minimo de seeds para decisao;
   - flags de partial/incomplete.
6. Incluir `gamma` e parametros de grade na chave de `_GRID_GPU_CACHE`.

## Conclusao

Os scripts rodam, mas nem todos os resultados atuais devem ser tratados como finais. V3 parece consistente. V2 precisa de revisao forte no protocolo Mackey-Glass. V4 precisa limpar NARMA e estatisticas. V5 precisa ser reexecutado ou remarcado como parcial ate incluir `gamma=0.2` e recuperar os seeds ausentes.
