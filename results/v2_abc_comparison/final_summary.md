# Embedded and Effective Hierarchical ABC QRC — final summary (v2, GPU)

Paper base: https://arxiv.org/abs/2505.02491. Execucao v2 com o protocolo completo em GPU; o v1 (CPU-bounded) fica como registro historico e NAO foi reaproveitado nas analises.

## Execution scope

- Hardware: GPU NVIDIA GeForce RTX 3080 Ti (12.629 GB), torch 2.5.1+cu121 CUDA 12.1, Python 3.12.11.
- Dispositivo comprovado: evolucao embedded em complex64 na GPU (19.9 ms/passo ABC 4096x4096; 3.46 ms/passo AB 256x256); modelos noaux 16x16 exatos em complex128 na CPU.
- Washout/train/test = 1000/1000/1000 em todas as analises principais.
- Seeds: replicacao do paper = 100; avaliacao pareada = 20; tuning Optuna = 64 trials (noaux/AB, 8 seeds) e 32 trials x 4 seeds (ABC embedded).
- ABC embedded N=4 (matriz densidade 4096x4096) FOI executado em todas as fases (tuning, capacidades, IPC, Mackey-Glass).
- Convergencia do washout verificada antes do treino: todas as configuracoes convergiram (<0.001).

## Direct answers (12 perguntas)

1. A vantagem AB do paper foi reproduzida? Gate: Omega=0.5 SUPEROU Omega=1.0 em MSE150 medio ({'0.0': 0.20199102746375633, '0.5': 0.029728253482965337, '1.0': 0.05791619886334946}); estatistica pareada: n=100, dif media=+0.02819, IC95=[0.02522,0.03107], Wilcoxon-Holm p=6.9e-15, dz=1.90, SIGNIFICATIVO.
2. Qual Omega foi melhor (MG, MSE150 medio)? 0.5.
3. A memoria nao markoviana AB superou o regime markoviano? STM tau>=10: n=100, dif media=+0.01348, IC95=[0.01205,0.01502], Wilcoxon-Holm p=7.68e-16, dz=1.75, SIGNIFICATIVO; nao-markovianidade: n=100, dif media=+2.026e-07, IC95=[1.21e-07,2.814e-07], Wilcoxon-Holm p=0.000362, dz=0.52, SIGNIFICATIVO.
4. ABC embedded supera AB embedded? Capacidade media: ABC-hier=0.2842 vs AB=0.3535; exemplo estatistico (por tarefa em paired_statistics.csv): n=20, dif media=+0.0003556, IC95=[0.0001869,0.0005353], Wilcoxon-Holm p=0.0382, dz=0.89, SIGNIFICATIVO. MG: n=20, dif media=-0.3071, IC95=[-0.8342,-0.03461], Wilcoxon-Holm p=0.00974, dz=-0.27, SIGNIFICATIVO.
5. ABC sem auxiliares supera AB sem auxiliares? n=20, dif media=-1.127e-05, IC95=[-2.344e-05,-4.26e-07], Wilcoxon-Holm p=1, dz=-0.41, nao significativo; medias: ABC-noaux-kraus=0.8392, AB-noaux-kraus=0.5735.
6. ABC embedded supera ABC sem auxiliares? n=20, dif media=-0.001162, IC95=[-0.001329,-0.001001], Wilcoxon-Holm p=0.000364, dz=-2.95, SIGNIFICATIVO; MG: n=20, dif media=-0.1789, IC95=[-0.7949,0.2089], Wilcoxon-Holm p=1, dz=-0.14, nao significativo.
7. As versoes apresentam escalas de memoria semelhantes? Ver effective_memory_scales.csv (21 registros; picos STM por modelo e autocorrelacao das camadas A/B/C do ABC embedded N=4).
8. A arquitetura sem auxiliares reproduz os revivals? Sem picos secundarios robustos detectados.
9. A versao sem auxiliares funciona na previsao autonoma? Sim (rollout sem valores futuros); validacao teacher-forced media por modelo: {'AB-Markov': 0.9742, 'AB-embedded': 0.9993, 'AB-noaux-kraus': 0.978, 'ABC-Markov': 0.9742, 'ABC-embedded-C-off': 0.9905, 'ABC-embedded-hierarchical': 0.9991, 'ABC-embedded-parallel': 0.9977, 'ABC-embedded-tied': 0.9989, 'ABC-noaux-hierarchical': 0.9627, 'ABC-noaux-kraus': 0.9627, 'ABC-noaux-shuffled-history': 0.8969, 'ABC-noaux-tied': 0.9995, 'M0-noaux': 0.9973}; VPT em mackey_glass_standard.csv.
10. Qual arquitetura utiliza menos qubits? M0/AB/ABC-noaux usam apenas N_A=4 qubits fisicos; embedded AB usa 8 e embedded ABC usa 12.
11. Qual arquitetura utiliza menos memoria total? M0-noaux; os noaux trocam qubits auxiliares por buffer classico de estados 16x16. Ver memory_resource_comparison.csv.
12. Qual arquitetura possui melhor relacao desempenho-custo? Melhor capacidade media global: ABC-noaux-hierarchical (0.8392 media). Melhor noaux: ABC-noaux-hierarchical. Custo por passo e memoria em computational_cost_comparison.csv.

## Controle negativo shuffled-history

- ABC-noaux-kraus vs shuffled-history: n=20, dif media=+0.0001711, IC95=[0.0001317,0.0002077], Wilcoxon-Holm p=0.000364, dz=1.99, SIGNIFICATIVO.
- O shuffle e refeito por seed e usa apenas estados passados do buffer (checks em sanity_checks.json). Se o controle ainda vencer em alguma tarefa, isso indica que a ordem temporal do buffer nao esta sendo explorada naquela tarefa, e esta reportado como tal.

## Hypothesis decisions (n>=20 obrigatorio para aceitar/rejeitar)

- H1: aceita. Detalhes em hypothesis_decisions.json.
- H2: nao aceita de forma robusta / parcial. Detalhes em hypothesis_decisions.json.
- H3: nao aceita. Detalhes em hypothesis_decisions.json.
- H4: vantagem embedded significativa em parte das tarefas. Detalhes em hypothesis_decisions.json.
- H5: equivalencia pratica nao demonstrada. Detalhes em hypothesis_decisions.json.
- H6: vantagem desempenho-custo noaux plausivel. Detalhes em hypothesis_decisions.json.

## Limitations

- Seeds de replicacao executadas: 100 de 100 planejadas (reducao explicita registrada em failed_runs.csv quando aplicavel).
- O readout e somente em A com 66 features Pauli para todos os modelos, como pre-registrado.
- Estados embedded evoluidos em complex64 na GPU; validacao pontual contra complex128 em sanity_checks.json.
- Nao ha afirmacao de vantagem quantica; resultados negativos e inconclusivos foram preservados.
