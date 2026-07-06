# C1 (G2) — Lei de escala do horizonte de memória (4 pontos, corrigida)

Gerado combinando as curvas STM de gamma=0.02/0.05/0.1 (v5, imutável) com gamma=0.2 AB-embedded recomputado (20/20 seeds). Threshold C>0.1.

## Pontos (AB-embedded, eta=pi/4)

| gamma | 1/gamma | tau_mem | tau_FM (M0) | excess |
|---|---|---|---|---|
| 0.02 | 50.0 | 19 | 17 | 2 |
| 0.05 | 20.0 | 19 | 19 | 0 |
| 0.1 | 10.0 | 16 | 19 | -3 |
| 0.2 | 5.0 | 16 | 17 | -1 |

## Ajustes (log-log)

- **fit_a: tau_mem ~ (1/gamma)^p** — expoente p = **0.089** (IC95 boot [0.025, 0.089]), intercepto=2.619, R2=0.773, n=4.
- **fit_b: (tau_mem - tau_FM) ~ (1/gamma)^p** — expoente p = **n/a** (n=1). Excesso = [2, 0, -3, -1]: tau_mem ~ tau_FM (M0), quase sem excesso positivo, logo o ajuste log-log é degenerado (só pontos com excesso>0 entram) — o ganho de horizonte do embedded sobre o M0 é marginal nestes gammas.

## Consistência com dimensão (v3) e topologia (v4) em gamma=0.1

Referência tau_mem(gamma=0.1) = **16.0**.
- v3 (d_B): {'16': 21, '2': 18, '32': 16, '4': 19, '64': 17, '8': 19}
- v4 (topologia): {'chain_dB16': 21, 'chain_dB64': 17, 'parallel_dB64': 20}
- pontos gamma=0.1 (v3+v4): [21, 18, 16, 19, 17, 19, 21, 17, 20]
- **consistentes com a referência (|Δ|<=5): True**

Os pontos de v3 (dimensão d_B) e v4 (topologia) são medidos a gamma=0.1 fixo, portanto 'caem na curva' se forem consistentes com o ponto gamma=0.1 da lei de escala (tau_mem~16.0). O teste acima confirma/refuta isso.

## Comparação com o fit original (3 pontos)

- original (3 pts): p=0.101, R2=0.678, n=3
- corrigido (4 pts): p=0.089, R2=0.773, n=4