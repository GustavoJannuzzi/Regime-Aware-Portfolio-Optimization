# .claude/memory.md — Memória Persistente do Projeto

Projeto: Regime-Aware Portfolio Optimization via Offline RL (RBFin)
Última atualização: 2026-08-27
Sessão atual: **NB-07 a NB-09 concluídos. Artigo compilado. Auditoria de consistência feita.**

---

## Status dos Notebooks

| Notebook | Status | Última sessão | Observações |
|----------|--------|---------------|-------------|
| 00_setup | DONE | 2026-04-13 | Todas as 12 deps OK |
| 01_literature | DONE | 2026-04-13 | 185 papers, 159 validados Crossref |
| 02_data | DONE | 2026-04-13 | 10 tickers B3, 2015–2024; 2487 dias |
| 03_eda | DONE | 2026-04-13 | 4 figs, 3 tabelas; regimes SELIC |
| 04_benchmarks | DONE | 2026-04-13 | 35 folds walk-forward (só os benchmarks) |
| 05_mdp | DONE | 2026-04-13 | state_dim=225; N_train=1485; N_eval=748 |
| 06_cql_train | DONE | 2026-04-13 | best cw=0.5 |
| 07_ablation | **DONE** | **2026-08-27** | A1–A4 completos, 5.000 steps cada, ~90 min |
| 08_results | **DONE** | **2026-08-27** | main/subperiod/regime_conditional + 6 figuras |
| 09_figures | **DONE** | **2026-08-27** | incorporado no run_nb08_09.py |
| fill_placeholders | **DONE** | **2026-08-27** | 26 placeholders preenchidos, zero sobrando |
| compile_latex | **DONE** | **2026-08-27** | main.pdf gerado |

---

## ⚠️ Auditoria de 2026-08-27 — onze correções

O texto tinha sido escrito ANTES dos resultados existirem, e afirmava coisas que
o código não faz. Corrigido tudo, mas vale saber o que era:

### Bugs de código

1. **`run_nb08_09.py` sobrescrevia o A4** com a série do NB-06 (2.000 steps),
   enquanto A1–A3 vinham do NB-07 (5.000). A escada comparava tratamento
   misturado com duração de treino. O A4 agora é o do NB-07; o do NB-06 sai
   como checagem de robustez em `a4_tuning_run_check.csv`.
2. **Colisão de colunas no `join`.** O NB-07 já grava os benchmarks dentro de
   `ablation_returns.parquet`; o NB-08 juntava com `benchmark_returns.parquet`
   de novo → `ValueError`, pipeline morto logo depois de 90 min de treino.
3. **Janelas de comparação diferentes.** Agente tem 748 pregões, benchmarks
   param em 2024-11-13. Cada coluna era medida no próprio intervalo. Agora
   tudo usa a interseção (720 pregões), com o número na legenda das tabelas.
4. **`fill_placeholders.py`: Sharpe do MVP fixo** em 0.3692, medido em
   2016–2024, comparado contra um A4 que só existe de 2022 em diante.
5. **`fill_placeholders.py`: `a or b or c` com NaN.** `NaN` é *truthy*, então o
   primeiro lookup falhado devolvia `NaN` e os fallbacks nunca rodavam. O
   `NaN` ia impresso para o artigo. Agora aborta se faltar valor.
6. **`run_nb07.py` descartava o modelo do A4**, achando que `cql_best.d3` era o
   mesmo. Não é. Sem o arquivo não dá para medir pesos do A4 depois.

### Erros factuais no artigo

7. **Protocolo.** O texto dizia que o agente era retreinado em 35 folds de
   walk-forward. Ele é treinado **uma vez** (2016–2021) e congelado sobre um
   holdout (2022–2024). Os 35 folds são só dos benchmarks. E isso é uma
   assimetria contra o agente: os benchmarks reestimam por trimestre.
8. **A1 não é DDPG online.** É CQL offline com `conservative_weight=0.001`. O
   degrau A1→A2 isola o **conservadorismo**, não online contra offline.
   Nenhum agente online foi treinado neste estudo.
9. **Dados macro inventados.** O texto citava um CDS soberano de 5 anos e o
   IVOL-BR. Nenhum dos dois existe no projeto. A dimensão chamada de "CDS
   proxy" é o z-score móvel da **Meta SELIC (SGS 432)**; o "credit spread" é a
   **SGS 20786** (spread bancário do crédito livre PF), não corporate vs.
   governo. A SELIC citada era a série 11, é a 432.
10. **Regra de regime.** Descrita como "subiu na última reunião do COPOM"; o
    código usa variação em **63 pregões com limiar de ±0,25 p.p.**
11. **Custo de transação — o mais grave.** O texto dizia que o agente
    "rebalanceia trimestralmente" e que o custo "dificilmente reverteria os
    resultados". O `evaluate()` aplica os pesos **dia a dia**. Giro medido:

    | | Giro anualizado | Custo a 10 bps |
    |---|---:|---:|
    | A1 | 38,45× | 3,85% a.a. |
    | A2 | 87,88× | 8,79% a.a. |
    | A4 (checkpoint NB-06) | 61,32× | 6,13% a.a. |
    | MVP | ~1,04× | ~0,10% a.a. |
    | 1/N | 0× | 0% |

    Contra retorno anualizado da ordem de 9%. **O custo é da mesma ordem do
    retorno inteiro que está sendo comparado.** Novo script:
    `scripts/measure_turnover.py`.

### O que ainda falta

- **Nenhuma medida de incerteza.** Um holdout, uma semente, zero intervalo de
  confiança, zero teste de diferença entre estratégias. Repetir a ablação em
  várias sementes e reportar a dispersão é a extensão mais valiosa, e roda em
  menos de 2h por semente.
- **Turnover do A4 do NB-07** não foi medido: o modelo foi descartado por causa
  do bug 6, corrigido depois do run. Uma reexecução do NB-07 já salva os quatro.
- **Resultado líquido de custos** nunca foi computado.

---

## Resultados finais (janela comum, 720 pregões, 2022-01-03 a 2024-11-13)

| Estratégia | Ret. anual | Sharpe | Sortino | CVaR 95% | Max DD | Calmar |
|---|---:|---:|---:|---:|---:|---:|
| **1/N** | **10,34%** | **0,6076** | **0,9808** | 0,0226 | **-17,82%** | **0,5800** |
| Risk Parity | 8,94% | 0,5561 | 0,9013 | 0,0211 | -18,06% | 0,4950 |
| A2 | 8,02% | 0,4464 | 0,7784 | 0,0221 | -24,91% | 0,3218 |
| A4 (proposta) | 7,88% | 0,4058 | 0,6482 | 0,0260 | -27,22% | 0,2896 |
| MVP | 5,16% | 0,3400 | 0,5545 | **0,0198** | -18,22% | 0,2829 |
| A3 | 3,91% | 0,2111 | 0,3500 | 0,0241 | -26,06% | 0,1500 |
| A1 | 2,62% | 0,1557 | 0,2511 | 0,0222 | -23,14% | 0,1132 |

**Efeitos da escada (Sharpe):** conservadorismo +0,291 · recompensa CVaR **-0,235** ·
estado macro +0,195.

**CVaR piora monotonicamente** quando a recompensa por CVaR entra: 0,0222 (A1),
0,0221 (A2), 0,0241 (A3), 0,0260 (A4). As duas configurações que otimizam risco
de cauda têm o pior risco de cauda das sete estratégias.

**O melhor agente é o A2**, que não usa nenhuma das duas contribuições do artigo.
Todos os agentes ficam abaixo de 1/N e de Risk Parity.

**Por fase do ciclo (Sharpe A4 vs 1/N):** subindo +0,281 vs **+1,205** ·
parados +0,493 vs +0,732 · caindo **+0,461** vs **-0,452**. A hipótese central
era ganho concentrado na alta de juros. Aconteceu o inverso: o agente só ganha
com juros caindo, que é a assinatura de uma carteira defensiva, não de
inteligência de regime.

**Artigo:** `outputs/main.pdf`, 30 páginas, compila com zero erro, zero
referência indefinida, zero citação indefinida, zero overfull.

---

## Próxima sessão

1. Rodar `python scripts/run_nb07.py` de novo (agora salva os 4 modelos) e
   `python scripts/measure_turnover.py` para ter o giro do A4 desta ablação.
2. Repetir a ablação com sementes 43, 44, 45 e reportar média e dispersão.
3. Implementar custo de transação no `evaluate()` e reportar líquido.
