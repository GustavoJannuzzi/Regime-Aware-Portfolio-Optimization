# Regime-Aware Portfolio Optimization via Offline RL

Artigo para a *Brazilian Review of Finance* (RBFin). Um agente de RL offline
(Conservative Q-Learning) com recompensa por CVaR e estado macro decide pesos de
carteira na B3. A pergunta é se ele bate os benchmarks clássicos.

**Resposta: não.** O valor do trabalho está na ablação, que liga uma peça por vez
e localiza exatamente onde o modelo quebra.

📄 **[`outputs/main.pdf`](outputs/main.pdf)** — 30 páginas, compilado.

## Resultado

Janela comum a todas as estratégias: 720 pregões, 2022-01-03 a 2024-11-13.

| Estratégia | Ret. anual | Sharpe | CVaR 95% | Max DD |
|---|---:|---:|---:|---:|
| **1/N** | **10,34%** | **0,608** | 0,0226 | **-17,8%** |
| Risk Parity | 8,94% | 0,556 | 0,0211 | -18,1% |
| A2 · CQL + Sharpe | 8,02% | 0,446 | 0,0221 | -24,9% |
| A4 · proposta completa | 7,88% | 0,406 | 0,0260 | -27,2% |
| MVP | 5,16% | 0,340 | **0,0198** | -18,2% |
| A3 · CQL + CVaR | 3,91% | 0,211 | 0,0241 | -26,1% |
| A1 · CQL sem freio | 2,62% | 0,156 | 0,0222 | -23,1% |

O $1/N$ domina a proposta em todas as métricas. O melhor agente é o **A2**, que
não usa nenhuma das duas contribuições que o artigo defendia.

## O que a ablação isola

Cada degrau muda **uma** peça. Mesmo conjunto, mesmos 5.000 passos, mesma semente.

| Degrau | Peça que entra | Δ Sharpe | Δ CVaR |
|---|---|---:|---:|
| A1 → A2 | conservadorismo do CQL | **+0,291** | ~0 |
| A2 → A3 | recompensa por CVaR | **-0,235** | pior |
| A3 → A4 | estado macro | +0,195 | pior |

**O achado central é o degrau do meio.** A recompensa por CVaR pontua os pesos
candidatos contra os 20 pregões anteriores a `t`; o estado do agente carrega
exatamente esses mesmos 20 pregões. A recompensa é, portanto, uma função
determinística do estado e da ação: não há retorno do ambiente. Um agente que a
maximiza não decide sobre o futuro, ele reordena dados que já enxerga, e converge
para um otimizador de cauda sobre janela curta. Não é viés de look-ahead. É
ausência de sinal, e o CVaR realizado piora monotonicamente conforme essa
recompensa entra.

A recompensa de A1 e A2 é o retorno realizado **no dia `t`**, que não está no
estado. Essa é feedback de verdade, e é onde o conservadorismo do CQL produz o
maior ganho do estudo.

## Rodar

```bash
pip install -r requirements.txt
python scripts/run_nb02.py && python scripts/run_nb03.py   # dados + EDA
python scripts/run_nb04.py                                  # benchmarks, 35 folds
python scripts/run_nb05.py && python scripts/run_nb06.py    # MDP + grid search
python scripts/run_nb07.py                                  # ablação A1-A4, ~90 min CPU
python scripts/run_nb08_09.py                               # tabelas + figuras
python scripts/measure_turnover.py                          # giro e concentração
python scripts/fill_placeholders.py                         # números no LaTeX
python scripts/compile_latex.py                             # main.pdf
```

## Ressalvas que o artigo declara

- **Nenhuma medida de incerteza.** Uma janela, uma semente, zero intervalo de
  confiança, zero teste de diferença. Em RL offline a variação entre sementes
  sozinha produz diferenças da ordem das reportadas.
- **Protocolos assimétricos.** Os benchmarks reestimam pesos a cada trimestre; o
  agente é treinado uma vez e congelado por três anos. A desvantagem é do agente.
- **Custo de transação decide tudo.** O agente rebalanceia **todo pregão**. Giro
  medido: 38 a 88 vezes o patrimônio por ano, contra ~1 vez dos benchmarks. A dez
  pontos-base por volta, 4% a 9% ao ano, sobre retornos de 2,6% a 8,0%. Nenhum
  número aqui é informativo sobre desempenho implementável.
- **Concentração.** Em 525 dos 748 dias de avaliação, um único ativo passa de
  metade da carteira. O freio do CQL limita a política ao suporte dos dados; ele
  não impõe diversificação, e não foi feito para isso.
- **Bloco macro magro.** Dois escalares (Meta Selic, SGS 432; spread do crédito
  livre PF, SGS 20786) e três indicadores de fase. **Não há série de CDS nem
  índice de volatilidade implícita** no estudo.

## Próximos passos, em ordem

1. Reespecificar a recompensa para pontuar a ação contra retornos que o agente
   ainda não observou, e rodar A3 e A4 de novo. Até lá, a segunda e a terceira
   perguntas do artigo estão em aberto, não respondidas.
2. Repetir a ablação em várias sementes e reportar a dispersão.
3. Restringir giro no objetivo ou cobrar custo na avaliação.

Histórico de sessões, auditoria e ponto de retomada: [`.claude/memory.md`](.claude/memory.md).
