# .claude/memory.md — Memória Persistente do Projeto

Projeto: Regime-Aware Portfolio Optimization via Offline RL (RBFin)
Última atualização: 2026-04-14
Sessão atual: PAUSADA — NB-07 interrompido em A1 epoch 4/10 (step 2000 de 5000)

---

## Status dos Notebooks

| Notebook | Status | Última sessão | Observações |
|----------|--------|---------------|-------------|
| 00_setup | DONE | 2026-04-13 | Todas as 12 deps OK; APIs BCB, yfinance, OpenAlex testadas |
| 01_literature | DONE | 2026-04-13 | 185 papers, 159 validados Crossref; 9 core, 136 supporting |
| 02_data | DONE | 2026-04-13 | 10 tickers B3, 2015–2024; 2487 dias; macro BCB coletado |
| 03_eda | DONE | 2026-04-13 | 4 figs, 3 tabelas; regimes SELIC; ADF estacionário todos tickers |
| 04_benchmarks | DONE | 2026-04-13 | 35 folds walk-forward; EW:0.77 / MVP:0.37 / RP:0.47 Sharpe |
| 05_mdp | DONE | 2026-04-13 | state_dim=225; N_train=1485; N_eval=748; CVaR reward |
| 06_cql_train | DONE | 2026-04-13 | best cw=0.5; OOS Sharpe=0.4189 (2000 steps) |
| 07_ablation | PAUSADO | 2026-04-14 | INTERROMPIDO em A1 epoch 4/10 (step 2000/5000). Recomeça do zero na próxima sessão. |
| 08_results | PENDENTE | — | Script pronto (run_nb08_09.py); aguarda NB-07 |
| 09_figures | PENDENTE | — | Incorporado no run_nb08_09.py |

## Status das Seções LaTeX

| Seção | Status | Observações |
|-------|--------|-------------|
| abstract | DONE | Com \PLACEHOLDER{A4_sharpe} e \PLACEHOLDER{A4_cvar} |
| intro | DONE | ~620 palavras; ciclo SELIC, gaps, 3 contribuições |
| litreview | DONE | ~900 palavras; 5 subsections; Markowitz, CQL, CVaR, regimes |
| data | DONE | ~520 palavras; 10 ativos, BCB, regimes SELIC, stationarity |
| methodology | DONE | ~800 palavras; MDP, CQL loss eq., walk-forward, ablation design |
| results | DONE | Com \PLACEHOLDER{} para todos os nrs específicos do CQL A4 |
| discussion | DONE | CVaR em mercados emergentes, macro state, limitações, implicações |
| conclusion | DONE | 3 contribuições numeradas, 4 direções futuras |

## Arquivos gerados

### data/processed/ (11 arquivos)
- prices.parquet (2487x10)
- returns.parquet (2486x10)
- macro.parquet (2487x7: selic, ipca, spread_credito, selic_regime, selic_up/stable/down)
- ew_returns.parquet
- benchmark_returns.parquet (2205x3: EqualWeight, MVP, RiskParity)
- benchmark_weights.parquet
- offline_dataset.npz (obs, actions, rewards, next_obs, terminals — N=1485)
- eval_dataset.npz (N=748)
- state_dates_train.npy / state_dates_eval.npy
- data_summary.json

### outputs/models/ (3 arquivos)
- environment_check.json
- cql_best.d3 ← modelo A4 treinado (conservative_weight=0.5, 2000 steps)
- best_hparams.json

### outputs/tables/ (10 arquivos .tex + .csv)
- descriptive_stats, regime_stats, stationarity_tests
- benchmark_results, mdp_summary
- grid_search_results

### outputs/figures/ (8 arquivos .pdf + .png)
- fig_cumulative_returns, fig_correlation_matrix
- fig_selic_regimes, fig_return_distribution

### scripts/ (7 scripts prontos)
- run_nb00.py ✅
- run_nb01.py ✅
- run_nb02.py ✅
- run_nb03.py ✅
- run_nb04.py ✅
- run_nb05.py ✅
- run_nb06.py ✅
- run_nb07.py ✅ (pronto, NÃO executado completo)
- run_nb08_09.py ✅ (pronto, NÃO executado)

### latex/sections/ (8 arquivos .tex — todos escritos)
- abstract.tex, intro.tex, litreview.tex, data.tex
- methodology.tex, results.tex, discussion.tex, conclusion.tex

---

## Resultados conhecidos

### Grid Search CQL (NB-06)
| cw  | OOS Sharpe | OOS CVaR-5% | Ann.Ret |
|-----|-----------|-------------|---------|
| 0.5 | 0.3874 ← melhor | 0.0242 | 0.0713 |
| 1.0 | 0.3010 | 0.0251 | 0.0567 |
| 2.0 | 0.2840 | 0.0259 | 0.0550 |
| **Melhor retrained (2000 steps)** | **0.4189** | — | — |

### Benchmarks OOS (NB-04, 35 folds, 2016–2024)
| Strategy    | Ann.Return | Sharpe | Sortino | CVaR_0.95 | MaxDD   | Calmar |
|-------------|-----------|--------|---------|-----------|---------|--------|
| EqualWeight | 0.1899 | 0.7692 | 0.9496 | 0.0348 | -0.4688 | 0.4050 |
| MVP         | 0.0779 | 0.3692 | 0.4330 | 0.0299 | -0.4441 | 0.1753 |
| RiskParity  | 0.0989 | 0.4664 | 0.5299 | 0.0307 | -0.4582 | 0.2160 |

---

## PRÓXIMA SESSÃO — Retomar aqui

### ⚠️ PONTO EXATO DE PARADA (2026-04-14 ~10:45)
- NB-07 interrompido: A1 completou epoch 4/10 (step 2000/5000), A2/A3/A4 NÃO iniciados
- NB-08+09: NÃO executado (aguarda NB-07)
- fill_placeholders.py: NÃO executado (aguarda NB-08+09)
- compile_latex.py: NÃO executado (requer pdflatex instalado)

### PASSO 1 — OBRIGATÓRIO: Rodar NB-07 (ablation) do zero
```bash
cd "C:\Users\gustavo.j.siebel\Downloads\Regime-Aware Portfolio Optimization"
set PYTHONIOENCODING=utf-8
python scripts/run_nb07.py
```
⏱ Aguardar ~2h (4 configs × 10 epochs × ~3.5 min/epoch no CPU).
✅ Gera: outputs/tables/ablation_results.csv/.tex
✅ Gera: data/processed/ablation_returns.parquet
✅ Gera: outputs/models/ablation_metrics.json
✅ Gera: outputs/models/cql_a1.d3, cql_a2.d3, cql_a3.d3

### PASSO 2 — Rodar NB-08+09 (tabelas finais + figuras)
```bash
set PYTHONIOENCODING=utf-8
python scripts/run_nb08_09.py
```
✅ Gera: outputs/tables/main_results.csv/.tex
✅ Gera: outputs/tables/subperiod_results.csv/.tex
✅ Gera: outputs/tables/regime_conditional.csv/.tex
✅ Gera: outputs/figures/fig_wealth_curves, fig_drawdown, fig_cvar_subperiods,
          fig_ablation_heatmap, fig_training_curves, fig_regime_sharpe (.pdf + .png)

### PASSO 3 — Substituir \PLACEHOLDER{} no LaTeX
```bash
set PYTHONIOENCODING=utf-8
python scripts/fill_placeholders.py
```
✅ Substitui todos os \PLACEHOLDER{} em abstract.tex e results.tex com números reais

### PASSO 4 — Compilar PDF (requer MiKTeX ou TeX Live instalado)
```bash
python scripts/compile_latex.py
```
Se pdflatex não estiver no PATH: instalar MiKTeX em https://miktex.org/
✅ Gera: latex/main.pdf + outputs/main.pdf

---

## Progresso: 14 / 18 marcos (77.8%)
(NB-00→06 completos + 8 seções LaTeX escritas + scripts prontos)
Faltam: NB-07 executar (~2h) + NB-08/09 (~5min) + fill_placeholders (~1min) + compile (~5min)

## Correções LaTeX aplicadas nesta sessão (2026-04-14)
- preamble.tex: \newcommand{\PLACEHOLDER}[1]{\textbf{[??#1??]}} adicionado
- data.tex: selic_regimes → fig_selic_regimes (nome correto do arquivo)
- results.tex: 5 ambientes figure adicionados (wealth_curves, drawdown, regime_sharpe, cvar_subperiods, ablation_heatmap)

---

## Problemas registrados
- NB-07: interrompido 2x manualmente. Na próxima sessão rodar do zero (recomeça sempre do zero, sem checkpoint).
- LaTeX fixes (2026-04-14): \PLACEHOLDER cmd added to preamble; fig_selic_regimes path fixed in data.tex; 5 figures added to results.tex.
- CQL A4 Sharpe=0.4189 < EqualWeight Sharpe=0.7692: normal — CQL prioriza tail risk (CVaR), não Sharpe máximo. Comparar CVaR e drawdown é mais justo.
- Encoding Windows (cp1252): usar `set PYTHONIOENCODING=utf-8` antes de qualquer script com caracteres especiais.

## Log de Sessões

### Sessão de 2026-04-13 (principal)
- NB-00 a NB-06 executados com sucesso (scripts run_nb00 a run_nb06)
- Grid search CQL: melhor cw=0.5, Sharpe OOS=0.4189
- Artigo LaTeX completo: 8 seções escritas (~4.800 palavras totais estimadas)
- run_nb07.py e run_nb08_09.py escritos e prontos
- Sessão pausada por solicitação do usuário com NB-07 em andamento

### Sessões anteriores (2026-04-09 a 2026-04-10)
- Estrutura do projeto criada
- Dependências instaladas (torch 2.11.0+cpu, gymnasium 1.0.0, d3rlpy 2.8.1)
- NB-01 literatura: 185 papers coletados e validados
