# CLAUDE.md — Projeto: Offline RL Portfolio Optimization (RBFin)

## IDENTIDADE DO PROJETO
- Artigo científico para Brazilian Review of Finance (RBFin)
- Categoria: Original Article | Área: Financial Econometrics / Risk Management
- Instituição: UFPR Mestrado | Autor: Gustavo
- Idioma do artigo: Inglês | Norma: ABNT / estilo RBFin

## PERGUNTA DE PESQUISA
Um agente Offline RL (CQL) com reward CVaR e estado macro produz carteiras com melhor
desempenho ajustado ao risco out-of-sample do que benchmarks clássicos no mercado
brasileiro (B3)?

## REGRAS ABSOLUTAS DO AGENTE
1. NUNCA modifique arquivos em data/raw/ — são imutáveis
2. SEMPRE salve outputs de células nos notebooks (tabelas, gráficos, logs)
3. SEMPRE valide citações via API antes de usar — NUNCA invente referências
4. SEMPRE que gerar tabela LaTeX, salve em outputs/tables/*.tex
5. SEMPRE que gerar figura, salve em outputs/figures/*.pdf E *.png
6. Commits granulares: um commit por notebook concluído
7. Log de progresso em .claude/memory.md após cada sessão

## CONVENÇÕES DE CÓDIGO
- Python: type hints, docstrings NumPy, black formatter
- Notebooks: uma seção por tópico com markdown explicativo antes de cada célula
- Seed global: RANDOM_SEED = 42 em todo experimento estocástico
- Figuras: sempre matplotlib publication-ready (dpi=300, fontsize=11, sem título interno)

## APIs CONFIGURADAS
- Semantic Scholar: SEMANTIC_SCHOLAR_API_KEY (ver .env)
- OpenAlex: gratuita, sem chave
- Unpaywall: email em UNPAYWALL_EMAIL (ver .env)
- BCB SGS: gratuita — https://api.bcb.gov.br/dados/serie/bcdata.sgs.{cod}/dados
- yfinance: sem autenticação

## ESTADO ATUAL DO PROJETO
Ver .claude/memory.md para status de cada notebook.

## SLASH COMMANDS DISPONÍVEIS
- /lit-search  — busca literatura por query e salva em literature_db.csv
- /validate-ref — valida DOI/título via Crossref
- /new-section  — cria novo .tex em latex/sections/ com template padrão
- /run-ablation — executa configurações A1-A4 e salva resultados
- /export-table — converte DataFrame em LaTeX e salva em outputs/tables/
- /fig-save     — salva figura ativa em outputs/figures/ nos dois formatos
- /status       — mostra progresso geral do projeto

## ESTRUTURA DO PROJETO
```
rbfin-offline-rl/
├── CLAUDE.md               ← Cérebro do agente: contexto, regras, memória
├── pyproject.toml
├── requirements.txt
├── .env.example            ← Chaves de API (nunca commitar .env)
├── .claude/
│   ├── settings.json       ← Config do Claude Code (permissões, MCPs)
│   ├── memory.md           ← Memória persistente entre sessões
│   └── commands/           ← Slash commands
├── data/
│   ├── raw/                ← Dados brutos (nunca modificar)
│   ├── processed/          ← Dados limpos prontos para uso
│   └── external/           ← Literatura: literature_db.csv
├── notebooks/
│   ├── 00_setup.ipynb
│   ├── 01_literature.ipynb
│   ├── 02_data.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_benchmarks.ipynb
│   ├── 05_mdp.ipynb
│   ├── 06_cql_train.ipynb
│   ├── 07_ablation.ipynb
│   ├── 08_results.ipynb
│   └── 09_figures.ipynb
├── src/
│   ├── data/loader.py
│   ├── data/macro.py
│   ├── data/cleaner.py
│   ├── models/mdp.py
│   ├── models/cql_agent.py
│   ├── models/benchmarks.py
│   ├── models/reward.py
│   ├── eval/metrics.py
│   ├── eval/walkforward.py
│   ├── eval/ablation.py
│   └── utils/plots.py
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── models/
└── latex/
    ├── main.tex
    ├── preamble.tex
    ├── references.bib
    └── sections/
```

## ABLATION STUDY — Configurações A1-A4
Todas offline, mesmo dataset, mesmos 5.000 steps, mesma seed. Muda uma peça por
degrau. **A1 não é DDPG online** — isso estava errado no plano original e no
texto do artigo: `run_nb07.py` instancia CQL com `conservative_weight=0.001`,
ou seja, offline com o termo conservador praticamente desligado. O degrau
A1→A2 isola o **conservadorismo**, não online contra offline. Nenhum agente
online foi treinado neste estudo.

| Config | Conservadorismo CQL | Reward | Macro |
|--------|--------------------|--------|-------|
| A1     | ~0 (0.001)         | Sharpe | Não   |
| A2     | grid search        | Sharpe | Não   |
| A3     | grid search        | CVaR   | Não   |
| A4     | grid search        | CVaR   | Sim   |

## VETOR DE ESTADO (NB-05)
- Retornos normalizados (janela T=20)
- Volatilidade realizada
- Momentum
- SELIC_dir (one-hot 3: subindo/estável/caindo)
- CDS_norm, spread_norm, IVOL_norm

## PROTOCOLO WALK-FORWARD
- Treino: 252 dias | Teste: 63 dias (rebalanceamento trimestral)
- Sem look-ahead bias: treino sempre anterior ao teste

## CRONOGRAMA DE SESSÕES
| Sessão | Objetivo |
|--------|----------|
| 1 | Estrutura + NB-00 |
| 2 | NB-01 (literatura) |
| 3 | NB-02 + NB-03 (dados + EDA) |
| 4 | NB-04 (benchmarks) |
| 5 | NB-05 (MDP) |
| 6 | NB-06 (CQL treino) |
| 7 | NB-07 (ablation) |
| 8 | NB-08 + NB-09 (resultados + figuras) |
| 9 | LaTeX: intro, data, methodology, results |
| 10 | LaTeX: litreview, discussion, conclusion |
