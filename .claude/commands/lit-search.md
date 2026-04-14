# Comando: /lit-search
# Uso: /lit-search <query> [--limit N] [--year-from YYYY]
#
# Instruções para o agente:
# 1. Receba a query do usuário
# 2. Execute src/utils/literature.py com a query
# 3. Consulte em sequência: Semantic Scholar -> OpenAlex -> arXiv
# 4. Para cada resultado, tente obter PDF via Unpaywall
# 5. Deduplicar por DOI
# 6. Appenda resultados em data/external/literature_db.csv
# 7. Imprima resumo: N novos artigos encontrados, N duplicados ignorados
# 8. Pergunte ao usuário quais artigos devem ser marcados como 'core'
#
# APIs utilizadas:
#   Semantic Scholar: https://api.semanticscholar.org/graph/v1/paper/search
#   OpenAlex: https://api.openalex.org/works?search=
#   arXiv: http://export.arxiv.org/api/query?search_query=
#   Unpaywall: https://api.unpaywall.org/v2/{doi}?email=
#
# Colunas do literature_db.csv:
#   id, title, authors, year, journal, doi, url, pdf_url,
#   citation_abnt, bibtex_key, bibtex_entry, abstract, keywords,
#   citation_count, source, validated, relevance_tag, notes, added_at
