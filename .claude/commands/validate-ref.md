# Comando: /validate-ref
# Uso: /validate-ref <doi_ou_titulo>
#
# Instruções para o agente:
# 1. Consulte a API Crossref: https://api.crossref.org/works/{doi}
# 2. Extraia: titulo, autores, ano, revista, volume, paginas, DOI
# 3. Formate a citação em ABNT e BibTeX
# 4. Se encontrado: atualize literature_db.csv com campo 'validated=True'
# 5. Se não encontrado: informe ao usuário e sugira busca manual
# 6. NUNCA confirme uma referência que não foi validada via API
#
# Formato ABNT:
#   SOBRENOME, Nome. Título do artigo. Revista, v. X, n. Y, p. ZZ-ZZ, AAAA.
#
# Formato BibTeX:
#   @article{PrimeiroAutorAno,
#     author  = {Sobrenome, Nome},
#     title   = {Título},
#     journal = {Revista},
#     year    = {AAAA},
#     volume  = {X},
#     pages   = {ZZ--ZZ},
#     doi     = {10.xxxx/xxxxx}
#   }
