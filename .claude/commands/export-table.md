# Comando: /export-table
# Uso: /export-table <nome_variavel_df> <nome_arquivo>
#
# Instruções para o agente:
# 1. Pegue o DataFrame da variável indicada no notebook ativo
# 2. Gere LaTeX com df.to_latex(caption=..., label=..., escape=False)
# 3. Aplique formatação: booktabs=True, column_format adequado
# 4. Salve em outputs/tables/<nome_arquivo>.tex
# 5. Salve também em outputs/tables/<nome_arquivo>.csv
# 6. Imprima o código \input{} para usar no LaTeX
#
# Template de saída LaTeX:
#   \begin{table}[htbp]
#     \centering
#     \caption{<caption>}
#     \label{tab:<nome_arquivo>}
#     \input{../outputs/tables/<nome_arquivo>}
#   \end{table}
#
# Notas:
#   - Sempre usar booktabs=True (toprule, midrule, bottomrule)
#   - Números: 4 casas decimais para retornos, 2 para métricas de risco
#   - Usar \multicolumn para cabeçalhos agrupados quando necessário
