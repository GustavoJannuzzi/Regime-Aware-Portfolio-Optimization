# Comando: /fig-save
# Uso: /fig-save <nome_arquivo> [--fig-var <nome_variavel>]
#
# Instruções para o agente:
# 1. Pegue a figura ativa (plt.gcf()) ou a variável especificada
# 2. Salve em outputs/figures/<nome_arquivo>.pdf (formato vetorial para LaTeX)
# 3. Salve também em outputs/figures/<nome_arquivo>.png com dpi=300
# 4. Verifique se o tamanho é adequado (padrão: figsize=(6.5, 4))
# 5. Imprima o código \includegraphics para usar no LaTeX
#
# Padrões obrigatórios:
#   - dpi=300
#   - figsize=(6.5, 4) para figuras de coluna simples
#   - figsize=(13, 4) para figuras de largura total
#   - fontsize=11
#   - Sem título interno (title='')
#   - Legenda limpa, sem borda
#   - bbox_inches='tight'
#
# Template de saída LaTeX:
#   \begin{figure}[htbp]
#     \centering
#     \includegraphics[width=\linewidth]{<nome_arquivo>}
#     \caption{<caption>}
#     \label{fig:<nome_arquivo>}
#   \end{figure}
