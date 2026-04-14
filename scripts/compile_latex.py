"""
Compila o artigo LaTeX: pdflatex + bibtex + pdflatex x2
Requer pdflatex e bibtex no PATH (MiKTeX ou TeX Live).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent
LATEX_DIR = ROOT / "latex"

def run(cmd: list[str], cwd: Path) -> int:
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr[-1000:])
    return result.returncode

def check_latex():
    r = subprocess.run(["pdflatex", "--version"], capture_output=True)
    return r.returncode == 0

if not check_latex():
    print("pdflatex nao encontrado no PATH.")
    print("Instale MiKTeX (https://miktex.org/) ou TeX Live e adicione ao PATH.")
    sys.exit(1)

print("=" * 60)
print("Compilando artigo LaTeX")
print("=" * 60)

# Passo 1: pdflatex (primeira passagem)
rc = run(["pdflatex", "-interaction=nonstopmode", "main.tex"], LATEX_DIR)

# Passo 2: bibtex
rc2 = run(["bibtex", "main"], LATEX_DIR)

# Passo 3 e 4: pdflatex (duas passagens para refs cruzadas)
run(["pdflatex", "-interaction=nonstopmode", "main.tex"], LATEX_DIR)
run(["pdflatex", "-interaction=nonstopmode", "main.tex"], LATEX_DIR)

pdf = LATEX_DIR / "main.pdf"
if pdf.exists():
    print(f"\nArtigo gerado com sucesso: {pdf}")
    # copiar para outputs/
    import shutil
    dest = ROOT / "outputs" / "main.pdf"
    shutil.copy(pdf, dest)
    print(f"Copiado para: {dest}")
else:
    print("\nERRO: main.pdf nao foi gerado. Verifique os logs acima.")
