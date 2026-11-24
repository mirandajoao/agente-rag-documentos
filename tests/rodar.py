"""Roda os testes automáticos do projeto (localmente e no GitHub Actions).

Uso:  python tests/rodar.py            # todos
      python tests/rodar.py moderacao  # só os que têm "moderacao" no nome

Os testes não chamam a OpenAI: usam modelos e embeddings falsos.
O de persistência só roda quando há um Postgres em TEST_DATABASE_URL.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

PASTA = Path(__file__).resolve().parent
REPO = PASTA.parent


def falhou(saida: str, codigo: int) -> bool:
    if codigo != 0:
        return True
    linhas = [l for l in saida.splitlines() if l.startswith("FALHAS:")]
    return any("nenhuma" not in l for l in linhas)


def main() -> int:
    filtro = sys.argv[1:]
    testes = sorted(PASTA.glob("teste_*.py"))
    if filtro:
        testes = [t for t in testes if any(f in t.stem for f in filtro)]
    # Os testes usam modelos falsos; o reranking só fica ligado no teste dele.
    os.environ.setdefault("RERANK", "off")
    resultado: list[tuple[str, bool, float]] = []
    for teste in testes:
        if teste.stem in {"teste_persistencia", "teste_documentos_banco", "teste_memoria"} and not os.getenv("TEST_DATABASE_URL"):
            print(f"-- {teste.stem}: pulado (defina TEST_DATABASE_URL)")
            continue
        inicio = time.time()
        r = subprocess.run([sys.executable, str(teste), str(REPO)], capture_output=True, text=True, timeout=600)
        saida = r.stdout + r.stderr
        ruim = falhou(r.stdout, r.returncode)
        resultado.append((teste.stem, ruim, time.time() - inicio))
        print(f"{'FALHOU' if ruim else 'ok    '} {teste.stem} ({time.time() - inicio:.1f}s)")
        if ruim:
            print(saida[-4000:])
    falhas = [n for n, ruim, _ in resultado if ruim]
    print(f"\n{len(resultado) - len(falhas)}/{len(resultado)} testes passaram")
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
