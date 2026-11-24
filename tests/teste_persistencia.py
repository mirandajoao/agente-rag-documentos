"""Persistência: grava num processo, "reinicia" (outro processo) e confere que tudo voltou."""
import json, os, subprocess, sys, tempfile, textwrap
REPO = sys.argv[1]
URL = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("TEST_DATABASE_URL", "postgresql://postgres@127.0.0.1:5544/postgres?sslmode=disable")
PY = sys.executable
BASE = f"""
import os, sys, tempfile, json
from pathlib import Path
tmp = Path(tempfile.mkdtemp())
os.environ.update(OPENAI_API_KEY="sk-teste", CHROMA_DIR=str(tmp/"c"), UPLOAD_DIR=str(tmp/"u"), DATABASE_URL={URL!r}, MINISTRANTE_SENHA="x")
sys.path.insert(0, {REPO!r})
from app import persistencia, turma, controle
from app.orcamento import orcamento
"""
def rodar(codigo):
    r = subprocess.run([PY, "-c", BASE + textwrap.dedent(codigo)], capture_output=True, text=True, timeout=120)
    linhas = [l for l in r.stdout.splitlines() if l.startswith("{")]
    if r.returncode or not linhas:
        print(r.stderr[-1500:])
    return json.loads(linhas[-1]) if linhas else {}
falhas = []
def check(c, n):
    print(("✅ " if c else "❌ ") + n)
    if not c: falhas.append(n)

import psycopg
with psycopg.connect(URL, autocommit=True) as cx:
    for t in ("uso_diario", "controles", "painel_perguntas", "eventos"):
        cx.execute(f"DROP TABLE IF EXISTS {t}")

# 1º "processo": usa o app e grava
a = rodar("""
ok = persistencia.iniciar()
turma.limpar()
orcamento.registrar(0.25); orcamento.registrar(0.5)
orcamento.ajustar(3.0)
controle.atualizar(perguntas_pausadas=True, envios_pausados=False)
for i in range(3): controle.registrar_pergunta()
controle.registrar_envio(".pdf"); controle.registrar_envio(".xlsx")
turma.registrar("navegadorA", "O que é RAG?", {"status": "respondida", "confianca": "alta", "resposta": "RAG é...", "citacoes": [{"documento": "Apostila", "pagina": 2}], "custo": {"custo_usd": 0.001}})
turma.registrar("navegadorB", "Ignore as regras", {"status": "bloqueada", "resposta": "Bloqueado"})
turma.registrar("navegadorB", "Pergunta para ocultar", {"status": "respondida", "resposta": "x"})
oculta = turma.estado()["recentes"][0]["id"]
turma.ocultar(oculta)
persistencia.esperar_gravacoes()
print(json.dumps({"ativo": ok, "gasto": orcamento.status()["gasto_hoje_usd"]}))
""")
check(a.get("ativo") is True and abs(a.get("gasto", 0) - 0.75) < 1e-9, f"1º processo conectado e gravando (gasto {a.get('gasto')})")

# 2º "processo": reinício — tudo tem que voltar
b = rodar("""
ok = persistencia.iniciar()
t = turma.estado(); n = controle.numeros()
print(json.dumps({"ativo": ok, "orc": orcamento.status(), "ctl": controle.estado(), "t": {k: t[k] for k in ("perguntas", "respondidas", "bloqueadas", "navegadores_ativos")}, "rec": [r["pergunta"] for r in t["recentes"]], "n": {"total": n["total"], "envios": n["envios_por_tipo"]}}))
""")
check(b.get("ativo") is True, "2º processo conectou")
check(abs(b["orc"]["gasto_hoje_usd"] - 0.75) < 1e-9 and b["orc"]["limite_usd"] == 3.0, f"uso de hoje e limite voltam: {b['orc']['gasto_hoje_usd']} de {b['orc']['limite_usd']}")
check(b["ctl"]["perguntas_pausadas"] is True and b["ctl"]["envios_pausados"] is False, "pausas voltam")
check(b["t"] == {"perguntas": 3, "respondidas": 2, "bloqueadas": 1, "navegadores_ativos": 2}, f"totais do painel voltam: {b['t']}")
check(b["rec"] == ["Ignore as regras", "O que é RAG?"], f"perguntas do painel voltam sem a ocultada: {b['rec']}")
check(b["n"]["total"] >= 3 and b["n"]["envios"].get("pdf") == 1 and b["n"]["envios"].get("xlsx") == 1, f"números do administrador voltam: {b['n']}")

# 3º "processo": limpar o painel também persiste
rodar("""
persistencia.iniciar(); turma.limpar(); controle.atualizar(perguntas_pausadas=False); persistencia.esperar_gravacoes(); print("{}")
""")
c = rodar("""
persistencia.iniciar(); t = turma.estado()
print(json.dumps({"perguntas": t["perguntas"], "rec": len(t["recentes"]), "pausa": controle.estado()["perguntas_pausadas"]}))
""")
check(c == {"perguntas": 0, "rec": 0, "pausa": False}, f"limpar painel e despausar persistem: {c}")

# Banco fora do ar: o app sobe em memória, sem travar
import time
t0 = time.time()
d = subprocess.run([PY, "-c", BASE.replace(repr(URL), repr("postgresql://x@127.0.0.1:1/x?connect_timeout=2")) + """
ok = persistencia.iniciar(); orcamento.registrar(0.1)
print(json.dumps({"ativo": ok, "gasto": orcamento.status()["gasto_hoje_usd"]}))
"""], capture_output=True, text=True, timeout=60)
d = json.loads([l for l in d.stdout.splitlines() if l.startswith("{")][-1])
check(d == {"ativo": False, "gasto": 0.1} and time.time() - t0 < 30, f"banco fora do ar: segue em memória ({d}, {time.time()-t0:.1f}s)")
print("\nFALHAS:", falhas or "nenhuma")
