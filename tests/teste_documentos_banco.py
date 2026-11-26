"""Documentos enviados no Postgres: envia num processo, "reinicia" e confere que voltaram."""
import json, os, subprocess, sys, tempfile, textwrap
REPO = sys.argv[1]
URL = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("TEST_DATABASE_URL", "postgresql://postgres@127.0.0.1:5544/postgres?sslmode=disable")
PY = sys.executable
BASE = f"""
import os, sys, tempfile, json, time
from pathlib import Path
tmp = Path(tempfile.mkdtemp())
os.environ.update(OPENAI_API_KEY="sk-teste", CHROMA_DIR=str(tmp/"c"), UPLOAD_DIR=str(tmp/"u"), DATABASE_URL={URL!r},
                  RETRIEVER_MAX_DISTANCE="2", RERANK="off", DOCUMENTOS_RETENCAO_DIAS="30")
sys.path.insert(0, {REPO!r})
from langchain_core.embeddings import DeterministicFakeEmbedding
import app.ingest as ingest
ingest.obter_embeddings = lambda: DeterministicFakeEmbedding(size=64)
from fastapi.testclient import TestClient
import app.api as api
from app import persistencia
from app.guardrails import LimitadorDeTaxa
api.limitador = LimitadorDeTaxa(1000); api.limitador_upload = LimitadorDeTaxa(1000)
ESP = "espacoBANCO01"
"""
def rodar(codigo):
    r = subprocess.run([PY, "-c", BASE + textwrap.dedent(codigo)], capture_output=True, text=True, timeout=300)
    linhas = [l for l in r.stdout.splitlines() if l.startswith("{")]
    if r.returncode or not linhas:
        print(r.stderr[-2500:])
    return json.loads(linhas[-1]) if linhas else {}

falhas = []
def check(cond, nome):
    print(("✅ " if cond else "❌ ") + nome)
    if not cond: falhas.append(nome)

import psycopg
with psycopg.connect(URL, autocommit=True) as c:
    for t in ("trechos_enviados", "documentos_enviados"):
        c.execute(f"DROP TABLE IF EXISTS {t}")

texto = ("Regulamento da Biblioteca Central. O prazo de empréstimo de livros é de quinze dias corridos. "
         "A multa por atraso é de um real por dia. ") * 20
a = rodar(f"""
with TestClient(api.app) as cli:
    r = cli.post("/api/documentos", data={{"espaco_id": ESP}}, files={{"arquivo": ("biblioteca.txt", {texto!r}.encode(), "text/plain")}})
    r2 = cli.post("/api/documentos", data={{"espaco_id": ESP}}, files={{"arquivo": ("outro.txt", b"Documento que vai ser removido. " * 40, "text/plain")}})
    cli.delete("/api/documentos", params={{"espaco_id": ESP, "arquivo": "outro.txt"}})
    persistencia.esperar_gravacoes(10)
    print(json.dumps({{"status": r.status_code, "docs": [d["arquivo"] for d in r.json()["documentos"] if not d["exemplo"]]}}))
""")
check(a.get("status") == 200 and "biblioteca.txt" in a.get("docs", []), "envio funciona com o banco ligado")

with psycopg.connect(URL, autocommit=True) as c:
    n_docs = c.execute("SELECT count(*) FROM documentos_enviados").fetchone()[0]
    n_trechos, dims = c.execute("SELECT count(*), max(coalesce(array_length(embedding::real[], 1), 0)) FROM trechos_enviados").fetchone()
check(n_docs == 1, "banco guarda o arquivo enviado e apaga o removido")
check(n_trechos > 0 and dims == 64, f"banco guarda os trechos com o embedding ({n_trechos} trechos, {dims} dimensões)")

b = rodar("""
with TestClient(api.app) as cli:
    docs = [d["arquivo"] for d in cli.get("/api/documentos", params={"espaco_id": ESP}).json()["documentos"] if not d["exemplo"]]
    arq = cli.get("/api/documentos/arquivo", params={"espaco_id": ESP, "arquivo": "biblioteca.txt"})
    from app.tools import busca_hibrida
    achados = busca_hibrida("prazo de empréstimo", ESP, exemplos=False)
    print(json.dumps({"docs": docs, "arquivo": arq.status_code, "trecho": "quinze dias" in arq.text,
                      "busca": [d.metadata.get("arquivo") for d, _ in achados]}))
""")
check(b.get("docs") == ["biblioteca.txt"], "depois do reinício o documento continua na lista (e o removido não)")
check(b.get("arquivo") == 200 and b.get("trecho"), "o arquivo original volta para abrir na página")
check(b.get("busca") and set(b["busca"]) == {"biblioteca.txt"}, "a busca acha os trechos restaurados sem vetorizar de novo")

with psycopg.connect(URL, autocommit=True) as c:
    c.execute("UPDATE documentos_enviados SET enviado_em = enviado_em - 31 * 86400")
d = rodar("""
with TestClient(api.app) as cli:
    docs = [d["arquivo"] for d in cli.get("/api/documentos", params={"espaco_id": ESP}).json()["documentos"] if not d["exemplo"]]
    print(json.dumps({"docs": docs}))
""")
with psycopg.connect(URL, autocommit=True) as c:
    resto = c.execute("SELECT count(*) FROM trechos_enviados").fetchone()[0]
check(d.get("docs") == [] and resto == 0, "documentos com mais de 30 dias são apagados na subida")
# Com o servidor no ar: a limpeza periódica apaga do banco, do índice e do disco o que venceu.
e = rodar("""
from app import acervo
with TestClient(api.app) as cli:
    cli.post("/api/documentos", data={"espaco_id": ESP}, files={"arquivo": ("vence.txt", b"Documento que vai vencer no prazo. " * 40, "text/plain")})
    cli.post("/api/documentos", data={"espaco_id": ESP}, files={"arquivo": ("fica.txt", b"Documento enviado agora, dentro do prazo. " * 40, "text/plain")})
    persistencia.esperar_gravacoes(10)
    with persistencia._conectar() as c:
        c.execute("UPDATE documentos_enviados SET enviado_em = enviado_em - 31 * 86400 WHERE arquivo = 'vence.txt'")
    apagados = acervo.limpar_vencidos()
    docs = [d["arquivo"] for d in cli.get("/api/documentos", params={"espaco_id": ESP}).json()["documentos"] if not d["exemplo"]]
    no_disco = sorted(p.name for p in (Path(os.environ["UPLOAD_DIR"]) / ESP).glob("*"))
    with persistencia._conectar() as c:
        no_banco = [r[0] for r in c.execute("SELECT arquivo FROM documentos_enviados WHERE espaco = %s", (ESP,)).fetchall()]
    print(json.dumps({"apagados": apagados, "docs": docs, "disco": no_disco, "banco": no_banco}))
""")
check(e.get("apagados") == 1 and e.get("docs") == ["fica.txt"], "com o servidor no ar, o documento vencido sai da lista")
check(e.get("disco") == ["fica.txt"] and e.get("banco") == ["fica.txt"], "e sai também do disco e do banco; o que está no prazo fica")
print("\nFALHAS:", falhas or "nenhuma")
