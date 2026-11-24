"""Visualizador: texto da página citada, sem repetir a sobreposição, e arquivos do PDF.js servidos com segurança."""
import os, sys, tempfile
from pathlib import Path
REPO = Path(sys.argv[1]); tmp = Path(tempfile.mkdtemp())
os.environ.update(OPENAI_API_KEY="sk-teste", CHROMA_DIR=str(tmp / "c"), UPLOAD_DIR=str(tmp / "u"), RETRIEVER_MAX_DISTANCE="2")
sys.path.insert(0, str(REPO))
from langchain_core.embeddings import DeterministicFakeEmbedding
import app.ingest as ingest
ingest.obter_embeddings = lambda: DeterministicFakeEmbedding(size=64)
import app.api as api
from app.guardrails import LimitadorDeTaxa
api.limitador = LimitadorDeTaxa(1000); api.limitador_upload = LimitadorDeTaxa(1000)
from fastapi.testclient import TestClient

falhas = []
def check(cond, nome):
    print(("✅ " if cond else "❌ ") + nome)
    if not cond: falhas.append(nome)

with TestClient(api.app) as c:
    r = c.get("/api/documentos/pagina", params={"arquivo": "apostila_ia_oficina.pdf", "exemplo": "true", "pagina": 5})
    d = r.json()
    check(r.status_code == 200 and d["pagina"] == 5 and d["pdf"] and d["paginas"][0] == 1, "página de um exemplo")
    check(d["texto"].count("Um embedding é uma representação") == 1, "sobreposição entre trechos não aparece duas vezes")
    check(c.get("/api/documentos/pagina", params={"arquivo": "apostila_ia_oficina.pdf", "exemplo": "true", "pagina": 999}).json()["pagina"] == 7,
          "página fora do documento vai para a mais próxima")

    frase = "O prazo de empréstimo é de quinze dias corridos, renovável uma vez. "
    texto = "".join(f"Parágrafo {i}. " + frase * 3 + "\n\n" for i in range(12))
    c.post("/api/documentos", data={"espaco_id": "espacoVIS0001", "tamanho_trecho": "400"}, files={"arquivo": ("regras.txt", texto.encode(), "text/plain")})
    d = c.get("/api/documentos/pagina", params={"arquivo": "regras.txt", "espaco_id": "espacoVIS0001", "pagina": 1}).json()
    check(d.get("texto", "").count("Parágrafo") == 12 and not d["pdf"], "texto inteiro reconstruído a partir de trechos sobrepostos")
    check(c.get("/api/documentos/pagina", params={"arquivo": "regras.txt", "espaco_id": "outroEspaco01", "pagina": 1}).status_code == 404,
          "outro navegador não lê o documento")
    check(c.get("/api/documentos/pagina", params={"arquivo": "../app/config.py", "exemplo": "true"}).status_code in (400, 404, 422), "nome de arquivo inválido é recusado")

    r = c.get("/vendor/pdfjs/pdf.min.mjs")
    check(r.status_code == 200 and "javascript" in r.headers["content-type"], "PDF.js servido pelo próprio app")
    check(c.get("/vendor/pdfjs/../../app/config.py").status_code == 404 and c.get("/vendor/%2e%2e/app/config.py").status_code == 404, "sem sair da pasta vendor")
    r = c.get("/ver")
    check(r.status_code == 200 and "faixas(" in r.text, "página do visualizador")
    check('"/ver?"' in c.get("/").text or "/ver?" in c.get("/").text, "links das fontes apontam para o visualizador")
print("\nFALHAS:", falhas or "nenhuma")
