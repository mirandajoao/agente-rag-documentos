import os, sys, tempfile
from pathlib import Path
REPO = Path(sys.argv[1]); tmp = Path(tempfile.mkdtemp())
os.environ.update(OPENAI_API_KEY="sk-teste", CHROMA_DIR=str(tmp/"c"), UPLOAD_DIR=str(tmp/"u"), RETRIEVER_MAX_DISTANCE="2")
sys.path.insert(0, str(REPO))
from langchain_core.embeddings import DeterministicFakeEmbedding
import app.ingest as ingest
ingest.obter_embeddings = lambda: DeterministicFakeEmbedding(size=64)
ingest.construir_indice()
from fastapi.testclient import TestClient
import app.api as api
from app.tools import busca_hibrida
with TestClient(api.app) as c:
    esp = "espacoTESTE123"
    r = c.post("/api/documentos", data={"espaco_id": esp, "exemplos": "false"}, files={"arquivo": ("notas.txt", "Frequência mínima de 75 por cento nas aulas de cálculo.".encode(), "text/plain")})
    print("upload sem exemplos:", r.status_code, [d["documento"] for d in r.json()["documentos"]])
    com = c.get("/api/documentos", params={"espaco_id": esp}).json()["documentos"]
    sem = c.get("/api/documentos", params={"espaco_id": esp, "exemplos": "false"}).json()["documentos"]
    print("lista com/sem exemplos:", len(com), len(sem))
    origens = lambda res: {d.metadata.get("espaco") for d, _ in res}
    print("busca com exemplos:", origens(busca_hibrida("frequência mínima", esp)))
    print("busca sem exemplos:", origens(busca_hibrida("frequência mínima", esp, exemplos=False)))
    assert len(sem) == 1 and len(com) > 1
    assert origens(busca_hibrida("frequência mínima", esp, exemplos=False)) == {esp}
    assert c.post("/api/feedback", json={}).status_code in (404, 405)
    print("OK")
