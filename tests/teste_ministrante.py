"""Modo ministrante: senha no servidor, pausas, limite do dia, números e recursos restritos."""
import os, sys, tempfile
from pathlib import Path
REPO = Path(sys.argv[1]); tmp = Path(tempfile.mkdtemp())
os.environ.update(OPENAI_API_KEY="sk-teste", CHROMA_DIR=str(tmp/"c"), UPLOAD_DIR=str(tmp/"u"), RETRIEVER_MAX_DISTANCE="2", MINISTRANTE_SENHA="Senha-Forte-123")
os.environ.pop("PAINEL_SENHA", None)
sys.path.insert(0, str(REPO))
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
import app.ingest as ingest
ingest.obter_embeddings = lambda: DeterministicFakeEmbedding(size=64)
import app.api as api, app.estudos as estudos
from app.guardrails import LimitadorDeTaxa
from app.schemas import ComparacaoOut, Custo
api.limitador = LimitadorDeTaxa(1000); api.limitador_upload = LimitadorDeTaxa(1000)
class FakeUso(FakeListChatModel):
    def invoke(self, *a, **k):
        return AIMessage(content="ok", usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
estudos.criar_chat = lambda *a, **k: FakeUso(responses=["ok"])
custo = Custo(modelo="x", chamadas_llm=1, tokens_entrada=1, tokens_entrada_cache=0, tokens_saida=1, tokens_embedding=0, custo_usd=0.0, custo_brl=0.0, cotacao_usd_brl=5.4)
api.comparar_sem_rag = lambda pergunta: ComparacaoOut(resposta="sem rag", custo=custo)
from fastapi.testclient import TestClient
falhas = []
def check(c, n):
    print(("✅ " if c else "❌ ") + n)
    if not c: falhas.append(n)
H = {"X-Ministrante": "Senha-Forte-123"}
with TestClient(api.app) as c:
    esp = "espacoMIN12345"
    check(c.get("/api/info").json()["ministrante_disponivel"] is True, "info avisa que o modo ministrante existe")
    check(c.post("/api/ministrante/entrar", json={"senha": "errada"}).status_code == 401, "senha errada: 401")
    r = c.post("/api/ministrante/entrar", json={"senha": "Senha-Forte-123"})
    check(r.status_code == 200 and "controles" in r.json() and "numeros" in r.json(), "senha certa devolve controles e números")
    check(c.get("/api/ministrante/estado").status_code == 401, "estado sem senha: 401")
    # recurso restrito
    check(c.post("/api/comparar-modelos", json={"espaco_id": esp, "pergunta": "teste"}).status_code == 401, "comparar modelos sem senha: 401")
    check(c.post("/api/comparar-modelos", json={"espaco_id": esp, "pergunta": "teste"}, headers=H).status_code == 200, "comparar modelos com senha: 200")
    # pausar perguntas
    r = c.post("/api/ministrante/controles", json={"perguntas_pausadas": True}, headers=H)
    check(r.status_code == 200 and r.json()["controles"]["perguntas_pausadas"], "pausa as perguntas")
    check(c.post("/api/ministrante/controles", json={"perguntas_pausadas": False}).status_code == 401, "controles sem senha: 401")
    r = c.post("/api/comparar", json={"pergunta": "oi", "sessao_id": "sessaoMIN001"})
    check(r.status_code == 503 and "pausadas" in r.json()["detail"], "aluno recebe aviso de pausa: " + r.json().get("detail", "")[:50])
    r = c.post("/api/chat/stream", json={"pergunta": "oi", "sessao_id": "sessaoMIN001", "espaco_id": esp})
    check(r.status_code == 503, "stream também pausado para o aluno")
    check(c.post("/api/comparar", json={"pergunta": "oi", "sessao_id": "sessaoMIN001"}, headers=H).status_code == 200, "ministrante pergunta mesmo com a pausa")
    check(c.get("/api/info").json()["controle"]["perguntas_pausadas"] is True, "info mostra a pausa para todos")
    c.post("/api/ministrante/controles", json={"perguntas_pausadas": False}, headers=H)
    check(c.post("/api/comparar", json={"pergunta": "oi", "sessao_id": "sessaoMIN001"}).status_code == 200, "despausa: aluno volta a perguntar")
    c.post("/api/chat/stream", json={"pergunta": "oi de novo", "sessao_id": "sessaoMIN001", "espaco_id": esp})
    # pausar envios
    c.post("/api/ministrante/controles", json={"envios_pausados": True}, headers=H)
    arq = {"arquivo": ("notas.txt", "Conteúdo de teste sobre vetores e embeddings. " * 10, "text/plain")}
    r = c.post("/api/documentos", data={"espaco_id": esp}, files=arq)
    check(r.status_code == 503 and "envios" in r.json()["detail"], "upload pausado para o aluno")
    r = c.post("/api/documentos/url", json={"espaco_id": esp, "url": "https://exemplo.com"})
    check(r.status_code == 503, "link pausado para o aluno")
    check(c.post("/api/documentos", data={"espaco_id": esp}, files=arq, headers=H).status_code == 200, "ministrante envia mesmo com a pausa")
    c.post("/api/ministrante/controles", json={"envios_pausados": False}, headers=H)
    # limite do dia
    r = c.post("/api/ministrante/controles", json={"limite_diario_usd": 3.5}, headers=H)
    check(r.json()["controles"]["limite_diario_usd"] == 3.5 and c.get("/api/info").json()["orcamento"]["limite_usd"] == 3.5, "ajusta o limite do dia para US$ 3,50")
    check(c.post("/api/ministrante/controles", json={"limite_diario_usd": 999}, headers=H).status_code == 422, "limite absurdo recusado")
    # números
    n = c.get("/api/ministrante/estado", headers=H).json()["numeros"]
    check(n["total"] >= 1 and n["envios_por_tipo"].get("txt") == 1 and len(n["por_hora"]) == 6, f"números: {n['total']} perguntas, envios {n['envios_por_tipo']}, {len(n['por_hora'])} horas")
    # moderação do painel com a mesma senha
    check(c.post("/api/turma/limpar", json={"senha": "Senha-Forte-123"}).status_code == 200, "mesma senha modera o painel")
    # força bruta
    codigos = [c.post("/api/ministrante/entrar", json={"senha": f"chute{i}"}).status_code for i in range(12)]
    check(429 in codigos, f"tentativas demais viram 429: {codigos[-3:]}")
print("\nFALHAS:", falhas or "nenhuma")
