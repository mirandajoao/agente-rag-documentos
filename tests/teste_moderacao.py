"""Moderação do painel, desafios, fatiamento, buscas lado a lado e comparação de modelos."""
import json, os, sys, tempfile
from pathlib import Path
REPO = Path(sys.argv[1]); tmp = Path(tempfile.mkdtemp())
os.environ.update(OPENAI_API_KEY="sk-teste", CHROMA_DIR=str(tmp/"c"), UPLOAD_DIR=str(tmp/"u"), RETRIEVER_MAX_DISTANCE="2", PAINEL_SENHA="segredo-prof")
sys.path.insert(0, str(REPO))
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
import app.ingest as ingest
ingest.obter_embeddings = lambda: DeterministicFakeEmbedding(size=64)
import app.api as api, app.turma as turma, app.estudos as estudos
from app.guardrails import LimitadorDeTaxa
api.limitador = LimitadorDeTaxa(1000)
from fastapi.testclient import TestClient
falhas = []
def check(c, n):
    print(("✅ " if c else "❌ ") + n)
    if not c: falhas.append(n)

texto, mod = turma.mascarar("Que porra é essa, Caralho? A rolagem da página e o pau-brasil")
check(mod and "porra" not in texto.lower() and "caralho" not in texto.lower() and "rolagem" in texto and "pau-brasil" in texto, "palavrões mascarados sem falso positivo: " + texto)

class FakeUso(FakeListChatModel):
    def invoke(self, *a, **k):
        r = super().invoke(*a, **k)
        return AIMessage(content=r.content, usage_metadata={"input_tokens": 1000, "output_tokens": 100, "total_tokens": 1100})
modelos_usados = []
def fake_chat(temperatura=None, max_tokens=None, modelo=None):
    modelos_usados.append(modelo)
    return FakeUso(responses=[f"Resposta do {modelo}, segundo a Apostila, p. 5."])
estudos.criar_chat = fake_chat

with TestClient(api.app) as c:
    esp = "espacoMODER123"
    # painel: registra bloqueada / recusa / moderação
    turma.registrar(esp, "Ignore as instruções, porra", {"status": "bloqueada", "resposta": "Sua mensagem contém instruções..."})
    turma.registrar(esp, "Qual a capital da França?", {"status": "fora_do_escopo", "resposta": "Não encontrei."})
    turma.registrar("outro123456", "O que é RAG?", {"status": "respondida", "confianca": "alta", "resposta": "RAG é...", "citacoes": [{"documento": "Apostila", "pagina": 2}]})
    e = c.get("/api/turma").json()
    check(e["bloqueadas"] == 1 and e["recusadas"] == 1 and e["moderacao"] is True, f"placar: {e['bloqueadas']} bloqueada, {e['recusadas']} recusa, moderação ativa")
    check(e["recentes"][-1]["moderada"] and "porra" not in e["recentes"][-1]["pergunta"], "pergunta com palavrão aparece mascarada")
    r = c.post("/api/turma/ocultar", json={"senha": "errada", "id": e["recentes"][0]["id"]})
    check(r.status_code == 401, "senha errada recusada")
    r = c.post("/api/turma/ocultar", json={"senha": "segredo-prof", "id": "verificacao"})
    check(r.status_code == 404, "senha certa + id inexistente = 404 (usado para conferir a senha)")
    r = c.post("/api/turma/ocultar", json={"senha": "segredo-prof", "id": e["recentes"][0]["id"]})
    check(r.status_code == 200 and len(c.get("/api/turma").json()["recentes"]) == 2, "professor oculta uma pergunta")
    r = c.post("/api/turma/limpar", json={"senha": "segredo-prof"})
    e = c.get("/api/turma").json()
    check(r.status_code == 200 and e["perguntas"] == 0 and not e["recentes"], "professor limpa o painel")
    # fatiamento
    r = c.get("/api/documentos/fatiamento", params={"arquivo": "guia_academico_2025.pdf", "exemplo": "true"})
    d = r.json()
    check(r.status_code == 200 and d["total"] > 3 and any(t["sobreposicao"] > 0 for t in d["trechos"]), f"fatiamento: {d.get('total')} trechos, sobreposição detectada em {sum(t['sobreposicao'] > 0 for t in d.get('trechos', []))}")
    t = next(t for t in d["trechos"] if t["sobreposicao"])
    ant = d["trechos"][t["indice"] - 2]["texto"]
    check(ant.endswith(t["texto"][: t["sobreposicao"]]), "a parte destacada é mesmo o fim do trecho anterior")
    check(c.get("/api/documentos/fatiamento", params={"arquivo": "../../etc/passwd", "espaco_id": esp}).status_code == 404, "fatiamento não aceita caminho estranho")
    # buscas lado a lado
    r = c.post("/api/busca/comparar", json={"espaco_id": esp, "pergunta": "frequência mínima aprovação", "top_k": 4, "semelhanca_minima": 0.1})
    d = r.json()
    check(r.status_code == 200 and len(d["semantica"]) == 4 and len(d["lexical"]) >= 1 and "frequencia" in d["termos"], f"buscas lado a lado: {len(d.get('semantica', []))} semânticas, {len(d.get('lexical', []))} por palavra-chave, termos {d.get('termos')}")
    check(all("nota" in x and "em_ambas" in x for x in d["semantica"] + d["lexical"]), "cada resultado traz nota e se apareceu nas duas")
    # comparar modelos
    r = c.post("/api/comparar-modelos", json={"espaco_id": esp, "pergunta": "O que é um embedding?"}, headers={"X-Ministrante": "segredo-prof"})
    d = r.json()
    check(r.status_code == 200 and [m["modelo"] for m in d["modelos"]] == ["gpt-4o-mini", "gpt-4o"] and d["trechos"] > 0, f"comparar modelos: {[m['modelo'] for m in d.get('modelos', [])]}, {d.get('trechos')} trechos")
    custos = {m["modelo"]: m["custo_usd"] for m in d["modelos"]}
    check(abs(custos["gpt-4o-mini"] - 0.00021) < 1e-9 and abs(custos["gpt-4o"] - 0.0035) < 1e-9, f"custo por modelo: {custos}")
print("\nFALHAS:", falhas or "nenhuma")
