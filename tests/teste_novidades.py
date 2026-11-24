"""OCR, planilhas, link (SSRF), mapa, painel da turma, parâmetros, prompt e streaming de tokens."""
import json, os, sys, tempfile
from pathlib import Path
REPO = Path(sys.argv[1]); tmp = Path(tempfile.mkdtemp())
os.environ.update(OPENAI_API_KEY="sk-teste", CHROMA_DIR=str(tmp/"c"), UPLOAD_DIR=str(tmp/"u"), RETRIEVER_MAX_DISTANCE="2")
sys.path.insert(0, str(REPO))
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
import app.ingest as ingest
ingest.obter_embeddings = lambda: DeterministicFakeEmbedding(size=64)
import app.tools as tools, app.agent as agent, app.api as api
from fastapi.testclient import TestClient

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

class FakeComTools(BaseChatModel):
    """Modelo falso que transmite como o da OpenAI: tool calls em pedaços e texto aos poucos."""
    @property
    def _llm_type(self): return "fake-stream"
    def bind_tools(self, tools, **kw): return self
    def _generate(self, messages, stop=None, run_manager=None, **kw):
        return ChatResult(generations=[ChatGeneration(message=next(fila))])
    def _stream(self, messages, stop=None, run_manager=None, **kw):
        msg = next(fila)
        if msg.tool_calls:
            tc = msg.tool_calls[0]
            yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=[{"name": tc["name"], "args": json.dumps(tc["args"]), "id": tc["id"], "index": 0}]))
            return
        for i in range(0, len(msg.content), 9):
            pedaco = ChatGenerationChunk(message=AIMessageChunk(content=msg.content[i:i + 9]))
            if run_manager: run_manager.on_llm_new_token(pedaco.text, chunk=pedaco)
            yield pedaco
fila = iter([])
def fake_construir():
    from langchain.agents import create_agent
    from langgraph.checkpoint.memory import InMemorySaver
    return create_agent(model=FakeComTools(messages=fila), tools=tools.FERRAMENTAS, system_prompt="x", checkpointer=InMemorySaver())
api.construir_agente = lambda: None
from app.guardrails import LimitadorDeTaxa
api.limitador_upload = LimitadorDeTaxa(1000); api.limitador = LimitadorDeTaxa(1000)
agent.construir_agente = fake_construir
falhas = []
def check(cond, nome):
    print(("✅ " if cond else "❌ ") + nome)
    if not cond: falhas.append(nome)

# --- arquivos de teste ---
from PIL import Image, ImageDraw, ImageFont
fonte = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 42) if Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf").exists() else ImageFont.load_default()
img = Image.new("RGB", (1700, 2200), "white"); dr = ImageDraw.Draw(img)
for i, linha in enumerate(["Relatorio de Estagio Supervisionado", "A carga horaria minima do estagio", "e de 300 horas, cumpridas no semestre.", "Orientador: professor da disciplina."]):
    dr.text((120, 200 + i * 90), linha, fill="black", font=fonte)
img.save(tmp/"escaneado.pdf", "PDF", resolution=200)
from openpyxl import Workbook
wb = Workbook(); ws = wb.active; ws.title = "Notas"
ws.append(["Aluno", "Disciplina", "Nota"]); ws.append(["Ana", "Cálculo", 9.5]); ws.append(["Bruno", "Física", 7.0])
wb.save(tmp/"notas.xlsx")
(tmp/"vendas.csv").write_text("produto;quantidade;preco\nCaneta azul;120;2,50\nCaderno;40;18,90\n", encoding="utf-8")

with TestClient(api.app) as c:
    esp = "espacoNOVO1234"
    envia = lambda nome, mime, extra=None: c.post("/api/documentos", data={"espaco_id": esp, **(extra or {})}, files={"arquivo": (nome, (tmp/nome).read_bytes(), mime)})
    r = envia("escaneado.pdf", "application/pdf")
    check(r.status_code == 200 and "OCR" in r.json()["mensagem"], "OCR em PDF escaneado: " + r.json().get("mensagem", r.text)[:110])
    trechos = ingest.trechos_do_documento(esp, "escaneado.pdf")
    check(any("300 horas" in t.page_content for t in trechos), "texto reconhecido contém '300 horas'")
    r = envia("notas.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    check(r.status_code == 200, "upload XLSX: " + r.json().get("mensagem", r.text)[:80])
    t = " ".join(d.page_content for d in ingest.trechos_do_documento(esp, "notas.xlsx"))
    check("Aluno: Ana | Disciplina: Cálculo | Nota: 9.5" in t, "linhas da planilha com o nome das colunas")
    r = envia("vendas.csv", "text/csv")
    t = " ".join(d.page_content for d in ingest.trechos_do_documento(esp, "vendas.csv"))
    check(r.status_code == 200 and "produto: Caderno | quantidade: 40 | preco: 18,90" in t, "upload CSV com ';'")
    # tamanho do trecho
    (tmp/"longo.txt").write_text(("Frase de teste sobre vetores e busca. " * 20 + "\n\n") * 20, encoding="utf-8")
    n_padrao = int(envia("longo.txt", "text/plain").json()["mensagem"].split("dividido em ")[1].split()[0])
    n_menor = int(envia("longo.txt", "text/plain", {"tamanho_trecho": "400"}).json()["mensagem"].split("dividido em ")[1].split()[0])
    check(n_menor > n_padrao, f"tamanho do trecho menor => mais trechos ({n_padrao} -> {n_menor})")
    c.delete("/api/documentos", params={"espaco_id": esp, "arquivo": "longo.txt"})

    # link: proteção SSRF
    for url in ["http://127.0.0.1:8000/", "http://localhost/", "http://169.254.169.254/latest/meta-data/", "http://10.0.0.5/", "file:///etc/passwd", "http://[::1]/"]:
        r = c.post("/api/documentos/url", json={"espaco_id": esp, "url": url})
        check(r.status_code == 422, f"bloqueia {url}: {r.json().get('detail', '')[:60]}")
    # link: fluxo completo com download simulado
    html = b"<html><head><title>Guia de RAG</title><script>x()</script></head><body><nav>menu</nav><article><h1>O que e RAG</h1><p>RAG combina busca com geracao de texto e cita a fonte de cada resposta.</p><p>Os trechos mais parecidos entram no prompt do modelo.</p></article></body></html>"
    api.baixar = lambda url: (html, "text/html", url)
    r = c.post("/api/documentos/url", json={"espaco_id": esp, "url": "https://exemplo.com/rag"})
    check(r.status_code == 200 and "página da web" in r.json()["mensagem"], "importa página da web: " + r.json().get("mensagem", r.text)[:80])
    nome = next(d["arquivo"] for d in r.json()["documentos"] if d["documento"] == "Guia de RAG")
    t = " ".join(d.page_content for d in ingest.trechos_do_documento(esp, nome))
    check("RAG combina busca" in t and "menu" not in t and "x()" not in t, "texto da página sem menu nem script")

    # mapa
    r = c.post("/api/mapa", json={"espaco_id": esp, "pergunta": "carga horária do estágio", "exemplos": False, "top_k": 3})
    d = r.json()
    check(r.status_code == 200 and len(d["pontos"]) > 3 and d["pergunta"] and sum(p["semelhanca"] is not None for p in d["pontos"]) == 3, f"mapa: {len(d.get('pontos', []))} pontos, pergunta e 3 destaques")
    check(all(-1.01 <= p["x"] <= 1.01 and -1.01 <= p["y"] <= 1.01 for p in d["pontos"]), "mapa normalizado em [-1, 1]")

    # agente: streaming de tokens, prompt completo, parâmetros e painel da turma
    final = json.dumps({"resposta": "A carga horária mínima é de 300 horas, segundo o relatório.", "citacoes": [], "status": "respondida", "confianca": "media"}, ensure_ascii=False)
    fila = iter([AIMessage(content="Vou buscar.", tool_calls=[{"name": "buscar_documentos", "args": {"pensamento": "Buscar a carga horária.", "consulta": "carga horária estágio"}, "id": "t1"}]), AIMessage(content=final)])
    capturados = {}
    original = tools.busca_hibrida
    def espia(*a, **k):
        capturados["parametros"] = k.get("parametros") if "parametros" in k else (a[4] if len(a) > 4 else None)
        return original(*a, **k)
    tools.busca_hibrida = espia
    r = c.post("/api/chat/stream", json={"pergunta": "Qual a carga horária do estágio?", "sessao_id": "sessaoTESTE1", "espaco_id": esp, "exemplos": False, "parametros": {"top_k": 2, "semelhanca_minima": 0.1}})
    eventos = [json.loads(l[6:]) for l in r.text.split("\n") if l.startswith("data: ")]
    tokens = [e["texto"] for e in eventos if e["tipo"] == "token"]
    check(len(tokens) >= 2 and tokens[-1].startswith("A carga horária mínima é de 300") and "{" not in tokens[0], f"streaming: {len(tokens)} eventos de token, último = {tokens[-1][:40] if tokens else '-'}")
    resp = eventos[-1]["resposta"]
    papeis = [p["papel"] for p in resp["prompt"]]
    check(papeis[0] == "sistema" and "usuario" in papeis and "ferramenta" in papeis, "prompt completo: " + " > ".join(papeis))
    check(capturados.get("parametros") == {"top_k": 2, "semelhanca_minima": 0.1}, f"parâmetros do laboratório chegam à busca: {capturados.get('parametros')}")
    t = c.get("/api/turma").json()
    check(t["perguntas"] >= 1 and t["recentes"][0]["pergunta"].startswith("Qual a carga") and esp not in json.dumps(t), "painel da turma registra sem identificar o espaço")
    check(c.get("/turma").status_code == 200, "página /turma")
    info = c.get("/api/info").json()
    check(info["busca"]["top_k"] >= 1 and ".xlsx" in info["upload"]["extensoes"], "info com padrões da busca e novos formatos")
print("\nFALHAS:", falhas or "nenhuma")
