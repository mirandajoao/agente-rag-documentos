"""Memória da conversa no Postgres: a conversa continua depois de um reinício."""
import json, os, subprocess, sys, textwrap
REPO = sys.argv[1]
URL = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("TEST_DATABASE_URL", "postgresql://postgres@127.0.0.1:5544/postgres?sslmode=disable")
PY = sys.executable
BASE = f"""
import os, sys, tempfile, json, time
from pathlib import Path
tmp = Path(tempfile.mkdtemp())
os.environ.update(OPENAI_API_KEY="sk-teste", CHROMA_DIR=str(tmp/"c"), UPLOAD_DIR=str(tmp/"u"), DATABASE_URL={URL!r}, RERANK="off")
sys.path.insert(0, {REPO!r})
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from app import persistencia, memoria, tools
import app.agent as agent
vistos = []
class Falso(BaseChatModel):
    @property
    def _llm_type(self): return "falso"
    def bind_tools(self, tools, **kw): return self
    def _generate(self, messages, stop=None, run_manager=None, **kw):
        vistos.append([m.content for m in messages])
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=json.dumps({{"resposta": "ok", "citacoes": [], "status": "respondida", "confianca": "media"}})))])
def construir():
    from langchain.agents import create_agent
    return create_agent(model=Falso(), tools=tools.FERRAMENTAS, system_prompt="x", checkpointer=memoria.criar_checkpointer())
from functools import lru_cache
agent.construir_agente = lru_cache(maxsize=1)(construir)
persistencia.iniciar()
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
    for t in ("conversas", "checkpoint_writes", "checkpoint_blobs", "checkpoints", "checkpoint_migrations"):
        c.execute(f"DROP TABLE IF EXISTS {t}")

a = rodar("""
r = agent.perguntar("Qual a frequência mínima do IFPA?", "sessaoMEM001")
persistencia.esperar_gravacoes(5)
print(json.dumps({"resposta": r.resposta, "tipo": memoria.tipo()}))
""")
check(a.get("resposta") == "ok" and a.get("tipo") == "banco", "primeira pergunta com a memória no banco")

b = rodar("""
r = agent.perguntar("E se eu não alcançar?", "sessaoMEM001")
outra = agent.perguntar("Pergunta de outra pessoa", "sessaoMEM002")
print(json.dumps({"resposta": r.resposta, "viu": vistos[0], "outra": vistos[1]}))
""")
check(any("frequência mínima" in str(m) for m in b.get("viu", [])), "depois do reinício o modelo recebe a pergunta anterior")
check(not any("frequência mínima" in str(m) for m in b.get("outra", [])), "outra sessão não vê a conversa dos outros")

with psycopg.connect(URL, autocommit=True) as c:
    c.execute("UPDATE conversas SET atualizada_em = atualizada_em - 8 * 86400 WHERE thread_id = 'sessaoMEM001'")
d = rodar("""
agent.construir_agente()
estado = agent.construir_agente().get_state({"configurable": {"thread_id": "sessaoMEM001"}}).values
outra = agent.construir_agente().get_state({"configurable": {"thread_id": "sessaoMEM002"}}).values
print(json.dumps({"msgs": len(estado.get("messages", [])), "outra": len(outra.get("messages", []))}))
""")
check(d.get("msgs") == 0 and d.get("outra", 0) > 0, "conversas paradas há mais de 7 dias são apagadas; as recentes ficam")

e = rodar("""
import psycopg
agent.construir_agente()
original = agent.construir_agente
chamadas = {"n": 0}
def quebra():
    chamadas["n"] += 1
    if chamadas["n"] == 1:
        class Quebrado:
            def invoke(self, *a, **k): raise psycopg.OperationalError("banco caiu")
        return Quebrado()
    return original()
quebra.cache_clear = original.cache_clear
agent.construir_agente = quebra
r = agent.perguntar("Pergunta com o banco fora", "sessaoMEM003")
print(json.dumps({"resposta": r.resposta, "tipo": memoria.tipo()}))
""")
check(e.get("resposta") == "ok" and e.get("tipo") == "memória do servidor", "banco caiu no meio: responde mesmo assim, com a memória do servidor")
print("\nFALHAS:", falhas or "nenhuma")
