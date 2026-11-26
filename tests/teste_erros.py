"""Erros: só falta de chave vira "chave ausente"; falha passageira do índice é repetida."""
import os, sys, tempfile
from pathlib import Path
REPO = Path(sys.argv[1]); tmp = Path(tempfile.mkdtemp())
os.environ.update(OPENAI_API_KEY="sk-teste", CHROMA_DIR=str(tmp / "c"), UPLOAD_DIR=str(tmp / "u"), RETRIEVER_MAX_DISTANCE="2")
sys.path.insert(0, str(REPO))
from langchain_core.embeddings import DeterministicFakeEmbedding
import app.ingest as ingest
ingest.obter_embeddings = lambda: DeterministicFakeEmbedding(size=64)
from app import tools
from app.agent import _resposta_de_erro
from app.config import ConfiguracaoInvalida

falhas = []
def check(cond, nome):
    print(("✅ " if cond else "❌ ") + nome)
    if not cond: falhas.append(nome)

r = _resposta_de_erro(RuntimeError("Cannot return the results in a contiguous 2D array"), "s1")
check("chave" not in r.resposta.lower() and r.status == "erro", "falha do índice não aparece como 'chave ausente'")
r = _resposta_de_erro(ConfiguracaoInvalida("OPENAI_API_KEY não configurada"), "s1")
check("chave de API ausente" in r.resposta, "falta de chave continua com a mensagem de configuração")

ingest.construir_indice()
vs = tools._vectorstore()
original = vs.similarity_search_with_score
chamadas = {"n": 0}
def falha_uma_vez(*a, **k):
    chamadas["n"] += 1
    if chamadas["n"] == 1:
        raise RuntimeError("Cannot return the results in a contiguous 2D array. Probably ef or M is too small")
    return original(*a, **k)
vs.similarity_search_with_score = falha_uma_vez
achados = tools.busca_hibrida("frequência mínima", "", exemplos=True)
check(chamadas["n"] >= 2 and len(achados) > 0, "busca repete a consulta depois de uma falha passageira")

def sempre_falha(*a, **k):
    raise RuntimeError("índice quebrado")
vs.similarity_search_with_score = sempre_falha
try:
    tools.busca_hibrida("frequência mínima", "", exemplos=True); check(False, "falha persistente vira erro")
except RuntimeError:
    check(True, "falha persistente vira erro (depois de 3 tentativas)")
def sem_chave(*a, **k):
    chamadas["chave"] = chamadas.get("chave", 0) + 1
    raise ConfiguracaoInvalida("sem chave")
vs.similarity_search_with_score = sem_chave
try:
    tools.busca_hibrida("x", "", exemplos=True)
except ConfiguracaoInvalida:
    pass
check(chamadas.get("chave") == 1, "falta de chave não é repetida")
print("\nFALHAS:", falhas or "nenhuma")
