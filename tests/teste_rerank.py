"""Reranking: reordena pela nota, descarta o que não ajuda, soma o custo e nunca derruba a busca."""
import os, sys, tempfile
from pathlib import Path
from types import SimpleNamespace
REPO = Path(sys.argv[1]); tmp = Path(tempfile.mkdtemp())
os.environ.update(OPENAI_API_KEY="sk-teste", CHROMA_DIR=str(tmp / "c"), UPLOAD_DIR=str(tmp / "u"), RETRIEVER_MAX_DISTANCE="2", RERANK="llm")
sys.path.insert(0, str(REPO))
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage, ToolMessage
import app.ingest as ingest
ingest.obter_embeddings = lambda: DeterministicFakeEmbedding(size=64)
from app import rerank, tools
from app.agent import extrair_trechos
from app.schemas import RespostaAgente

falhas = []
def check(cond, nome):
    print(("✅ " if cond else "❌ ") + nome)
    if not cond: falhas.append(nome)

import re
class Falso:
    def __init__(self, texto, erro=False):
        self.texto, self.erro, self.recebido = texto, erro, None
    def invoke(self, mensagens):
        if self.erro:
            raise RuntimeError("fora do ar")
        self.recebido = mensagens
        texto = self.texto(len(re.findall(r"^\[\d+\] ", mensagens[1].content, re.M))) if callable(self.texto) else self.texto
        return AIMessage(content=texto, usage_metadata={"input_tokens": 900, "output_tokens": 12, "total_tokens": 912})

docs = [(Document(page_content=f"trecho {i} " + "x" * 2000, metadata={"fonte": "Doc", "pagina": i, "espaco": "e1"}), f"semântica 0.{5 - i}") for i in range(4)]

modelo = Falso('{"notas": [2, 9, 0, 6]}')
rerank._modelo = lambda: modelo
r = rerank.reordenar("frequência mínima", docs)
check([d.metadata["pagina"] for d, _, _ in r.resultados] == [1, 3], "ordena pela nota e tira as notas abaixo de 3")
check([n for _, _, n in r.resultados] == [9, 6] and r.descartados == 2, "guarda a nota e conta os descartados")
check(r.uso and r.uso[0]["input_tokens"] == 900, "devolve os tokens gastos")
corpo = modelo.recebido[1].content
check(len(corpo) < 4 * 800 and "Busca: frequência mínima" in corpo, "corta cada trecho antes de mandar ao modelo")

rerank._modelo = lambda: Falso("", erro=True)
r = rerank.reordenar("x", docs)
check(not r.usado and len(r.resultados) == 4, "modelo fora do ar: segue com a ordem da busca")
rerank._modelo = lambda: Falso('{"notas": [1, 2]}')
check(not rerank.reordenar("x", docs).usado, "quantidade errada de notas: ignora o reranking")
rerank._modelo = lambda: Falso('```json\n{"notas": [7, 7, 7, 1]}\n```')
check([d.metadata["pagina"] for d, _, _ in rerank.reordenar("x", docs).resultados] == [0, 1, 2], "aceita JSON em bloco de código e mantém empates na ordem")
check(rerank.ativo() and not rerank.ativo({"rerank": False}), "laboratório consegue desligar só numa pergunta")

# Pela ferramenta do agente
ingest.construir_indice()
runtime = SimpleNamespace(config={"configurable": {"thread_id": "sessaoRR1", "espaco_id": "", "exemplos": True, "parametros": {}}})
rerank._modelo = lambda: Falso(lambda n: '{"notas": [' + ", ".join(["8"] + ["0"] * (n - 1)) + "]}")
saida = tools.buscar_documentos.func("frequência mínima para aprovação", "teste", runtime)
check(saida.count("\n\n---\n\n") == 0 and "· nota 8" in saida, "ferramenta entrega só o trecho com nota e mostra a nota")
check(tools.consumir_uso_extra("sessaoRR1"), "custo do reranking entra na conta da pergunta")
rerank._modelo = lambda: Falso(lambda n: '{"notas": [' + ", ".join(["0"] * n) + "]}")
saida = tools.buscar_documentos.func("frequência mínima para aprovação", "teste", runtime)
check(saida.startswith("SEM_EVIDENCIA") and "reranking" in saida, "nenhum trecho útil: SEM_EVIDENCIA")
runtime.config["configurable"]["parametros"] = {"rerank": False}
saida = tools.buscar_documentos.func("frequência mínima para aprovação", "teste", runtime)
check("· nota" not in saida and not saida.startswith("SEM_EVIDENCIA"), "desligado pelo laboratório: busca sem notas")

# A interface recebe a nota
msg = ToolMessage(content="[1] Doc | página 3 | relevância semântica 0.62 · nota 9 | origem: exemplo\ntexto\n\n---\n\n[2] Doc | página 4 | relevância lexical · nota 5 | origem: exemplo\noutro", tool_call_id="1", name="buscar_documentos")
t = extrair_trechos([msg], RespostaAgente(resposta="ok"))
check(t[0].relevancia == 0.62 and t[0].nota == 9 and t[1].nota == 5 and t[1].tipo_busca == "lexical", "painel mostra semelhança e nota")
msg = ToolMessage(content="[1] Doc | página 3 | relevância semântica 0.62 | origem: exemplo\ntexto", tool_call_id="1", name="buscar_documentos")
t = extrair_trechos([msg], RespostaAgente(resposta="ok"))
check(t[0].relevancia == 0.62 and t[0].nota is None, "sem reranking: igual a antes")
print("\nFALHAS:", falhas or "nenhuma")
