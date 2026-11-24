"""Notebook do Colab: roda todas as células (menos instalação, chave e upload) com modelos falsos."""
import io, json, sys
from contextlib import redirect_stdout
from pathlib import Path
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

REPO = Path(sys.argv[1])
nb = json.load(open(REPO / "notebooks" / "rag_passo_a_passo.ipynb", encoding="utf-8"))
codigos = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
falhas = []
def check(cond, nome):
    print(("✅ " if cond else "❌ ") + nome)
    if not cond: falhas.append(nome)

respostas = ['{"notas": [9, 2, 7, 0]}', '{"notas": [9, 0, 6, 1]}', "Um embedding é um vetor de números (p. 5).",
             "Resposta sem fonte.", '{"notas": [0, 0, 0, 0]}', "Não encontrei essa informação no documento."]
ns = {"llm": FakeListChatModel(responses=respostas), "embeddings": DeterministicFakeEmbedding(size=64),
      "CAMINHO": str(REPO / "data" / "pdfs" / "apostila_ia_oficina.pdf"), "PROVEDOR": "gemini"}
saida = io.StringIO()
try:
    with redirect_stdout(saida):
        for i, c in enumerate(codigos[4:], start=5):  # pula: pip, chave, modelos, upload
            exec(compile(c, f"celula{i}", "exec"), ns)
    ok = True
except Exception as exc:
    ok = False
    print("erro:", type(exc).__name__, exc)
texto = saida.getvalue()
check(ok, f"todas as {len(codigos) - 4} células rodam")
check("nota 9/10 → fica" in texto and "nota 0/10 → SAI" in texto, "etapa do reranking mostra quem fica e quem sai")
check("Páginas usadas como fonte" in texto, "resposta com RAG mostra as páginas")
check("USAR_RERANKING" in "".join(codigos) and any("## 8. Reranking" in "".join(c["source"]) for c in nb["cells"]), "notebook tem a etapa de reranking")
print("\nFALHAS:", falhas or "nenhuma")
