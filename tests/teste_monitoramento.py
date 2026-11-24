"""Sentry: liga só com SENTRY_DSN e não manda nada que o usuário escreveu."""
import os, sys, tempfile
from pathlib import Path
REPO = Path(sys.argv[1]); tmp = Path(tempfile.mkdtemp())
os.environ.update(OPENAI_API_KEY="sk-teste", CHROMA_DIR=str(tmp / "c"), UPLOAD_DIR=str(tmp / "u"),
                  SENTRY_DSN="https://chave@o0.ingest.sentry.io/0")
sys.path.insert(0, str(REPO))
from langchain_core.embeddings import DeterministicFakeEmbedding
import app.ingest as ingest
ingest.obter_embeddings = lambda: DeterministicFakeEmbedding(size=64)
falhas = []
def check(cond, nome):
    print(("✅ " if cond else "❌ ") + nome)
    if not cond: falhas.append(nome)

from app import monitoramento
import sentry_sdk
from sentry_sdk.transport import Transport
capturados = []
class Captura(Transport):
    def capture_envelope(self, envelope):
        capturados.append(envelope)
check(monitoramento.iniciar(transport=Captura), "liga com SENTRY_DSN")

evento = {"request": {"method": "POST", "url": "https://app/api/chat?espaco_id=segredo", "data": {"pergunta": "meu CPF é 123"},
                      "headers": {"X-Ministrante": "senha"}, "cookies": {"a": "b"}, "query_string": "espaco_id=segredo"},
          "user": {"ip_address": "1.2.3.4"},
          "exception": {"values": [{"type": "ValueError", "stacktrace": {"frames": [{"function": "f", "vars": {"pergunta": "meu CPF é 123"}}]}}]}}
limpo = monitoramento._sem_dados_pessoais(evento, {})
texto = str(limpo)
check("CPF" not in texto and "senha" not in texto and "segredo" not in texto and "1.2.3.4" not in texto, "remove corpo, cabeçalhos, query, usuário e variáveis")
check(limpo["request"] == {"method": "POST", "url": "https://app/api/chat"}, "mantém só método e caminho")
check(limpo["exception"]["values"][0]["type"] == "ValueError", "mantém o tipo do erro")

from fastapi.testclient import TestClient
from app.api import app
SEGREDOS = ["pergunta-" + str(7 * 1111), "espaco-" + str(6 * 1111), "senha-" + str(5 * 1111)]
def _erro():
    pergunta_secreta = SEGREDOS[0]  # noqa: F841
    raise RuntimeError("falha de teste")
app.add_api_route("/__erro_teste", _erro)
app.router.routes.insert(0, app.router.routes.pop())
with TestClient(app, raise_server_exceptions=False) as c:
    r = c.get("/__erro_teste?espaco_id=" + SEGREDOS[1], headers={"X-Ministrante": SEGREDOS[2]})
    check(r.status_code == 500, "erro vira 500")
sentry_sdk.flush()
enviado = " ".join(str(e) for e in capturados)
for env in capturados:
    for item in env.items:
        enviado += item.payload.get_bytes().decode("utf-8", "ignore")
check("falha de teste" in enviado, "o erro chega ao Sentry")
check(not any(t in enviado for t in SEGREDOS), "sem dados do usuário no que foi enviado")
print("\nFALHAS:", falhas or "nenhuma")
