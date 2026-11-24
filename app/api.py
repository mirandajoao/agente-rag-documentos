"""API HTTP (FastAPI) + interface web do assistente.

Local:    uvicorn app.api:app --reload
Railway:  uvicorn app.api:app --host 0.0.0.0 --port $PORT  (ver Dockerfile)
"""

from __future__ import annotations

import io
import json
import logging
import re
import shutil
import unicodedata
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

import segno
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

from app.agent import construir_agente, perguntar, perguntar_stream
from app.config import BASE_DIR, settings
from app.guardrails import LimitadorDeTaxa, limitador
from app import controle, persistencia, turma
from app.ingest import (
    ESPACO_EXEMPLOS,
    EXTENSOES_ACEITAS,
    FORMATOS_ACEITOS,
    DocumentoInvalido,
    construir_indice,
    documentos_do_espaco,
    indexar_arquivo_detalhado,
    indice_existe,
    remover_documentos,
    titulo_amigavel,
)
from app.extras import comparar_sem_rag, sugerir_perguntas
from app.estudos import comparar_buscas, comparar_modelos, fatiamento
from app.mapa import montar_mapa
from app.web import LinkInvalido, baixar, html_para_texto
from app.orcamento import MENSAGEM_LIMITE, orcamento
from app.schemas import (
    ComparacaoIn,
    ComparacaoOut,
    DocumentoOut,
    DocumentosOut,
    ControlesIn,
    EstudoIn,
    MinistranteEntrarIn,
    LinkIn,
    MapaIn,
    ModeracaoIn,
    PerguntaIn,
    RespostaOut,
    SugestoesIn,
    SugestoesOut,
)
from app.tools import marcar_base_alterada

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
log = logging.getLogger("api")

STATIC = BASE_DIR / "static"
ESTADO = {"pronto": False, "erro": None}


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Banco (opcional): devolve o uso do dia, os controles e o painel salvos antes do reinício.
    try:
        await run_in_threadpool(persistencia.iniciar)
    except Exception:
        log.exception("Falha ao iniciar a persistência; seguindo só em memória")
    # Indexa os PDFs na primeira subida (ex.: novo deploy no Railway, cujo disco é efêmero).
    try:
        if not indice_existe():
            log.info("Índice vetorial não encontrado — construindo a partir de %s", settings.pdf_dir)
            await run_in_threadpool(construir_indice)
        construir_agente()
        ESTADO["pronto"] = True
        log.info("Agente pronto (modelo=%s, embeddings=%s)", settings.chat_model, settings.embedding_model)
    except Exception as exc:  # a API sobe mesmo assim, para o /health mostrar o problema
        ESTADO["erro"] = str(exc)
        log.exception("Falha ao inicializar o agente")
    yield


app = FastAPI(title="Converse com os seus documentos", version="2.0.0", lifespan=lifespan)
limitador_upload = LimitadorDeTaxa(settings.rate_limit_upload_por_minuto)
limitador_ip = LimitadorDeTaxa(settings.rate_limit_ip_por_minuto)
limitador_upload_ip = LimitadorDeTaxa(settings.rate_limit_upload_ip_por_minuto)



@app.get("/health")
def health():
    corpo = {"status": "ok" if ESTADO["pronto"] else "degradado", "modelo": settings.chat_model}
    if ESTADO["erro"]:
        corpo["erro"] = ESTADO["erro"]
    return JSONResponse(corpo, status_code=200 if ESTADO["pronto"] else 503)


def _cliente(request: Request) -> str:
    return request.headers.get("x-forwarded-for", request.client.host if request.client else "?").split(",")[0].strip()


@app.post("/api/chat", response_model=RespostaOut)
async def chat(dados: PerguntaIn, request: Request) -> RespostaOut:
    _limitar(request, dados.espaco_id or dados.sessao_id)
    _checar_pausa(request)
    controle.registrar_pergunta()
    _exigir_pronto_e_orcamento()
    # O agente é síncrono; rodar em threadpool evita travar o event loop.
    documento = dados.documento.model_dump() if dados.documento else None
    parametros = dados.parametros.model_dump(exclude_none=True) if dados.parametros else None
    resp = await run_in_threadpool(perguntar, dados.pergunta, dados.sessao_id, dados.espaco_id, documento, dados.exemplos, parametros)
    orcamento.registrar(resp.custo.custo_usd if resp.custo else 0)
    turma.registrar(dados.espaco_id or dados.sessao_id, dados.pergunta, resp.model_dump())
    return resp


def _limitar(request: Request, chave: str | None, upload: bool = False) -> None:
    """Limite por navegador (espaço/sessão) + teto por IP.

    Num laboratório, todos os alunos saem pela mesma rede (mesmo IP público):
    limitar só por IP faria a turma inteira dividir a mesma cota.
    """
    pessoa, rede = (limitador_upload, limitador_upload_ip) if upload else (limitador, limitador_ip)
    ip = _cliente(request)
    if not pessoa.permitir(f"nav:{chave}" if chave else f"ip:{ip}"):
        raise HTTPException(429, "Muitos envios em sequência. Aguarde um minuto." if upload else "Muitas perguntas em sequência. Aguarde um minuto.")
    if not rede.permitir(ip):
        raise HTTPException(429, "Muitos acessos desta rede agora. Aguarde um minuto.")


limitador_senha = LimitadorDeTaxa(10)  # tentativas de senha erradas por minuto, por IP


def _ministrante(request: Request) -> bool:
    """True se a chamada traz a senha dos ministrantes (cabeçalho X-Ministrante) correta."""
    return controle.senha_ok(request.headers.get("x-ministrante"))


def _exigir_ministrante(request: Request) -> None:
    if not controle.disponivel():
        raise HTTPException(403, "O Painel do Administrador está desligado: defina MINISTRANTE_SENHA no servidor.")
    if _ministrante(request):
        return
    if not limitador_senha.permitir(_cliente(request)):
        raise HTTPException(429, "Tentativas demais. Aguarde um minuto.")
    raise HTTPException(401, "Só o administrador pode usar isto (senha ausente ou incorreta).")


def _checar_pausa(request: Request, envio: bool = False) -> None:
    """Controles de emergência: os ministrantes continuam usando mesmo com a pausa ligada."""
    estado = controle.estado()
    pausado = estado["envios_pausados"] if envio else estado["perguntas_pausadas"]
    if pausado and not _ministrante(request):
        raise HTTPException(503, controle.PAUSA_ENVIOS if envio else controle.PAUSA_PERGUNTAS)


def _exigir_pronto_e_orcamento() -> None:
    if not ESTADO["pronto"]:
        raise HTTPException(503, "O assistente ainda está inicializando ou mal configurado.")
    if orcamento.excedido():
        raise HTTPException(429, MENSAGEM_LIMITE)


@app.post("/api/chat/stream")
async def chat_stream(dados: PerguntaIn, request: Request):
    """Mesmo que /api/chat, mas em Server-Sent Events: cada passo ReAct chega assim que acontece."""
    _limitar(request, dados.espaco_id or dados.sessao_id)
    _checar_pausa(request)
    controle.registrar_pergunta()
    _exigir_pronto_e_orcamento()
    documento = dados.documento.model_dump() if dados.documento else None

    parametros = dados.parametros.model_dump(exclude_none=True) if dados.parametros else None

    def eventos():
        for evento in perguntar_stream(dados.pergunta, dados.sessao_id, dados.espaco_id, documento, dados.exemplos, parametros):
            if evento["tipo"] == "final":
                orcamento.registrar(((evento["resposta"].get("custo") or {}).get("custo_usd")) or 0)
                turma.registrar(dados.espaco_id or dados.sessao_id, dados.pergunta, evento["resposta"])
            yield f"event: {evento['tipo']}\ndata: {json.dumps(evento, ensure_ascii=False)}\n\n"

    # StreamingResponse percorre o gerador síncrono em threadpool, sem travar o event loop.
    return StreamingResponse(
        eventos(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/comparar", response_model=ComparacaoOut)
async def comparar(dados: ComparacaoIn, request: Request) -> ComparacaoOut:
    """A mesma pergunta respondida SEM RAG (só o modelo), para a demonstração lado a lado."""
    _limitar(request, dados.sessao_id)
    _checar_pausa(request)
    _exigir_pronto_e_orcamento()
    try:
        resp = await run_in_threadpool(comparar_sem_rag, dados.pergunta)
    except Exception as exc:
        log.exception("Falha na comparação sem RAG")
        raise HTTPException(502, "Não foi possível consultar o modelo agora.") from exc
    orcamento.registrar(resp.custo.custo_usd)
    return resp


# ---------------------------------------------------------------------------
# Documentos enviados pelo usuário
# ---------------------------------------------------------------------------

RE_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


def _validar_espaco(espaco_id: str) -> str:
    if not RE_ID.match(espaco_id or ""):
        raise HTTPException(422, "Identificador de espaço inválido.")
    return espaco_id


def _nome_seguro(nome: str) -> str:
    """Nome de arquivo sem caminho, acentos ou caracteres especiais (evita path traversal)."""
    nome = Path(nome or "").name
    nome = unicodedata.normalize("NFKD", nome)
    nome = "".join(c for c in nome if not unicodedata.combining(c))
    base, ext = Path(nome).stem, Path(nome).suffix.lower()
    base = re.sub(r"[^A-Za-z0-9_-]+", "_", base).strip("_")[:80] or "documento"
    return base + ext


def _listar(espaco_id: str, mensagem: str = "", exemplos: bool = True) -> DocumentosOut:
    docs = documentos_do_espaco(espaco_id, incluir_exemplos=settings.incluir_exemplos and exemplos)
    return DocumentosOut(espaco_id=espaco_id, documentos=[DocumentoOut(**d) for d in docs], mensagem=mensagem)


@app.post("/api/documentos/sugestoes", response_model=SugestoesOut)
async def sugestoes(dados: SugestoesIn, request: Request) -> SugestoesOut:
    """3 perguntas de exemplo sobre um documento recém-enviado."""
    _limitar(request, dados.espaco_id)
    _checar_pausa(request, envio=True)
    _exigir_pronto_e_orcamento()
    try:
        resp = await run_in_threadpool(sugerir_perguntas, dados.espaco_id, _nome_seguro(dados.arquivo))
    except Exception as exc:
        log.exception("Falha ao sugerir perguntas")
        raise HTTPException(502, "Não foi possível sugerir perguntas agora.") from exc
    if resp is None:
        raise HTTPException(404, "Documento não encontrado.")
    orcamento.registrar(resp.custo.custo_usd)
    return resp


TIPOS_ARQUIVO = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".txt": "text/plain; charset=utf-8",
    ".md": "text/plain; charset=utf-8",
}


@app.get("/api/documentos/arquivo", include_in_schema=False)
def abrir_documento(espaco_id: str = Query(""), arquivo: str = Query(...), exemplo: bool = Query(False)):
    """Serve o arquivo original para abrir na página citada (o navegador entende `#page=N` em PDFs)."""
    nome = _nome_seguro(arquivo)
    if exemplo:
        pasta = settings.pdf_dir
    else:
        pasta = settings.upload_dir / _validar_espaco(espaco_id)
    caminho = (pasta / nome).resolve()
    if not caminho.is_file() or caminho.parent != pasta.resolve() or caminho.suffix.lower() not in TIPOS_ARQUIVO:
        raise HTTPException(404, "Arquivo não encontrado. Ele pode ter sido removido ou o servidor reiniciou.")
    return FileResponse(
        caminho,
        media_type=TIPOS_ARQUIVO[caminho.suffix.lower()],
        headers={"Content-Disposition": f'inline; filename="{nome}"', "Cache-Control": "private, max-age=300"},
    )


@app.get("/api/documentos", response_model=DocumentosOut)
async def listar_documentos(espaco_id: str = Query(...), exemplos: bool = Query(True)) -> DocumentosOut:
    _validar_espaco(espaco_id)
    if not ESTADO["pronto"]:
        raise HTTPException(503, "O assistente ainda está inicializando ou mal configurado.")
    return await run_in_threadpool(_listar, espaco_id, "", exemplos)


async def _salvar_e_indexar(
    espaco_id: str, nome: str, conteudo: bytes, titulo: str, exemplos: bool, tamanho_trecho: int | None, origem: str = ""
) -> DocumentosOut:
    """Valida, grava no espaço do usuário e indexa. Usado pelo upload e pelo envio por link."""
    ext = Path(nome).suffix.lower()
    if ext not in EXTENSOES_ACEITAS:
        raise HTTPException(415, f"Formato não suportado. Envie um arquivo {FORMATOS_ACEITOS}.")
    if not conteudo:
        raise HTTPException(422, "O arquivo está vazio.")
    if ext == ".pdf" and not conteudo.startswith(b"%PDF"):
        raise HTTPException(415, "O arquivo não parece ser um PDF válido.")
    if ext in {".docx", ".pptx", ".xlsx"} and not conteudo.startswith(b"PK"):
        raise HTTPException(415, "O arquivo não parece ser um documento do Office válido (.docx/.pptx/.xlsx). Arquivos .doc/.ppt/.xls antigos precisam ser salvos no formato novo.")

    existentes = await run_in_threadpool(documentos_do_espaco, espaco_id, False)
    if nome not in {d["arquivo"] for d in existentes} and len(existentes) >= settings.max_documentos_por_espaco:
        raise HTTPException(409, f"Limite de {settings.max_documentos_por_espaco} documentos atingido. Remova um para enviar outro.")

    pasta = settings.upload_dir / espaco_id
    pasta.mkdir(parents=True, exist_ok=True)
    destino = pasta / nome
    destino.write_bytes(conteudo)
    try:
        trechos, paginas_ocr = await run_in_threadpool(indexar_arquivo_detalhado, destino, espaco_id, titulo, tamanho_trecho)
    except DocumentoInvalido as exc:
        destino.unlink(missing_ok=True)
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        destino.unlink(missing_ok=True)
        log.exception("Falha ao indexar %s", nome)
        raise HTTPException(500, "Não foi possível processar o documento agora. Tente novamente.") from exc
    finally:
        marcar_base_alterada()
    # Custo aproximado dos embeddings do upload (~300 tokens por trecho de 1.200 caracteres).
    orcamento.registrar(trechos * 300 * 0.02 / 1_000_000)
    controle.registrar_envio(ext)

    detalhes = f" ({paginas_ocr} página{'s' if paginas_ocr != 1 else ''} lida{'s' if paginas_ocr != 1 else ''} por OCR)" if paginas_ocr else ""
    mensagem = f"“{titulo}”{origem} foi lido{detalhes} e dividido em {trechos} trecho{'s' if trechos != 1 else ''}. Já pode perguntar!"
    return await run_in_threadpool(_listar, espaco_id, mensagem, exemplos)


@app.post("/api/documentos", response_model=DocumentosOut)
async def enviar_documento(
    request: Request,
    espaco_id: str = Form(...),
    arquivo: UploadFile = File(...),
    exemplos: bool = Form(True),
    tamanho_trecho: int | None = Form(None, ge=200, le=4000),
) -> DocumentosOut:
    _validar_espaco(espaco_id)
    if not ESTADO["pronto"]:
        raise HTTPException(503, "O assistente ainda está inicializando ou mal configurado.")
    _limitar(request, espaco_id, upload=True)
    _checar_pausa(request, envio=True)
    if orcamento.excedido():
        raise HTTPException(429, MENSAGEM_LIMITE)

    nome = _nome_seguro(arquivo.filename or "")
    if Path(nome).suffix.lower() not in EXTENSOES_ACEITAS:
        raise HTTPException(415, f"Formato não suportado. Envie um arquivo {FORMATOS_ACEITOS}.")
    # Lê no máximo o limite + 1 byte: arquivo maior é recusado sem ocupar memória à toa.
    limite = settings.max_upload_mb * 1024 * 1024
    conteudo = await arquivo.read(limite + 1)
    if len(conteudo) > limite:
        raise HTTPException(413, f"O arquivo passa de {settings.max_upload_mb} MB.")
    titulo = titulo_amigavel(Path(Path(arquivo.filename or nome).name))
    return await _salvar_e_indexar(espaco_id, nome, conteudo, titulo, exemplos, tamanho_trecho)


@app.post("/api/documentos/url", response_model=DocumentosOut)
async def enviar_link(dados: LinkIn, request: Request) -> DocumentosOut:
    """Importa uma página da web (ou um PDF por link) como documento do usuário."""
    _validar_espaco(dados.espaco_id)
    if not ESTADO["pronto"]:
        raise HTTPException(503, "O assistente ainda está inicializando ou mal configurado.")
    _limitar(request, dados.espaco_id, upload=True)
    _checar_pausa(request, envio=True)
    if orcamento.excedido():
        raise HTTPException(429, MENSAGEM_LIMITE)
    try:
        conteudo, tipo, final = await run_in_threadpool(baixar, dados.url)
    except LinkInvalido as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        log.info("Falha ao baixar %s: %s", dados.url, exc)
        raise HTTPException(422, "Não consegui abrir esse link agora. Confira o endereço ou tente outro.") from exc
    if tipo == "application/pdf":
        base = Path(final.split("?")[0].rstrip("/")).stem or "pagina"
        titulo, nome = titulo_amigavel(Path(base)), _nome_seguro(base + ".pdf")
    else:
        if tipo in {"text/plain", "text/markdown"}:
            titulo, texto = Path(final.split("?")[0].rstrip("/")).stem or "pagina", f"Fonte: {final}\n\n" + conteudo.decode("utf-8", "replace")
        else:
            titulo, texto = await run_in_threadpool(html_para_texto, conteudo, final)
        titulo = titulo[:80]
        nome, conteudo = _nome_seguro(titulo + ".md"), texto.encode("utf-8")
    return await _salvar_e_indexar(dados.espaco_id, nome, conteudo, titulo, dados.exemplos, dados.tamanho_trecho, " (página da web)")


@app.delete("/api/documentos", response_model=DocumentosOut)
async def remover_documento(espaco_id: str = Query(...), arquivo: str | None = Query(None), exemplos: bool = Query(True)) -> DocumentosOut:
    _validar_espaco(espaco_id)
    if not ESTADO["pronto"]:
        raise HTTPException(503, "O assistente ainda está inicializando ou mal configurado.")
    nome = _nome_seguro(arquivo) if arquivo else None
    try:
        await run_in_threadpool(remover_documentos, espaco_id, nome)
    except DocumentoInvalido as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        marcar_base_alterada()
    pasta = settings.upload_dir / espaco_id
    if nome:
        (pasta / nome).unlink(missing_ok=True)
    else:
        shutil.rmtree(pasta, ignore_errors=True)
    return await run_in_threadpool(_listar, espaco_id, "Documento removido." if nome else "Seus documentos foram removidos.", exemplos)


@app.post("/api/mapa")
async def mapa(dados: MapaIn, request: Request):
    """Pontos 2D dos trechos (PCA dos embeddings) e da pergunta, para o modo aula."""
    _validar_espaco(dados.espaco_id)
    if not ESTADO["pronto"]:
        raise HTTPException(503, "O assistente ainda está inicializando ou mal configurado.")
    _limitar(request, dados.espaco_id)
    try:
        return await run_in_threadpool(montar_mapa, dados.espaco_id, dados.pergunta, dados.exemplos, dados.top_k)
    except Exception as exc:
        log.exception("Falha ao montar o mapa")
        raise HTTPException(500, "Não foi possível montar o mapa agora.") from exc


@app.get("/api/turma")
def painel_turma():
    """Perguntas recentes da turma (sem identificar ninguém), para o projetor."""
    return turma.estado()


limitador_moderacao = LimitadorDeTaxa(20)


def _moderador(dados: ModeracaoIn, request: Request) -> None:
    if not settings.painel_senha:
        raise HTTPException(403, "A moderação está desligada: defina a variável MINISTRANTE_SENHA no servidor.")
    if not limitador_moderacao.permitir(_cliente(request)):
        raise HTTPException(429, "Tentativas demais. Aguarde um minuto.")
    if not turma.senha_ok(dados.senha):
        raise HTTPException(401, "Senha do professor incorreta.")


@app.post("/api/turma/ocultar")
def ocultar_pergunta(dados: ModeracaoIn, request: Request):
    _moderador(dados, request)
    if not dados.id or not turma.ocultar(dados.id):
        raise HTTPException(404, "Pergunta não encontrada (talvez já tenha saído do painel).")
    return {"ok": True}


@app.post("/api/turma/limpar")
def limpar_painel(dados: ModeracaoIn, request: Request):
    _moderador(dados, request)
    turma.limpar()
    return {"ok": True}


def _estado_ministrante() -> dict:
    t = turma.estado()
    return {
        "controles": {**controle.estado(), "limite_diario_usd": orcamento.status()["limite_usd"]},
        "persistencia": persistencia.ativo(),
        "orcamento": orcamento.status(),
        "numeros": {
            **controle.numeros(),
            "navegadores_ativos": t["navegadores_ativos"],
            "bloqueadas": t["bloqueadas"],
            "recusadas": t["recusadas"],
        },
    }


@app.post("/api/ministrante/entrar")
def ministrante_entrar(dados: MinistranteEntrarIn, request: Request):
    """Confere a senha dos ministrantes (sem guardar nada no servidor)."""
    if not controle.disponivel():
        raise HTTPException(403, "O Painel do Administrador está desligado: defina MINISTRANTE_SENHA no servidor.")
    if not limitador_senha.permitir(_cliente(request)):
        raise HTTPException(429, "Tentativas demais. Aguarde um minuto.")
    if not controle.senha_ok(dados.senha):
        raise HTTPException(401, "Senha incorreta.")
    return {"ok": True, **_estado_ministrante()}


@app.get("/api/ministrante/estado")
def ministrante_estado(request: Request):
    _exigir_ministrante(request)
    return _estado_ministrante()


@app.post("/api/ministrante/controles")
def ministrante_controles(dados: ControlesIn, request: Request):
    """Pausar perguntas/envios e ajustar o limite de gasto do dia."""
    _exigir_ministrante(request)
    controle.atualizar(dados.perguntas_pausadas, dados.envios_pausados)
    if dados.limite_diario_usd is not None:
        orcamento.ajustar(dados.limite_diario_usd)
    log.info("MINISTRANTE controles: %s", dados.model_dump(exclude_none=True))
    return _estado_ministrante()


@app.post("/api/busca/comparar")
async def busca_comparada(dados: EstudoIn, request: Request):
    """Modo aula: o que a busca semântica e a por palavras-chave acham, cada uma sozinha."""
    _validar_espaco(dados.espaco_id)
    if not ESTADO["pronto"]:
        raise HTTPException(503, "O assistente ainda está inicializando ou mal configurado.")
    _limitar(request, dados.espaco_id)
    return await run_in_threadpool(comparar_buscas, dados.pergunta, dados.espaco_id, dados.exemplos, dados.top_k, dados.semelhanca_minima)


@app.get("/api/documentos/fatiamento")
async def ver_fatiamento(espaco_id: str = Query(""), arquivo: str = Query(...), exemplo: bool = Query(False)):
    """Modo aula: como o documento foi dividido em trechos."""
    if not exemplo:
        _validar_espaco(espaco_id)
    dados = await run_in_threadpool(fatiamento, ESPACO_EXEMPLOS if exemplo else espaco_id, _nome_seguro(arquivo))
    if not dados["total"]:
        raise HTTPException(404, "Documento não encontrado.")
    return dados


@app.post("/api/comparar-modelos")
async def comparar_dois_modelos(dados: EstudoIn, request: Request):
    """Modo aula: dois modelos respondendo à mesma pergunta com os mesmos trechos (só ministrantes: o modelo maior custa caro)."""
    _exigir_ministrante(request)
    _validar_espaco(dados.espaco_id)
    _limitar(request, dados.espaco_id)
    _exigir_pronto_e_orcamento()
    resp = await run_in_threadpool(comparar_modelos, dados.pergunta, dados.espaco_id, dados.exemplos)
    orcamento.registrar(sum(m.get("custo_usd", 0) for m in resp["modelos"]))
    return resp


@app.get("/turma", include_in_schema=False)
def pagina_turma():
    return FileResponse(STATIC / "turma.html")


def url_publica(request: Request) -> str:
    """URL que os alunos devem abrir (QR code). Prioriza a configurada no ambiente."""
    if settings.public_url:
        return settings.public_url.rstrip("/")
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host", request.headers.get("host", request.url.netloc))
    return f"{proto}://{host}"


@lru_cache(maxsize=8)
def _qr_svg(url: str) -> bytes:
    buf = io.BytesIO()
    segno.make(url, error="m").save(buf, kind="svg", scale=8, border=2, dark="#111111", light="#ffffff", xmldecl=False)
    return buf.getvalue()


@app.get("/api/info")
def info(request: Request):
    return {
        "url": url_publica(request),
        "modelo": settings.chat_model,
        "cotacao_usd_brl": settings.usd_brl,
        "upload": {
            "extensoes": sorted(EXTENSOES_ACEITAS),
            "max_mb": settings.max_upload_mb,
            "max_paginas": settings.max_upload_paginas,
            "max_documentos": settings.max_documentos_por_espaco,
        },
        "orcamento": orcamento.status(),
        "busca": {
            "top_k": settings.top_k,
            "semelhanca_minima": round(1 - settings.max_distance, 2),
            "tamanho_trecho": settings.chunk_size,
        },
        "painel_turma": settings.painel_turma,
        "ministrante_disponivel": controle.disponivel(),
        "controle": controle.estado(),
    }


@app.get("/qr.svg")
def qr_code(request: Request):
    return Response(_qr_svg(url_publica(request)), media_type="image/svg+xml", headers={"Cache-Control": "public, max-age=3600"})


ICONES = {
    "favicon.ico": "image/x-icon",
    "favicon.svg": "image/svg+xml",
    "apple-touch-icon.png": "image/png",
    "icon-512.png": "image/png",
}


@app.get("/como-funciona", include_in_schema=False)
def como_funciona():
    return FileResponse(STATIC / "como-funciona.html")


@app.get("/{nome}", include_in_schema=False)
def icone(nome: str):
    if nome not in ICONES:
        raise HTTPException(404)
    return FileResponse(STATIC / nome, media_type=ICONES[nome], headers={"Cache-Control": "public, max-age=86400"})


@app.get("/")
def index():
    return FileResponse(Path(STATIC / "index.html"))
