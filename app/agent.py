"""Agente autônomo ReAct (Thought -> Action -> Observation) com LangChain/LangGraph.

- Modelo: OpenAI (gpt-4o-mini, padrão) ou Gemini, via app/provedor.py
- Ferramentas: buscar_documentos, listar_documentos (filtradas pelo espaço do usuário)
- Memória de sessão: InMemorySaver (checkpointer do LangGraph), uma thread por sessao_id
- Saída: JSON reparado com json-repair e validado com Pydantic
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from typing import Callable, Iterator

import openai
from json_repair import repair_json
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from pydantic import ValidationError

from app.config import settings
from app.custos import calcular_custo
from app.guardrails import citacao_foi_recuperada, validar_pergunta
from app.prompts import SYSTEM_PROMPT
from app.provedor import criar_chat
from app.schemas import Custo, PassoReAct, RespostaAgente, RespostaOut, TrechoLido
from app.tools import FERRAMENTAS, consumir_uso_extra

log = logging.getLogger("agente")

# FinOps: o histórico enviado ao modelo é limitado a este orçamento de tokens.
# A conversa inteira continua salva no checkpointer; só o que vai para a API é aparado.
ORCAMENTO_HISTORICO_TOKENS = 6000


@wrap_model_call
def aparar_historico(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    """Envia ao modelo só as mensagens mais recentes que cabem no orçamento de tokens."""
    mensagens = request.messages
    if count_tokens_approximately(mensagens) > ORCAMENTO_HISTORICO_TOKENS:
        aparadas = trim_messages(
            mensagens,
            strategy="last",
            token_counter=count_tokens_approximately,
            max_tokens=ORCAMENTO_HISTORICO_TOKENS,
            start_on="human",  # nunca começa com ToolMessage órfã
            allow_partial=False,
        )
        if aparadas:
            request = request.override(messages=aparadas)
    return handler(request)


@lru_cache(maxsize=1)
def construir_agente():
    modelo = criar_chat()
    return create_agent(
        model=modelo,
        tools=FERRAMENTAS,
        system_prompt=SYSTEM_PROMPT,
        middleware=[aparar_historico],
        checkpointer=InMemorySaver(),
    )


# ---------------------------------------------------------------------------
# Extração do raciocínio ReAct (para logs e para a interface)
# ---------------------------------------------------------------------------

def _mensagens_do_turno(mensagens: list[BaseMessage]) -> list[BaseMessage]:
    """Retorna as mensagens geradas após a última pergunta do usuário."""
    for i in range(len(mensagens) - 1, -1, -1):
        if isinstance(mensagens[i], HumanMessage):
            return mensagens[i + 1 :]
    return mensagens


def _texto(msg: BaseMessage) -> str:
    c = msg.content
    if isinstance(c, str):
        return c
    return "".join(p.get("text", "") for p in c if isinstance(p, dict))


def extrair_passos(mensagens: list[BaseMessage]) -> tuple[list[PassoReAct], list[str]]:
    passos: list[PassoReAct] = []
    observacoes: list[str] = []
    for msg in mensagens:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            if _texto(msg).strip():
                passos.append(PassoReAct(tipo="thought", conteudo=_texto(msg).strip()))
            for call in msg.tool_calls:
                args = dict(call.get("args") or {})
                pensamento = args.pop("pensamento", None)
                if pensamento:
                    passos.append(PassoReAct(tipo="thought", conteudo=pensamento))
                args_txt = ", ".join(f'{k}="{v}"' for k, v in args.items())
                passos.append(
                    PassoReAct(tipo="action", ferramenta=call["name"], conteudo=f"{call['name']}({args_txt})")
                )
        elif isinstance(msg, ToolMessage):
            obs = _texto(msg)
            observacoes.append(obs)
            resumo = obs if len(obs) <= 1500 else obs[:1500] + " [...]"
            passos.append(PassoReAct(tipo="observation", ferramenta=msg.name, conteudo=resumo))
    return passos, observacoes


# ---------------------------------------------------------------------------
# Saída estruturada tolerante a falhas
# ---------------------------------------------------------------------------

def interpretar_resposta_final(texto: str) -> tuple[RespostaAgente, list[str]]:
    """Converte a mensagem final do modelo em RespostaAgente.

    1. json-repair corrige JSON malformado (aspas, vírgulas, ```json``` etc.)
    2. Pydantic valida tipos e valores permitidos
    3. Se tudo falhar, devolve o texto cru com confiança baixa (nunca quebra a API)
    """
    avisos: list[str] = []
    try:
        dados = repair_json(texto, return_objects=True)
        if not isinstance(dados, dict) or "resposta" not in dados:
            raise ValueError("JSON sem o campo 'resposta'")
        return RespostaAgente.model_validate(dados), avisos
    except (ValidationError, ValueError, TypeError, json.JSONDecodeError) as exc:
        log.warning("Saída fora do schema, usando fallback: %s", exc)
        avisos.append("A resposta do modelo não veio no formato estruturado; exibindo texto bruto.")
        return RespostaAgente(resposta=texto.strip() or "(sem resposta)", confianca="baixa"), avisos


RE_CABECALHO_TRECHO = re.compile(
    r"^\[\d+\] (?P<doc>.+?) \| página (?P<pag>\d+)(?: \| Art\. \d+)? \| relevância (?P<rel>[^|\n]+?)(?: \| origem: (?P<origem>[^\n]+))?$",
    re.MULTILINE,
)


def extrair_trechos(turno: list[BaseMessage], resp: RespostaAgente) -> list[TrechoLido]:
    """Lê os trechos devolvidos pela busca neste turno, para mostrar na interface."""
    citados = {(c.documento, c.pagina) for c in resp.citacoes}
    trechos: list[TrechoLido] = []
    vistos: set[tuple[str, int, str]] = set()
    for msg in turno:
        if not isinstance(msg, ToolMessage) or msg.name not in {"buscar_documentos", "resumir_documento"}:
            continue
        for bloco in _texto(msg).split("\n\n---\n\n"):
            m = RE_CABECALHO_TRECHO.search(bloco)
            if not m:
                continue
            corpo = bloco[m.end():].strip()
            chave = (m["doc"], int(m["pag"]), corpo[:80])
            if chave in vistos:
                continue
            vistos.add(chave)
            rel = m["rel"].strip()
            numero = re.search(r"[-\d.]+$", rel)
            trechos.append(
                TrechoLido(
                    documento=m["doc"],
                    pagina=int(m["pag"]),
                    relevancia=max(0.0, min(1.0, float(numero.group()))) if rel.startswith("semântica") and numero else None,
                    tipo_busca="semântica" if rel.startswith("semântica") else ("resumo" if rel.startswith("resumo") else "lexical"),
                    origem="exemplo" if (m["origem"] or "").strip() == "exemplo" else "usuario",
                    texto=corpo if len(corpo) <= 700 else corpo[:697] + "...",
                    citado=any(m["doc"] == d and (p is None or int(p) == int(m["pag"])) for d, p in citados),
                )
            )
    return trechos


RE_ARTIGO_CITADO = re.compile(r"\b(?:Art\.?|artigo)\s*(\d+)", re.IGNORECASE)


def verificar_citacoes(resp: RespostaAgente, observacoes: list[str]) -> list[str]:
    """Guardrail de saída: só aceita fontes e artigos que vieram da busca.

    `observacoes` são TODOS os retornos de ferramenta da sessão, para que uma
    pergunta de acompanhamento possa reaproveitar evidência já recuperada.
    """
    avisos: list[str] = []
    verificadas = [c for c in resp.citacoes if citacao_foi_recuperada(c.documento, c.pagina, observacoes)]
    removidas = len(resp.citacoes) - len(verificadas)
    if removidas:
        avisos.append(f"{removidas} citação(ões) descartada(s) por não corresponder(em) aos trechos recuperados.")
    resp.citacoes = verificadas

    if resp.status != "respondida":
        resp.citacoes = []
        return avisos

    if not observacoes:
        resp.confianca = "baixa"
        avisos.append("Resposta gerada sem consulta à base documental.")
    elif not verificadas:
        resp.confianca = "baixa"
        avisos.append("Nenhuma citação verificável: confira a informação no documento original.")

    # Números de artigo mencionados no texto precisam existir nos trechos recuperados.
    recuperados = {int(n) for obs in observacoes for n in RE_ARTIGO_CITADO.findall(obs)}
    inventados = sorted({int(n) for n in RE_ARTIGO_CITADO.findall(resp.resposta)} - recuperados)
    if inventados:
        resp.confianca = "baixa"
        lista = ", ".join(f"Art. {n}" for n in inventados)
        avisos.append(f"{lista} não aparece(m) nos trechos consultados; trate essa referência com cautela.")
    return avisos


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------

def _erro(sessao_id: str, mensagem: str, status: str = "erro") -> RespostaOut:
    # Sem custo calculável: bloqueios da blindagem acontecem antes de qualquer chamada à API.
    custo = Custo(modelo=settings.chat_model, cotacao_usd_brl=settings.usd_brl)
    return RespostaOut(sessao_id=sessao_id, resposta=mensagem, status=status, confianca="baixa", custo=custo)


def _preparar(pergunta: str, sessao_id: str, espaco_id: str | None, documento: dict | None, exemplos: bool = True):
    """Valida a pergunta e monta a entrada e o config do agente. Devolve (entrada, config, erro)."""
    validacao = validar_pergunta(pergunta)
    if not validacao.ok:
        log.info("[%s] entrada bloqueada: %s", sessao_id, validacao.motivo)
        return None, None, _erro(sessao_id, validacao.motivo, status="bloqueada")

    # thread_id = memória da conversa; espaco_id = quais documentos enviados esta pessoa pode consultar.
    # O espaço e o documento escolhido vão pelo config (e não pela pergunta), então o modelo não consegue trocá-los.
    config = {
        "configurable": {"thread_id": sessao_id, "espaco_id": espaco_id or "", "documento": documento or None, "exemplos": exemplos},
        "recursion_limit": settings.recursion_limit,
    }
    texto = validacao.texto
    if documento and documento.get("arquivo"):
        # Contexto visível ao modelo; a restrição de fato é aplicada pelas ferramentas.
        nome = documento.get("documento") or documento["arquivo"]
        texto = f"[O usuário escolheu na interface o documento «{nome}» para esta pergunta.]\n{texto}"
    return {"messages": [HumanMessage(texto)]}, config, None


def _resposta_de_erro(exc: Exception, sessao_id: str) -> RespostaOut:
    if isinstance(exc, RuntimeError):  # chave ausente
        log.error("Configuração inválida: %s", exc)
        return _erro(sessao_id, "O assistente não está configurado (chave de API ausente).")
    if isinstance(exc, openai.AuthenticationError):
        log.error("Chave da OpenAI inválida ou revogada")
        return _erro(sessao_id, "Falha de autenticação com o provedor de IA. Avise os administradores.")
    if isinstance(exc, openai.RateLimitError) or "RESOURCE_EXHAUSTED" in str(exc) or "429" in type(exc).__name__ + str(exc)[:40]:
        log.warning("Cota/limite da OpenAI atingido")
        return _erro(sessao_id, "O limite de uso da API foi atingido. Tente novamente em alguns instantes.")
    if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError)):
        log.warning("Timeout/conexão com a OpenAI")
        return _erro(sessao_id, "O provedor de IA não respondeu a tempo. Tente novamente.")
    if isinstance(exc, GraphRecursionError):
        log.warning("Limite de passos ReAct atingido")
        return _erro(
            sessao_id,
            "Não consegui chegar a uma resposta segura dentro do limite de passos. Reformule a pergunta de forma mais específica.",
        )
    log.exception("Erro inesperado no agente", exc_info=exc)
    return _erro(sessao_id, "Ocorreu um erro interno ao processar sua pergunta.")


def _mapa_arquivos(espaco_id: str | None) -> dict[str, tuple[str, bool]]:
    """Nome do documento -> (arquivo, é exemplo), para os links "abrir na página"."""
    from app.ingest import ESPACO_EXEMPLOS, documentos_do_espaco

    try:
        docs = documentos_do_espaco(espaco_id or ESPACO_EXEMPLOS, incluir_exemplos=bool(espaco_id))
    except Exception:
        return {}
    return {d["documento"]: (d["arquivo"], d["exemplo"]) for d in docs}


def _finalizar(mensagens: list[BaseMessage], sessao_id: str, espaco_id: str | None) -> RespostaOut:
    turno = _mensagens_do_turno(mensagens)
    passos, _ = extrair_passos(turno)
    observacoes_sessao = [_texto(m) for m in mensagens if isinstance(m, ToolMessage)]
    final = next((m for m in reversed(turno) if isinstance(m, AIMessage) and not m.tool_calls), None)

    resp, avisos = interpretar_resposta_final(_texto(final) if final else "")
    avisos += verificar_citacoes(resp, observacoes_sessao)
    custo = calcular_custo(turno, consumir_uso_extra(sessao_id))
    log.info(
        "[%s] CUSTO %d chamadas | %d in (%d cache) | %d out | %d emb | US$ %.6f",
        sessao_id, custo.chamadas_llm, custo.tokens_entrada, custo.tokens_entrada_cache,
        custo.tokens_saida, custo.tokens_embedding, custo.custo_usd,
    )

    for p in passos:
        log.info("[%s] %-11s %s", sessao_id, p.tipo.upper(), p.conteudo.replace("\n", " ")[:200])

    trechos = extrair_trechos(turno, resp)
    arquivos = _mapa_arquivos(espaco_id) if trechos else {}
    for t in trechos:
        t.arquivo = arquivos.get(t.documento, ("", False))[0]
    return RespostaOut(sessao_id=sessao_id, trechos=trechos, passos=passos, avisos=avisos, custo=custo, **resp.model_dump())


def perguntar(
    pergunta: str, sessao_id: str, espaco_id: str | None = None, documento: dict | None = None, exemplos: bool = True
) -> RespostaOut:
    entrada, config, erro = _preparar(pergunta, sessao_id, espaco_id, documento, exemplos)
    if erro:
        return erro
    try:
        estado = construir_agente().invoke(entrada, config=config)
    except Exception as exc:
        return _resposta_de_erro(exc, sessao_id)
    return _finalizar(estado["messages"], sessao_id, espaco_id)


def perguntar_stream(
    pergunta: str, sessao_id: str, espaco_id: str | None = None, documento: dict | None = None, exemplos: bool = True
) -> Iterator[dict]:
    """Mesma lógica de `perguntar`, mas emitindo cada passo ReAct assim que ele acontece.

    Eventos: {"tipo": "passo", "passo": {...}} durante o raciocínio e, no fim,
    {"tipo": "final", "resposta": {...RespostaOut...}}.
    """
    entrada, config, erro = _preparar(pergunta, sessao_id, espaco_id, documento, exemplos)
    if erro:
        yield {"tipo": "final", "resposta": erro.model_dump()}
        return
    try:
        agente = construir_agente()
        for atualizacao in agente.stream(entrada, config=config, stream_mode="updates"):
            for parcial in (atualizacao or {}).values():
                mensagens = (parcial or {}).get("messages") if isinstance(parcial, dict) else None
                if not mensagens:
                    continue
                passos, _ = extrair_passos(mensagens)
                for passo in passos:
                    yield {"tipo": "passo", "passo": passo.model_dump()}
        estado = agente.get_state(config).values
    except Exception as exc:
        yield {"tipo": "final", "resposta": _resposta_de_erro(exc, sessao_id).model_dump()}
        return
    yield {"tipo": "final", "resposta": _finalizar(estado["messages"], sessao_id, espaco_id).model_dump()}
