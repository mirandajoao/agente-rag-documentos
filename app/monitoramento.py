"""Monitoramento de erros com o Sentry (opcional, ligado por SENTRY_DSN).

Só vão para o Sentry o tipo do erro, a mensagem e a pilha de chamadas.
Nada do que o usuário escreveu ou enviou: sem corpo das requisições,
sem variáveis locais, sem cabeçalhos, cookies ou parâmetros da URL.
"""

from __future__ import annotations

import logging
import os

from app.config import settings

log = logging.getLogger("monitoramento")
ATIVO = False


def _sem_dados_pessoais(evento: dict, _hint: dict) -> dict:
    requisicao = evento.get("request")
    if isinstance(requisicao, dict):
        evento["request"] = {k: v for k, v in requisicao.items() if k in {"method", "url"}}
        if "url" in evento["request"]:
            evento["request"]["url"] = str(evento["request"]["url"]).split("?", 1)[0]
    evento.pop("user", None)
    for excecao in (evento.get("exception") or {}).get("values") or []:
        for quadro in (excecao.get("stacktrace") or {}).get("frames") or []:
            quadro.pop("vars", None)
    return evento


def iniciar(transport=None) -> bool:
    global ATIVO
    if ATIVO:
        return True
    if not settings.sentry_dsn:
        return False
    try:
        import sentry_sdk

        sentry_sdk.init(
            dsn=settings.sentry_dsn,
            environment=os.getenv("RAILWAY_ENVIRONMENT_NAME", "local"),
            release=os.getenv("RAILWAY_GIT_COMMIT_SHA") or None,
            send_default_pii=False,
            include_local_variables=False,
            max_request_body_size="never",
            traces_sample_rate=0.0,
            before_send=_sem_dados_pessoais,
            **({"transport": transport} if transport is not None else {}),
        )
        ATIVO = True
        log.info("Sentry ligado")
    except Exception:
        log.exception("Não foi possível ligar o Sentry; seguindo sem ele")
    return ATIVO
