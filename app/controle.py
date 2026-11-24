"""Modo ministrante: senha, controles de emergência e números da turma.

A senha (MINISTRANTE_SENHA, ou PAINEL_SENHA) vai no cabeçalho X-Ministrante de
cada chamada e é conferida no servidor: esconder botões na tela não protege nada.
Tudo fica em memória (zera quando o servidor reinicia).
"""

from __future__ import annotations

import hmac
import os
import threading
import time
from collections import Counter, deque
from datetime import datetime
from zoneinfo import ZoneInfo

from app import persistencia
from app.config import settings

_lock = threading.Lock()
_estado = {"perguntas_pausadas": False, "envios_pausados": False}
_perguntas: deque[float] = deque(maxlen=20_000)  # horário de cada pergunta
_envios: Counter[str] = Counter()  # tipos de arquivo enviados (sem nomes: podem ser pessoais)

PAUSA_PERGUNTAS = "As perguntas estão pausadas pelos ministrantes por alguns instantes. Já já voltam!"
PAUSA_ENVIOS = "Os envios de documentos estão pausados pelos ministrantes por alguns instantes."


def disponivel() -> bool:
    return bool(settings.painel_senha)


def senha_ok(senha: str | None) -> bool:
    return bool(settings.painel_senha) and bool(senha) and hmac.compare_digest(senha.encode(), settings.painel_senha.encode())


def estado() -> dict:
    with _lock:
        return dict(_estado)


def atualizar(perguntas_pausadas: bool | None = None, envios_pausados: bool | None = None) -> dict:
    with _lock:
        if perguntas_pausadas is not None:
            _estado["perguntas_pausadas"] = perguntas_pausadas
        if envios_pausados is not None:
            _estado["envios_pausados"] = envios_pausados
        atual = dict(_estado)
    if perguntas_pausadas is not None:
        persistencia.salvar_controle("perguntas_pausadas", perguntas_pausadas)
    if envios_pausados is not None:
        persistencia.salvar_controle("envios_pausados", envios_pausados)
    return atual


def registrar_pergunta() -> None:
    agora = time.time()
    with _lock:
        _perguntas.append(agora)
    persistencia.salvar_evento("pergunta", hora=agora)


def registrar_envio(extensao: str) -> None:
    tipo = extensao.lower().lstrip(".") or "?"
    with _lock:
        _envios[tipo] += 1
    persistencia.salvar_evento("envio", tipo)


def restaurar(controles: dict, perguntas: list[float], envios: dict[str, int]) -> None:
    """Estado salvo no banco, aplicado quando o servidor sobe."""
    with _lock:
        for chave in ("perguntas_pausadas", "envios_pausados"):
            if chave in controles:
                _estado[chave] = bool(controles[chave])
        _perguntas.clear()
        _perguntas.extend(perguntas)
        _envios.clear()
        _envios.update(envios)


def numeros(horas: int = 6) -> dict:
    """Perguntas por hora (últimas `horas`, no fuso do orçamento) e tipos de arquivo enviados."""
    fuso = ZoneInfo(os.getenv("FUSO_ORCAMENTO", "America/Belem"))
    agora = time.time()
    inicio_hora = agora - (agora % 3600)
    with _lock:
        tempos = list(_perguntas)
        envios = dict(_envios)
    baldes = []
    for i in range(horas - 1, -1, -1):
        ini = inicio_hora - i * 3600
        rotulo = datetime.fromtimestamp(ini, fuso).strftime("%Hh")
        baldes.append({"hora": rotulo, "perguntas": sum(1 for t in tempos if ini <= t < ini + 3600)})
    ultimos_10min = sum(1 for t in tempos if agora - t <= 600)
    return {"por_hora": baldes, "ultimos_10min": ultimos_10min, "total": len(tempos), "envios_por_tipo": envios, "envios_total": sum(envios.values())}
