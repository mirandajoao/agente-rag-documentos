"""Painel da turma (/turma): perguntas recentes de todos, ao vivo, no projetor.

Fica só em memória e não guarda quem perguntou: nem IP, nem espaço, nem sessão
aparecem no painel. O espaço entra só como um "hash" curto, para contar quantos
navegadores estão ativos.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import deque

from app.config import settings

_lock = threading.Lock()
_recentes: deque[dict] = deque(maxlen=60)
_ativos: dict[str, float] = {}
_totais = {"perguntas": 0, "respondidas": 0, "custo_usd": 0.0}
JANELA_ATIVOS = 15 * 60  # navegador que perguntou nos últimos 15 min conta como ativo


def registrar(chave: str | None, pergunta: str, resposta: dict) -> None:
    if not settings.painel_turma:
        return
    citados = []
    for c in resposta.get("citacoes") or []:
        nome = f"{c.get('documento')}" + (f", p. {c['pagina']}" if c.get("pagina") else "")
        if nome not in citados:
            citados.append(nome)
    custo = float(((resposta.get("custo") or {}).get("custo_usd")) or 0)
    agora = time.time()
    with _lock:
        if chave:
            _ativos[hashlib.sha1(chave.encode()).hexdigest()[:10]] = agora
        _totais["perguntas"] += 1
        _totais["respondidas"] += resposta.get("status") == "respondida"
        _totais["custo_usd"] += custo
        _recentes.appendleft({
            "hora": agora,
            "pergunta": pergunta.strip()[:240],
            "resposta": (resposta.get("resposta") or "")[:280],
            "status": resposta.get("status"),
            "confianca": resposta.get("confianca"),
            "fontes": citados[:3],
        })


def estado() -> dict:
    agora = time.time()
    with _lock:
        for k in [k for k, t in _ativos.items() if agora - t > JANELA_ATIVOS]:
            del _ativos[k]
        return {
            "ativo": settings.painel_turma,
            "agora": agora,
            "navegadores_ativos": len(_ativos),
            **_totais,
            "recentes": list(_recentes),
        }
