"""Painel da turma (/turma): perguntas recentes de todos, ao vivo, no projetor.

Fica só em memória e não guarda quem perguntou: nem IP, nem espaço, nem sessão
aparecem no painel. O espaço entra só como um "hash" curto, para contar quantos
navegadores estão ativos.

Moderação: palavrões são mascarados automaticamente, e o professor (com a senha
PAINEL_SENHA) pode ocultar uma pergunta ou limpar o painel.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import threading
import time
import unicodedata
import uuid
from collections import deque

from app import persistencia
from app.config import settings

_lock = threading.Lock()
_recentes: deque[dict] = deque(maxlen=60)
_ativos: dict[str, float] = {}
_totais = {"perguntas": 0, "respondidas": 0, "custo_usd": 0.0, "bloqueadas": 0, "recusadas": 0}
JANELA_ATIVOS = 15 * 60  # navegador que perguntou nos últimos 15 min conta como ativo

# Lista curta e conservadora: o objetivo é evitar constrangimento no telão, não censurar.
PALAVROES = {
    "porra", "caralho", "merda", "bosta", "puta", "puto", "putaria", "foder", "foda", "fodase", "fodido",
    "cacete", "buceta", "boceta", "piroca", "cuzao", "arrombado", "arrombada",
    "viado", "veado", "otario", "otaria", "babaca", "idiota", "imbecil", "retardado", "corno", "vagabundo",
    "vagabunda", "vadia", "safado", "safada", "desgraca", "desgracado", "fdp", "pqp", "vsf", "tnc", "krl", "crl",
    "fuck", "shit", "bitch",
}


def _sem_acento(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in texto if not unicodedata.combining(c))


def mascarar(texto: str) -> tuple[str, bool]:
    """Troca palavrões por "•••". Devolve (texto, se mascarou algo)."""
    mascarou = False

    def troca(m: re.Match) -> str:
        nonlocal mascarou
        palavra = m.group(0)
        if _sem_acento(palavra) in PALAVROES:
            mascarou = True
            return palavra[0] + "•" * max(2, len(palavra) - 1)
        return palavra

    return re.sub(r"[A-Za-zÀ-ÿ]+", troca, texto), mascarou


def registrar(chave: str | None, pergunta: str, resposta: dict) -> None:
    if not settings.painel_turma:
        return
    citados = []
    for c in resposta.get("citacoes") or []:
        nome = f"{c.get('documento')}" + (f", p. {c['pagina']}" if c.get("pagina") else "")
        if nome not in citados:
            citados.append(nome)
    custo = float(((resposta.get("custo") or {}).get("custo_usd")) or 0)
    status = resposta.get("status")
    pergunta_limpa, m1 = mascarar(pergunta.strip()[:240])
    resposta_limpa, m2 = mascarar((resposta.get("resposta") or "")[:280])
    agora = time.time()
    navegador = hashlib.sha1(chave.encode()).hexdigest()[:10] if chave else None
    with _lock:
        if navegador:
            _ativos[navegador] = agora
        _totais["perguntas"] += 1
        _totais["respondidas"] += status == "respondida"
        _totais["bloqueadas"] += status == "bloqueada"
        _totais["recusadas"] += status in {"sem_evidencia", "fora_do_escopo"}
        _totais["custo_usd"] += custo
        item = {
            "id": uuid.uuid4().hex[:10],
            "hora": agora,
            "pergunta": pergunta_limpa,
            "resposta": resposta_limpa,
            "status": status,
            "confianca": resposta.get("confianca"),
            "fontes": citados[:3],
            "moderada": m1 or m2,
        }
        _recentes.appendleft(item)
    persistencia.salvar_pergunta_painel(item, navegador, custo)


def senha_ok(senha: str) -> bool:
    from app.controle import senha_ok as ok

    return ok(senha)


def ocultar(item_id: str) -> bool:
    with _lock:
        for i, item in enumerate(_recentes):
            if item["id"] == item_id:
                del _recentes[i]
                break
        else:
            return False
    persistencia.ocultar_pergunta_painel(item_id)
    return True


def limpar() -> None:
    with _lock:
        _recentes.clear()
        _ativos.clear()
        for k in _totais:
            _totais[k] = 0.0 if k == "custo_usd" else 0
    persistencia.salvar_controle("painel_limpo_em", time.time())


def restaurar(linhas: list[dict]) -> None:
    """Perguntas salvas no banco (desde a última limpeza), em ordem de horário."""
    with _lock:
        _recentes.clear()
        _ativos.clear()
        for k in _totais:
            _totais[k] = 0.0 if k == "custo_usd" else 0
        for r in linhas:
            status = r.get("status")
            _totais["perguntas"] += 1
            _totais["respondidas"] += status == "respondida"
            _totais["bloqueadas"] += status == "bloqueada"
            _totais["recusadas"] += status in {"sem_evidencia", "fora_do_escopo"}
            _totais["custo_usd"] += float(r.get("custo_usd") or 0)
            if r.get("navegador"):
                _ativos[r["navegador"]] = max(_ativos.get(r["navegador"], 0), r["hora"])
            if not r.get("oculta"):
                _recentes.appendleft({k: r[k] for k in ("id", "hora", "pergunta", "resposta", "status", "confianca", "fontes", "moderada")})


def estado() -> dict:
    agora = time.time()
    with _lock:
        for k in [k for k, t in _ativos.items() if agora - t > JANELA_ATIVOS]:
            del _ativos[k]
        return {
            "ativo": settings.painel_turma,
            "moderacao": bool(settings.painel_senha),
            "agora": agora,
            "navegadores_ativos": len(_ativos),
            **_totais,
            "recentes": list(_recentes),
        }
