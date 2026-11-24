"""Memória da conversa: onde o agente guarda o histórico de cada sessão.

Sem banco, o histórico fica na memória do servidor (InMemorySaver) e some a cada
reinício. Com DATABASE_URL, fica no Postgres (PostgresSaver do LangGraph): a
pessoa pode recarregar a página, o servidor pode reiniciar, e a próxima pergunta
continua a mesma conversa ("e se eu não alcançar essa frequência?").

O banco nunca derruba a aula: se ele falhar durante uma pergunta, o agente passa
a usar a memória do servidor até o próximo reinício.
Conversas paradas há mais de MEMORIA_RETENCAO_DIAS são apagadas na subida.
"""

from __future__ import annotations

import atexit
import logging
import os
import time

from langgraph.checkpoint.memory import InMemorySaver

from app import persistencia

log = logging.getLogger("memoria")

RETENCAO_DIAS = int(os.getenv("MEMORIA_RETENCAO_DIAS", "7") or 7)
_estado = {"tipo": "memória do servidor", "desligada": False}
_pool = None


def tipo() -> str:
    return _estado["tipo"]


def _limpar_antigas(saver) -> int:
    limite = time.time() - RETENCAO_DIAS * 86400
    with persistencia._conectar() as conexao:
        conexao.execute(
            "CREATE TABLE IF NOT EXISTS conversas (thread_id TEXT PRIMARY KEY, atualizada_em DOUBLE PRECISION NOT NULL)"
        )
        antigas = [t for (t,) in conexao.execute("SELECT thread_id FROM conversas WHERE atualizada_em < %s", (limite,)).fetchall()]
    for thread_id in antigas:
        saver.delete_thread(thread_id)
        persistencia.gravar("DELETE FROM conversas WHERE thread_id = %s", (thread_id,))
    return len(antigas)


def criar_checkpointer():
    """PostgresSaver quando há banco; InMemorySaver quando não há (ou quando ele falhou)."""
    global _pool
    if not persistencia.ativo() or _estado["desligada"]:
        _estado["tipo"] = "memória do servidor"
        return InMemorySaver()
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg_pool import ConnectionPool

        if _pool is None:
            _pool = ConnectionPool(
                persistencia.URL,
                min_size=1,
                max_size=4,
                kwargs={"autocommit": True, "prepare_threshold": None, "connect_timeout": 10},
                check=ConnectionPool.check_connection,  # o Neon fecha conexões paradas: testa antes de usar
                open=True,
                timeout=15,
            )
            atexit.register(fechar)
        saver = PostgresSaver(_pool)
        saver.setup()
        apagadas = _limpar_antigas(saver)
        _estado["tipo"] = "banco"
        log.info("Memória das conversas no banco (%d conversa(s) antigas apagadas)", apagadas)
        return saver
    except Exception as exc:
        log.warning("Memória no banco indisponível (%s): usando a memória do servidor", type(exc).__name__)
        _estado["tipo"] = "memória do servidor"
        return InMemorySaver()


def registrar_uso(thread_id: str) -> None:
    """Marca a conversa como usada agora (para a limpeza das antigas)."""
    if _estado["tipo"] == "banco":
        persistencia.gravar(
            "INSERT INTO conversas (thread_id, atualizada_em) VALUES (%s, %s) "
            "ON CONFLICT (thread_id) DO UPDATE SET atualizada_em = EXCLUDED.atualizada_em",
            (thread_id, time.time()),
        )


def falha_do_banco(exc: Exception) -> bool:
    """True se o erro veio do banco da memória (e não do modelo ou da busca)."""
    if _estado["tipo"] != "banco":
        return False
    try:
        import psycopg
        from psycopg_pool import PoolTimeout

        return isinstance(exc, (psycopg.Error, PoolTimeout))
    except Exception:
        return False


def desligar() -> None:
    """Depois de uma falha do banco: o resto da execução usa a memória do servidor."""
    _estado["desligada"] = True
    _estado["tipo"] = "memória do servidor"
    log.warning("Memória no banco desligada após falha; usando a memória do servidor")


def fechar() -> None:
    """Fecha as conexões da memória (na saída do processo)."""
    global _pool
    if _pool is not None:
        try:
            _pool.close(timeout=2)
        except Exception:
            pass
        _pool = None
