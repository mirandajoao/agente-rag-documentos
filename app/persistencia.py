"""Persistência opcional no Postgres (ex.: Neon): o que hoje zera a cada deploy passa a sobreviver.

Guarda:
- o gasto de cada dia (orçamento) e o limite ajustado pelo administrador;
- os controles do Painel do Administrador (pausas);
- as perguntas do painel da turma (sem identificar ninguém) e quando ele foi limpo;
- os eventos de perguntas e envios (para os números do administrador).

Como funciona: durante a execução tudo continua em memória (rápido); cada mudança
é gravada no banco por uma thread em segundo plano e, quando o servidor sobe,
o estado é carregado de volta. Sem DATABASE_URL (ou com o banco fora do ar), o
app segue exatamente como antes, só em memória: o banco nunca derruba a aula.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time

log = logging.getLogger("persistencia")

URL = os.getenv("DATABASE_URL", "").strip()
_fila: queue.Queue = queue.Queue(maxsize=5000)
_thread: threading.Thread | None = None
_ativo = False

TABELAS = [
    """CREATE TABLE IF NOT EXISTS uso_diario (
        dia DATE PRIMARY KEY,
        gasto_usd DOUBLE PRECISION NOT NULL DEFAULT 0,
        atualizado_em TIMESTAMPTZ NOT NULL DEFAULT now()
    )""",
    """CREATE TABLE IF NOT EXISTS controles (
        chave TEXT PRIMARY KEY,
        valor TEXT NOT NULL,
        atualizado_em TIMESTAMPTZ NOT NULL DEFAULT now()
    )""",
    """CREATE TABLE IF NOT EXISTS painel_perguntas (
        id TEXT PRIMARY KEY,
        hora DOUBLE PRECISION NOT NULL,
        navegador TEXT,
        pergunta TEXT NOT NULL,
        resposta TEXT,
        status TEXT,
        confianca TEXT,
        fontes TEXT,
        moderada BOOLEAN NOT NULL DEFAULT false,
        custo_usd DOUBLE PRECISION NOT NULL DEFAULT 0,
        oculta BOOLEAN NOT NULL DEFAULT false
    )""",
    "CREATE INDEX IF NOT EXISTS painel_perguntas_hora ON painel_perguntas (hora)",
    """CREATE TABLE IF NOT EXISTS eventos (
        id BIGSERIAL PRIMARY KEY,
        tipo TEXT NOT NULL,
        detalhe TEXT,
        hora DOUBLE PRECISION NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS eventos_tipo_hora ON eventos (tipo, hora)",
]


def ativo() -> bool:
    return _ativo


def _conectar():
    import psycopg

    # prepare_threshold=None: sem "prepared statements" automáticos, que podem falhar
    # atrás do pooler do Neon (PgBouncer em modo transação).
    return psycopg.connect(URL, connect_timeout=10, autocommit=True, prepare_threshold=None)


def _trabalhador() -> None:
    """Grava a fila no banco, reconectando quando a conexão cai."""
    conexao = None
    while True:
        sql, params = _fila.get()
        for tentativa in range(3):
            try:
                if conexao is None or conexao.closed:
                    conexao = _conectar()
                conexao.execute(sql, params)
                break
            except Exception as exc:  # banco fora do ar: tenta de novo e, se não der, segue
                log.warning("Falha ao gravar no banco (tentativa %d): %s", tentativa + 1, type(exc).__name__)
                try:
                    if conexao is not None:
                        conexao.close()
                except Exception:
                    pass
                conexao = None
                time.sleep(1 + tentativa)


def gravar(sql: str, params: tuple = ()) -> None:
    """Agenda uma escrita (não bloqueia a requisição). Sem banco, não faz nada."""
    if not _ativo:
        return
    try:
        _fila.put_nowait((sql, params))
    except queue.Full:
        log.warning("Fila de gravação cheia; descartando uma escrita")


# --- Escritas usadas pelos outros módulos ---------------------------------------

def salvar_gasto(dia: str, gasto_usd: float) -> None:
    gravar(
        "INSERT INTO uso_diario (dia, gasto_usd) VALUES (%s, %s) "
        "ON CONFLICT (dia) DO UPDATE SET gasto_usd = EXCLUDED.gasto_usd, atualizado_em = now()",
        (dia, gasto_usd),
    )


def salvar_controle(chave: str, valor) -> None:
    gravar(
        "INSERT INTO controles (chave, valor) VALUES (%s, %s) "
        "ON CONFLICT (chave) DO UPDATE SET valor = EXCLUDED.valor, atualizado_em = now()",
        (chave, json.dumps(valor)),
    )


def salvar_pergunta_painel(item: dict, navegador: str | None, custo_usd: float) -> None:
    gravar(
        "INSERT INTO painel_perguntas (id, hora, navegador, pergunta, resposta, status, confianca, fontes, moderada, custo_usd) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
        (item["id"], item["hora"], navegador, item["pergunta"], item.get("resposta"), item.get("status"),
         item.get("confianca"), json.dumps(item.get("fontes") or []), bool(item.get("moderada")), custo_usd),
    )


def ocultar_pergunta_painel(item_id: str) -> None:
    gravar("UPDATE painel_perguntas SET oculta = true WHERE id = %s", (item_id,))


def salvar_evento(tipo: str, detalhe: str | None = None, hora: float | None = None) -> None:
    gravar("INSERT INTO eventos (tipo, detalhe, hora) VALUES (%s, %s, %s)", (tipo, detalhe, hora or time.time()))


# --- Carga no início ------------------------------------------------------------

def _carregar(conexao) -> dict:
    """Lê o estado salvo. Devolve um dicionário que os módulos aplicam em memória."""
    estado: dict = {}
    estado["controles"] = {k: json.loads(v) for k, v in conexao.execute("SELECT chave, valor FROM controles").fetchall()}
    estado["gastos"] = {str(d): float(g) for d, g in conexao.execute("SELECT dia, gasto_usd FROM uso_diario").fetchall()}
    limpo_em = float(estado["controles"].get("painel_limpo_em") or 0)
    linhas = conexao.execute(
        "SELECT id, hora, navegador, pergunta, resposta, status, confianca, fontes, moderada, custo_usd, oculta "
        "FROM painel_perguntas WHERE hora > %s ORDER BY hora",
        (limpo_em,),
    ).fetchall()
    estado["painel"] = [
        {"id": r[0], "hora": float(r[1]), "navegador": r[2], "pergunta": r[3], "resposta": r[4], "status": r[5],
         "confianca": r[6], "fontes": json.loads(r[7] or "[]"), "moderada": bool(r[8]), "custo_usd": float(r[9] or 0),
         "oculta": bool(r[10])}
        for r in linhas
    ]
    desde = time.time() - 24 * 3600
    estado["perguntas"] = [float(h) for (h,) in conexao.execute(
        "SELECT hora FROM eventos WHERE tipo = 'pergunta' AND hora > %s ORDER BY hora", (desde,)
    ).fetchall()]
    estado["envios"] = {d or "?": int(n) for d, n in conexao.execute(
        "SELECT detalhe, count(*) FROM eventos WHERE tipo = 'envio' GROUP BY detalhe"
    ).fetchall()}
    return estado


def iniciar() -> bool:
    """Cria as tabelas, carrega o estado salvo nos módulos e liga a gravação em segundo plano."""
    global _ativo, _thread
    if not URL:
        log.info("Sem DATABASE_URL: uso, painel e controles ficam só em memória")
        return False
    try:
        with _conectar() as conexao:
            for sql in TABELAS:
                conexao.execute(sql)
            from app import acervo

            acervo.preparar(conexao)
            estado = _carregar(conexao)
    except Exception as exc:
        log.warning("Banco indisponível (%s): seguindo só em memória", type(exc).__name__)
        return False

    from app import controle, turma
    from app.orcamento import orcamento

    orcamento.restaurar(estado["gastos"], estado["controles"].get("limite_diario_usd"))
    controle.restaurar(estado["controles"], estado["perguntas"], estado["envios"])
    turma.restaurar(estado["painel"])

    _ativo = True
    if _thread is None:
        _thread = threading.Thread(target=_trabalhador, name="persistencia", daemon=True)
        _thread.start()
    log.info(
        "Banco conectado: %d pergunta(s) no painel, %d evento(s) nas últimas 24h, gasto de %d dia(s) restaurados",
        len(estado["painel"]), len(estado["perguntas"]), len(estado["gastos"]),
    )
    return True


def esperar_gravacoes(timeout: float = 5.0) -> None:
    """Para testes: espera a fila esvaziar."""
    fim = time.time() + timeout
    while not _fila.empty() and time.time() < fim:
        time.sleep(0.05)
    time.sleep(0.2)
