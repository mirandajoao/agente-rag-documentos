"""Documentos enviados guardados no Postgres (com DATABASE_URL): sobrevivem a deploys e reinícios.

O ChromaDB continua sendo o índice de busca (rápido, em memória e disco local),
mas o disco do Railway é apagado a cada deploy. Por isso, cada documento enviado
também vai para o banco:

- documentos_enviados: o arquivo original (para abrir na página citada);
- trechos_enviados: o texto, os metadados e o embedding de cada trecho, numa
  coluna `vector` do pgvector (ou REAL[], se o banco não tiver a extensão).

Quando o servidor sobe, os trechos voltam para o ChromaDB com os embeddings
salvos (sem pagar de novo pela vetorização) e os arquivos voltam para o disco.
Documentos mais antigos que DOCUMENTOS_RETENCAO_DIAS são apagados do banco.
Os de exemplo não entram: eles já vêm do repositório.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from app import persistencia
from app.config import settings

log = logging.getLogger("acervo")

_tipo_vetor = {"coluna": "vector"}  # "vector" (pgvector) ou "real[]"

TABELAS = [
    """CREATE TABLE IF NOT EXISTS documentos_enviados (
        espaco TEXT NOT NULL,
        arquivo TEXT NOT NULL,
        titulo TEXT,
        conteudo BYTEA NOT NULL,
        enviado_em DOUBLE PRECISION NOT NULL,
        PRIMARY KEY (espaco, arquivo)
    )""",
    """CREATE TABLE IF NOT EXISTS trechos_enviados (
        id TEXT PRIMARY KEY,
        espaco TEXT NOT NULL,
        arquivo TEXT NOT NULL,
        modelo TEXT NOT NULL,
        texto TEXT NOT NULL,
        metadados TEXT NOT NULL,
        embedding {coluna} NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS trechos_enviados_doc ON trechos_enviados (espaco, arquivo)",
]


def _vetor_sql(vetor) -> str:
    return "[" + ",".join(f"{float(x):.7g}" for x in vetor) + "]"


def _ler_vetor(valor) -> list[float]:
    if isinstance(valor, (list, tuple)):
        return [float(x) for x in valor]
    return [float(x) for x in str(valor).strip("[]{}").split(",") if x.strip()]


def preparar(conexao) -> None:
    """Cria as tabelas. Tenta o pgvector; sem ele, guarda o embedding como REAL[]."""
    try:
        conexao.execute("CREATE EXTENSION IF NOT EXISTS vector")
        _tipo_vetor["coluna"] = "vector"
    except Exception as exc:
        log.info("pgvector indisponível (%s): embeddings como REAL[]", type(exc).__name__)
        _tipo_vetor["coluna"] = "real[]"
    existente = conexao.execute(
        "SELECT udt_name FROM information_schema.columns WHERE table_name = 'trechos_enviados' AND column_name = 'embedding'"
    ).fetchone()
    if existente:
        _tipo_vetor["coluna"] = "vector" if existente[0] == "vector" else "real[]"
    for sql in TABELAS:
        conexao.execute(sql.replace("{coluna}", _tipo_vetor["coluna"]))


def _cast() -> str:
    return "%s::vector" if _tipo_vetor["coluna"] == "vector" else "%s::real[]"


def salvar(espaco: str, caminho: Path, titulo: str, ids: list[str], textos: list[str], metadados: list[dict], vetores) -> None:
    """Agenda a gravação do arquivo e dos trechos (substitui uma versão anterior com o mesmo nome)."""
    if not persistencia.ativo():
        return
    arquivo = caminho.name
    persistencia.gravar("DELETE FROM trechos_enviados WHERE espaco = %s AND arquivo = %s", (espaco, arquivo))
    persistencia.gravar(
        "INSERT INTO documentos_enviados (espaco, arquivo, titulo, conteudo, enviado_em) VALUES (%s, %s, %s, %s, %s) "
        "ON CONFLICT (espaco, arquivo) DO UPDATE SET titulo = EXCLUDED.titulo, conteudo = EXCLUDED.conteudo, enviado_em = EXCLUDED.enviado_em",
        (espaco, arquivo, titulo, caminho.read_bytes(), time.time()),
    )
    usar_vector = _tipo_vetor["coluna"] == "vector"
    for i, t, m, v in zip(ids, textos, metadados, vetores):
        persistencia.gravar(
            f"INSERT INTO trechos_enviados (id, espaco, arquivo, modelo, texto, metadados, embedding) VALUES (%s, %s, %s, %s, %s, %s, {_cast()}) "
            "ON CONFLICT (id) DO NOTHING",
            (i, espaco, arquivo, settings.embedding_model, t, json.dumps(m, ensure_ascii=False),
             _vetor_sql(v) if usar_vector else [float(x) for x in v]),
        )


def remover(espaco: str, arquivo: str | None = None) -> None:
    if not persistencia.ativo():
        return
    if arquivo:
        filtro, params = "espaco = %s AND arquivo = %s", (espaco, arquivo)
    else:
        filtro, params = "espaco = %s", (espaco,)
    persistencia.gravar(f"DELETE FROM trechos_enviados WHERE {filtro}", params)
    persistencia.gravar(f"DELETE FROM documentos_enviados WHERE {filtro}", params)


def restaurar() -> dict:
    """Devolve ao ChromaDB e ao disco os documentos salvos. Chamado na subida do servidor."""
    if not persistencia.ativo():
        return {"documentos": 0, "trechos": 0}
    from app.ingest import _trava_escrita, abrir_vectorstore

    limite = time.time() - settings.documentos_retencao_dias * 86400
    with persistencia._conectar() as conexao:
        vencidos = conexao.execute("SELECT espaco, arquivo FROM documentos_enviados WHERE enviado_em < %s", (limite,)).fetchall()
        for espaco, arquivo in vencidos:
            conexao.execute("DELETE FROM trechos_enviados WHERE espaco = %s AND arquivo = %s", (espaco, arquivo))
            conexao.execute("DELETE FROM documentos_enviados WHERE espaco = %s AND arquivo = %s", (espaco, arquivo))
        docs = conexao.execute("SELECT espaco, arquivo, conteudo FROM documentos_enviados").fetchall()
        linhas = conexao.execute(
            "SELECT t.id, t.texto, t.metadados, t.embedding::text FROM trechos_enviados t "
            "JOIN documentos_enviados d ON d.espaco = t.espaco AND d.arquivo = t.arquivo WHERE t.modelo = %s",
            (settings.embedding_model,),
        ).fetchall()

    for espaco, arquivo, conteudo in docs:
        pasta = settings.upload_dir / espaco
        pasta.mkdir(parents=True, exist_ok=True)
        (pasta / arquivo).write_bytes(bytes(conteudo))

    colecao = abrir_vectorstore()._collection
    lote = 200
    with _trava_escrita:
        for i in range(0, len(linhas), lote):
            parte = linhas[i : i + lote]
            colecao.upsert(
                ids=[r[0] for r in parte],
                documents=[r[1] for r in parte],
                metadatas=[json.loads(r[2]) for r in parte],
                embeddings=[_ler_vetor(r[3]) for r in parte],
            )
    if vencidos:
        log.info("%d documento(s) com mais de %d dias apagados do banco", len(vencidos), settings.documentos_retencao_dias)
    log.info("Documentos enviados restaurados do banco: %d arquivo(s), %d trecho(s)", len(docs), len(linhas))
    return {"documentos": len(docs), "trechos": len(linhas)}
