"""Mapa dos embeddings: os trechos como pontos em 2D (modo aula).

Cada trecho é um vetor de 1.536 números (text-embedding-3-small). Para desenhar,
a projeção PCA escolhe as 2 direções em que os trechos mais variam. A pergunta é
projetada no mesmo plano: os trechos mais parecidos com ela são destacados.
A distância no desenho é aproximada (2 de 1.536 dimensões); a semelhança
mostrada no destaque é a de verdade (cosseno no espaço completo).
"""

from __future__ import annotations

import numpy as np

from app.config import settings
from app.ingest import ESPACO_EXEMPLOS, abrir_vectorstore, obter_embeddings

MAX_PONTOS = 600


def montar_mapa(espaco: str, pergunta: str = "", exemplos: bool = True, top_k: int | None = None) -> dict:
    espacos = [espaco] + ([ESPACO_EXEMPLOS] if exemplos and settings.incluir_exemplos else [])
    dados = abrir_vectorstore()._collection.get(
        where={"espaco": {"$in": espacos}}, include=["embeddings", "metadatas", "documents"]
    )
    vetores = dados.get("embeddings")
    if vetores is None or len(vetores) == 0:
        return {"pontos": [], "pergunta": None, "documentos": []}
    vetores = np.asarray(vetores, dtype=np.float32)
    metas, textos = dados["metadatas"], dados["documents"]

    # Muitos trechos: amostra espaçada (o desenho fica legível e a conta, rápida).
    idx = np.arange(len(vetores))
    if len(idx) > MAX_PONTOS:
        idx = np.linspace(0, len(vetores) - 1, MAX_PONTOS).astype(int)
    base = vetores[idx]

    consulta = None
    if pergunta.strip():
        consulta = np.asarray(obter_embeddings().embed_query(pergunta.strip()), dtype=np.float32)

    # PCA: centraliza e usa os 2 primeiros componentes (SVD).
    media = base.mean(axis=0)
    _, _, vt = np.linalg.svd(base - media, full_matrices=False)
    eixos = vt[:2]
    xy = (base - media) @ eixos.T
    escala = float(np.abs(xy).max()) or 1.0

    proximos: dict[int, float] = {}
    if consulta is not None:
        normas = np.linalg.norm(vetores, axis=1) * (np.linalg.norm(consulta) or 1)
        cos = (vetores @ consulta) / np.where(normas == 0, 1, normas)
        for i in np.argsort(-cos)[: (top_k or settings.top_k)]:
            proximos[int(i)] = float(cos[i])
        # Garante que os mais parecidos apareçam mesmo fora da amostra.
        faltam = [i for i in proximos if i not in set(idx.tolist())]
        if faltam:
            idx = np.concatenate([idx, np.asarray(faltam)])
            xy = np.vstack([xy, (vetores[faltam] - media) @ eixos.T])

    documentos: list[str] = []
    pontos = []
    for pos, i in enumerate(idx.tolist()):
        m = metas[i] or {}
        doc = m.get("fonte", "?")
        if doc not in documentos:
            documentos.append(doc)
        pontos.append({
            "x": round(float(xy[pos][0]) / escala, 4),
            "y": round(float(xy[pos][1]) / escala, 4),
            "doc": documentos.index(doc),
            "pagina": m.get("pagina"),
            "texto": (textos[i] or "")[:160],
            "semelhanca": round(proximos[i], 3) if i in proximos else None,
        })
    ponto_pergunta = None
    if consulta is not None:
        p = (consulta - media) @ eixos.T
        ponto_pergunta = {"x": round(float(p[0]) / escala, 4), "y": round(float(p[1]) / escala, 4), "texto": pergunta.strip()[:160]}
    return {"pontos": pontos, "pergunta": ponto_pergunta, "documentos": documentos}
