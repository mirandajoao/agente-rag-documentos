"""Pipeline de indexação RAG.

Etapas:
1. Ingestão documental  -> PDF (PyPDFLoader), Word (.docx), PowerPoint (.pptx) e texto (.txt/.md)
2. Fatiamento           -> RecursiveCharacterTextSplitter com separadores estruturais
3. Vetorização          -> OpenAI text-embedding-3-small (ou Gemini, ver app/provedor.py)
4. Armazenamento        -> ChromaDB persistido localmente em chroma_db/

Os documentos vêm de duas origens, separadas pelo metadado "espaco":
- "exemplos": PDFs de data/pdfs/, indexados na subida da aplicação e visíveis a todos;
- <espaco_id>: arquivos enviados por um usuário na interface, visíveis só para ele.

Uso:
    python -m app.ingest            # indexa os exemplos (pula se já existir índice)
    python -m app.ingest --rebuild  # apaga e reconstrói o índice
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import shutil
import sys
import threading
from bisect import bisect_right
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import exigir_chave_api, settings

log = logging.getLogger("ingest")

ESPACO_EXEMPLOS = "exemplos"
EXTENSOES_ACEITAS = {".pdf", ".docx", ".pptx", ".xlsx", ".csv", ".txt", ".md"}
FORMATOS_ACEITOS = "PDF, DOCX, PPTX, XLSX, CSV, TXT ou MD"

# Separadores em ordem de prioridade. Primeiro as fronteiras "fortes" de
# qualquer documento (títulos em Markdown, capítulos, seções e artigos de
# normas), depois parágrafos, linhas, frases e palavras. Em um texto comum os
# marcadores de norma simplesmente não aparecem e o corte cai nos parágrafos;
# em uma norma, cada artigo tende a ficar inteiro em um único trecho.
SEPARADORES = [
    "\n# ",
    "\n## ",
    "\n### ",
    "\nTÍTULO ",
    "\nCAPÍTULO ",
    "\nCapítulo ",
    "\nSeção ",
    "\nSEÇÃO ",
    "\nArt. ",
    "\n§ ",
    "\n\n",
    "\n",
    ". ",
    "; ",
    " ",
    "",
]

# Nome amigável dos documentos de exemplo, inferido pelo nome do arquivo.
TITULOS_EXEMPLOS = {
    "apostila_ia": "Apostila – Introdução à Inteligência Artificial e RAG",
    "944": "Resolução CONSUP nº 944/2023 – Regulamento Didático-Pedagógico da Graduação",
    "945": "Resolução CONSUP nº 945/2023 – RDP do Ensino da Educação Básica",
    "ppc": "PPC – Bacharelado em Ciência da Computação (IFPA Ananindeua)",
    "guia": "Guia Acadêmico IFPA 2025",
    "calendario": "Calendário Acadêmico IFPA 2026",
}

# Escritas concorrentes no ChromaDB (vários uploads ao mesmo tempo) passam por aqui.
_trava_escrita = threading.Lock()


class DocumentoInvalido(ValueError):
    """Arquivo que não pode ser indexado (sem texto, grande demais, formato errado)."""


def titulo_amigavel(caminho: Path) -> str:
    return re.sub(r"\s+", " ", caminho.stem.replace("_", " ").replace("-", " ")).strip() or caminho.name


def titulo_do_exemplo(caminho: Path) -> str:
    nome = caminho.stem.lower()
    for chave, titulo in TITULOS_EXEMPLOS.items():
        if chave in nome:
            return titulo
    return titulo_amigavel(caminho)


def limpar_texto(texto: str) -> str:
    """Normaliza artefatos comuns de extração de PDF sem perder a estrutura."""
    texto = texto.replace("\x00", "")
    texto = re.sub(r"-\n(?=[a-zà-ú])", "", texto)  # hifenização no fim de linha
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


def _paginas_docx(caminho: Path) -> list[tuple[int, str]]:
    """Word não tem páginas fixas: o documento inteiro vira a "página" 1.

    Títulos (estilos Heading/Título) ganham "# " para o fatiamento respeitar as seções.
    """
    from docx import Document as Docx

    doc = Docx(str(caminho))
    linhas: list[str] = []
    for par in doc.paragraphs:
        texto = par.text.strip()
        if not texto:
            continue
        estilo = (par.style.name if par.style is not None else "").lower()
        linhas.append(f"# {texto}" if estilo.startswith(("heading", "título", "titulo", "title")) else texto)
    for tabela in doc.tables:
        for linha in tabela.rows:
            celulas = [c.text.strip() for c in linha.cells if c.text.strip()]
            if celulas:
                linhas.append(" | ".join(dict.fromkeys(celulas)))
    return [(1, "\n\n".join(linhas))]


def _paginas_pptx(caminho: Path) -> list[tuple[int, str]]:
    """PowerPoint: cada slide vira uma "página" (o número citado é o do slide)."""
    from pptx import Presentation

    paginas: list[tuple[int, str]] = []
    for numero, slide in enumerate(Presentation(str(caminho)).slides, start=1):
        partes: list[str] = []
        for forma in slide.shapes:
            if getattr(forma, "has_text_frame", False) and forma.text_frame.text.strip():
                partes.append(forma.text_frame.text.strip())
            if getattr(forma, "has_table", False):
                for linha in forma.table.rows:
                    celulas = [c.text.strip() for c in linha.cells if c.text.strip()]
                    if celulas:
                        partes.append(" | ".join(celulas))
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame.text.strip():
            partes.append("Notas: " + slide.notes_slide.notes_text_frame.text.strip())
        if partes:
            paginas.append((numero, f"# Slide {numero}\n" + "\n".join(partes)))
    return paginas


def _ocr_paginas(caminho: Path, numeros: list[int]) -> dict[int, str]:
    """OCR (Tesseract) das páginas indicadas de um PDF escaneado. Devolve {página: texto}.

    Sem o Tesseract instalado, devolve {} e o arquivo segue o fluxo normal.
    """
    if not numeros:
        return {}
    try:
        import pypdfium2 as pdfium
        import pytesseract
    except ImportError:
        log.warning("OCR indisponível: instale pypdfium2 e pytesseract")
        return {}
    textos: dict[int, str] = {}
    try:
        pdf = pdfium.PdfDocument(str(caminho))
        for numero in numeros[: settings.max_paginas_ocr]:
            imagem = pdf[numero - 1].render(scale=300 / 72).to_pil()  # ~300 dpi
            textos[numero] = pytesseract.image_to_string(imagem, lang=settings.ocr_idiomas)
        pdf.close()
    except Exception as exc:  # sem tesseract, idioma ausente, PDF estranho: segue sem OCR
        log.warning("OCR falhou em %s: %s", caminho.name, exc)
    return textos


def _linhas_em_paginas(titulo: str, cabecalho: list[str], linhas: list[list[str]], inicio_pagina: int) -> list[tuple[int, str]]:
    """Planilha -> "páginas" de 40 linhas, cada linha como "Coluna: valor | ...".

    Repetir o nome da coluna em cada linha deixa o trecho compreensível sozinho
    (o fatiamento pode separar a linha do cabeçalho).
    """
    paginas: list[tuple[int, str]] = []
    for i in range(0, len(linhas), 40):
        bloco = linhas[i : i + 40]
        texto = [f"# {titulo} (linhas {i + 2} a {i + 1 + len(bloco)})"]
        for linha in bloco:
            pares = [f"{c or f'Coluna {j + 1}'}: {v}" for j, (c, v) in enumerate(zip(cabecalho + [""] * len(linha), linha)) if v]
            if pares:
                texto.append(" | ".join(pares))
        paginas.append((inicio_pagina + len(paginas), "\n".join(texto)))
    return paginas


def _paginas_planilha(caminho: Path) -> list[tuple[int, str]]:
    """Excel (cada aba) ou CSV. A "página" citada é o bloco de 40 linhas."""
    limite = settings.max_linhas_planilha
    tabelas: list[tuple[str, list[list[str]]]] = []
    if caminho.suffix.lower() == ".xlsx":
        from openpyxl import load_workbook

        livro = load_workbook(str(caminho), read_only=True, data_only=True)
        for aba in livro.worksheets:
            linhas = []
            for linha in aba.iter_rows(values_only=True):
                valores = ["" if v is None else str(v).strip() for v in linha]
                if any(valores):
                    linhas.append(valores)
                if len(linhas) > limite:
                    break
            tabelas.append((f"Planilha {aba.title}", linhas))
        livro.close()
    else:
        import csv

        bruto = caminho.read_bytes()
        try:
            texto = bruto.decode("utf-8-sig")
        except UnicodeDecodeError:
            texto = bruto.decode("latin-1")
        try:
            dialeto = csv.Sniffer().sniff(texto[:5000], delimiters=",;\t|")
        except csv.Error:
            dialeto = csv.excel
        linhas = [[c.strip() for c in l] for l in csv.reader(texto.splitlines(), dialeto) if any(c.strip() for c in l)]
        tabelas.append((caminho.stem, linhas[: limite + 1]))
    paginas: list[tuple[int, str]] = []
    for nome, linhas in tabelas:
        if len(linhas) < 1:
            continue
        cabecalho, corpo = linhas[0], linhas[1:] or [linhas[0]]
        paginas += _linhas_em_paginas(nome, cabecalho, corpo, len(paginas) + 1)
    return paginas


def ler_arquivo(caminho: Path, titulo: str, espaco: str, max_paginas: int | None = None) -> Document:
    """Lê um arquivo e devolve UM Document com o texto inteiro.

    Fatiar o documento inteiro (e não página a página) evita cortar um parágrafo
    ou artigo na virada de página. A página de cada trecho é recuperada depois
    pelos offsets guardados em metadata["inicios_pagina"].
    """
    ext = caminho.suffix.lower()
    paginas_ocr = 0
    if ext not in EXTENSOES_ACEITAS:
        raise DocumentoInvalido(f"Formato {ext or '(sem extensão)'} não suportado. Envie {FORMATOS_ACEITOS}.")

    if ext == ".pdf":
        try:
            paginas = [(int(d.metadata.get("page", 0)) + 1, d.page_content) for d in PyPDFLoader(str(caminho)).load()]
        except Exception as exc:
            raise DocumentoInvalido(f"Não foi possível ler o PDF ({type(exc).__name__}). O arquivo pode estar corrompido ou protegido.") from exc
        if max_paginas and len(paginas) > max_paginas:
            raise DocumentoInvalido(f"O PDF tem {len(paginas)} páginas; o limite é {max_paginas}.")
        # Páginas sem texto (escaneadas): reconhece o texto da imagem com OCR.
        vazias = [n for n, t in paginas if len(limpar_texto(t)) < 30]
        if vazias:
            reconhecidas = _ocr_paginas(caminho, vazias)
            paginas = [(n, reconhecidas.get(n) or t) for n, t in paginas]
            paginas_ocr = sum(1 for t in reconhecidas.values() if len(limpar_texto(t)) >= 30)
    elif ext in {".docx", ".pptx"}:
        try:
            paginas = _paginas_docx(caminho) if ext == ".docx" else _paginas_pptx(caminho)
        except Exception as exc:
            raise DocumentoInvalido(f"Não foi possível ler o arquivo ({type(exc).__name__}). Ele pode estar corrompido ou protegido.") from exc
        if max_paginas and len(paginas) > max_paginas:
            raise DocumentoInvalido(f"A apresentação tem {len(paginas)} slides; o limite é {max_paginas}.")
    elif ext in {".xlsx", ".csv"}:
        try:
            paginas = _paginas_planilha(caminho)
        except Exception as exc:
            raise DocumentoInvalido(f"Não foi possível ler a planilha ({type(exc).__name__}). Ela pode estar corrompida ou protegida.") from exc
    else:
        bruto = caminho.read_bytes()
        try:
            texto = bruto.decode("utf-8")
        except UnicodeDecodeError:
            texto = bruto.decode("latin-1")
        paginas = [(1, texto)]  # arquivo de texto = uma "página"

    partes: list[str] = []
    inicios: list[tuple[int, int]] = []  # (offset no texto completo, página humana)
    offset = 0
    for numero, conteudo in paginas:
        texto = limpar_texto(conteudo)
        if len(texto) < 30:  # páginas em branco / só cabeçalho
            continue
        inicios.append((offset, numero))
        partes.append(texto)
        offset += len(texto) + 1  # +1 pela quebra de linha do join

    if not partes:
        raise DocumentoInvalido(
            "Não encontrei texto no arquivo, nem com OCR. Se for um PDF escaneado com baixa qualidade ou slides só com imagens, o texto precisa ser digitado ou reconhecido antes."
        )
    return Document(
        page_content="\n".join(partes),
        metadata={"fonte": titulo, "arquivo": caminho.name, "espaco": espaco, "inicios_pagina": inicios, "paginas_ocr": paginas_ocr},
    )


def carregar_exemplos(pasta: Path) -> list[Document]:
    """Lê os documentos de exemplo de data/pdfs/. Pasta vazia não é erro."""
    arquivos = sorted(p for p in pasta.glob("*") if p.suffix.lower() in EXTENSOES_ACEITAS) if pasta.exists() else []
    if not arquivos:
        log.warning("Nenhum documento de exemplo em %s — a base começa vazia e recebe só uploads.", pasta)
        return []

    documentos: list[Document] = []
    for arq in arquivos:
        try:
            doc = ler_arquivo(arq, titulo_do_exemplo(arq), ESPACO_EXEMPLOS)
        except DocumentoInvalido as exc:  # um arquivo ruim não derruba a indexação inteira
            log.error("Ignorando %s: %s", arq.name, exc)
            continue
        log.info("  %-55s %3d páginas com texto", arq.name, len(doc.metadata["inicios_pagina"]))
        documentos.append(doc)
    return documentos


RE_ARTIGO = re.compile(r"^\s*Art\.\s*(\d+)", re.MULTILINE)


def _pagina_do_offset(inicios: list[tuple[int, int]], pos: int) -> int:
    i = bisect_right([o for o, _ in inicios], pos) - 1
    return inicios[max(i, 0)][1]


def fatiar(documentos: list[Document], tamanho: int | None = None) -> list[Document]:
    tamanho = tamanho or settings.chunk_size
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=tamanho,
        chunk_overlap=min(settings.chunk_overlap, tamanho // 4),
        separators=SEPARADORES,
        keep_separator="start",
        length_function=len,
        add_start_index=True,
    )
    chunks: list[Document] = []
    for doc in documentos:
        texto = doc.page_content
        inicios = doc.metadata["inicios_pagina"]
        espaco = doc.metadata.get("espaco", ESPACO_EXEMPLOS)
        artigos = [(m.start(), int(m.group(1))) for m in RE_ARTIGO.finditer(texto)]
        posicoes = [p for p, _ in artigos]

        for c in splitter.create_documents([texto]):
            inicio = c.metadata["start_index"]
            meta = {
                "fonte": doc.metadata["fonte"],
                "arquivo": doc.metadata["arquivo"],
                "espaco": espaco,
                "pagina": _pagina_do_offset(inicios, inicio),
                "start_index": inicio,
            }
            conteudo = c.page_content.strip()
            # Só para normas: artigo em vigor no início do trecho (último "Art. N" antes dele).
            i = bisect_right(posicoes, inicio + 2) - 1
            if i >= 0:
                meta["artigo"] = artigos[i][1]
                if not conteudo.startswith("Art."):
                    # Contexto explícito evita que o modelo "chute" o número do artigo
                    # quando o trecho começa no meio dele (ex.: "§ 2º ...").
                    conteudo = f"(continuação do Art. {artigos[i][1]}) {conteudo}"
            # ID determinístico: reindexar o mesmo conteúdo não duplica vetores.
            meta_id = f"{espaco}|{meta['arquivo']}|{inicio}"
            chunks.append(Document(page_content=conteudo, metadata=meta, id=hashlib.sha1(meta_id.encode()).hexdigest()))
    return chunks


def obter_embeddings():
    from app.provedor import criar_embeddings

    return criar_embeddings()


def abrir_vectorstore() -> Chroma:
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=obter_embeddings(),
        persist_directory=str(settings.chroma_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )


def indice_existe() -> bool:
    try:
        return abrir_vectorstore()._collection.count() > 0
    except Exception:
        return False


def _gravar(vs: Chroma, chunks: list[Document]) -> None:
    lote = 200  # lotes pequenos evitam estourar o limite de tokens por requisição
    for i in range(0, len(chunks), lote):
        parte = chunks[i : i + lote]
        vs.add_documents(parte, ids=[c.id for c in parte])
        log.info("    %d/%d trechos indexados", min(i + lote, len(chunks)), len(chunks))


def construir_indice(rebuild: bool = False) -> int:
    """Indexa os documentos de exemplo de data/pdfs/."""
    exigir_chave_api()

    if rebuild and settings.chroma_dir.exists():
        log.info("Removendo índice anterior em %s", settings.chroma_dir)
        shutil.rmtree(settings.chroma_dir)

    if not rebuild and indice_existe():
        n = abrir_vectorstore()._collection.count()
        log.info("Índice já existe com %d trechos. Use --rebuild para recriar.", n)
        return n

    log.info("1/4 Ingestão dos documentos de exemplo em %s", settings.pdf_dir)
    documentos = carregar_exemplos(settings.pdf_dir)

    log.info("2/4 Fatiamento (chunk_size=%d, chunk_overlap=%d)", settings.chunk_size, settings.chunk_overlap)
    chunks = fatiar(documentos)
    tokens_estimados = sum(len(c.page_content) for c in chunks) // 4
    log.info(
        "    %d trechos | ~%d tokens | custo estimado de embedding ≈ US$ %.4f",
        len(chunks), tokens_estimados, tokens_estimados / 1_000_000 * 0.02,
    )

    log.info("3/4 Vetorização com %s", settings.embedding_model)
    log.info("4/4 Persistindo no ChromaDB em %s", settings.chroma_dir)
    vs = abrir_vectorstore()
    if chunks:
        with _trava_escrita:
            _gravar(vs, chunks)
    return vs._collection.count()


# ---------------------------------------------------------------------------
# Documentos enviados pelos usuários
# ---------------------------------------------------------------------------

def indexar_arquivo(caminho: Path, espaco: str, titulo: str, tamanho_trecho: int | None = None) -> int:
    """Lê, fatia e indexa um arquivo enviado. Devolve o número de trechos criados."""
    return indexar_arquivo_detalhado(caminho, espaco, titulo, tamanho_trecho)[0]


def indexar_arquivo_detalhado(caminho: Path, espaco: str, titulo: str, tamanho_trecho: int | None = None) -> tuple[int, int]:
    """Como indexar_arquivo, mas devolve (trechos criados, páginas lidas por OCR)."""
    exigir_chave_api()
    doc = ler_arquivo(caminho, titulo, espaco, max_paginas=settings.max_upload_paginas)
    chunks = fatiar([doc], tamanho_trecho)
    log.info("Upload [%s] %s: %d páginas, %d trechos", espaco[:8], caminho.name, len(doc.metadata["inicios_pagina"]), len(chunks))
    with _trava_escrita:
        # Reenviar um arquivo com o mesmo nome substitui a versão anterior.
        abrir_vectorstore()._collection.delete(where={"$and": [{"espaco": espaco}, {"arquivo": caminho.name}]})
        _gravar(abrir_vectorstore(), chunks)
    _guardar_no_banco(espaco, caminho, titulo, chunks)
    return len(chunks), int(doc.metadata.get("paginas_ocr") or 0)


def _guardar_no_banco(espaco: str, caminho: Path, titulo: str, chunks: list[Document]) -> None:
    """Com DATABASE_URL, guarda o arquivo e os trechos (com os embeddings já calculados) no Postgres."""
    from app import acervo, persistencia

    if not persistencia.ativo() or not chunks:
        return
    try:
        dados = abrir_vectorstore()._collection.get(ids=[c.id for c in chunks], include=["documents", "metadatas", "embeddings"])
        acervo.salvar(espaco, caminho, titulo, list(dados["ids"]), list(dados["documents"]), list(dados["metadatas"]), list(dados["embeddings"]))
    except Exception:  # o banco é um extra: o documento já está pronto para perguntas
        log.exception("Falha ao guardar o documento no banco")


def remover_documentos(espaco: str, arquivo: str | None = None) -> None:
    """Apaga do índice os trechos de um arquivo (ou de todos) de um espaço."""
    if espaco == ESPACO_EXEMPLOS:
        raise DocumentoInvalido("Os documentos de exemplo não podem ser removidos.")
    filtro = {"espaco": espaco} if not arquivo else {"$and": [{"espaco": espaco}, {"arquivo": arquivo}]}
    with _trava_escrita:
        abrir_vectorstore()._collection.delete(where=filtro)
    from app import acervo

    acervo.remover(espaco, arquivo)


def documentos_do_espaco(espaco: str, incluir_exemplos: bool = True) -> list[dict]:
    """Lista os documentos visíveis para um espaço: os próprios e (opcionalmente) os de exemplo."""
    espacos = [espaco] + ([ESPACO_EXEMPLOS] if incluir_exemplos else [])
    dados = abrir_vectorstore()._collection.get(where={"espaco": {"$in": espacos}}, include=["metadatas"])
    vistos: dict[tuple[str, str], dict] = {}
    for m in dados.get("metadatas") or []:
        chave = (m.get("espaco", ""), m.get("arquivo", ""))
        item = vistos.setdefault(
            chave,
            {"documento": m.get("fonte", "?"), "arquivo": m.get("arquivo", "?"), "exemplo": chave[0] == ESPACO_EXEMPLOS, "trechos": 0, "paginas": 0},
        )
        item["trechos"] += 1
        item["paginas"] = max(item["paginas"], int(m.get("pagina") or 0))
    return sorted(vistos.values(), key=lambda d: (d["exemplo"], d["documento"].lower()))


def trechos_do_documento(espaco: str, arquivo: str) -> list[Document]:
    """Todos os trechos de um documento, na ordem em que aparecem no texto."""
    dados = abrir_vectorstore()._collection.get(
        where={"$and": [{"espaco": espaco}, {"arquivo": arquivo}]}, include=["documents", "metadatas"]
    )
    docs = [Document(page_content=t, metadata=m or {}) for t, m in zip(dados["documents"], dados["metadatas"])]
    return sorted(docs, key=lambda d: int(d.metadata.get("start_index") or 0))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Indexa os documentos de exemplo no ChromaDB.")
    parser.add_argument("--rebuild", action="store_true", help="apaga e reconstrói o índice")
    args = parser.parse_args()
    try:
        total = construir_indice(rebuild=args.rebuild)
    except Exception as exc:
        log.error("Erro na indexação: %s", exc)
        sys.exit(1)
    log.info("Pronto: %d trechos no índice '%s'.", total, settings.collection_name)


if __name__ == "__main__":
    main()
