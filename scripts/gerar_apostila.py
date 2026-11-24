"""Gera data/pdfs/apostila_ia_oficina.pdf a partir de docs/apostila_ia_oficina.md.

Cada capítulo ("## ...") começa numa página nova, para as citações mostrarem
páginas diferentes. Uso: python scripts/gerar_apostila.py
"""

from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

RAIZ = Path(__file__).resolve().parent.parent
ORIGEM = RAIZ / "docs" / "apostila_ia_oficina.md"
DESTINO = RAIZ / "data" / "pdfs" / "apostila_ia_oficina.pdf"

AZUL, TINTA, CINZA = HexColor("#2F5BFF"), HexColor("#0B1530"), HexColor("#4A5571")
TITULO = ParagraphStyle("titulo", fontName="Helvetica-Bold", fontSize=20, leading=25, textColor=TINTA, spaceAfter=10)
CAPITULO = ParagraphStyle("capitulo", fontName="Helvetica-Bold", fontSize=15, leading=19, textColor=AZUL, spaceAfter=10)
CORPO = ParagraphStyle("corpo", fontName="Helvetica", fontSize=11, leading=16, textColor=TINTA, alignment=TA_JUSTIFY, spaceAfter=9)
DESTAQUE = ParagraphStyle("destaque", parent=CORPO, fontName="Helvetica-Bold", textColor=AZUL, spaceBefore=4)
INTRO = ParagraphStyle("intro", parent=CORPO, textColor=CINZA)


def rodape(canvas, doc):
    # Só o número da página, desenhado como forma (não como texto): assim ele não
    # "suja" o texto extraído de cada página nem os trechos indexados.
    canvas.saveState()
    canvas.setFillColor(AZUL)
    canvas.circle(A4[0] / 2, 1.3 * cm, 0.12 * cm, stroke=0, fill=1)
    canvas.restoreState()


def main() -> None:
    historia, primeiro_capitulo = [], True
    for linha in ORIGEM.read_text(encoding="utf-8").splitlines():
        linha = linha.strip()
        if not linha:
            continue
        if linha.startswith("## "):
            if not primeiro_capitulo:
                historia.append(PageBreak())
            primeiro_capitulo = False
            historia.append(Paragraph(linha[3:], CAPITULO))
        elif linha.startswith("# "):
            historia += [Paragraph(linha[2:], TITULO), Spacer(1, 4)]
        elif linha.startswith("Ponto-chave:"):
            historia.append(Paragraph(linha, DESTAQUE))
        else:
            historia.append(Paragraph(linha, INTRO if primeiro_capitulo else CORPO))
    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(
        str(DESTINO), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=2.2 * cm, bottomMargin=2.2 * cm,
        title="Apostila - Introdução à Inteligência Artificial e RAG", author="Oficina Converse com os seus documentos",
    ).build(historia, onFirstPage=rodape, onLaterPages=rodape)
    print(f"Gerado: {DESTINO}")


if __name__ == "__main__":
    main()
