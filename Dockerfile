FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# OCR de PDFs escaneados (Tesseract com português e inglês)
RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-por tesseract-ocr-eng \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY static ./static
COPY data ./data

# Usuário sem privilégios
RUN useradd --create-home agente && chown -R agente /app
USER agente

# O índice ChromaDB é construído na primeira subida (precisa da OPENAI_API_KEY).
# O Railway injeta $PORT; localmente cai para 8000.
CMD ["sh", "-c", "uvicorn app.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
