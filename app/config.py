"""Configuração central do agente.

Todos os parâmetros vêm de variáveis de ambiente (arquivo .env localmente ou
painel "Variables" no Railway). Nenhuma chave de API fica no código-fonte.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _int(nome: str, padrao: int) -> int:
    try:
        return int(os.getenv(nome, padrao))
    except ValueError:
        return padrao


def _float(nome: str, padrao: float) -> float:
    try:
        return float(os.getenv(nome, padrao))
    except ValueError:
        return padrao


# Provedor de IA: "openai" (padrão, pago e estável) ou "gemini" (Google, com plano gratuito).
PROVEDOR = os.getenv("PROVEDOR_IA", "openai").strip().lower()
GEMINI = PROVEDOR == "gemini"


@dataclass(frozen=True)
class Settings:
    # --- Credenciais (somente via ambiente) ---
    provedor: str = PROVEDOR
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")

    # --- Modelos ---
    # OpenAI (FinOps: os mais econômicos com bom tool calling):
    #   gpt-4o-mini: US$ 0,15 / 1M tokens de entrada e US$ 0,60 / 1M de saída.
    #   text-embedding-3-small: US$ 0,02 / 1M tokens, 1536 dimensões.
    # Gemini: os nomes mudam com frequência; confira o ID exato no Google AI Studio.
    chat_model: str = (
        os.getenv("GEMINI_MODEL", "gemini-flash-latest") if GEMINI else os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    )
    embedding_model: str = (
        os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
        if GEMINI
        else os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )
    temperature: float = _float("LLM_TEMPERATURE", 0.0)
    max_output_tokens: int = _int("LLM_MAX_TOKENS", 900)
    request_timeout: int = _int("LLM_TIMEOUT", 45)
    max_retries: int = _int("LLM_MAX_RETRIES", 2)

    # --- Fatiamento (justificativa FinOps no README / relatório) ---
    chunk_size: int = _int("CHUNK_SIZE", 1200)
    chunk_overlap: int = _int("CHUNK_OVERLAP", 200)

    # --- Recuperação ---
    top_k: int = _int("RETRIEVER_TOP_K", 4)
    bm25_k: int = _int("BM25_TOP_K", 3)
    # Distância de cosseno máxima aceita (0 = idêntico, 2 = oposto).
    # Trechos acima desse limiar são descartados como "sem evidência".
    max_distance: float = _float("RETRIEVER_MAX_DISTANCE", 0.75)

    # --- Agente ---
    recursion_limit: int = _int("AGENT_RECURSION_LIMIT", 12)

    # --- Blindagem de entrada ---
    max_question_chars: int = _int("MAX_QUESTION_CHARS", 1000)
    # Limites por navegador (espaço/sessão). Num laboratório, todos saem pelo mesmo IP público,
    # então o limite por IP é só um teto geral, bem mais alto.
    rate_limit_per_minute: int = _int("RATE_LIMIT_PER_MINUTE", 15)
    rate_limit_ip_por_minuto: int = _int("RATE_LIMIT_IP_PER_MINUTE", 300)
    rate_limit_upload_ip_por_minuto: int = _int("RATE_LIMIT_UPLOAD_IP_PER_MINUTE", 90)

    # --- Custos exibidos ao usuário ---
    # Cotação usada só para exibir o custo em reais (ajuste no painel de variáveis).
    usd_brl: float = _float("USD_BRL", 5.40)

    # --- URL pública (QR code). No Railway, RAILWAY_PUBLIC_DOMAIN é injetada automaticamente ---
    public_url: str = os.getenv("PUBLIC_URL", "") or (
        f"https://{os.environ['RAILWAY_PUBLIC_DOMAIN']}" if os.getenv("RAILWAY_PUBLIC_DOMAIN") else ""
    )

    # --- Upload de documentos pelo usuário ---
    max_upload_mb: int = _int("MAX_UPLOAD_MB", 15)
    max_upload_paginas: int = _int("MAX_UPLOAD_PAGINAS", 200)
    # OCR: páginas de PDF sem texto (escaneadas) passam pelo Tesseract, até este limite por arquivo.
    max_paginas_ocr: int = _int("MAX_PAGINAS_OCR", 30)
    ocr_idiomas: str = os.getenv("OCR_IDIOMAS", "por+eng")
    # Planilhas: no máximo estas linhas por arquivo (o resto é ignorado).
    max_linhas_planilha: int = _int("MAX_LINHAS_PLANILHA", 3000)
    # Painel da turma (/turma): mostra as perguntas recentes, sem identificar quem perguntou.
    # Resposta aparecendo palavra por palavra (STREAM_TOKENS=false volta a mostrar só a resposta pronta).
    stream_tokens: bool = os.getenv("STREAM_TOKENS", "true").strip().lower() in {"1", "true", "sim", "yes"}
    # Senha do professor no painel da turma (ocultar perguntas e limpar). Vazia = moderação manual desligada.
    # Senha dos ministrantes: Modo ministrante no app e moderação do painel. Vazia = recursos desligados.
    painel_senha: str = (os.getenv("MINISTRANTE_SENHA") or os.getenv("PAINEL_SENHA") or "").strip()
    # "Comparar modelos" (modo aula): dois modelos respondendo com os mesmos trechos.
    modelos_comparacao: str = os.getenv("MODELOS_COMPARACAO", "").strip()
    painel_turma: bool = os.getenv("PAINEL_TURMA", "true").strip().lower() in {"1", "true", "sim", "yes"}
    max_documentos_por_espaco: int = _int("MAX_DOCUMENTOS_POR_ESPACO", 5)
    rate_limit_upload_por_minuto: int = _int("RATE_LIMIT_UPLOAD_PER_MINUTE", 6)
    # Se verdadeiro, os documentos de exemplo (data/pdfs) também entram nas buscas de todos.
    incluir_exemplos: bool = os.getenv("INCLUIR_DOCUMENTOS_EXEMPLO", "true").strip().lower() in {"1", "true", "sim", "yes"}

    # --- Monitoramento de erros (opcional): DSN do projeto no Sentry. Vazio = desligado ---
    sentry_dsn: str = os.getenv("SENTRY_DSN", "").strip()

    # --- Reranking: reordena os trechos achados e descarta os que não ajudam a responder ---
    # "llm" (padrão): o mesmo modelo de chat dá uma nota de 0 a 10 a cada trecho; "off" desliga.
    rerank: str = os.getenv("RERANK", "llm").strip().lower()
    rerank_nota_minima: int = _int("RERANK_NOTA_MINIMA", 3)

    # --- Documentos enviados guardados no banco (com DATABASE_URL) por até N dias ---
    # Quem testa o app pelo link público não precisa que o arquivo fique guardado por muito tempo.
    documentos_retencao_dias: int = _int("DOCUMENTOS_RETENCAO_DIAS", 3)

    # --- Caminhos ---
    pdf_dir: Path = Path(os.getenv("PDF_DIR", BASE_DIR / "data" / "pdfs"))
    upload_dir: Path = Path(os.getenv("UPLOAD_DIR", BASE_DIR / "data" / "uploads"))
    chroma_dir: Path = Path(os.getenv("CHROMA_DIR", BASE_DIR / "chroma_db"))
    # Cada provedor gera embeddings de tamanhos diferentes, então cada um usa sua coleção.
    collection_name: str = os.getenv("CHROMA_COLLECTION", "documentos_gemini" if GEMINI else "documentos")


settings = Settings()


class ConfiguracaoInvalida(RuntimeError):
    """Falta a chave da API ou o provedor está errado: erro de configuração, não da pergunta."""


def exigir_chave_api() -> None:
    """Falha cedo, com mensagem clara, se a chave não estiver configurada."""
    if settings.provedor == "gemini":
        if not settings.google_api_key:
            raise ConfiguracaoInvalida(
                "GOOGLE_API_KEY não configurada. Gere uma chave gratuita no Google AI Studio "
                "(aistudio.google.com) e cadastre-a no .env ou em Variables no Railway."
            )
        return
    if settings.provedor != "openai":
        raise ConfiguracaoInvalida(f"PROVEDOR_IA inválido: {settings.provedor!r}. Use 'openai' ou 'gemini'.")
    if not settings.openai_api_key or not settings.openai_api_key.startswith("sk-"):
        raise ConfiguracaoInvalida(
            "OPENAI_API_KEY não configurada. Localmente, copie .env.example para .env "
            "e preencha a chave; no Railway, cadastre-a em Variables."
        )
