"""Fábrica de modelos: troca OpenAI <-> Gemini com uma variável de ambiente.

    PROVEDOR_IA=openai  (padrão) -> ChatOpenAI + OpenAIEmbeddings
    PROVEDOR_IA=gemini           -> ChatGoogleGenerativeAI + GoogleGenerativeAIEmbeddings

O resto da aplicação (agente, ferramentas, resumo, comparação) só chama estas
duas funções, então nada mais muda quando o provedor muda.
"""

from __future__ import annotations

from app.config import exigir_chave_api, settings


def criar_chat(temperatura: float | None = None, max_tokens: int | None = None):
    exigir_chave_api()
    temperatura = settings.temperature if temperatura is None else temperatura
    max_tokens = max_tokens or settings.max_output_tokens
    if settings.provedor == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.chat_model,
            google_api_key=settings.google_api_key,
            temperature=temperatura,
            max_output_tokens=max_tokens,
            timeout=settings.request_timeout,
            max_retries=settings.max_retries,
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.openai_api_key,
        temperature=temperatura,
        max_tokens=max_tokens,
        timeout=settings.request_timeout,
        max_retries=settings.max_retries,
        stream_usage=True,  # tokens também na resposta em streaming (cálculo de custo)
    )


def criar_embeddings():
    if settings.provedor == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model=settings.embedding_model, google_api_key=settings.google_api_key)
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
        max_retries=settings.max_retries,
        request_timeout=settings.request_timeout,
    )
