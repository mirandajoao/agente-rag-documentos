"""Contratos de dados (Pydantic) da API e da resposta do agente."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

StatusResposta = Literal["respondida", "sem_evidencia", "fora_do_escopo", "bloqueada", "erro"]


class Citacao(BaseModel):
    documento: str = Field(..., description="Nome do documento de onde veio o trecho")
    pagina: int | None = Field(None, description="Página do PDF onde o trecho aparece")
    trecho: str = Field("", description="Trecho literal curto que fundamenta a resposta")

    @field_validator("trecho")
    @classmethod
    def encurtar(cls, v: str) -> str:
        v = " ".join(v.split())
        return v if len(v) <= 400 else v[:397] + "..."


class RespostaAgente(BaseModel):
    """Formato JSON que o agente deve produzir na resposta final."""

    resposta: str = Field(..., min_length=1, description="Resposta ao usuário em português")
    citacoes: list[Citacao] = Field(default_factory=list)
    status: StatusResposta = "respondida"
    confianca: Literal["alta", "media", "baixa"] = "media"


class PassoReAct(BaseModel):
    tipo: Literal["thought", "action", "observation"]
    conteudo: str
    ferramenta: str | None = None


class Custo(BaseModel):
    """Custo de API gerado por uma pergunta (todas as chamadas do ciclo ReAct)."""

    modelo: str
    chamadas_llm: int = 0
    tokens_entrada: int = 0
    tokens_entrada_cache: int = 0
    tokens_saida: int = 0
    tokens_embedding: int = 0
    custo_usd: float = 0.0
    custo_brl: float = 0.0
    cotacao_usd_brl: float


ID_PATTERN = r"^[A-Za-z0-9_-]+$"


class DocumentoAlvo(BaseModel):
    """Documento escolhido no seletor "Perguntar sobre" (None = todos)."""

    arquivo: str = Field(..., min_length=1, max_length=120, pattern=r"^[A-Za-z0-9_.-]+$")
    documento: str = Field("", max_length=200)
    exemplo: bool = False


class ParametrosBusca(BaseModel):
    """Laboratório do modo aula: muda a busca só desta pergunta."""

    top_k: int | None = Field(None, ge=1, le=12)
    semelhanca_minima: float | None = Field(None, ge=0, le=0.95)


class PerguntaIn(BaseModel):
    pergunta: str = Field(..., min_length=1)
    sessao_id: str = Field(..., min_length=8, max_length=64, pattern=ID_PATTERN)
    # Espaço de documentos do usuário (fica igual entre conversas no mesmo navegador).
    espaco_id: str | None = Field(None, min_length=8, max_length=64, pattern=ID_PATTERN)
    documento: DocumentoAlvo | None = None
    # False quando o usuário removeu os documentos de exemplo da tela (ficam fora da busca dele).
    exemplos: bool = True
    parametros: ParametrosBusca | None = None


class ComparacaoIn(BaseModel):
    pergunta: str = Field(..., min_length=1)
    sessao_id: str = Field(..., min_length=8, max_length=64, pattern=ID_PATTERN)


class ComparacaoOut(BaseModel):
    resposta: str
    custo: Custo


class LinkIn(BaseModel):
    espaco_id: str = Field(..., min_length=8, max_length=64, pattern=ID_PATTERN)
    url: str = Field(..., min_length=8, max_length=2000)
    exemplos: bool = True
    tamanho_trecho: int | None = Field(None, ge=200, le=4000)


class MapaIn(BaseModel):
    espaco_id: str = Field(..., min_length=8, max_length=64, pattern=ID_PATTERN)
    pergunta: str = Field("", max_length=500)
    exemplos: bool = True
    top_k: int | None = Field(None, ge=1, le=12)


class ModeracaoIn(BaseModel):
    senha: str = Field(..., min_length=1, max_length=200)
    id: str | None = Field(None, max_length=40)


class EstudoIn(BaseModel):
    """Ferramentas do modo aula: comparar buscas e comparar modelos."""

    espaco_id: str = Field(..., min_length=8, max_length=64, pattern=ID_PATTERN)
    pergunta: str = Field(..., min_length=1, max_length=500)
    exemplos: bool = True
    top_k: int | None = Field(None, ge=1, le=12)
    semelhanca_minima: float | None = Field(None, ge=0, le=0.95)


class SugestoesIn(BaseModel):
    espaco_id: str = Field(..., min_length=8, max_length=64, pattern=ID_PATTERN)
    arquivo: str = Field(..., min_length=1, max_length=120, pattern=r"^[A-Za-z0-9_.-]+$")


class SugestoesOut(BaseModel):
    documento: str
    perguntas: list[str] = Field(default_factory=list)
    custo: Custo


class DocumentoOut(BaseModel):
    documento: str
    arquivo: str
    exemplo: bool = False
    trechos: int = 0
    paginas: int = 0


class DocumentosOut(BaseModel):
    espaco_id: str
    documentos: list[DocumentoOut] = Field(default_factory=list)
    mensagem: str = ""


class TrechoLido(BaseModel):
    """Trecho que a busca devolveu ao agente neste turno (painel "O que a IA leu")."""

    documento: str
    pagina: int | None = None
    relevancia: float | None = Field(None, description="Semelhança de 0 a 1 (só na busca semântica)")
    tipo_busca: Literal["semântica", "lexical", "resumo"] = "semântica"
    origem: Literal["usuario", "exemplo"] = "usuario"
    texto: str = ""
    citado: bool = False
    arquivo: str = Field("", description="Nome do arquivo no servidor, para abrir na página citada")


class PromptParte(BaseModel):
    """Uma parte do que foi enviado ao modelo (modo aula: "Ver o prompt")."""

    papel: Literal["sistema", "historico", "usuario", "assistente", "ferramenta"]
    conteudo: str


class RespostaOut(RespostaAgente):
    sessao_id: str
    trechos: list[TrechoLido] = Field(default_factory=list)
    passos: list[PassoReAct] = Field(default_factory=list)
    avisos: list[str] = Field(default_factory=list)
    custo: Custo | None = None
    prompt: list[PromptParte] = Field(default_factory=list)
