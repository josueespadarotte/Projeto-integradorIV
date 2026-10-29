from __future__ import annotations
import logging
import os
from typing import List
from contextlib import asynccontextmanager  # <-- ADICIONADO AQUI
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("backend")


# ---------------------------------------------------------------------------
# Integração com o componente de IA
# ---------------------------------------------------------------------------
from backend.rag import answer as responder, _get_pipeline  # <-- ADICIONADO O _get_pipeline AQUI


# ---------------------------------------------------------------------------
# Modelos Pydantic
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Pergunta em linguagem natural sobre acidentes rodoviários ou normas de trânsito.",
        examples=["Quantos acidentes fatais ocorreram no Espírito Santo em 2023?"],
    )


    @field_validator("question")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("A pergunta não pode ser vazia.")
        return v


class ChatResponse(BaseModel):
    answer: str = Field(..., description="Resposta gerada pela IA com base nos dados recuperados.")
    sources: List[str] = Field(
        default_factory=list,
        description="Trechos ou documentos que serviram de base para a resposta.",
    )


class HealthResponse(BaseModel):
    status: str = Field(default="ok", description="Estado da API.")


# ---------------------------------------------------------------------------
# Aplicação e Carregamento Antecipado (Startup)
# ---------------------------------------------------------------------------
@asynccontextmanager  # <-- ADICIONADO AQUI
async def lifespan(app: FastAPI):
    logger.info("⏳ INICIANDO CARREGAMENTO DA IA. POR FAVOR, AGUARDE...")
    _get_pipeline()  # Força o RAG a carregar tudo AGORA
    logger.info("✅ MODELO CARREGADO! PRONTO PARA PERGUNTAS.")
    yield


app = FastAPI(
    title="Assistente de Acidentes Rodoviários",
    description=(
        "Backend do assistente conversacional baseado em RAG + Groq. "
        "Recebe perguntas em linguagem natural e responde com base nos dados "
        "de acidentes e normas de trânsito selecionados pelo grupo."
    ),
    version="0.1.0",
    lifespan=lifespan  # <-- ADICIONADO AQUI
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Infraestrutura"],
    summary="Verificação de saúde",
    description="Retorna se a API está funcionando.",
)
def health() -> HealthResponse:
    return HealthResponse()


@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["Inteligência Artificial"],
    summary="Consulta em linguagem natural",
    description=(
        "Recebe uma pergunta em linguagem natural, aciona o pipeline RAG + LLM "
        "e devolve a resposta junto com as fontes utilizadas."
    ),
    responses={
        422: {"description": "Pergunta ausente ou vazia."},
        502: {"description": "Falha no componente de IA ou resposta vazia."},
    },
)
def chat(payload: ChatRequest) -> ChatResponse:
    try:
        result = responder(payload.question)
    except RuntimeError as exc:
        logger.exception("Falha ao processar /chat")
        raise HTTPException(status_code=502, detail=f"Erro no componente de IA: {exc}") from exc
    except Exception as exc:
        logger.exception("Erro inesperado em /chat")
        raise HTTPException(status_code=500, detail="Erro interno ao processar a pergunta.") from exc

    answer_text = str(result.get("resposta", "")).strip()
    sources = list(result.get("fontes") or [])

    if not answer_text:
        raise HTTPException(
            status_code=502,
            detail="O componente de IA retornou uma resposta vazia.",
        )

    return ChatResponse(answer=answer_text, sources=sources)


# ---------------------------------------------------------------------------
# Execução direta: python -m backend.main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "true").lower() == "true",
    )