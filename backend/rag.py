
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("backend.rag")

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CSV_ACCIDENTS = DATA_DIR / "prf_es_clean.csv"
NORMS_DIR = DATA_DIR / "normas"
CACHE_DIR = DATA_DIR / "rag_cache"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 5
EMBEDDING_MODEL = "tardellirs/brazembed-pt-br-retrieval"

# Pipeline carregado sob demanda (evita custo no import do main.py).
_pipeline: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# 1. Carregamento dos documentos
# ---------------------------------------------------------------------------
def _load_accidents(csv_path: Path) -> list[dict[str, str]]:
    """Converte cada linha do CSV em uma frase com metadados de fonte."""
    if not csv_path.exists():
        logger.warning("CSV de acidentes não encontrado em %s", csv_path)
        return []

    df = pd.read_csv(csv_path)
    df = df.fillna("não informado")

    documents: list[dict[str, str]] = []
    for idx, row in df.iterrows():
        text = (
            f"Acidente em {row['data_inversa']}, na rodovia BR-{row['br']}, "
            f"km {row['km']}, município de {row['municipio']}. "
            f"Tipo: {row['tipo_acidente']}. "
            f"Classificação: {row['classificacao_acidente']}. "
            f"Causa: {row['causa_acidente']}. "
            f"Mortos: {row['mortos']}, feridos leves: {row['feridos_leves']}, "
            f"feridos graves: {row['feridos_graves']}."
        )
        documents.append(
            {
                "texto": text,
                "fonte": f"prf_es_clean.csv#linha_{idx + 1}",
            }
        )
    return documents


def _load_norms(directory: Path) -> list[dict[str, str]]:
    """Lê cada arquivo .txt de normas e devolve um documento por arquivo."""
    if not directory.exists():
        logger.warning(
            "Pasta de normas não encontrada em %s. "
            "O RAG responderá apenas com base nos acidentes.",
            directory,
        )
        return []

    documents: list[dict[str, str]] = []
    for path in sorted(directory.glob("*.txt")):
        try:
            content = path.read_text(encoding="utf-8")
        except Exception:
            logger.exception("Falha ao ler o arquivo de norma %s", path)
            continue
        documents.append(
            {
                "texto": content,
                "fonte": path.name,
            }
        )
    return documents


def load_documents() -> list[dict[str, str]]:
    """Ponto único de entrada para carregar todos os documentos."""
    accidents = _load_accidents(CSV_ACCIDENTS)
    norms = _load_norms(NORMS_DIR)
    logger.info("Documentos carregados: %d acidentes, %d normas", len(accidents), len(norms))
    return accidents + norms


# ---------------------------------------------------------------------------
# 2. Chunking
# ---------------------------------------------------------------------------
def chunk_documents(
    documents: list[dict[str, str]],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict[str, str]]:
    """Divide textos longos em chunks, preservando a fonte de origem."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks: list[dict[str, str]] = []
    for doc in documents:
        pieces = splitter.split_text(doc["texto"])
        for i, piece in enumerate(pieces):
            chunks.append(
                {
                    "texto": piece,
                    "fonte": f"{doc['fonte']}#chunk_{i}",
                }
            )
    return chunks


# ---------------------------------------------------------------------------
# 3. Embeddings + índice FAISS (com cache em disco)
# ---------------------------------------------------------------------------
def _get_cache_paths() -> tuple[Path, Path]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / "chunks.json", CACHE_DIR / "index.faiss"


def _build_index(
    chunks: list[dict[str, str]],
    model: SentenceTransformer,
) -> tuple[list[dict[str, str]], faiss.Index]:
    texts = [c["texto"] for c in chunks]
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # produto interno ≈ cosseno
    index.add(embeddings)

    # Persiste para não recalcular a cada reinício.
    chunks_path, index_path = _get_cache_paths()
    chunks_path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    faiss.write_index(index, str(index_path))
    logger.info("Índice construído e salvo em %s", CACHE_DIR)
    return chunks, index


def _load_index_from_cache() -> tuple[list[dict[str, str]], faiss.Index] | None:
    chunks_path, index_path = _get_cache_paths()
    if not chunks_path.exists() or not index_path.exists():
        return None
    try:
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        index = faiss.read_index(str(index_path))
        logger.info("Índice carregado do cache (%d chunks).", len(chunks))
        return chunks, index
    except Exception:
        logger.exception("Falha ao carregar o cache do RAG; será reconstruído.")
        return None


# ---------------------------------------------------------------------------
# 4. Inicialização lazy do pipeline
# ---------------------------------------------------------------------------
def _get_pipeline() -> dict[str, Any]:
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    model = SentenceTransformer(EMBEDDING_MODEL)

    cache = _load_index_from_cache()
    if cache is not None:
        chunks, index = cache
    else:
        documents = load_documents()
        chunks = chunk_documents(documents)
        if not chunks:
            raise RuntimeError(
                "Nenhum chunk foi gerado. Verifique se data/prf_es_clean.csv "
                "e a pasta data/normas/ existem."
            )
        chunks, index = _build_index(chunks, model)

    _pipeline = {"model": model, "chunks": chunks, "index": index}
    return _pipeline


# ---------------------------------------------------------------------------
# 5. Busca dos trechos mais relevantes
# ---------------------------------------------------------------------------
def search_chunks(question: str, top_k: int = TOP_K) -> list[dict[str, str]]:
    """Retorna os top_k chunks mais similares à pergunta."""
    pipeline = _get_pipeline()
    model: SentenceTransformer = pipeline["model"]
    chunks: list[dict[str, str]] = pipeline["chunks"]
    index: faiss.Index = pipeline["index"]

    question_embedding = model.encode(
        [question],
        normalize_embeddings=True,
    ).astype("float32")

    distances, indices = index.search(question_embedding, top_k)
    results: list[dict[str, str]] = []
    for idx in indices[0]:
        if idx != -1:
            results.append(chunks[int(idx)])
    return results


# ---------------------------------------------------------------------------
# 6. Função pública usada pelo main.py
# ---------------------------------------------------------------------------
def _build_prompt(question: str, chunks: list[dict[str, str]]) -> str:
    context = "\n\n".join(
        f"[Fonte: {c['fonte']}]\n{c['texto']}" for c in chunks
    )
    return f"""Você é um assistente especializado em segurança rodoviária e normas de trânsito brasileiras.
Responda à pergunta do usuário usando EXCLUSIVAMENTE o contexto abaixo.
Se o contexto não contiver informação suficiente, diga que não sabe e sugira consultar a fonte oficial.
Sempre cite a fonte (nome do arquivo ou linha do CSV) ao final da resposta.

Contexto:
{context}

Pergunta: {question}

Resposta (com citação de fontes):"""


def answer(question: str) -> dict[str, Any]:
    """
    Contrato esperado pelo backend/main.py:
    retorna {"resposta": str, "fontes": list[str]}.
    """
    question = (question or "").strip()
    if not question:
        return {"resposta": "Pergunta vazia.", "fontes": []}

    chunks = search_chunks(question, top_k=TOP_K)
    if not chunks:
        return {
            "resposta": "Não encontrei informação suficiente na base para responder.",
            "fontes": [],
        }

    prompt = _build_prompt(question, chunks)

    # Import tardio para evitar circularidade e permitir que o rag.py exista
    # antes do llm.py ser implementado.
    try:
        from backend.llm import generate_response
    except ImportError as exc:
        raise RuntimeError(
            "backend.llm.generate_response não encontrado. "
            "Implemente backend/llm.py antes de usar o RAG completo."
        ) from exc

    response = generate_response(prompt)

    # Fontes únicas, preservando a ordem de relevância.
    sources: list[str] = []
    for c in chunks:
        if c["fonte"] not in sources:
            sources.append(c["fonte"])

    return {"resposta": response, "fontes": sources}


# ---------------------------------------------------------------------------
# Teste manual: python -m backend.rag
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_question = "Quantos acidentes com mortos ocorreram na BR-101?"
    print("Pergunta:", test_question)
    for chunk in search_chunks(test_question, top_k=3):
        print("-", chunk["fonte"], "→", chunk["texto"][:120], "...")