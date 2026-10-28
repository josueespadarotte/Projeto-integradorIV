
from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv
from groq import Groq

logger = logging.getLogger("backend.llm")

# Carrega as variáveis do arquivo .env (se existir).
load_dotenv()

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEFAULT_MODEL = "llama-3.3-70b-versatile"  # modelo atual e recomendado
DEFAULT_TEMPERATURE = 0.1                  # baixa temperatura para respostas factuais
DEFAULT_MAX_TOKENS = 1024

# Cliente inicializado sob demanda para não falhar no import caso a chave
# ainda não esteja configurada em algum ambiente de teste.
_client: Groq | None = None


def _get_client() -> Groq:
    """Retorna o cliente Groq, criando-o na primeira chamada."""
    global _client
    if _client is not None:
        return _client

    if not GROQ_API_KEY:
        raise RuntimeError(
            "A variável de ambiente GROQ_API_KEY não foi encontrada. "
            "Adicione-a ao arquivo .env na raiz do projeto."
        )

    _client = Groq(api_key=GROQ_API_KEY)
    logger.info("Cliente Groq inicializado com sucesso.")
    return _client


# ---------------------------------------------------------------------------
# Função pública usada pelo rag.py
# ---------------------------------------------------------------------------
def generate_response(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """
    Envia o prompt para o modelo do Groq e retorna o texto da resposta.

    O prompt já vem montado pelo rag.py com o contexto e as fontes,
    então esta função apenas faz a chamada à API e extrai o conteúdo.
    """
    if not prompt or not prompt.strip():
        return ""

    client = _get_client()

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Você é um assistente especializado em segurança "
                        "rodoviária e normas de trânsito brasileiras. "
                        "Responda sempre em português do Brasil, de forma "
                        "clara e objetiva, citando as fontes fornecidas."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception:
        logger.exception("Falha ao chamar a API do Groq.")
        raise

    response_text = completion.choices[0].message.content or ""
    logger.info(
        "Resposta gerada pelo modelo %s (%d caracteres).",
        model,
        len(response_text),
    )
    return response_text.strip()


# ---------------------------------------------------------------------------
# Teste manual: python -m backend.llm
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_prompt = (
        "Contexto:\n"
        "[Fonte: prf_es_clean.csv#linha_1]\n"
        "Acidente em 2023-05-10, na rodovia BR-101, km 45, "
        "município de Vitória. Mortos: 0, feridos leves: 2.\n\n"
        "Pergunta: Houve acidente na BR-101 em Vitória?\n\n"
        "Resposta (com citação de fontes):"
    )

    print("Prompt de teste enviado:\n", test_prompt)
    print("-" * 60)
    print("Resposta do modelo:\n")
    print(generate_response(test_prompt))