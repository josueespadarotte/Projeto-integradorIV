
from __future__ import annotations

import logging
import os
import time
from typing import Any

from dotenv import load_dotenv
from groq import Groq, RateLimitError

logger = logging.getLogger("backend.llm")

# Carrega as variáveis do arquivo .env (se existir).
load_dotenv()

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ATUALIZADO: O modelo llama-3.3-70b-versatile foi descontinuado.
# Usando o modelo atual recomendado pela Groq (verifique no console).
DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 2048 # Aumentado para o novo modelo

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
    max_retries: int = 3,
) -> str:
    """
    Envia o prompt para o modelo do Groq e retorna o texto da resposta.
    Implementa retry (espera) em caso de rate limit.
    """
    if not prompt or not prompt.strip():
        return ""

    client = _get_client()

    for attempt in range(max_retries):
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
            
            # Monitoramento de Tokens (IA Responsável)
            usage = completion.usage
            if usage:
                logger.info(
                    "Uso de tokens -> Prompt: %d | Resposta: %d | Total: %d",
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    usage.total_tokens
                )

            response_text = completion.choices[0].message.content or ""
            return response_text.strip()

        except RateLimitError as e:
            # Tratamento de Rate Limit
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # Espera 2s, depois 4s...
                logger.warning(
                    "Rate limit atingido no Groq. Tentando novamente em %d segundos...",
                    wait_time
                )
                time.sleep(wait_time)
            else:
                logger.error("Falha após %d tentativas devido a rate limit.", max_retries)
                raise RuntimeError("Muitas requisições à API. Tente novamente mais tarde.") from e

        except Exception:
            logger.exception("Falha ao chamar a API do Groq.")
            raise

    return ""

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