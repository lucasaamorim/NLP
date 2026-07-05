import json
import os
import re
import logging
import time
import urllib.error
import urllib.request

from dotenv import load_dotenv

load_dotenv()

_DEFAULT_PROVIDER = os.environ.get("LLM_PROVIDER", "mistral").strip().lower()
_DEFAULT_MODEL = os.environ.get("MISTRAL_MODEL", "mistral-small-latest")
_MISTRAL_CHAT_URL = "https://api.mistral.ai/v1/chat/completions"
_LOGGER = logging.getLogger(__name__)


class _MistralAPIError(RuntimeError):
    """Falha na camada de transporte, HTTP ou resposta bruta da API."""


class _TagValidationError(ValueError):
    """Resposta da IA chegou, mas não atende ao formato esperado."""


def _get_api_key(provider):
    if provider == "mistral":
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Defina a variável de ambiente MISTRAL_API_KEY antes de rodar."
            )
        return api_key

    raise RuntimeError(f"Provedor de LLM não suportado: {provider}")


def _extract_json_array(text):
    """
        O modelo às vezes envolve a resposta em ```json ... ``` apesar da instrução.
        Esta função extrai o primeiro array JSON válido do texto.
    """
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ValueError(f"Não encontrei um array JSON na resposta: {text[:200]!r}")
    return json.loads(match.group(0))


def _call_mistral(prompt, model, temperature):
    api_key = _get_api_key("mistral")
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
    }
    request = urllib.request.Request(
        _MISTRAL_CHAT_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise _MistralAPIError(
            f"Erro HTTP do Mistral ({exc.code}): {error_body[:500]}"
        ) from exc
    except urllib.error.URLError as exc:
        raise _MistralAPIError(f"Erro de conexão com o Mistral: {exc}") from exc

    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise _MistralAPIError(
            f"Resposta inválida do Mistral (JSON malformado): {body[:500]}"
        ) from exc

    choices = data.get("choices", [])
    if not choices:
        raise _MistralAPIError(f"Resposta do Mistral sem choices válidas: {body[:500]}")

    message = choices[0].get("message", {})
    content = message.get("content")
    if not content:
        raise _MistralAPIError(f"Resposta do Mistral sem conteúdo: {body[:500]}")

    return content


def tag_tokens(prompt, n_tokens, max_retries=None, temperature=0.0, provider=None, model=None):
    """
        Envia o prompt já pronto (ver prompts.py) e retorna a lista de tags
        prevista pelo modelo. Faz retry até obter uma resposta válida.
    """
    provider = (provider or _DEFAULT_PROVIDER).strip().lower()
    model = model or _DEFAULT_MODEL
    last_error = None
    attempt = 0

    while True:
        attempt += 1
        try:
            if provider != "mistral":
                raise RuntimeError(f"Provedor de LLM não suportado: {provider}")

            response_text = _call_mistral(prompt, model=model, temperature=temperature)
            tags = _extract_json_array(response_text)

            if len(tags) != n_tokens:
                raise _TagValidationError(
                    f"Esperava {n_tokens} tags, recebi {len(tags)} (tentativa {attempt})"
                )
            return tags

        except _MistralAPIError as e:
            last_error = e
            _LOGGER.warning(
                "Falha na tentativa %s ao chamar a IA (%s/%s): %s",
                attempt,
                provider,
                model,
                e,
            )
            if max_retries is not None and attempt >= max_retries:
                raise RuntimeError(f"Falhou após {max_retries} tentativas: {last_error}")
            time.sleep(12)
