"""Provider clients shared by generators, diagnostic judges and benchmark adapters."""
from __future__ import annotations

import re
import time
import uuid

import requests

from autopi.models.endpoints import ModelEndpoint
from autopi.models.recovery import get_decoder


def visible_text(content: str | None) -> str:
    """Keep the visible answer when an endpoint leaks a reasoning delimiter."""
    content = re.sub(r"<think>.*?</think>", "", content or "", flags=re.S)
    if "</think>" in content:
        content = content.rpartition("</think>")[2]
    return content.strip()


def instrument_openai_client(client, endpoint: ModelEndpoint, *, journal=None, role="target"):
    """Apply settings to this client only, and optionally recover prompted-tool text."""
    decoder = get_decoder(endpoint.tokenizer_path) if endpoint.tokenizer_path else None
    original = client.chat.completions.create

    def create(*args, **kwargs):
        kwargs = dict(kwargs)
        if endpoint.extra_body:
            kwargs["extra_body"] = {**(kwargs.get("extra_body") or {}), **endpoint.extra_body}
        if endpoint.reasoning_effort:
            kwargs["reasoning_effort"] = endpoint.reasoning_effort
        if endpoint.provider == "openai-compatible" and endpoint.thinking != "server":
            extra = dict(kwargs.get("extra_body") or {})
            template = dict(extra.get("chat_template_kwargs") or {})
            template["enable_thinking"] = endpoint.thinking == "on"
            extra["chat_template_kwargs"] = template
            kwargs["extra_body"] = extra
        if endpoint.provider == "ollama" and endpoint.thinking != "server":
            # Ollama's OpenAI-compatible API reads the thinking mode from reasoning_effort:
            # "none" turns thinking off, an effort level (the alias's own, else "medium") on.
            if endpoint.thinking == "off":
                kwargs["reasoning_effort"] = "none"
            else:
                kwargs.setdefault("reasoning_effort", "medium")
        for key in ("temperature", "top_p", "max_tokens"):
            value = getattr(endpoint, key)
            if value is not None:
                kwargs[key] = value
        for key in endpoint.omit_parameters:
            kwargs.pop(key, None)
        if decoder:
            kwargs = decoder.prepare(kwargs)
        call_id = uuid.uuid4().hex
        if journal:
            journal.emit("llm_request", role=role, call_id=call_id, endpoint=endpoint.base_url,
                         request=kwargs, tokenizer_sha256=decoder.sha256 if decoder else None)
        try:
            response = original(*args, **kwargs)
            if journal:
                journal.emit("llm_response_original", role=role, call_id=call_id, response=response)
            if decoder:
                recovered = decoder.recover(response.model_dump(mode="json"))
                for choice, replacement in zip(response.choices, recovered["choices"], strict=True):
                    choice.message.content = replacement["message"]["content"]
                    choice.message.tool_calls = None
                if journal:
                    journal.emit("llm_response_recovered", role=role, call_id=call_id, response=recovered)
            return response
        except Exception as exc:
            if journal:
                journal.emit("llm_error", role=role, call_id=call_id, error_type=type(exc).__name__, error=str(exc))
            raise

    client.chat.completions.create = create
    return client


def query_endpoint(endpoint: ModelEndpoint, prompt: str, *, temperature=0.7, max_tokens=1024,
                   journal=None, role="refiner") -> str:
    """Text generation with an explicit provider, model id and optional remote URL."""
    temperature = endpoint.temperature if endpoint.temperature is not None else temperature
    max_tokens = endpoint.max_tokens if endpoint.max_tokens is not None else max_tokens
    if endpoint.provider in {"openai-compatible", "openai"}:
        url = (endpoint.base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
        payload = {"model": endpoint.model, "messages": [{"role": "user", "content": prompt}],
                   "temperature": temperature, "max_tokens": max_tokens}
        if endpoint.top_p is not None:
            payload["top_p"] = endpoint.top_p
        payload.update(endpoint.extra_body)
        if endpoint.reasoning_effort:
            payload["reasoning_effort"] = endpoint.reasoning_effort
        if endpoint.provider == "openai-compatible" and endpoint.thinking != "server":
            payload["chat_template_kwargs"] = {**payload.get("chat_template_kwargs", {}), "enable_thinking": endpoint.thinking == "on"}
        for key in endpoint.omit_parameters:
            payload.pop(key, None)
        decoder = get_decoder(endpoint.tokenizer_path) if endpoint.tokenizer_path else None
        if decoder:
            payload = decoder.prepare(payload)
        call_id = uuid.uuid4().hex
        if journal:
            journal.emit("llm_request", role=role, call_id=call_id, endpoint=url, request=payload,
                         tokenizer_sha256=decoder.sha256 if decoder else None)
        for attempt in range(endpoint.retries + 1):
            try:
                response = requests.post(url, json=payload, timeout=endpoint.timeout,
                                         headers={"Authorization": "Bearer " + endpoint.api_key()})
                if journal:
                    journal.emit("llm_response_original", role=role, call_id=call_id,
                                 retry=attempt, status_code=response.status_code, body=response.text)
                response.raise_for_status()
                data = response.json()
                if decoder:
                    # Generator text keeps special tokens: they can be the structure it wrote.
                    data = decoder.recover(data, keep_special_tokens=True)
                    if journal:
                        journal.emit("llm_response_recovered", role=role, call_id=call_id, response=data)
                content = data["choices"][0]["message"].get("content")
                # Structural markers may be the requested injection construction.
                # Preserve the generator's text; its extractor handles delimiters.
                return (content or "").strip()
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
                retryable = not isinstance(exc, requests.HTTPError) or exc.response.status_code >= 500 or exc.response.status_code == 429
                if journal:
                    journal.emit("llm_error", role=role, call_id=call_id, retry=attempt, error=str(exc))
                if not retryable or attempt >= endpoint.retries:
                    raise
                time.sleep(min(2 ** attempt, 10))
    if endpoint.provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=endpoint.api_key(), timeout=endpoint.timeout,
                                     max_retries=endpoint.retries, **({"base_url": endpoint.base_url} if endpoint.base_url else {}))
        # No sampling parameters: the Anthropic API (version 2026-10-21) rejects them on every model.
        payload = dict(model=endpoint.model, messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens)
        for key in endpoint.omit_parameters:
            payload.pop(key, None)
        if journal:
            journal.emit("llm_request", role=role, request=payload)
        response = client.messages.create(**payload)
        if journal:
            journal.emit("llm_response_original", role=role, response=response)
        return "".join(block.text for block in response.content if block.type == "text").strip()
    if endpoint.provider == "google":
        from google import genai

        kwargs = {"api_key": endpoint.api_key(), "http_options": {
            "timeout": int(endpoint.timeout * 1000), "retry_options": {"attempts": endpoint.retries + 1}}}
        if endpoint.base_url:
            kwargs["http_options"]["base_url"] = endpoint.base_url
        client = genai.Client(**kwargs)
        if journal:
            journal.emit("llm_request", role=role, model=endpoint.model, prompt=prompt)
        config = {"temperature": temperature, "max_output_tokens": max_tokens}
        if endpoint.top_p is not None:
            config["top_p"] = endpoint.top_p
        response = client.models.generate_content(model=endpoint.model, contents=prompt, config=config)
        if journal:
            journal.emit("llm_response_original", role=role, response=response)
        return (response.text or "").strip()
    if endpoint.provider == "ollama":
        payload = {"model": endpoint.model, "messages": [{"role": "user", "content": prompt}],
                   "stream": False, "options": {"temperature": temperature, "num_predict": max_tokens}}
        if endpoint.thinking != "server":
            payload["think"] = endpoint.thinking == "on"
        if journal:
            journal.emit("llm_request", role=role, request=payload)
        response = requests.post(endpoint.base_url.rstrip("/") + "/api/chat", json=payload, timeout=endpoint.timeout)
        response.raise_for_status()
        if journal:
            journal.emit("llm_response_original", role=role, body=response.text)
        return (response.json()["message"]["content"] or "").strip()
    raise ValueError(f"provider {endpoint.provider!r} cannot generate text; configure an explicit endpoint")
