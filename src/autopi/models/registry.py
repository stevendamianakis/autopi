"""Model resolution for AgentDojo targets and standalone text generation."""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict

TARGET_THINKING_CHOICES = ("off", "on", "server")
_THINKING_ALIASES = {"0": "off", "false": "off", "no": "off", "1": "on", "true": "on", "yes": "on", "default": "server"}


def resolve_target_thinking(value: str | None = None) -> str:
    """Resolve an explicit target thinking request to ``off``, ``on`` or ``server`` (send nothing
    and let the server decide). ``off`` is the setting of the reported runs and the fallback here,
    but the CLI calls this only when a thinking mode was actually requested (by the flag, the
    ``AUTOPI_TARGET_THINKING`` environment variable, or a ``vllm:`` spec); a named alias otherwise
    keeps its own thinking. An explicit ``value`` wins over the environment variable."""
    v = (value or os.environ.get("AUTOPI_TARGET_THINKING") or "off").strip().lower()
    v = _THINKING_ALIASES.get(v, v)
    if v not in TARGET_THINKING_CHOICES:
        raise ValueError(f"target thinking must be one of {', '.join(TARGET_THINKING_CHOICES)}; got {v!r}")
    return v


def resolve_target_max_tokens(value: int | None = None) -> int | None:
    """Output cap of a locally served target, or ``None`` for no cap (the default).
    An explicit ``value`` wins over ``AUTOPI_TARGET_MAX_TOKENS``. ``LOCAL_LLM_MAX_TOKENS``
    is NOT read, so launch scripts that export it keep running without a cap."""
    if value is None:
        raw = os.environ.get("AUTOPI_TARGET_MAX_TOKENS", "").strip()
        value = int(raw) if raw else None
    if value is not None and value <= 0:
        raise ValueError(f"target max tokens must be positive; got {value}")
    return value


def send_developer_as_system(client) -> None:
    """AgentDojo's native client sends its system prompt with the OpenAI ``developer`` role.
    A self-hosted model's chat template knows only ``system``: recent vLLM converts the role,
    but other OpenAI-compatible servers may reject the message or render it as a turn the model
    was never trained on. Send it as ``system``, as AgentDojo's prompted client does."""
    create = client.chat.completions.create

    def _create(*args, **kwargs):
        messages = kwargs.get("messages")
        if isinstance(messages, list) and any(isinstance(m, dict) and m.get("role") == "developer" for m in messages):
            kwargs["messages"] = [dict(m, role="system") if isinstance(m, dict) and m.get("role") == "developer" else m
                                  for m in messages]
        return create(*args, **kwargs)

    client.chat.completions.create = _create


def build_target_pipeline(
    spec: str,
    *,
    defense: str | None = None,
    tool_output_format: str | None = None,
    target_thinking: str | None = None,
    target_max_tokens: int | None = None,
    endpoint=None,
    journal=None,
):
    """Use AgentDojo's pipeline with an explicitly configured model client."""
    from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig

    from autopi.models.clients import instrument_openai_client
    from autopi.models.endpoints import override_endpoint, resolve_endpoint

    endpoint = endpoint or resolve_endpoint(spec)
    endpoint = override_endpoint(endpoint, thinking=target_thinking, max_tokens=target_max_tokens)
    endpoint.refuse_returned_reasoning()
    if endpoint.provider in {"openai-compatible", "openai", "ollama"}:
        import openai
        from agentdojo.agent_pipeline.llms.local_llm import LocalLLM
        from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM

        if endpoint.tokenizer_path and endpoint.tool_mode != "prompted":
            raise ValueError("target raw-token recovery (--target-tokenizer or tokenizer_path) reads "
                             "prompted tool calls; add --tool-calling prompted (tool_mode: prompted)")
        url = endpoint.base_url
        if endpoint.provider == "ollama":
            url = url.rstrip("/") + "/v1"
        client = openai.OpenAI(api_key=endpoint.api_key(), base_url=url,
                               timeout=endpoint.timeout, max_retries=endpoint.retries)
        instrument_openai_client(client, endpoint, journal=journal)
        if endpoint.tool_mode == "native" and endpoint.provider in {"openai-compatible", "ollama"}:
            send_developer_as_system(client)
        llm_cls = LocalLLM if endpoint.tool_mode == "prompted" else OpenAILLM
        llm = llm_cls(client, endpoint.model)
        llm.name = endpoint.model
    elif endpoint.provider == "anthropic":
        import anthropic
        from agentdojo.agent_pipeline.llms.anthropic_llm import AnthropicLLM

        from autopi.models.clients import require_no_reasoning_effort

        try:
            from httpx2 import Limits
        except ImportError:
            # Anthropic SDK 0.x/1.x uses httpx, and need not have httpx2 installed.
            from httpx import Limits

        require_no_reasoning_effort(endpoint)
        # AgentDojo's Anthropic element streams through an async client (it copies a sync
        # client into a new one), so the settings and the trace are applied to that call.
        # AgentDojo runs each query in its own event loop (``asyncio.run``) on this one client.
        # A kept-alive connection then belongs to a closed loop, and the SDK sends the next
        # request twice (once on the stale connection, once on a fresh one). Disabling HTTP
        # keep-alive makes every request open its own connection, so each is sent once.
        http_client = anthropic.DefaultAsyncHttpxClient(
            timeout=endpoint.timeout, limits=Limits(max_keepalive_connections=0))
        client = anthropic.AsyncAnthropic(api_key=endpoint.api_key(), max_retries=endpoint.retries,
                                          http_client=http_client,
                                          **({"base_url": endpoint.base_url} if endpoint.base_url else {}))
        _instrument_anthropic(client.messages, endpoint, journal)
        llm = AnthropicLLM(client, endpoint.model)
        llm.name = endpoint.model
    elif endpoint.provider == "google":
        from agentdojo.agent_pipeline.llms.google_llm import GoogleLLM
        from google import genai

        from autopi.models.clients import google_config_fields, require_no_reasoning_effort

        require_no_reasoning_effort(endpoint)
        # Validate the alias's extra fields now, as the SDK does for the generator: a misspelt field
        # is refused at setup, and a camelCase one (thinkingConfig) is sent instead of dropped.
        extra_fields = google_config_fields(endpoint)
        options = {"timeout": int(endpoint.timeout * 1000), "retry_options": {"attempts": endpoint.retries + 1}}
        if endpoint.base_url:
            options["base_url"] = endpoint.base_url
        client = genai.Client(api_key=endpoint.api_key(), http_options=options)
        _instrument_google(client.models, endpoint, journal, extra_fields)
        llm = GoogleLLM(endpoint.model, client)
        llm.name = endpoint.model
    else:
        llm = spec  # Additional provider/model enums supported by the installed benchmark.
    return AgentPipeline.from_config(PipelineConfig(
        llm=llm, model_id=endpoint.model, defense=defense,
        system_message_name=None, system_message=None, tool_output_format=tool_output_format,
    ))


_SAMPLING_PARAMETERS = ("temperature", "top_p", "top_k")


def _instrument_anthropic(messages, endpoint, journal):
    """Prepare ``messages.stream`` of an async Anthropic client, the call AgentDojo makes, and
    trace the request and the final message. AgentDojo always passes a temperature, but the
    Anthropic API (version 2026-10-21, anthropic SDK 2.x) rejects sampling parameters on every
    model and the SDK no longer accepts them, so they are removed. The output cap is the
    endpoint's ``max_tokens``, or AgentDojo's 1,024."""
    original = messages.stream

    def stream(*args, **kwargs):
        for key in _SAMPLING_PARAMETERS:
            kwargs.pop(key, None)
        if endpoint.max_tokens is not None:
            kwargs["max_tokens"] = endpoint.max_tokens
        if endpoint.extra_body:
            # Passthrough for Anthropic-only request fields, e.g. enabling extended thinking.
            kwargs["extra_body"] = {**(kwargs.get("extra_body") or {}), **endpoint.extra_body}
        for key in endpoint.omit_parameters:
            kwargs.pop(key, None)
        if journal:
            journal.emit("llm_request", role="target", request=kwargs)
        return _TracedAnthropicStream(original(*args, **kwargs), journal)

    messages.stream = stream


class _TracedAnthropicStream:
    """The async context manager returned by ``messages.stream``, with the final message
    (or the error) written to the trace."""

    def __init__(self, manager, journal):
        self._manager, self._journal, self._stream = manager, journal, None

    def _error(self, exc):
        if self._journal:
            self._journal.emit("llm_error", role="target", error_type=type(exc).__name__, error=str(exc))

    async def __aenter__(self):
        try:
            self._stream = await self._manager.__aenter__()
        except Exception as exc:
            self._error(exc)
            raise
        return self

    async def __aexit__(self, *exc_info):
        return await self._manager.__aexit__(*exc_info)

    async def get_final_message(self):
        try:
            message = await self._stream.get_final_message()
        except Exception as exc:
            self._error(exc)
            raise
        if self._journal:
            self._journal.emit("llm_response_original", role="target", response=message)
        return message

    def __getattr__(self, name):
        return getattr(self._stream, name)


class _GeminiThoughtSignatures:
    """Gemini 3 and later reject a request whose function calls of the current turn lack the
    thought signatures they came with, and AgentDojo's Gemini client rebuilds those calls
    without them. The signatures of each reply are remembered, keyed by the conversation that
    preceded it, and put back on the same calls when that conversation is sent again."""

    def __init__(self, size: int = 256):
        self._by_prefix: OrderedDict[str, list] = OrderedDict()
        self._size = size

    @staticmethod
    def _key(contents) -> str:
        plain = []
        for content in contents:
            data = content.model_dump(mode="json", exclude_none=True) if hasattr(content, "model_dump") else content
            if isinstance(data, dict):
                for part in data.get("parts") or []:
                    if isinstance(part, dict):
                        part.pop("thought_signature", None)
            plain.append(data)
        return hashlib.sha256(json.dumps(plain, sort_keys=True, default=str).encode()).hexdigest()

    def remember(self, contents, response) -> None:
        try:
            parts = response.candidates[0].content.parts or []
        except (AttributeError, IndexError, TypeError):
            return
        found = [getattr(part, "thought_signature", None) for part in parts if getattr(part, "function_call", None)]
        if any(found) and isinstance(contents, list):
            key = self._key(contents)
            self._by_prefix[key] = found
            self._by_prefix.move_to_end(key)
            while len(self._by_prefix) > self._size:
                self._by_prefix.popitem(last=False)

    def attach(self, contents):
        if not isinstance(contents, list):
            return contents
        out = list(contents)
        for index, content in enumerate(out):
            parts = getattr(content, "parts", None) or []
            calls = [part for part in parts if getattr(part, "function_call", None)]
            if getattr(content, "role", None) != "model" or not calls or all(part.thought_signature for part in calls):
                continue
            found = self._by_prefix.get(self._key(out[:index]))
            if not found:
                continue
            signatures, rebuilt = iter(found), []
            for part in parts:
                if getattr(part, "function_call", None):
                    signature = next(signatures, None)
                    if signature and not part.thought_signature:
                        part = part.model_copy(update={"thought_signature": signature})
                rebuilt.append(part)
            out[index] = content.model_copy(update={"parts": rebuilt})
        return out


def _instrument_google(resource, endpoint, journal, extra_fields=None):
    original = resource.generate_content
    signatures = _GeminiThoughtSignatures()

    from autopi.models.clients import google_config_fields, google_omitted_fields

    omitted = google_omitted_fields(endpoint)
    if extra_fields is None:
        extra_fields = google_config_fields(endpoint)

    def call(*args, **kwargs):
        updates = {key: value for key, value in {
            "temperature": endpoint.temperature, "top_p": endpoint.top_p,
            "max_output_tokens": endpoint.max_tokens}.items() if value is not None}
        if extra_fields:
            # Extra GenerateContentConfig fields (e.g. thinking_config), validated by the SDK's
            # own model, for Google native calls.
            updates.update(extra_fields)
        # An omitted field is cleared, AgentDojo's own value included (its Gemini client always
        # sets temperature 0 and a 65,535-token cap); the SDK sends no field that is None.
        updates.update({key: None for key in omitted})
        config = kwargs.get("config")
        if updates:
            if hasattr(config, "model_copy"):
                kwargs["config"] = config.model_copy(update=updates)
            else:
                kwargs["config"] = {key: value for key, value in {**(config or {}), **updates}.items()
                                    if value is not None}
        if "contents" in kwargs:
            kwargs["contents"] = signatures.attach(kwargs["contents"])
        if journal:
            journal.emit("llm_request", role="target", request=kwargs)
        result = original(*args, **kwargs)
        signatures.remember(kwargs.get("contents"), result)
        if journal:
            journal.emit("llm_response_original", role="target", response=result)
        return result

    resource.generate_content = call
