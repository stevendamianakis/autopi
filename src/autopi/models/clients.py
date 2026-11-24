"""Provider clients shared by the generator, the judge and the benchmark adapters."""
from __future__ import annotations

import os
import re
import time
import uuid
from collections import OrderedDict

import requests

from autopi.models.endpoints import ModelEndpoint, redact_url
from autopi.models.recovery import get_decoder


def google_omitted_fields(endpoint: ModelEndpoint) -> set[str]:
    """The GenerateContentConfig field names of ``omit_parameters`` (``max_tokens`` is called
    ``max_output_tokens`` there)."""
    return {{"max_tokens": "max_output_tokens"}.get(key, key) for key in endpoint.omit_parameters}


def google_config_fields(endpoint: ModelEndpoint) -> dict:
    """A Google alias's ``extra_body`` as validated ``GenerateContentConfig`` fields: the names may
    be snake_case or camelCase, as the SDK accepts them, nested settings become the SDK's own
    types (so they are sent under their API names), and an unknown field is refused. The Google
    generator gets the same check from the SDK when its request is built."""
    if not endpoint.extra_body:
        return {}
    from google.genai import types
    from pydantic import ValidationError

    from autopi.models.endpoints import EndpointConfigError

    try:
        config = types.GenerateContentConfig.model_validate(endpoint.extra_body)
    except ValidationError as exc:
        error = exc.errors()[0]
        where = ".".join(str(part) for part in error.get("loc", ()))
        raise EndpointConfigError(
            f"extra_body of the Google model {endpoint.model!r} is not a valid GenerateContentConfig "
            f"({where}: {error.get('msg')})") from None
    return {name: getattr(config, name) for name in config.model_fields_set}


def _omitted_request_fields(endpoint: ModelEndpoint) -> set[str]:
    """The request fields ``omit_parameters`` removes on OpenAI-style requests. With
    ``max_tokens_field: max_completion_tokens``, omitting ``max_tokens`` also removes the cap
    AutoPI sends under that name (a ``max_completion_tokens`` set in ``extra_body`` with the
    default field is the alias's own and is kept)."""
    omitted = set(endpoint.omit_parameters)
    if "max_tokens" in omitted and endpoint.max_tokens_field == "max_completion_tokens":
        omitted.add("max_completion_tokens")
    return omitted


def _unset(value) -> bool:
    """A request field that is not set: None, or the OpenAI SDK's NOT_GIVEN or omit sentinel."""
    return value is None or type(value).__name__ in {"NotGiven", "Omit"}


def require_no_reasoning_effort(endpoint: ModelEndpoint) -> None:
    """The native Anthropic and Google clients have no ``reasoning_effort`` request field, so
    the setting would be dropped silently. It reaches requests only on OpenAI-compatible paths
    (OpenAI-compatible, OpenAI and Ollama endpoints, and every IPI Arena client)."""
    from autopi.models.endpoints import EndpointConfigError

    if endpoint.reasoning_effort:
        raise EndpointConfigError(
            f"reasoning_effort is not sent by the native {endpoint.provider} client; put the "
            "provider's own field (for example its thinking settings) in extra_body instead")


# Reasoning delimiters a server can leave in a reply when no reasoning parser splits them off:
# <think> (Qwen, DeepSeek R1 and others), [THINK] (Magistral), ◁think▷ (Kimi) and <seed:think>
# (Seed-OSS). Each is (open, close). Callers remove reasoning only from an OpenAI-compatible or
# Ollama endpoint that may think (``ModelEndpoint.may_leak_reasoning``), only when the server did
# not return the reasoning in a field of its own (a reasoning parser does), and only reasoning
# that opens the reply. A tag later in a reply is content (an injected token the agent quotes in
# a tool argument, a judge quoting the agent) and is kept.
_REASONING_DELIMITERS = (
    ("<think>", "</think>"), ("[THINK]", "[/THINK]"), ("◁think▷", "◁/think▷"), ("<seed:think>", "</seed:think>"),
)
_LEADING_REASONING = re.compile(
    r"\A\s*(?:" + "|".join(re.escape(o) + ".*?" + re.escape(c) for o, c in _REASONING_DELIMITERS) + r")\s*", re.S)
_UNCLOSED_REASONING = re.compile(
    r"\A\s*(?:" + "|".join(re.escape(o) for o, _ in _REASONING_DELIMITERS) + ")")
# The markers that open AutoPI's own answers: a generated line and a judge verdict.
_ANSWER_MARKER = re.compile(r"\[TEMPLATE_START\]|FAILURE_MODE:")


def _without_leading_blocks(text: str) -> str:
    while True:
        match = _LEADING_REASONING.match(text)
        if not match:
            return text
        text = text[match.end():]


def visible_text(content: str | None, *, drop_unclosed: bool = False, lone_closer: bool = False) -> str:
    """The visible answer of a reply from an endpoint that may think and returned no reasoning
    field of its own.

    Only reasoning that opens the reply is removed: one or more leading blocks of any family
    (``<think>``, ``[THINK]``, ``◁think▷``, ``<seed:think>``). With ``drop_unclosed``, a leading
    block that never closes (the reply was cut off while reasoning) is reasoning to the end of the
    reply. With ``lone_closer`` (thinking explicitly on, where a template can put ``<think>`` in
    the prompt so that the reply holds only ``</think>``), the text before a first ``</think>``
    that no ``<think>`` precedes is removed too. A tag anywhere else is content, for example an
    injected token the agent quotes or a judge quoting the agent."""
    text = content or ""
    if lone_closer:
        head, sep, tail = text.partition("</think>")
        if sep and "<think>" not in head:
            text = tail
    text = _without_leading_blocks(text)
    if drop_unclosed and _UNCLOSED_REASONING.match(text):
        return ""
    return text.strip()


def returned_reasoning(message) -> bool:
    """Whether the server returned the reply's reasoning in a field of its own, as vLLM and other
    servers with a reasoning parser do (``reasoning_content`` or ``reasoning``). The reply text
    then holds no reasoning, and reasoning-like tags in it are content."""
    for name in ("reasoning_content", "reasoning"):
        value = getattr(message, name, None)
        if value is None:
            value = (getattr(message, "model_extra", None) or {}).get(name)
        if value:
            return True
    return False


def strip_leading_reasoning(text: str | None) -> str:
    """Remove reasoning that a generator or judge reply opens with, so that a draft
    written inside it is not parsed as the answer: leading reasoning blocks, a block that never
    closes (all reasoning, so nothing is left), and, for a template that put the opener in the
    prompt, the text before a closing tag when AutoPI's answer starts after it. A closing tag
    inside the answer itself (a generated line can carry such tokens) is kept as written."""
    out = _without_leading_blocks(text or "")
    if _UNCLOSED_REASONING.match(out):
        return ""
    for opener, closer in _REASONING_DELIMITERS:
        head, sep, tail = out.partition(closer)
        if sep and opener not in head and _ANSWER_MARKER.search(tail):
            return tail
    return out


def http_status(exc: BaseException) -> int | None:
    """The HTTP status of a provider or transport error, from whichever attribute carries it."""
    for value in (getattr(exc, "status_code", None), getattr(getattr(exc, "response", None), "status_code", None),
                  getattr(exc, "code", None)):
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


_MODERATION_TEXT = re.compile(r"moderation|flagged|guardrail|content[ _-]?polic|content[ _]filter", re.IGNORECASE)


def is_content_policy_rejection(exc: BaseException) -> bool:
    """A request the provider refused on content-policy grounds, as opposed to one it could not
    accept: Azure OpenAI's ``content_filter`` (inner code ``ResponsibleAIPolicyViolation``) and
    OpenAI's ``invalid_prompt`` for a flagged prompt (both HTTP 400), and a moderation or
    guardrail block answered with HTTP 403 (OpenRouter)."""
    status = http_status(exc)
    if status not in (400, 403):
        return False
    code = getattr(exc, "code", None)
    reply = getattr(exc, "response", None)
    text = " ".join(str(part) for part in (getattr(exc, "body", "") or "", getattr(reply, "text", "") or "", exc))
    if status == 403:
        return bool(_MODERATION_TEXT.search(text))
    if code == "content_filter" or "ResponsibleAIPolicyViolation" in text or '"content_filter"' in text:
        return True
    return (code == "invalid_prompt" or "invalid_prompt" in text) and "flagged" in text.lower()


def _remember_tool_call_extras(response, extras: OrderedDict, size: int = 1024) -> None:
    """Keep the provider fields of each returned tool call under the call's id. Gemini's
    OpenAI-compatible API puts the thought signature there (``extra_content``)."""
    for choice in getattr(response, "choices", None) or []:
        for call in getattr(getattr(choice, "message", None), "tool_calls", None) or []:
            extra = getattr(call, "extra_content", None)
            if extra is None:
                extra = (getattr(call, "model_extra", None) or {}).get("extra_content")
            call_id = getattr(call, "id", None)
            if extra and call_id:
                extras[call_id] = extra
                extras.move_to_end(call_id)
    while len(extras) > size:
        extras.popitem(last=False)


def _with_tool_call_extras(messages, extras: OrderedDict):
    """Put those fields back on the same tool calls when the conversation is sent again:
    Gemini 3 and later reject a follow-up request without the thought signatures."""
    if not extras or not isinstance(messages, list):
        return messages

    def needs(call):
        return isinstance(call, dict) and "extra_content" not in call and call.get("id") in extras

    out, changed = [], False
    for message in messages:
        calls = message.get("tool_calls") if isinstance(message, dict) else None
        if calls and any(needs(call) for call in calls):
            message = dict(message, tool_calls=[
                dict(call, extra_content=extras[call["id"]]) if needs(call) else call for call in calls])
            changed = True
        out.append(message)
    return out if changed else messages


# Google's documented placeholder thought signature for a function-call part that the API did
# not generate, such as a prefilled tool call in a benchmark conversation. Gemini 3 and later
# otherwise reject the request (HTTP 400) for a missing current-turn signature. Google notes
# this may reduce model performance; it is a last resort for calls with no real signature.
_GOOGLE_PLACEHOLDER_SIGNATURE = {"google": {"thought_signature": "skip_thought_signature_validator"}}


def _with_placeholder_signatures(messages):
    """For a Google OpenAI-compatible endpoint, add the placeholder signature to assistant tool
    calls that still carry none (prefilled calls the API never produced a signature for)."""
    if not isinstance(messages, list):
        return messages

    def needs(call):
        return isinstance(call, dict) and not call.get("extra_content")

    out, changed = [], False
    for message in messages:
        calls = message.get("tool_calls") if isinstance(message, dict) else None
        if calls and any(needs(call) for call in calls):
            message = dict(message, tool_calls=[
                dict(call, extra_content=_GOOGLE_PLACEHOLDER_SIGNATURE) if needs(call) else call for call in calls])
            changed = True
        out.append(message)
    return out if changed else messages


def instrument_openai_client(client, endpoint: ModelEndpoint, *, journal=None, role="target"):
    """Apply settings to this client only, return provider tool-call fields (Gemini's thought
    signatures) with the calls they came with, and optionally recover prompted-tool text."""
    decoder = get_decoder(endpoint.tokenizer_path) if endpoint.tokenizer_path else None
    original = client.chat.completions.create
    tool_call_extras: OrderedDict = OrderedDict()

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
            # AgentDojo's client passes reasoning_effort=NOT_GIVEN, which counts as unset here.
            if endpoint.thinking == "off":
                kwargs["reasoning_effort"] = "none"
            elif _unset(kwargs.get("reasoning_effort")):
                kwargs["reasoning_effort"] = "medium"
        if role == "target":
            # An alias's decoding settings shape the target's requests. IPI Arena's scoring judge
            # and WorldSim keep the temperature and token caps IPI Arena sets on each call, which
            # are part of its evaluation.
            for key in ("temperature", "top_p", "max_tokens"):
                value = getattr(endpoint, key)
                if value is not None:
                    kwargs[key] = value
        if endpoint.max_tokens_field == "max_completion_tokens" and "max_tokens" in kwargs:
            # OpenAI's reasoning models take the output cap only as max_completion_tokens.
            value = kwargs.pop("max_tokens")
            if value is not None:
                kwargs["max_completion_tokens"] = value
        for key in _omitted_request_fields(endpoint):
            kwargs.pop(key, None)
        if endpoint.provider == "anthropic":
            # IPI Arena's client always sends temperature 0. Through the OpenAI-compatible API,
            # Claude 4.7 and later reject a temperature below 1 and any top_p but the default
            # (HTTP 400), so sampling parameters are left out, as on AutoPI's other Anthropic
            # requests, and the API's default sampling applies.
            for key in ("temperature", "top_p", "top_k"):
                kwargs.pop(key, None)
        if "messages" in kwargs:
            kwargs["messages"] = _with_tool_call_extras(kwargs["messages"], tool_call_extras)
            if endpoint.provider == "google":
                # IPI Arena prefills assistant tool calls the Gemini API never signed; add the
                # documented placeholder so Gemini 3+ does not reject the request (HTTP 400).
                kwargs["messages"] = _with_placeholder_signatures(kwargs["messages"])
        if decoder:
            kwargs = decoder.prepare(kwargs)
        call_id = uuid.uuid4().hex
        if journal:
            journal.emit("llm_request", role=role, call_id=call_id, endpoint=redact_url(endpoint.base_url),
                         request=kwargs, tokenizer_sha256=decoder.sha256 if decoder else None)
        try:
            response = original(*args, **kwargs)
            _remember_tool_call_extras(response, tool_call_extras)
            if journal:
                journal.emit("llm_response_original", role=role, call_id=call_id, response=response)
            if decoder:
                recovered = decoder.recover(response.model_dump(mode="json"))
                # Recovery requires thinking off, so the recovered text holds no reasoning to remove;
                # reasoning tags in it are content and are kept.
                for choice, replacement in zip(response.choices, recovered["choices"], strict=True):
                    choice.message.content = replacement["message"]["content"]
                    choice.message.tool_calls = None
                if journal:
                    journal.emit("llm_response_recovered", role=role, call_id=call_id, response=recovered)
            elif endpoint.tool_mode == "prompted" and endpoint.may_leak_reasoning:
                # AgentDojo's prompted parser (LocalLLM) scans message.content for
                # ``<function=...>`` text. A target that may think can open its reply with
                # reasoning; remove that so a hypothetical call written inside the reasoning is
                # not parsed as a real tool call (a reply cut off while reasoning is all
                # reasoning). Only reasoning that opens the reply is removed, never tags later in
                # the reply, and nothing when the server returned the reasoning in a field of its
                # own. A thinking-off target does not reason, so nothing is removed either: a
                # reasoning tag in its reply is content, such as an injected token it quotes.
                for choice in getattr(response, "choices", None) or []:
                    message = getattr(choice, "message", None)
                    if (message is not None and getattr(message, "content", None)
                            and not returned_reasoning(message)):
                        message.content = visible_text(message.content, drop_unclosed=True,
                                                       lone_closer=endpoint.thinking == "on")
            return response
        except Exception as exc:
            if journal:
                journal.emit("llm_error", role=role, call_id=call_id, error_type=type(exc).__name__, error=str(exc))
            raise

    client.chat.completions.create = create
    return client


# A rate-limited request (HTTP 429) is retried at least this many times, waiting as long as the
# server's Retry-After asks (at most _MAX_RETRY_AFTER seconds each time), before the run stops.
_RATE_LIMIT_RETRIES = 5
_MAX_RETRY_AFTER = 120.0


def _retry_after_seconds(response) -> float | None:
    """The wait a ``Retry-After`` header asks for, in seconds (a number or an HTTP date)."""
    value = (getattr(response, "headers", None) or {}).get("Retry-After")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        from datetime import datetime, timezone
        from email.utils import parsedate_to_datetime

        return max(0.0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return None


def _post_json(url: str, payload: dict, endpoint: ModelEndpoint, *, headers=None, journal=None,
               role="refiner", call_id=None):
    """POST with the endpoint's timeout and retries and return the JSON reply. A timeout, a
    connection error or a server error is retried with backoff, HTTP 429 at least
    ``_RATE_LIMIT_RETRIES`` times and as long as the server's ``Retry-After`` asks; any other
    HTTP error is raised at once."""
    attempt = 0
    while True:
        try:
            response = requests.post(url, json=payload, timeout=endpoint.timeout, headers=headers)
            if journal:
                journal.emit("llm_response_original", role=role, call_id=call_id,
                             retry=attempt, status_code=response.status_code, body=response.text)
            response.raise_for_status()
            return response.json()
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
            reply = exc.response if isinstance(exc, requests.HTTPError) else None
            status = getattr(reply, "status_code", None)
            retryable = status is None or status >= 500 or status == 429
            limit = max(endpoint.retries, _RATE_LIMIT_RETRIES) if status == 429 else endpoint.retries
            if journal:
                journal.emit("llm_error", role=role, call_id=call_id, retry=attempt, error=str(exc))
            if not retryable or attempt >= limit:
                raise
            wait = _retry_after_seconds(reply) if status in (429, 503) else None
            time.sleep(min(wait, _MAX_RETRY_AFTER) if wait is not None else min(2 ** attempt, 30 if status == 429 else 10))
            attempt += 1


def query_endpoint(endpoint: ModelEndpoint, prompt: str, *, temperature=0.7, max_tokens=1024,
                   journal=None, role="refiner") -> str:
    """Text generation with an explicit provider, model id and optional remote URL."""
    temperature = endpoint.temperature if endpoint.temperature is not None else temperature
    max_tokens = endpoint.max_tokens if endpoint.max_tokens is not None else max_tokens
    if endpoint.provider in {"openai-compatible", "openai"}:
        # An openai endpoint without its own URL goes where the OpenAI SDK sends the target:
        # OPENAI_BASE_URL when it is set, else the public API.
        default = (os.environ.get("OPENAI_BASE_URL") if endpoint.provider == "openai" else None) or "https://api.openai.com/v1"
        url = (endpoint.base_url or default).rstrip("/") + "/chat/completions"
        payload = {"model": endpoint.model, "messages": [{"role": "user", "content": prompt}],
                   "temperature": temperature, "max_tokens": max_tokens}
        if endpoint.top_p is not None:
            payload["top_p"] = endpoint.top_p
        payload.update(endpoint.extra_body)
        if endpoint.reasoning_effort:
            payload["reasoning_effort"] = endpoint.reasoning_effort
        if endpoint.provider == "openai-compatible" and endpoint.thinking != "server":
            payload["chat_template_kwargs"] = {**payload.get("chat_template_kwargs", {}), "enable_thinking": endpoint.thinking == "on"}
        if endpoint.max_tokens_field == "max_completion_tokens":
            payload["max_completion_tokens"] = payload.pop("max_tokens")
        for key in _omitted_request_fields(endpoint):
            payload.pop(key, None)
        decoder = get_decoder(endpoint.tokenizer_path) if endpoint.tokenizer_path else None
        if decoder:
            payload = decoder.prepare(payload)
        call_id = uuid.uuid4().hex
        if journal:
            journal.emit("llm_request", role=role, call_id=call_id, endpoint=redact_url(url), request=payload,
                         tokenizer_sha256=decoder.sha256 if decoder else None)
        data = _post_json(url, payload, endpoint, headers={"Authorization": "Bearer " + endpoint.api_key()},
                          journal=journal, role=role, call_id=call_id)
        if decoder:
            # Generator text keeps special tokens: they can be the structure it wrote.
            data = decoder.recover(data, keep_special_tokens=True)
            if journal:
                journal.emit("llm_response_recovered", role=role, call_id=call_id, response=data)
        content = data["choices"][0]["message"].get("content")
        # Structural markers may be the requested injection construction.
        # Preserve the generator's text; its extractor handles delimiters.
        return (content or "").strip()
    if endpoint.provider == "anthropic":
        import anthropic

        require_no_reasoning_effort(endpoint)
        client = anthropic.Anthropic(api_key=endpoint.api_key(), timeout=endpoint.timeout,
                                     max_retries=endpoint.retries, **({"base_url": endpoint.base_url} if endpoint.base_url else {}))
        # No sampling parameters: the Anthropic API (version 2026-10-21) rejects them on every model.
        payload = dict(model=endpoint.model, messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens)
        for key in endpoint.omit_parameters:
            payload.pop(key, None)
        if endpoint.extra_body:
            # Anthropic-only request fields (for example thinking settings) go into the body.
            payload["extra_body"] = dict(endpoint.extra_body)
        if journal:
            journal.emit("llm_request", role=role, request=payload)
        response = client.messages.create(**payload)
        if journal:
            journal.emit("llm_response_original", role=role, response=response)
        return "".join(block.text for block in response.content if block.type == "text").strip()
    if endpoint.provider == "google":
        from google import genai

        require_no_reasoning_effort(endpoint)
        kwargs = {"api_key": endpoint.api_key(), "http_options": {
            "timeout": int(endpoint.timeout * 1000), "retry_options": {"attempts": endpoint.retries + 1}}}
        if endpoint.base_url:
            kwargs["http_options"]["base_url"] = endpoint.base_url
        client = genai.Client(**kwargs)
        config = {"temperature": temperature, "max_output_tokens": max_tokens}
        if endpoint.top_p is not None:
            config["top_p"] = endpoint.top_p
        config.update(endpoint.extra_body)              # extra GenerateContentConfig fields
        for key in google_omitted_fields(endpoint):
            config.pop(key, None)
        if journal:
            journal.emit("llm_request", role=role, model=endpoint.model, prompt=prompt, config=config)
        response = client.models.generate_content(model=endpoint.model, contents=prompt, config=config)
        if journal:
            journal.emit("llm_response_original", role=role, response=response)
        return (response.text or "").strip()
    if endpoint.provider == "ollama":
        options = {"temperature": temperature, "num_predict": max_tokens}
        if endpoint.top_p is not None:
            options["top_p"] = endpoint.top_p
        payload = {"model": endpoint.model, "messages": [{"role": "user", "content": prompt}],
                   "stream": False, "options": options}
        if endpoint.thinking != "server":
            payload["think"] = endpoint.thinking == "on"
        extra = dict(endpoint.extra_body or {})
        if isinstance(extra.get("options"), dict):
            # Ollama's own options (for example num_ctx) join AutoPI's sampling settings.
            options.update(extra.pop("options"))
        payload.update(extra)
        ollama_names = {"max_tokens": "num_predict"}
        for key_name in endpoint.omit_parameters:
            payload.pop(key_name, None)
            options.pop(ollama_names.get(key_name, key_name), None)
        key = endpoint.api_key()  # "EMPTY" when no api_key_env is configured
        call_id = uuid.uuid4().hex
        if journal:
            journal.emit("llm_request", role=role, call_id=call_id, request=payload)
        data = _post_json(endpoint.base_url.rstrip("/") + "/api/chat", payload, endpoint,
                          headers={"Authorization": "Bearer " + key} if key != "EMPTY" else None,
                          journal=journal, role=role, call_id=call_id)
        return (data["message"]["content"] or "").strip()
    raise ValueError(f"provider {endpoint.provider!r} cannot generate text; configure an explicit endpoint")
