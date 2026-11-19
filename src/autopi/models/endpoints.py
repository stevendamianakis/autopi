"""Explicit, per-role model configuration. No benchmark or machine-specific routing."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import yaml


class EndpointConfigError(ValueError):
    """A model endpoint is misconfigured (e.g. a required API key is not set). Distinct from a
    transient failure: it recurs on every request, so a run stops rather than skipping."""


@dataclass(frozen=True)
class ModelEndpoint:
    provider: str
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    tool_mode: str = "native"
    thinking: str = "server"
    tokenizer_path: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    timeout: float = 240
    retries: int = 2
    reasoning_effort: str | None = None
    extra_body: dict = field(default_factory=dict)
    omit_parameters: tuple[str, ...] = ()

    def __post_init__(self):
        if self.provider not in {"openai-compatible", "openai", "anthropic", "google", "ollama", "agentdojo"}:
            raise ValueError(f"unsupported provider {self.provider!r}")
        if self.provider == "ollama" and not self.base_url:
            # The same default server as an ollama:MODEL spec.
            object.__setattr__(self, "base_url", os.environ.get(
                "OLLAMA_BASE_URL", f"http://localhost:{os.environ.get('LOCAL_LLM_PORT', '11434')}"))
        if not self.model or not self.model.strip():
            raise ValueError("a model id is required; AutoPI never substitutes a different model")
        if self.provider == "openai-compatible" and not self.base_url:
            raise ValueError("openai-compatible endpoints require base_url")
        if not isinstance(self.extra_body, dict):
            raise ValueError("extra_body must be a mapping of provider-specific request fields")
        if set(self.extra_body) & {"model", "messages", "tools", "functions", "stream"}:
            raise ValueError("extra_body cannot replace model, messages, tools, functions or stream")
        if not isinstance(self.omit_parameters, (list, tuple)) or not all(isinstance(key, str) for key in self.omit_parameters):
            raise ValueError("omit_parameters must be a list of parameter names")
        if set(self.omit_parameters) & {"model", "messages", "tools", "functions"}:
            raise ValueError("omit_parameters cannot remove the model, conversation or tool definitions")
        if self.tool_mode not in {"prompted", "native"}:
            raise ValueError("tool_mode must be prompted or native")
        if self.tool_mode == "prompted" and self.provider not in {"openai-compatible", "ollama"}:
            # Prompted tools are parsed from text by AgentDojo's LocalLLM, which talks an
            # OpenAI-compatible chat API. A hosted anthropic/google/openai target uses that
            # provider's native tool API, so `prompted` there would silently run native (or,
            # for openai, send tool-role messages the API rejects). Fail fast instead.
            raise ValueError("tool_mode: prompted is only for openai-compatible or ollama endpoints; "
                             "hosted anthropic/google/openai targets use native tools")
        if self.thinking not in {"off", "on", "server"}:
            raise ValueError("thinking must be off, on, or server")
        if self.provider not in {"openai-compatible", "ollama"} and self.thinking != "server":
            raise ValueError("thinking off/on is an endpoint template option; use server for hosted providers")
        if self.provider == "anthropic" and (self.temperature is not None or self.top_p is not None):
            raise ValueError("anthropic endpoints take no temperature or top_p: the Anthropic API "
                             "(version 2026-10-21) rejects sampling parameters on every model")
        if self.provider == "anthropic" and "max_tokens" in self.omit_parameters:
            raise ValueError("anthropic endpoints cannot omit max_tokens: the Anthropic API requires it")
        if self.tokenizer_path and (self.provider != "openai-compatible" or self.thinking != "off"):
            raise ValueError("raw token recovery requires an OpenAI-compatible endpoint with thinking: off")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.timeout <= 0 or self.retries < 0:
            raise ValueError("timeout must be positive and retries must be nonnegative")

    def api_key(self) -> str:
        name = self.api_key_env or {
            "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY", "openai-compatible": "OPENAI_COMPATIBLE_API_KEY",
        }.get(self.provider)
        value = os.environ.get(name, "") if name else ""
        if self.api_key_env and not value:
            raise EndpointConfigError(f"set the {self.api_key_env} environment variable for this endpoint")
        if self.provider in {"openai", "anthropic", "google"} and not value:
            raise EndpointConfigError(f"set {name} for provider {self.provider}")
        return value or "EMPTY"

    def public_config(self) -> dict:
        """Config includes an environment variable NAME, never its secret value."""
        config = asdict(self)
        config["omit_parameters"] = list(self.omit_parameters)
        return config


def load_model_catalog(path: str | Path | None) -> dict[str, ModelEndpoint]:
    if path is None:
        return {}
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != {"models"} or not isinstance(data["models"], dict):
        raise ValueError("model config must contain one 'models' mapping")
    result = {}
    for alias, settings in data["models"].items():
        if not isinstance(alias, str) or not isinstance(settings, dict):
            raise ValueError("each model alias must map to endpoint settings")
        settings = dict(settings)
        # PyYAML also accepts the unquoted YAML 1.1 scalars on/off as booleans.
        if type(settings.get("thinking")) is bool:
            settings["thinking"] = "on" if settings["thinking"] else "off"
        if settings.get("tokenizer_path"):
            tokenizer = Path(settings["tokenizer_path"]).expanduser()
            settings["tokenizer_path"] = str((path.parent / tokenizer).resolve()) if not tokenizer.is_absolute() else str(tokenizer)
        try:
            result[alias] = ModelEndpoint(**settings)
        except TypeError as exc:
            raise ValueError(f"invalid settings for model alias {alias!r}: {exc}") from exc
    return result


def resolve_endpoint(spec: str, catalog: dict[str, ModelEndpoint] | None = None, *, role: str = "target") -> ModelEndpoint:
    if catalog and spec in catalog:
        return catalog[spec]
    if spec == "openai-compatible":
        model = os.environ.get("OPENAI_COMPATIBLE_MODEL_ID", "")
        url = os.environ.get("OPENAI_COMPATIBLE_BASE_URL")
        if not url:
            raise ValueError("set OPENAI_COMPATIBLE_BASE_URL and OPENAI_COMPATIBLE_MODEL_ID, or use --models-config")
        return ModelEndpoint("openai-compatible", model, url)
    if spec.startswith("vllm:"):
        parts = spec.split(":", 2)
        if len(parts) != 3 or not parts[1].isdigit() or not 0 < int(parts[1]) < 65536 or not parts[2]:
            raise ValueError("use vllm:PORT:MODEL_ID, or name an endpoint in --models-config")
        return ModelEndpoint("openai-compatible", parts[2], f"http://localhost:{parts[1]}/v1",
                             tool_mode="prompted", thinking="off")
    if spec.startswith("local:") or spec.startswith("ollama:"):
        return ModelEndpoint("ollama", spec.split(":", 1)[1],
                             os.environ.get("OLLAMA_BASE_URL", f"http://localhost:{os.environ.get('LOCAL_LLM_PORT', '11434')}"),
                             tool_mode="prompted")
    if spec == "local":
        # Bare "local" names no model and no port, so there is nothing to route to.
        raise ValueError("'local' names no model; use vllm:PORT:MODEL_ID, local:MODEL_ID (Ollama), "
                         "or a configured --models-config alias")
    for provider in ("openai", "anthropic", "google"):
        if spec.startswith(provider + ":"):
            return ModelEndpoint(provider, spec.split(":", 1)[1])
    # Backward compatible unprefixed hosted names. Explicit aliases work with ANY model id.
    if spec.startswith(("gpt-", "o1", "o3", "o4")):
        return ModelEndpoint("openai", spec)
    if spec.startswith("claude-"):
        return ModelEndpoint("anthropic", spec)
    if spec.startswith("gemini-"):
        return ModelEndpoint("google", spec)
    if role == "target":
        return ModelEndpoint("agentdojo", spec)
    raise ValueError(f"unknown model {spec!r}; use a configured alias or an explicit provider:MODEL_ID")


def override_endpoint(endpoint: ModelEndpoint, **changes) -> ModelEndpoint:
    return replace(endpoint, **{key: value for key, value in changes.items() if value is not None})
