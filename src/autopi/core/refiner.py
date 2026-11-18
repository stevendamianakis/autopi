"""Text-in/text-out client for the generator and diagnostic judge.

Prefer RefinerConfig(endpoint=ModelEndpoint(...)). The legacy port/environment
fields remain for existing experiment drivers; they resolve to the same shared
provider client, with no benchmark dependency.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests as requests  # Compatibility: callers may replace this module's HTTP transport.

from autopi.models.endpoints import ModelEndpoint, resolve_endpoint


@dataclass
class RefinerConfig:
    endpoint: ModelEndpoint | None = None
    journal: Any = None
    role: str = "refiner"
    temperature: float = 0.7
    vllm_port: str | None = None
    vllm_nothink: bool | None = None
    vllm_max_tokens: int | None = None
    vllm_timeout: int | None = None
    ollama_port: str | None = None

    def _vllm_port(self):
        return self.vllm_port if self.vllm_port is not None else os.environ.get("VLLM_REFINER_PORT")

    def _ollama_port(self):
        return self.ollama_port if self.ollama_port is not None else os.environ.get("LOCAL_LLM_PORT", "11434")

    def _vllm_nothink(self):
        return self.vllm_nothink if self.vllm_nothink is not None else os.environ.get("VLLM_REFINER_NOTHINK") == "1"

    def _vllm_max_tokens(self):
        return self.vllm_max_tokens if self.vllm_max_tokens is not None else int(os.environ.get("VLLM_REFINER_MAX_TOKENS", "1024"))

    def _vllm_timeout(self):
        return self.vllm_timeout if self.vllm_timeout is not None else float(os.environ.get("VLLM_REFINER_TIMEOUT", "240"))

    def resolve(self, model: str):
        if self.endpoint is not None:
            return self.endpoint
        if self._vllm_port():
            extra = {} if self._vllm_nothink() else {
                "chat_template_kwargs": {"reasoning_effort": os.environ.get("VLLM_REFINER_EFFORT", "low")}}
            return ModelEndpoint("openai-compatible", model,
                                 f"http://localhost:{self._vllm_port()}/v1",
                                 thinking="off" if self._vllm_nothink() else "server",
                                 temperature=self.temperature, max_tokens=self._vllm_max_tokens(),
                                 timeout=self._vllm_timeout(), extra_body=extra)
        if model.startswith(("vllm:", "openai:", "anthropic:", "google:", "ollama:",
                             "gpt-", "o1", "o3", "o4", "claude-", "gemini-")):
            return resolve_endpoint(model, role="refiner")
        return ModelEndpoint("ollama", model,
                             os.environ.get("OLLAMA_BASE_URL", f"http://localhost:{self._ollama_port()}"),
                             thinking="off", temperature=self.temperature,
                             max_tokens=self._vllm_max_tokens(), timeout=self._vllm_timeout())


def query_refiner_llm(model: str, prompt: str, cfg: RefinerConfig | None = None) -> str:
    from autopi.models.clients import query_endpoint

    cfg = cfg or RefinerConfig()
    return query_endpoint(cfg.resolve(model), prompt, temperature=cfg.temperature,
                          max_tokens=cfg._vllm_max_tokens(), journal=cfg.journal, role=cfg.role)
