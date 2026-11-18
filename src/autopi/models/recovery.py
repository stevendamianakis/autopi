"""Opt-in recovery of prompted-tool text stripped by an endpoint's tool parser.

The tokenizer must be the exact tokenizer used by the serving process. Native tool
requests and thinking-on requests cannot use this recovery path.
"""
from __future__ import annotations

import copy
import hashlib
from functools import lru_cache
from pathlib import Path


class ResponseRecoveryError(RuntimeError):
    fatal = True


@lru_cache(maxsize=8)
def _load_decoder(path, mtime_ns, size):
    return RawTokenDecoder(path)


def get_decoder(path):
    stat = Path(path).stat()
    return _load_decoder(str(Path(path).resolve()), stat.st_mtime_ns, stat.st_size)


class RawTokenDecoder:
    def __init__(self, tokenizer_path: str):
        from tokenizers import Tokenizer

        path = Path(tokenizer_path)
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.tokenizer = Tokenizer.from_file(str(path))

    def prepare(self, payload: dict) -> dict:
        payload = copy.deepcopy(payload)
        extra = payload.get("extra_body") or {}
        template = extra.get("chat_template_kwargs", payload.get("chat_template_kwargs", {}))
        if payload.get("stream") or payload.get("tools") or payload.get("functions"):
            raise ResponseRecoveryError("raw recovery supports non-streaming text/prompted tools only")
        if template.get("enable_thinking") is not False:
            raise ResponseRecoveryError("raw recovery requires explicit enable_thinking=False")
        if "extra_body" in payload:
            payload["extra_body"]["return_token_ids"] = True
        else:
            payload["return_token_ids"] = True
        return payload

    def recover(self, response: dict) -> dict:
        result = copy.deepcopy(response)
        choices = result.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ResponseRecoveryError("endpoint returned no choices")
        for choice in choices:
            ids = choice.get("token_ids")
            if not isinstance(ids, list) or any(type(value) is not int or value < 0 for value in ids):
                raise ResponseRecoveryError("missing or invalid choice.token_ids; refusing stripped response")
            if not isinstance(choice.get("message"), dict):
                raise ResponseRecoveryError("endpoint returned no message")
            try:
                choice["message"]["content"] = self.tokenizer.decode(ids, skip_special_tokens=True)
            except Exception as exc:
                raise ResponseRecoveryError(f"cannot decode generated tokens: {exc}") from exc
            choice["message"]["tool_calls"] = None
        return result
