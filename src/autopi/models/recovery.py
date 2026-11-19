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
        self.special_ids = {index for index, token in self.tokenizer.get_added_tokens_decoder().items()
                            if token.special}

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

    def recover(self, response: dict, *, keep_special_tokens: bool = False) -> dict:
        """Replace each choice's text with its decoded generated tokens.

        By default special tokens are skipped, as a vLLM server does in its own text. With
        ``keep_special_tokens`` (generator output) a special token such as ``<|im_start|>``
        stays in the text, since a structural injection can be made of it. Only the
        end-of-turn token that stopped generation is dropped: the last token, when it is
        special and the reply did not end at the length limit."""
        result = copy.deepcopy(response)
        choices = result.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ResponseRecoveryError("endpoint returned no choices")
        for choice in choices:
            ids = choice.get("token_ids")
            if not isinstance(ids, list) or any(type(value) is not int or value < 0 for value in ids):
                raise ResponseRecoveryError("missing or invalid choice.token_ids; refusing stripped response")
            vocab_size = getattr(self.tokenizer, "get_vocab_size", None)
            if callable(vocab_size) and ids and max(ids) >= vocab_size(with_added_tokens=True):
                # The tokenizer would silently drop an unknown id, losing generated text.
                raise ResponseRecoveryError("choice.token_ids holds an id outside the tokenizer's vocabulary; "
                                            "the configured tokenizer does not match the served model")
            if not isinstance(choice.get("message"), dict):
                raise ResponseRecoveryError("endpoint returned no message")
            try:
                if keep_special_tokens:
                    end = len(ids)
                    if end and ids[-1] in getattr(self, "special_ids", set()) and choice.get("finish_reason") != "length":
                        end -= 1
                    text = self.tokenizer.decode(ids[:end], skip_special_tokens=False)
                else:
                    text = self.tokenizer.decode(ids, skip_special_tokens=True)
                choice["message"]["content"] = text
            except Exception as exc:
                raise ResponseRecoveryError(f"cannot decode generated tokens: {exc}") from exc
            choice["message"]["tool_calls"] = None
        return result
