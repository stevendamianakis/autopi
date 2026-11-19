"""Incremental request/response evidence, without authorization headers or keys."""
from __future__ import annotations

import dataclasses
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path


class TraceWriteError(RuntimeError):
    fatal = True


def plain(value):
    if hasattr(value, "model_dump"):
        return plain(value.model_dump(mode="json"))
    if dataclasses.is_dataclass(value):
        return plain(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()
                if not re.search(r"authorization|api[_-]?key$|password|secret", str(key), re.I)}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class Journal:
    def __init__(self, root: str | Path, extra_secret_envs=None):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / f"events-{uuid.uuid4().hex}.jsonl"
        self.context = {}
        # Environment variables whose value must be scrubbed from every event, in addition to
        # the ones whose name matches the default pattern. Used for a custom ``api_key_env``.
        self.extra_secret_envs = set(extra_secret_envs or ())

    def emit(self, event: str, **fields):
        record = dict(self.context, event=event, utc=datetime.now(timezone.utc).isoformat(), **plain(fields))
        text = json.dumps(record, ensure_ascii=False) + "\n"
        # Also remove a credential if a provider includes it inside an exception string, for
        # a variable named like a secret or named as one of the run's configured api_key_env.
        for key, value in os.environ.items():
            if len(value) >= 8 and (key in self.extra_secret_envs
                                    or re.search(r"API_KEY|TOKEN|SECRET|PASSWORD", key)):
                text = text.replace(value, "<redacted>")
        try:
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as exc:
            raise TraceWriteError(f"cannot save trace {self.path}: {exc}") from exc
