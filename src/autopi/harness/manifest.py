"""Record source provenance and prevent accidental mixing of incompatible runs."""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import uuid
from pathlib import Path


def source_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".json", ".yaml"} and "__pycache__" not in path.parts:
            digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def benchmark_data_digest(adapter) -> str:
    """SHA-256 of the scenario definitions a run draws from.

    IPI Arena: the scenario files the adapter loaded. AgentDojo: the user prompts, injection
    goals and default injection-vector contents of every suite at the adapter's benchmark
    version, so the value does not depend on which suites or tasks one worker selects.
    """
    behaviors = getattr(adapter, "_behaviors", None)
    if behaviors is not None:
        payload = behaviors
    elif getattr(adapter, "benchmark_version", None) is not None and hasattr(adapter, "_suites"):
        from agentdojo.task_suite.load_suites import get_suites

        payload = {
            "benchmark_version": adapter.benchmark_version,
            "suites": {
                name: {
                    "user_tasks": {key: task.PROMPT for key, task in suite.user_tasks.items()},
                    "injection_tasks": {key: task.GOAL for key, task in suite.injection_tasks.items()},
                    "injection_vectors": suite.get_injection_vector_defaults(),
                }
                for name, suite in get_suites(adapter.benchmark_version).items()
            },
        }
    else:
        payload = {}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def save_run_manifest(logdir: Path, settings: dict):
    logdir.mkdir(parents=True, exist_ok=True)
    versions, sources = {}, {}
    modules = {"autopi": "autopi", "agentdojo": "agentdojo", "ipi_arena_bench": "ipi-arena-bench",
               "openai": "openai", "tokenizers": "tokenizers"}
    for module_name, distribution in modules.items():
        try:
            versions[distribution] = importlib.metadata.version(distribution)
            if module_name in {"autopi", "agentdojo", "ipi_arena_bench"}:
                module = importlib.import_module(module_name)
                sources[module_name] = source_digest(Path(module.__file__).parent)
        except (ImportError, importlib.metadata.PackageNotFoundError):
            continue
    tokenizers = {}
    for role, model in settings["models"].items():
        path = model.get("tokenizer_path")
        if path:
            tokenizers[role] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    record = {"format_version": 1, "settings": settings, "versions": versions, "sources": sources,
              "tokenizer_sha256": tokenizers}
    manifest = logdir / "run_manifest.json"
    if not manifest.exists() and any((logdir / "refiner_logs").rglob("*.json")) and not manifest.exists():
        raise ValueError("existing results have no run manifest; choose a fresh --logdir")
    temporary = logdir / (".manifest-" + uuid.uuid4().hex)
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        try:
            os.link(temporary, manifest)
        except FileExistsError:
            if json.loads(manifest.read_text(encoding="utf-8")) != record:
                raise ValueError(
                    "output directory belongs to different settings/code/tokenizers; choose a fresh --logdir"
                ) from None
    finally:
        temporary.unlink()
