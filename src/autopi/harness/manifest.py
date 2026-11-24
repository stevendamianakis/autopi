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
    # Hash the full scenario set the benchmark provides, not the subset one worker selected, so
    # the digest is the same for every shard/task selection (IPI: all behaviors; AgentDojo: all
    # suites of the version). ``_all_behaviors`` is the unfiltered IPI set; ``_behaviors`` is the
    # selection, used only as a fallback for an adapter that does not expose the full set.
    behaviors = getattr(adapter, "_all_behaviors", None)
    if behaviors is None:
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


_DIFFERENT_RUN = "output directory belongs to different settings/code/tokenizers; choose a fresh --logdir"


def check_run_manifest(logdir: Path, settings: dict) -> None:
    """Raise ValueError, without writing anything, when ``logdir`` holds a run with other settings,
    code or tokenizers (or results without a manifest). The CLI calls it before the first request,
    so a refused run leaves nothing (not even a trace) in the other run's folder."""
    manifest = logdir / "run_manifest.json"
    if not manifest.exists():
        if any((logdir / "refiner_logs").rglob("*.json")):
            raise ValueError("existing results have no run manifest; choose a fresh --logdir")
        return
    if json.loads(manifest.read_text(encoding="utf-8")) != _manifest_record(settings):
        raise ValueError(_DIFFERENT_RUN)


def save_run_manifest(logdir: Path, settings: dict):
    logdir.mkdir(parents=True, exist_ok=True)
    record = _manifest_record(settings)
    manifest = logdir / "run_manifest.json"
    if not manifest.exists() and any((logdir / "refiner_logs").rglob("*.json")):
        raise ValueError("existing results have no run manifest; choose a fresh --logdir")
    temporary = logdir / (".manifest-" + uuid.uuid4().hex)
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        try:
            os.link(temporary, manifest)
        except FileExistsError:
            if json.loads(manifest.read_text(encoding="utf-8")) != record:
                raise ValueError(_DIFFERENT_RUN) from None
    finally:
        temporary.unlink()


def _manifest_record(settings: dict) -> dict:
    versions, sources = {}, {}
    modules = {"autopi": "autopi", "agentdojo": "agentdojo", "ipi_arena_bench": "ipi-arena-bench",
               "openai": "openai", "anthropic": "anthropic", "google_genai": "google-genai",
               "tokenizers": "tokenizers"}
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
        # A role the run does not use (AutoPI's judge when it is off, IPI Arena's judge on
        # AgentDojo) is recorded as None.
        path = (model or {}).get("tokenizer_path")
        if path:
            tokenizers[role] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return {"format_version": 1, "settings": settings, "versions": versions, "sources": sources,
            "tokenizer_sha256": tokenizers}
