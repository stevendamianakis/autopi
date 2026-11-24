"""AutoPI: taxonomy-guided adaptive red teaming for prompt injection in LLM agents."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("autopi")
except PackageNotFoundError:  # a source tree that is not installed
    __version__ = "unknown"
