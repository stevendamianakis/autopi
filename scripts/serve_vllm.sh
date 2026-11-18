#!/usr/bin/env bash
# All model/server choices are supplied by the researcher.
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo 'Usage: scripts/serve_vllm.sh MODEL_OR_PATH [vllm serve options...]' >&2
  exit 2
fi
exec vllm serve "$@"
