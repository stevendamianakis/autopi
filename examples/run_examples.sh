#!/usr/bin/env bash
set -euo pipefail
: "${AUTOPI_MODELS_CONFIG:?Set AUTOPI_MODELS_CONFIG to your model-alias YAML file}"
: "${AUTOPI_OUTPUT_DIR:?Set AUTOPI_OUTPUT_DIR to a fresh results directory}"
autopi run --benchmark agentdojo --models-config "$AUTOPI_MODELS_CONFIG" \
  --target target --refiner generator --strategy configs/strategies/full.yaml \
  --logdir "$AUTOPI_OUTPUT_DIR/agentdojo"
# IPI Arena's 8 browser scenarios also need playwright and its Chromium (README, IPI Arena);
# add --suite browser once they are installed.
autopi run --benchmark ipiarena --models-config "$AUTOPI_MODELS_CONFIG" \
  --target arena-target --refiner generator --judge arena-judge --worldsim arena-judge \
  --seed-source agentdojo --suite tool --suite coding --strategy configs/strategies/full.yaml \
  --logdir "$AUTOPI_OUTPUT_DIR/ipiarena"
