#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -x "$REPO_DIR/.venv/bin/python" ]; then
  PYTHON="$REPO_DIR/.venv/bin/python"
elif [ -x "$REPO_DIR/.venv/Scripts/python.exe" ]; then
  PYTHON="$REPO_DIR/.venv/Scripts/python.exe"
else
  PYTHON="python"
fi

"$PYTHON" -m src.pipeline.run_pipeline \
  --config "$REPO_DIR/configs/scale_1k.yaml" \
  --mode scale_sweep \
  --output-root "$REPO_DIR/outputs"

"$PYTHON" -m src.pipeline.run_pipeline \
  --config "$REPO_DIR/configs/scale_10k.yaml" \
  --mode scale_sweep \
  --output-root "$REPO_DIR/outputs"

"$PYTHON" -m src.pipeline.run_pipeline \
  --config "$REPO_DIR/configs/scale_100k.yaml" \
  --mode scale_sweep \
  --output-root "$REPO_DIR/outputs"

