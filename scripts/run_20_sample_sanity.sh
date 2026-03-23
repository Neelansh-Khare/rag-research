#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -x "$REPO_DIR/.venv/bin/python" ]; then
  PYTHON="$REPO_DIR/.venv/bin/python"
else
  PYTHON="python"
fi

"$PYTHON" -m src.pipeline.run_pipeline \
  --config "$REPO_DIR/configs/frozen_baseline.yaml" \
  --mode sanity \
  --output-root "$REPO_DIR/outputs/baseline"

