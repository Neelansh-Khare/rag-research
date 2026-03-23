#!/usr/bin/env bash
set -euo pipefail

# Default baseline: run the (frozen) config in baseline mode.
bash "$(dirname "$0")/run_200_sample_baseline.sh"

