#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/python"
if command -v uv >/dev/null 2>&1; then uv run python -m pfs "$@"; else python3 -m pfs "$@"; fi
