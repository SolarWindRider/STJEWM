#!/usr/bin/env bash
# Thin wrapper: just call the actual sweep script.
# Keep this name stable as the canonical entry point.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_stress_sweep.sh" "$@"