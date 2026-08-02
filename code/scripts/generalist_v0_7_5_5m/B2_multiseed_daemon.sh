#!/bin/bash
# Detach helper: re-exec self with setsid so it survives parent shell exit.
# Usage: setsid_orchestrator.sh <orchestrator_script>
exec setsid bash "$@"
