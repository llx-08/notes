#!/usr/bin/env bash
# One-time per clone: use tracked hooks under .githooks/
set -euo pipefail
cd "$(dirname "$0")/.."
git config core.hooksPath .githooks
echo "Git hooks path set to .githooks (pre-commit runs Hexo sync before each commit)."
echo "To skip once: git commit --no-verify"
