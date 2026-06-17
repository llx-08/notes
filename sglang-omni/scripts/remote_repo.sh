#!/usr/bin/env bash
set -euo pipefail

HOST="${SGLANG_OMNI_SSH_HOST:-dsw-dpsk-v32}"
REPO="${SGLANG_OMNI_REPO:-/mnt/data/llx/sglang-omni}"

if [[ $# -eq 0 ]]; then
  printf 'Usage: %s COMMAND\n' "$0" >&2
  printf 'Example: %s %q\n' "$0" 'git status --short --branch' >&2
  exit 2
fi

printf -v quoted_repo '%q' "$REPO"
remote_command="cd $quoted_repo && $*"

ssh -o BatchMode=yes -o ConnectTimeout=12 "$HOST" "$remote_command"
