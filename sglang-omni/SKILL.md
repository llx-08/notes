---
name: sglang-omni
description: Guide for working on the remote DSW SGLang-Omni repository at /mnt/data/llx/sglang-omni over SSH. Use when Codex needs to inspect, modify, launch, test, or debug SGLang-Omni code; understand its pipeline/config/stage/scheduler/serve/router/profiler architecture; work on Qwen3-Omni, Qwen3.5-Omni, TTS, ASR, multimodal model integrations; or run remote DSW validation commands.
---

# SGLang-Omni

## Remote Context

Work in the remote repo unless the user explicitly says otherwise:

- SSH host: `dsw-dpsk-v32` (same target as `blade-server` in the local SSH config)
- Repo path: `/mnt/data/llx/sglang-omni`
- Observed branch/commit when this skill was created: `main` at `ec1173a feat: align qwen3.5 omni speech inference`

Always refresh the remote state before relying on these notes:

```bash
/Users/llx/.codex/skills/sglang-omni/scripts/remote_repo.sh 'git status --short --branch && git log -1 --oneline --decorate'
```

Use `git grep` or `find` inside the remote repo for code search; use `rg` only after confirming it is installed. The DSW login banner appears on SSH stdout; ignore the banner and read the command output after it.

## Workflow

1. Inspect the current remote state with the helper script, then read the directly relevant files. Do not assume the reference notes are newer than the repo.
2. Pick the narrowest reference:
   - Read [repo-map.md](references/repo-map.md) for framework architecture, key files, and ownership boundaries.
   - Read [qwen35-omni.md](references/qwen35-omni.md) for Qwen3.5-Omni launchers, stage topology, request builders, preflight, and code2wav behavior.
   - Read [ops-and-tests.md](references/ops-and-tests.md) for install, serve, router, profiling, and validation commands.
3. Keep changes within the owning layer:
   - Put model-family behavior under `sglang_omni/models/<model>/`.
   - Put generic topology/runtime behavior under `sglang_omni/config/` or `sglang_omni/pipeline/`.
   - Put HTTP/OpenAI schema behavior under `sglang_omni/serve/` and client aggregation behavior under `sglang_omni/client/`.
   - Put external worker-pool routing behavior under `sglang_omni_router/`.
4. Preserve config invariants:
   - `StageConfig` must declare one of `next` or `terminal=True`.
   - Non-TP stages should declare `process`.
   - `wait_for` requires `merge_fn`; request-specific fan-in should use `wait_for_fn`.
   - Keep `stream_to` as the static superset when using `stream_done_to_fn`.
5. Validate with the smallest meaningful remote command. Prefer unit tests for contract changes; use model/benchmark tests only when the change touches real serving quality or GPU behavior.

## Remote Helper

Run a command from the repo root:

```bash
/Users/llx/.codex/skills/sglang-omni/scripts/remote_repo.sh 'git grep -n "class PipelineConfig" -- sglang_omni/config'
```

Override host or path when needed:

```bash
SGLANG_OMNI_SSH_HOST=wan-dsw SGLANG_OMNI_REPO=/other/path \
  /Users/llx/.codex/skills/sglang-omni/scripts/remote_repo.sh 'pwd'
```

For edits, prefer non-interactive commands and patches. Check `git status --short` before and after; do not overwrite unrelated remote work.
