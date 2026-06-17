# Operations and Tests

## Contents

- Remote helper
- Installation
- API server
- TTS
- Qwen3-Omni
- Router
- Profiling
- Tests and CI
- Qwen3.5 validation

## Remote Helper

Run commands from `/mnt/data/llx/sglang-omni` on the DSW host:

```bash
/Users/llx/.codex/skills/sglang-omni/scripts/remote_repo.sh 'pwd && git status --short --branch'
```

Override defaults:

```bash
SGLANG_OMNI_SSH_HOST=dsw-dpsk-v32 \
SGLANG_OMNI_REPO=/mnt/data/llx/sglang-omni \
/Users/llx/.codex/skills/sglang-omni/scripts/remote_repo.sh 'git grep -n "TODO" -- sglang_omni'
```

The DSW login banner is printed on SSH stdout. Ignore it and inspect the command output after the banner.

## Installation

Docker is the documented recommended path:

```bash
docker pull frankleeeee/sglang-omni:dev
docker run -it \
  --shm-size 32g \
  --gpus all \
  --ipc host \
  --network host \
  --privileged \
  frankleeeee/sglang-omni:dev \
  /bin/zsh
```

Inside the container:

```bash
git clone git@github.com:sgl-project/sglang-omni.git
cd sglang-omni
uv venv .venv -p 3.12
source .venv/bin/activate
uv pip install -v -e .
```

Manual installs need UCX with CUDA/verbs, a flash-attn wheel matching `torch==2.9.1`, and the repo dependencies from `pyproject.toml`.

## API Server

Basic server:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --host 0.0.0.0 \
  --port 8000
```

With config:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --config examples/configs/qwen3_omni_colocated_h20.yaml \
  --colocate \
  --port 8008
```

Health and model list:

```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8000/v1/models
```

Chat request:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 128,
    "stream": false
  }'
```

Streaming request:

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [{"role": "user", "content": "Write a short greeting."}],
    "stream": true
  }'
```

## TTS

Launch examples:

```bash
sgl-omni serve --model-path fishaudio/s2-pro --config examples/configs/s2pro_tts.yaml --port 8000
sgl-omni serve --model-path mistralai/Voxtral-4B-TTS-2603 --config examples/configs/voxtral_tts.yaml --port 8000
sgl-omni serve --model-path Qwen/Qwen3-TTS-12Hz-0.6B-Base --config examples/configs/qwen3_tts_0_6b.yaml --port 8000
sgl-omni serve --model-path OpenMOSS-Team/MOSS-TTS-v1.5 --config examples/configs/moss_tts.yaml --port 8000
```

Speech request:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, how are you?"}' \
  --output output.wav
```

Voice cloning:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Get the trust fund to the bank early.",
    "references": [{
      "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }]
  }' \
  --output output.wav
```

Raw PCM streaming:

```bash
curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Get the trust fund to the bank early.",
    "stream": true,
    "stream_format": "audio",
    "response_format": "pcm"
  }' \
  --output output.pcm
```

Qwen3-TTS Base needs reference audio. CustomVoice can use checkpoint speakers, and VoiceDesign needs `task_type="VoiceDesign"` plus non-empty `instructions`.

## Qwen3-Omni

Text-only:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --text-only \
  --port 8008
```

Fused text path for MMSU-style audio-input/text-output workloads:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --config examples/configs/qwen3_omni_mmsu.yaml \
  --text-only \
  --port 8008
```

Speech colocated:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --config examples/configs/qwen3_omni_colocated_h20.yaml \
  --colocate \
  --port 8008
```

Manual multi-GPU speech:

```bash
python examples/run_qwen3_omni_speech_server.py \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --gpu-thinker 0 \
  --gpu-talker 1 \
  --gpu-code-predictor 1 \
  --gpu-code2wav 1 \
  --port 8008
```

Use `--mem-fraction-static`, `--thinker-mem-fraction-static`, or `--talker-mem-fraction-static` only when auto-sizing is not good enough.

Image/text request:

```bash
curl -X POST http://localhost:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [{"role": "user", "content": "How many cars are there in the picture?"}],
    "images": ["tests/data/cars.jpg"],
    "modalities": ["text"],
    "max_tokens": 16
  }'
```

For semantic content supplied entirely by media, set the user message content to an empty string.

## Router

Managed local worker pool:

```bash
sgl-omni-router \
  --host 0.0.0.0 \
  --port 8008 \
  --launcher-config examples/configs/qwen3_omni_router.yaml \
  --policy round_robin \
  --health-failure-threshold 2 \
  --health-success-threshold 1 \
  --health-check-interval-secs 10 \
  --log-level info
```

Manual workers:

```bash
CUDA_VISIBLE_DEVICES=0 sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --model-name qwen3-omni \
  --config examples/configs/qwen3_omni_colocated_h20.yaml \
  --colocate \
  --host 0.0.0.0 \
  --port 8011

CUDA_VISIBLE_DEVICES=1 sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --model-name qwen3-omni \
  --config examples/configs/qwen3_omni_colocated_h20.yaml \
  --colocate \
  --host 0.0.0.0 \
  --port 8012

sgl-omni-router \
  --host 0.0.0.0 \
  --port 8008 \
  --worker-urls http://127.0.0.1:8011 http://127.0.0.1:8012 \
  --policy round_robin
```

Router health:

```bash
curl -i http://127.0.0.1:8008/live
curl -i http://127.0.0.1:8008/ready
curl -i http://127.0.0.1:8008/health
curl -s http://127.0.0.1:8008/workers
curl -s http://127.0.0.1:8008/v1/models
```

Runtime worker management:

```bash
curl -s http://127.0.0.1:8008/workers \
  -H "Content-Type: application/json" \
  -d '{"url":"http://127.0.0.1:8013","model":"qwen3-omni"}'

curl -s -X PUT http://127.0.0.1:8008/workers/http%3A%2F%2F127.0.0.1%3A8013 \
  -H "Content-Type: application/json" \
  -d '{"disabled":true}'
```

## Profiling

Request-event profiling:

```bash
curl -X POST http://localhost:8000/start_request_profile \
  -d '{"run_id":"demo","event_dir":"/tmp/profiles/demo/events"}'

# run traffic

curl -X POST http://localhost:8000/stop_request_profile -d '{}'
python -m sglang_omni.profiler /tmp/profiles/demo/events --format table
```

Torch + event profiling:

```bash
curl -X POST http://localhost:8000/start_profile \
  -d '{
    "run_id": "demo-run",
    "trace_path_template": "/tmp/profiles/demo-run/trace",
    "event_dir": "/tmp/profiles/demo-run/events",
    "enable_torch": true
  }'

curl -X POST http://localhost:8000/stop_profile -d '{}'
```

Use `enable_torch=false` or `/start_request_profile` for low-overhead event traces. Expensive torch profiler env vars are opt-in: `SGLANG_TORCH_PROFILER_RECORD_SHAPES`, `SGLANG_TORCH_PROFILER_PROFILE_MEMORY`, `SGLANG_TORCH_PROFILER_WITH_STACK`, `SGLANG_TORCH_PROFILER_WITH_FLOPS`.

## Tests and CI

Fast unit tests:

```bash
pytest tests/unit_test -q
```

All non-benchmark tests, matching PR Test workflow:

```bash
bash .github/scripts/run_flaky_pytest.sh pytest tests/ -v -m "not benchmark" -x
```

Benchmark/model tests:

```bash
pytest tests/test_model -m benchmark -v -s
```

TTS model CI stages:

```bash
pytest tests/test_model/test_tts_ci.py -v -s -x --concurrency 16 --tts-stage tts-stage-1-nonstream
pytest tests/test_model/test_tts_ci.py -v -s -x --concurrency 16 --tts-stage tts-stage-2-stream
pytest tests/test_model/test_tts_consistency_artifacts.py -v -s -x
```

Qwen3-Omni CI tests are under `tests/test_model/test_qwen3_omni_*_ci.py`. These require real servers, GPUs, and model snapshots.

When adding tests:

- Put fast contract tests under `tests/unit_test/<owner>/`.
- Do not add root-level `tests/test_*.py`.
- Reuse fixtures in `tests/unit_test/fixtures/` before inventing fake schedulers/relays.
- Protect public contracts and ownership boundaries, not incidental structure.

## Qwen3.5 Validation

Focused Qwen3.5 script:

```bash
bash scripts/validate_qwen35_omni.sh
```

Remote execution:

```bash
/Users/llx/.codex/skills/sglang-omni/scripts/remote_repo.sh 'bash scripts/validate_qwen35_omni.sh'
```

Preflight:

```bash
python scripts/qwen35_omni_preflight.py --model-path "$MODEL"
python scripts/qwen35_omni_preflight.py --model-path "$MODEL" --text-only
python scripts/qwen35_omni_preflight.py --vllm-profile path/to/profile.json --disable-mtp
```

Qwen3.5 audio-only compare:

```bash
bash scripts/run_qwen35_omni_audio_only_compare.sh
```

Inspect output artifacts:

- `alignment_summary.json`
- `alignment_report.md`
- `vllm_docker_worker.log`
- `sglang_server.log`
- generated WAV files
