# SGLang-Omni Repo Map

## Contents

- Source of truth
- Architecture layers
- Directory map
- Config and topology contracts
- Pipeline and communication contracts
- Serving and client layer
- Router layer
- Model integration pattern
- Files to inspect by task

## Source of Truth

This reference was derived from the remote DSW repo at `/mnt/data/llx/sglang-omni`, branch `main`, commit `ec1173a`. Refresh the remote code before acting:

```bash
/Users/llx/.codex/skills/sglang-omni/scripts/remote_repo.sh 'git status --short --branch && git log -1 --oneline --decorate'
```

Primary docs in the repo:

- `README.md`
- `docs/developer_reference/main.md`
- `docs/developer_reference/config.md`
- `docs/developer_reference/pipeline.md`
- `docs/developer_reference/communication.md`
- `docs/developer_reference/apiserver_design.md`
- `docs/developer_reference/tts_model_integration.md`
- `docs/basic_usage/tts.md`
- `docs/basic_usage/qwen3_omni.md`
- `docs/basic_usage/omni_router.md`

## Architecture Layers

The main runtime path is:

```text
HTTP API -> Client -> Coordinator -> Stage -> Scheduler -> ModelRunner -> model forward
```

Layer responsibilities:

- `serve/`: FastAPI/OpenAI-compatible schemas, routes, SSE framing, HTTP errors, launcher lifecycle.
- `client/`: Converts public `GenerateRequest` objects to `OmniRequest`, submits to the coordinator, aggregates text/audio/usage/stream chunks.
- `pipeline/coordinator.py`: Global request lifecycle, entry-stage submission, terminal result collection, stream forwarding, abort broadcast.
- `pipeline/stage/` and `pipeline/stage_workers.py`: Stage IO shell, fan-in, relay reads/writes, local dispatch, scheduler inbox/outbox bridging, child process launch.
- `scheduling/`: Scheduler loops and SGLang-backed AR scheduling. Schedulers expose `inbox`, `outbox`, `start()`, `stop()`, and `abort(request_id)`.
- `model_runner/`: Shared AR forward path; model-specific runners prepare multimodal embeddings or feedback loops.
- `models/<model>/`: Model-family config, stage factories, request builders, payload types, custom runners/components.
- `config/`: `PipelineConfig`, `StageConfig`, runtime arg injection, placement, topology, process planning.
- `relay/` and `pipeline/relay_io.py`: Data-plane transfer for payload tensors and stream chunks.
- `proto/`: Request, stage, payload, and control-plane message types.

## Directory Map

```text
sglang_omni/
  cli/                  # Typer CLI: sgl-omni serve/config
  client/               # Internal client and audio encoding helpers
  config/               # Declarative pipeline schema, config manager, placement
  model_runner/         # SGLang/AR runner abstraction and model worker policy
  models/               # Model integrations: qwen3_omni, qwen3_5_omni, tts, asr, etc.
  pipeline/             # Coordinator, stages, multi-process runner, relay IO
  preprocessing/        # Shared media preprocessors/connectors
  profiler/             # Event recorder, torch profiler, reports
  proto/                # Message/request/stage datatypes
  relay/                # shm, nccl, nixl, mooncake relay backends
  scheduling/           # Simple/streaming/omni schedulers and SGLang backend glue
  serve/                # OpenAI-compatible API server
  utils/                # HF, audio, GPU memory, misc helpers
sglang_omni_router/     # External router for worker pools
examples/               # Configs and model-specific launch wrappers
tests/                  # Unit and GPU/model CI suites
scripts/                # CI, validation, Qwen3.5 alignment/preflight helpers
docs/                   # User and developer docs
```

## Config and Topology Contracts

`PipelineConfig` describes the whole pipeline: model path, stage list, relay backend, endpoints, runtime overrides, placement, and terminal-stage logic. `StageConfig` describes one logical stage.

Important `StageConfig` fields:

- `name`: Unique logical stage name.
- `factory`: Dotted path to the scheduler/stage factory.
- `factory_args`: Static args forwarded to the factory.
- `runtime_arg_map`: Maps runtime/YAML/CLI override names to factory args.
- `next`: Static downstream target or targets for normal results.
- `terminal`: Sends results to the coordinator.
- `route_fn`: Request-aware routing override; keep `next` as the static topology declaration.
- `process`: OS process group. Non-TP stages should declare it explicitly.
- `gpu`: CPU when unset, single GPU when int, tensor-parallel ranks when list.
- `tp_size`: Must match `len(gpu)` when `gpu` is a list.
- `wait_for`, `wait_for_fn`, `merge_fn`: Fan-in declaration and request-aware source selection.
- `stream_to`, `stream_done_to_fn`: Streaming target declaration and request-aware done target selection.
- `project_payload`: Target-stage projection functions before relay/local dispatch.
- `relay`: Per-stage relay override.

Runtime prep responsibilities live mainly in `config/runtime.py`, `config/topology.py`, `config/placement.py`, `pipeline/runtime_config.py`, and `pipeline/mp_runner.py`. It validates topology, allocates endpoints, resolves dotted functions, injects `model_path`/`gpu_id` when accepted by factories, builds relay config, wires stream targets, and constructs picklable child-process specs.

Tensor parallelism is stage-local. Rank 0 owns external stage IO; follower ranks stay internal to the TP group. TP stages get exclusive OS processes and a per-stage NCCL port.

## Pipeline and Communication Contracts

The coordinator is stage-implementation agnostic. It submits new requests to the entry stage, tracks pending/running/completed/failed/aborted requests, merges terminal results when needed, and broadcasts aborts.

`Stage` owns IO, not model execution. It handles ZMQ control messages, relay reads/writes, local object dispatch, fan-in aggregation, stream routing, abort handling, profiler control, and scheduler outbox draining. Avoid branching on scheduler types in `Stage`; use the scheduler queue contract.

Scheduler choices:

- `SimpleScheduler`: CPU/non-AR stages, optional batching with `batch_compute_fn`.
- `OmniScheduler`: SGLang-backed AR stages using SGLang batch selection, KV cache, prefill/decode, tree cache, and overlap scheduling.
- Streaming/custom schedulers: Stateful vocoders or specialized streaming stages.

Control plane:

- `pipeline/control_plane.py`: ZMQ sockets and msgpack serialization.
- `proto/messages.py`: `SubmitMessage`, `DataReadyMessage`, `CompleteMessage`, `StreamMessage`, `ShutdownMessage`, profiler control, `AbortMessage`.

Data plane:

- `pipeline/relay_io.py`: Extracts tensors from `StagePayload`, serializes tensor-free payloads, writes flat tensor buffers, restores tensors on receive, and handles stream chunks/signals.
- `relay/{shm,nccl,nixl,mooncake}.py`: Backend transport only. Backends should not route, fan in, or interpret model payloads.
- Same-process edges may use `LOCAL_OBJECT`; receivers must treat shared objects and tensor leaves as read-only.
- Same-GPU stream chunks can use CUDA IPC through `ForkingPickler`.

## Serving and Client Layer

Key files:

- `sglang_omni/serve/openai_api.py`: FastAPI app, routes, request conversion, response formatting.
- `sglang_omni/serve/protocol.py`: Request/response schemas.
- `sglang_omni/serve/launcher.py`: Compiles pipeline, starts runtime, creates client/app, runs Uvicorn, stops runtime.
- `sglang_omni/client/client.py`: Submits requests and aggregates text/audio/usage/stream chunks.
- `sglang_omni/cli/serve.py`: CLI surface and config/override handling.

`create_app(client, model_name=...)` only builds the FastAPI layer around an existing client. `launch_server(pipeline_config, ...)` owns the standard lifecycle.

Core routes:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/audio/speech`
- profiler routes: `/start_profile`, `/stop_profile`, `/start_request_profile`, `/stop_request_profile`
- realtime routes live under `serve/realtime/`

Chat requests become `GenerateRequest` via `_build_chat_generate_request()`. TTS requests become `GenerateRequest` via `build_speech_generate_request()` with `output_modalities=["audio"]` and `task="tts"` metadata. The client then converts to `OmniRequest` and talks to the coordinator.

## Router Layer

`sglang_omni_router` is an external HTTP router for a pool of complete Omni V1 worker servers. It does not load model weights or split a single request across workers.

Key files:

- `sglang_omni_router/config.py`: URL/capability/policy validation.
- `sglang_omni_router/app.py`: FastAPI routes, worker management, liveness/readiness.
- `sglang_omni_router/proxy.py`: Request forwarding and header filtering.
- `sglang_omni_router/selector.py`: `round_robin`, `least_request`, `random`.
- `sglang_omni_router/health.py`: Background health checks.
- `sglang_omni_router/launcher/local.py`: Managed local subprocess worker launcher.
- `sglang_omni_router/serve.py`: CLI entrypoint `sgl-omni-router`.

Worker capabilities are one or more of `chat`, `speech`, `streaming`, `image_input`, `audio_input`, `video_input`, `audio_output`. Text-only managed workers should not advertise `speech` or `audio_output`.

## Model Integration Pattern

Model-specific behavior should stay under `sglang_omni/models/<model>/`.

Common layout:

```text
config.py             # PipelineConfig subclass, StageConfig list, EntryClass, Variants
stages.py             # stage factories referenced by StageConfig.factory
request_builders.py   # request-to-scheduler and inter-stage payload transforms
payload_types.py      # typed state passed between stages
sglang_model.py       # SGLang-side model class, when needed
model_runner.py       # custom AR runner, when default runner does not fit
```

Registry flow:

1. Set `architecture` on a `PipelineConfig` subclass.
2. Export `EntryClass = YourPipelineConfig`.
3. `models/registry.py` auto-discovers model subpackages and matches HF `config.json::architectures`.
4. If the HF config is not supported by stock Transformers, register an `AutoConfig` class before SGLang infrastructure creation.
5. If SGLang cannot resolve the architecture, add the model class in `model_runner/sglang_model_runner.py::_register_omni_model` and pass the matching `model_arch_override` when needed.

TTS minimum pipeline shape:

```text
preprocessing -> tts_engine -> vocoder
```

Add an `audio_encoder` when reference encoding is heavy. For per-chunk streaming from engine to vocoder, set `stream_to=["vocoder"]` on the engine and `can_accept_stream_before_payload=True` on the vocoder.

Abort cleanup must be idempotent and wired into every scheduler that touches request-keyed shared state. Clean up when preprocessing aborts, AR aborts before consuming handoff, and preprocessing finishes after an abort.

## Files to Inspect by Task

- Pipeline topology/config bug: `sglang_omni/config/schema.py`, `config/runtime.py`, `config/topology.py`, `pipeline/runtime_config.py`, `pipeline/mp_runner.py`, relevant model `config.py`.
- Stage routing/fan-in/streaming bug: `pipeline/stage/runtime.py`, `pipeline/relay_io.py`, `pipeline/local_dispatch.py`, `proto/messages.py`, relevant `request_builders.py`.
- Scheduler behavior: `scheduling/simple_scheduler.py`, `scheduling/streaming_simple_scheduler.py`, `scheduling/omni_scheduler.py`, `scheduling/sglang_backend/*`.
- AR forward behavior: `model_runner/base.py`, `model_runner/thinker_model_runner.py`, `model_runner/sglang_model_runner.py`, relevant model runner.
- OpenAI endpoint behavior: `serve/openai_api.py`, `serve/protocol.py`, `client/client.py`, `client/types.py`.
- Router behavior: `sglang_omni_router/{app,config,proxy,selector,health,worker}.py`.
- Profiler behavior: `profiler/event_recorder.py`, `profiler/profiler_control.py`, `profiler/views.py`, `docs/developer_reference/profiler.md`.
- Qwen3.5-Omni: read [qwen35-omni.md](qwen35-omni.md).
- Tests: `tests/README.md` and [ops-and-tests.md](ops-and-tests.md).
