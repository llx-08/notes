# Qwen3.5-Omni Notes

## Contents

- Source of truth
- Main files
- Stage topology
- Defaults and runtime knobs
- Split checkpoint path behavior
- Request and output routing
- Thinker and talker scheduler construction
- Code2Wav behavior
- Launch and preflight
- Validation and common failure points

## Source of Truth

This reference was derived from `/mnt/data/llx/sglang-omni` on remote DSW at commit `ec1173a`. Refresh before edits:

```bash
/Users/llx/.codex/skills/sglang-omni/scripts/remote_repo.sh 'git log -1 --oneline && git grep -n "QWEN3_5_OMNI" -- sglang_omni/models/qwen3_5_omni examples scripts'
```

## Main Files

Core package:

- `sglang_omni/models/qwen3_5_omni/config.py`
- `sglang_omni/models/qwen3_5_omni/stages.py`
- `sglang_omni/models/qwen3_5_omni/bootstrap.py`
- `sglang_omni/models/qwen3_5_omni/request_builders.py`
- `sglang_omni/models/qwen3_5_omni/merge.py`
- `sglang_omni/models/qwen3_5_omni/preflight.py`
- `sglang_omni/models/qwen3_5_omni/components/preprocessor.py`
- `sglang_omni/models/qwen3_5_omni/components/image_encoder.py`
- `sglang_omni/models/qwen3_5_omni/components/audio_encoder.py`
- `sglang_omni/models/qwen3_5_omni/components/talker.py`
- `sglang_omni/models/qwen3_5_omni/components/subtalker.py`
- `sglang_omni/models/qwen3_5_omni/components/code2wav_scheduler.py`
- `sglang_omni/models/qwen3_5_omni/components/qwen3_omni_next_codec_decoder.py`

Launchers and configs:

- `examples/run_qwen3_5_omni_server.py`
- `examples/run_qwen3_5_omni_speech_server.py`
- `examples/configs/qwen3_5_omni_colocated_h20.yaml`
- `examples/qwen3_5_omni_README.md`
- `examples/QUICKSTART_QWEN35_OMNI.md`
- `examples/QWEN35_OMNI_RUN_AND_TEST.md`
- `scripts/launch_qwen35_omni_sglang_server.sh`
- `scripts/qwen35_omni_preflight.py`
- `scripts/validate_qwen35_omni.sh`
- `scripts/qwen35_omni_alignment.py`

Tests:

- `tests/unit_test/qwen3_5_omni/*`
- shared Qwen3 tests in `tests/unit_test/qwen3_omni/*`
- `tests/unit_test/serve/test_openai_api.py`

## Stage Topology

Architecture strings:

- Primary: `Qwen3OmniNextForConditionalGeneration`
- Aliases: `Qwen3OmniNextThinkerForConditionalGeneration`, `Qwen3OmniNextThinkerMTP`
- `EntryClass = Qwen35OmniSpeechPipelineConfig`
- `Variants = {"text", "speech", "speech-colocated"}`

Text-only pipeline is a 6-stage thinker path:

```text
preprocessing -> image_encoder
              -> audio_encoder
              -> mm_aggregate -> thinker -> decode
```

Speech pipeline is an 8-stage text + audio path:

```text
preprocessing -> image_encoder
              -> audio_encoder
              -> mm_aggregate -> thinker -> decode
                              -> talker_ar -> code2wav
```

Key stage details:

- `preprocessing`: `create_preprocessing_executor`, routes only to active media encoders plus aggregate, maps video/audio runtime params to preprocessor kwargs.
- `image_encoder` and `audio_encoder`: batched `SimpleScheduler` stages with `StageOutputCache`; reuse Qwen3 encoder batching/cache helpers.
- `mm_aggregate`: fan-in over preprocessing/image/audio using `wait_for_fn` and `merge_for_thinker`. Speech mode also projects to `talker_ar`.
- `thinker`: SGLang AR stage. Streams hidden/text chunks to `decode` and, in speech mode, `talker_ar`.
- `decode`: terminal text stage; accepts streams before full payload.
- `talker_ar`: SGLang AR talker; accepts stream before payload and streams codec chunks to `code2wav`.
- `code2wav`: terminal streaming vocoder stage; creates `Qwen35Code2WavScheduler` when DAC files are present, otherwise creates a missing scheduler that fails clearly only when audio is requested.

`Qwen35OmniSpeechColocatedPipelineConfig` uses the same logical 8 stages with thinker and talker on GPU 0, but stage process names still follow `_SPEECH_DEFAULT_PROCESSES`. The H20 YAML supplies memory budgets for colocation.

## Defaults and Runtime Knobs

Important constants in `config.py`:

- Thinker max seq len: `192000`
- Talker max seq len: `32768`
- Max prefill tokens: `32768`
- Chunked prefill size: `32768`
- Limit multimodal per prompt: `{"audio": 960, "image": 960, "video": 960}`
- Max running requests: `32`
- Code2Wav torch compile default: enabled
- Canonical model name: `qwen3.5-omni`
- Common aliases normalize to canonical name: `qwen3-5-omni`, `qwen3_5_omni`, `qwen35-omni`, `qwen35_omni`, `qwen35omni`
- Env default: `SGLANG_JIT_DEEPGEMM_PRECOMPILE=0`

Important stage override maps:

- Preprocessing maps `max_seq_len`, image pixel limits, video FPS/frame/pixel knobs, `audio_target_sr`, timestamp interval, downsample settings.
- Thinker maps `max_seq_len` to `thinker_max_seq_len`.
- Talker maps `max_seq_len` to `talker_max_seq_len`.
- Code2Wav maps vLLM-style names such as `send_chunk_size`, `code2wav_stream_chunk_size`, `code2wav_frequency`, `code2wav_odeint_method`, `code2wav_dit_quant`, and torch-compile toggles to scheduler factory args.

`examples/configs/qwen3_5_omni_colocated_h20.yaml` sets:

- `config_cls: Qwen35OmniSpeechColocatedPipelineConfig`
- `relay_backend: shm`
- preprocessing defaults: `image_max_pixels: 401408`, `video_fps: 2`, `video_max_frames: 128`, `audio_target_sr: 16000`, `audio_timestamp_interval: 60`, `audio_downsample_times: 4`, `audio_downsample_chunk_size: 100`
- memory fractions: image/audio encoders `0.025`, thinker `0.75`, talker `0.12`, code2wav `0.02`
- code2wav defaults: `send_chunk_size: 8`, torch compile enabled, `rk4`, relaxed ODE, `50hz`, `fp8`

## Split Checkpoint Path Behavior

`stages.py::_resolve_qwen35_stage_model_path()` detects local split checkpoints:

- `preprocessing`, `image_encoder`, `audio_encoder`, `thinker` prefer `root/thinker` when it contains `config.json`.
- `talker_ar` prefers `root/talker_lm`, then `root/talker`, when it contains `config.json`.
- If no split subdir exists, use the original model path.

When a split talker path is used, `create_talker_ar_executor_from_config()` sets `root_model_path` to the root and clears `weight_prefix`, because split talker weights are often not prefixed with `talker.`.

## Request and Output Routing

`request_builders.py` reuses most Qwen3-Omni request builder behavior and adds Qwen3.5-specific compatibility.

Important public-ish functions:

- `should_generate_audio_output(request)`: Determines whether speech/audio output should be active.
- `resolve_mm_aggregate_next_stages(...)`: In speech mode, routes aggregate output toward thinker and talker only when audio output is requested.
- `resolve_thinker_stream_done_targets(...)`: Selects active stream completion targets.
- `resolve_terminal_stages(request)`: Returns request-specific terminal stages.
- `make_thinker_scheduler_adapters(...)`: Builds thinker request and result adapters.
- `make_thinker_stream_output_builder(...)`: Builds streaming hidden/text output from thinker.
- `make_talker_scheduler_adapters(...)`: Builds talker request/result/stream adapters.

Request compatibility rules:

- Audio output config may appear in OpenAI `audio` shape or `extra_params`/stage params.
- Voice keys include `speaker`, `voice`, `voice_type`; request-side aliases include Chinese and English names. Examples: `cherry -> f245`, `ethan -> m02`, `serena -> f05`, `chelsie -> f030`.
- Language keys include `language_id`, `language`, `lang`, `language_type`, `target_language`, `target_lang`; human names expand to config codes such as `chinese -> zh`, `english -> en`, `filipino -> tagalog/tl`.
- Voice-style prefixes such as `<voice_style>...</voice_style>` are parsed and removed before tokenization when needed.
- Voice clone/xvector info can be supplied directly or loaded from paths.
- `QWEN35_TALKER_TEXT_FEEDBACK_STRIDE` may override the default feedback stride.

`merge.py::merge_for_thinker()` wraps Qwen3 merge logic and preserves Qwen3.5 video/audio metadata:

- `audio_is_dependent`
- per-video `use_audio_in_video`

It adds these to thinker `model_inputs` and keeps normalized metadata in `state.mm_inputs`.

## Thinker and Talker Scheduler Construction

`bootstrap.py` builds Qwen3.5-specific SGLang infrastructure.

Thinker:

- Architecture override: `Qwen3OmniNextThinkerForConditionalGeneration`
- Default hidden capture layers: `[0, 24]`
- In speech mode, hidden capture layers are resolved from talker `accept_hidden_layer`; layer `0` is always included for embed/text hidden.
- If hidden capture and CUDA graph are both desired, CUDA graph capture is deferred: enable hidden returns, disable graph for infrastructure creation, then re-enable and initialize device graphs.
- `SGLangOutputProcessor` emits hidden only when the request should generate audio.
- Uses `ThinkerModelRunner`.

Talker:

- Architecture override: `Qwen3OmniNextTalkerModel`
- Uses Qwen3 talker scheduler/model-runner infrastructure: `QwenTalkerScheduler`, `QwenTalkerModelRunner`, `configure_talker_server_args`.
- Supports `feedback_enabled`, partial start, `partial_start_min_chunks`, and split-checkpoint `root_model_path`.

Both thinker and talker construct server args via `build_sglang_server_args()` with defaults including disabled CUDA graph and Qwen3.5 prefill/chunked-prefill limits, then merge CLI/YAML overrides.

## Code2Wav Behavior

`components/code2wav_scheduler.py` plugs the Qwen3OmniNext DAC/codec decoder into the existing Qwen3 streaming `Code2WavScheduler`.

Defaults:

- Codec EOS token id: `2150`
- Sample rate: `24000`
- Stream chunk size: `4`
- Dynamic chunk sizes: `(2, 4, 6, 8)`
- Dynamic chunk steps: `(8, 4, 2, 1)`
- ODE methods: `euler`, `rk4`
- Frequencies: `50hz`, `25hz`
- DIT quantization: `fp8`

Accepted Code2Wav directories under a model path:

- `qwen3_5_omni_codec_decode_online_0306`
- `qwen3_5_omni_codec_decode_online_0226`
- `code2wav`
- `codec_decoder`
- `dac`
- `codec`
- root model dir

Accepted config names include `config.yaml`, `config.yml`, `codec_decoder.yaml`, `dac.yaml`. Accepted checkpoint names include `model_weights.pt`, `checkpoint.pt`, `model.pt`, `model.pth`, `state_dict.pt`, `pytorch_model.bin`, `generator.pt`, `codec_decoder.pt`, `dac.pt`.

The loader is searched in:

1. `sglang_omni.models.qwen3_5_omni.components.qwen3_omni_next_codec_decoder`
2. `vllm.model_executor.models.qwen3_omni_next_codec_decoder`

If no loadable DAC files or loader are found, the missing scheduler remains importable and emits a clear missing-code2wav error when audio output is requested. This lets text-only paths work even without code2wav assets.

## Launch and Preflight

Quick DSW commands:

```bash
/Users/llx/.codex/skills/sglang-omni/scripts/remote_repo.sh 'bash scripts/validate_qwen35_omni.sh'
```

Thinker-only:

```bash
cd /mnt/data/llx/sglang-omni
TORCHDYNAMO_DISABLE=1 .venv/bin/python examples/run_qwen3_5_omni_server.py \
  --model-path "$MODEL" \
  --gpu-thinker 1 --gpu-image-encoder 1 --gpu-audio-encoder 1 \
  --thinker-max-seq-len 32768 \
  --port 8101 --model-name qwen3_5-omni
```

Speech:

```bash
cd /mnt/data/llx/sglang-omni
TORCHDYNAMO_DISABLE=1 .venv/bin/python examples/run_qwen3_5_omni_speech_server.py \
  --model-path "$MODEL" \
  --gpu-thinker 1 --gpu-talker 2 --gpu-code2wav 2 \
  --gpu-image-encoder 3 --gpu-audio-encoder 3 \
  --thinker-max-seq-len 8192 \
  --port 8101 --model-name qwen3_5-omni
```

The shared wrapper defaults:

```bash
bash scripts/launch_qwen35_omni_sglang_server.sh
```

Important wrapper env vars:

- `CONTAINER` default `b5f665f3d883`
- `WORKDIR` default `/myapp/sglang-omni`
- `MODEL_PATH` default `/myapp/models/qwen3_5_omni_23b_final_multilingual_all_voice_bf16_0315`
- `CODE2WAV_PATH` default `$MODEL_PATH/qwen3_5_omni_codec_decode_online_0306`
- `PORT` default `8161`
- `VOICE_TYPE` default `m02`
- `GPU_THINKER` default `0`
- `GPU_TALKER` and `GPU_CODE2WAV` default `1`
- `THINKER_MAX_SEQ_LEN` default `192000`
- `NO_CODE2WAV_TORCH_COMPILE` default `1` for debug startup
- `EXTRA_ARGS` appends launcher args

Checkpoint/profile preflight:

```bash
python scripts/qwen35_omni_preflight.py --model-path "$MODEL"
python scripts/qwen35_omni_preflight.py --model-path "$MODEL" --text-only
python scripts/qwen35_omni_preflight.py --vllm-profile path/to/profile.json --disable-mtp
```

## Validation and Common Failure Points

Focused validation script:

```bash
bash scripts/validate_qwen35_omni.sh
```

It runs:

- `pytest tests/unit_test/qwen3_5_omni -q`
- `pytest tests/unit_test/qwen3_omni/test_cli.py -q`
- `pytest tests/unit_test/serve/test_openai_api.py -q`
- `py_compile` for Qwen3.5 files and touched shared launcher/serve/client/runtime files
- a line-length/trailing-whitespace check on Qwen3.5 and related files

When editing a specific area, prefer the closest unit test first:

- Request parsing/voice/language/style: `tests/unit_test/qwen3_5_omni/test_request_builders.py`
- Talker/feedback/code2wav: `tests/unit_test/qwen3_5_omni/test_talker.py`, `test_code2wav_scheduler.py`, `test_subtalker.py`
- Config/CLI: `tests/unit_test/qwen3_5_omni/test_config.py`, `test_cli.py`, `test_config_manager.py`
- Preprocessor/encoders: `test_preprocessor.py`, `test_encoder_components.py`
- Launchers: `test_example_launcher.py`, `test_text_example_launcher.py`

Common failure points:

- `Not enough memory`: lower thinker max seq len, adjust `mem_fraction_static`, or split encoders to another GPU.
- Stream has text but no finish/audio: inspect talker and code2wav errors in server logs.
- Code2Wav missing: run preflight and check for accepted DAC config/checkpoint names under the model/code2wav path.
- Split checkpoint fails: verify `root/thinker/config.json` and `root/talker_lm/config.json` or `root/talker/config.json`.
- Voice alias not respected: inspect `_VOICE_TO_SPK_MAPPING`, `_CODE2WAV_VOICE_TYPE_MAPPING`, and voice-control suffix stripping.
- Live translation language wrong: inspect language keys and `_LANGUAGE_NAME_ALIASES`.
- Video/audio mixed input wrong: inspect `merge_for_thinker()` preservation of `audio_is_dependent` and `use_audio_in_video`.
- Open-ended Qwen3.5 alignment differs from vLLM wording: compare generated artifacts, not just binary pass/fail; the audio-only joke wrapper allows alignment failure by default.
