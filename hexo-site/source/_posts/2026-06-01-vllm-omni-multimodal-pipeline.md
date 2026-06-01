---
title: Qwen3.5-Omni 多模态数据全流程（dashscope-serving → dashllm → vLLM，含 PD 分离）
date: 2026-06-01
tags: []
---

# Qwen3.5-Omni 多模态数据全流程（dashscope-serving → dashllm → vLLM，含 PD 分离）

> 整理自代码级溯源；所有引用都附 file:line。围绕实时音频流 + PD 分离场景。

---

## 0. 一图概览

```
client ──HTTP──▶ dashscope-serving
                  │  1. 解协议（image_url/video_url/input_audio）
                  │  2. download URL（global_lru_cache）
                  │  3. image_process/audio_process/video_process
                  │  4. dump 出 multi_modal_data + mm_processor_kwargs
                  ▼
              dashllm._backend_vllm.generate
                  │
                  ├──omni PD──▶ _call_prefill ─▶ P engine
                  │                              │  input_preprocessor（HF processor + placeholder 展开）
                  │                              │  add_request(max_tokens=1, do_remote_decode=True)
                  │                              │  model runner: encoder → KV 写入 P GPU
                  │                              │  prefill 完 → _dash_done[reqid] 等 D 来取 KV
                  │                              ▼
                  │                          (P abort/timeout 时清 _dash_done
                  │                            ↑↑↑ 这就是加 _gone_reqs 的位置)
                  │
                  └──────────▶ _call_decode  ─▶ D engine
                                              │ StreamInfer: input_ids + extra_params (无 multi_modal_data)
                                              │ KVT 控制面 _dash_prefill_rpc(reqid) ──RPC──▶ P._step_dinfoq
                                              │   ↑                                          │
                                              │   └─ 0: 拉 KV ───────────────────────────┐
                                              │   └─ 410 CODE_REQGONE: 立即 raise        │
                                              │      → release loading slot              │
                                              │                                          │
                                              │ KVT 数据面: P GPU KV blocks  ──RDMA──▶ D GPU
                                              │ model runner: encoder 跳过（mm 段已在 KV 里）
                                              │ 走正常 decode 出 token
                                              ▼
                                          回给 dashllm → dashscope-serving → client
```

---

## 一、HTTP 入口 → dashscope-serving 预处理

### 1.1 协议层把 image/video/audio 解成内部 dict

入口在 `aquila_stream_server_for_vl.py`（挂 `/mm`）。OpenAI 风格 content 由 `openai_protocol_tool.py` 分发：

```python
# /mnt/data/llx/dashscope-serving/multimodal_serving/schema/protocol_tool/openai_protocol_tool.py:1034
elif content_type == 'input_audio':
    return self._process_input_audio_content(contents, msg_index)
elif content_type == 'image_url':
    return self._process_image_url_content(contents, msg_index)
elif content_type == 'video_url':
    return self._process_video_url_content(contents, msg_index)
```

`image_url` 直接拿 URL 字段，`video_url` 类似，`input_audio` 把 base64/路径包进 `AudioData`。

### 1.2 三段式 step：collect → download → process

总管线在 `preprocessor.py`：

```python
# /mnt/data/llx/dashscope-serving/multimodal_serving/preprocessor/preprocessor.py:978
@step_log(step='preprocessor_process')
def __call__(self, inputs: MultiModalInput, parameters: MultiModalParameters,
             step_log_params: StepLogParameters, encoder_cache=None,
             safety_params=None) -> Result:
    for step in self.steps:
        step(inputs, context)
```

`step_log` 装饰器会写 `{step}_start` / `{step}_end` 日志。注意：**没有** `download_start` / `multimodal_data_process_start` 这两个 step 名，实际是 `preprocessor_process_start`、`image_process_start`、`audio_process_start` 等。

**下载**：

```python
# /mnt/data/llx/dashscope-serving/multimodal_serving/preprocessor/multimodal_data/multimodal_data_list_preprocessor.py:157
def process(self, step_input: MultiModalInput, context: ProcessingContext):
    global_loop.run_coroutine(self._download(multimodal_data_list, parameters, step_log_params))
        for data in multimodal_data_list.all:
            all_task.append(multimodal_data_download_wrapper(data, parameters, step_log_params, session))
        await asyncio.gather(*all_task)
```

OSS 私链命中 `global_lru_cache`（**maxsize 仅 8**）可跳过下载，但 **omni 视频被显式排除**（要从 video 抽音频，缓存会破坏这一步）：

```python
# /mnt/data/llx/dashscope-serving/multimodal_serving/preprocessor/multimodal_data/multimodal_data_preprocessor.py:465
if original_oss_link and not mm_conf.QWEN3_OMNI:
    global_lru_cache.put(original_oss_link, multimodal_data.result)
```

**预处理**（按模态分发）：

| 模态 | step 名 | 产物 |
|------|---------|------|
| image | `image_process_start/end` | RGB `np.ndarray`（OpenCV resize） |
| audio | `audio_process_start/end` | `np.float32` waveform `[-1, 1]` |
| video | `video_file_process_*` | frame tensor / 帧列表 |

omni 的 audio token 数算法（每秒 7 token）：

```python
# /mnt/data/llx/dashscope-serving/multimodal_serving/preprocessor/multimodal_data/audio/audio_data_preprocessor.py:80
if mm_conf.QWEN3_5_OMNI:
    return audio.shape[0] // 16000 * 7
```

### 1.3 最终装出来给 vLLM 的 prompt dict

`MultimodalDataList.dump()` 产出三块：`multi_modal_data` / `mm_processor_kwargs` / `mm_extra_kwargs`：

```python
# /mnt/data/llx/dashscope-serving/multimodal_serving/schema/multimodal_data.py:196
data_map = {
    k: v
    for k, v in dict(audio=[x.data for x in self.audios],
                     image=[x.data for x in self.images],
                     video=[x.data for x in self.videos]).items() if v
}
if mm_conf.QWEN3_OMNI:
    data_map['use_audio_in_video'] = self.use_audio_in_video
```

`mm_processor_kwargs` 里典型字段：`fps`、`do_sample_frames=False`、`use_audio_in_video`、`dependent_audio`。

dashllm 本地模式直接把这堆字段透传到引擎：

```python
# /mnt/data/llx/dashllm/dashllm/core/backend/engine/_vllm_omni_rpc.py:499
prompt = TokensPrompt(
    prompt_token_ids=input_ids,
    multi_modal_data=multi_modal_data,
    mm_processor_kwargs=mm_processor_kwargs,
)
```

---

## 二、vLLM 引擎入口：input_preprocessor

omni 的 thinker engine 在 `add_request()` 里决定是否跑 `input_processor.process_inputs()`：

```python
# /mnt/data/llx/vllm/vllm/engine/omni3_5_thinker_engine.py:2564
is_prefill_node = self.kv_role == "kv_producer"
need_preprocess_and_transfer = use_t2t and (not thinker_only or is_prefill_node)
need_prepare_v6d = use_t2t and not thinker_only

if need_preprocess_and_transfer:
    logger.info("Thinker: request %s input_preprocessor start", request_id)
    processed_request = self.thinker_engine.input_processor.process_inputs(...)
    logger.info("Thinker: request %s input_preprocessor finished, cost:%.1f", ...)
```

**关键点（PD 模式下的差异）**：D 节点 (`thinker_only=True` 且 `kv_role != kv_producer`) 走 `else` 分支，**完全跳过** `input_preprocessor`：

```python
# /mnt/data/llx/vllm/vllm/engine/omni3_5_thinker_engine.py:2622
else:
    async def run_output_handler_and_add_request():
        self.thinker_engine._run_output_handler()
        return await self.thinker_engine.add_request(
            request_id=request_id,
            prompt=prompt,
            params=params,
            ...
        )
```

`input_processor.process_inputs()` 做的事：

- 调 HF processor 把 `multi_modal_data` 转成 `MultiModalKwargsItem`（含 `pixel_values`、`audio_features` 等）
- 把 prompt 中的 modality 占位符（`<image>`、`<video>`、`<audio>`）展开成 vision/audio pad token 串
- 算出 `mrope_position_delta`、`mm_prompt_len` 等元数据

omni 模型侧把字符串占位拼成正式 token 串的位置：

```python
# /mnt/data/llx/vllm/vllm/model_executor/models/qwen3_omni_next_thinker.py:717
if envs.VLLM_OMNI_USE_V35_RTC_PROMPT_STYLE:
    if metadata["video_backend"] == "rtc_first_frame":
        video_token_str = audio_bos_token + video_token_str
```

预热时给出参考形态：

```python
# /mnt/data/llx/vllm/vllm/engine/omni3_5_thinker_engine.py:589
audio_placeholder = (
    "<|vision_start|><|video_pad|>"
    "<|vision_end|><|audio_start|>"
    "<|audio_pad|><|audio_end|>"
)
```

### 中间的 `MediaWithPath` 优化

`VLLM_USE_FILE_PATH_MMD_CACHE=True` 时，`ImageLoader` 不再以图像内容做 hash，而是用文件路径，避免每次都 blake3 几 MB 的图像：

```python
# /mnt/data/llx/vllm/vllm/multimodal/base.py:42
class MediaWithPath(Generic[_T]):
    """
    Wrapper that couples a media object with its original file path.
    This enables fast path-based hashing while keeping the media object
    intact for HF Processor compatibility.
    """
    media: _T
    original_path: str
```

⚠️ **坑**：RTC 反复复用同一路径但内容不同会错误命中缓存，omni 实时场景下要确认 dashscope-serving 那边不会复用 path。

---

## 三、GPU model runner：MM encoder 执行 + 拼回 embeddings

只有 **first PP rank + supports_mm_inputs** 才会跑 encoder：

```python
# /mnt/data/llx/vllm/vllm/v1/worker/gpu_model_runner.py:3589
if self.supports_mm_inputs and is_first_rank and not is_encoder_decoder:
    with self.maybe_get_ec_connector_output(...) as ec_connector_output:
        self._execute_mm_encoder(scheduler_output)
        mm_embeds, is_mm_embed = self._gather_mm_embeddings(scheduler_output)
    inputs_embeds_scheduled = self.model.embed_input_ids(
        self.input_ids.gpu[:num_scheduled_tokens],
        multimodal_embeddings=mm_embeds,
        is_multimodal=is_mm_embed,
    )
```

`VLLM_OMNI_ENABLE_ENCODER_BATCH` 控制 batch 化逻辑：

```python
# /mnt/data/llx/vllm/vllm/v1/worker/gpu_model_runner.py:2782
if envs.VLLM_OMNI_ENABLE_ENCODER_BATCH:
    logger.info("resort mm input to enable encoder batch input")
    indexed_kwargs = [(i, item) for i, item in enumerate(mm_kwargs)]
    sorted_indexed = sorted(indexed_kwargs, key=lambda x: (x[1].modality, x[0]))
```

Encoder 出来的 embedding 按 `mm_hash` 写入 `encoder_cache`，再 scatter 回 placeholder token 的位置：

```python
# /mnt/data/llx/vllm/vllm/v1/worker/gpu_model_runner.py:2984
if start_pos + num_encoder_tokens <= num_computed_tokens:
    # The encoder output is already processed and stored
    # in the decoder's KV cache.
    continue
encoder_output = self.encoder_cache.get(mm_hash, None)
assert encoder_output is not None, f"Encoder cache miss for {mm_hash}."
```

非常重要的一句：**`already processed and stored in the decoder's KV cache → continue`**。也就是说，**MM embedding 一旦走完 prefill 进了 KV，对 D 来说它就是 KV 的一部分，不再需要 encoder 重跑**。

---

## 四、PD 分离下 P/D 的分工与跨节点传输

### 4.1 P 节点：做完整的 encoder + prefill

P 是 `kv_producer`，`_call_prefill` 显式带 `multi_modal_data`：

```python
# /mnt/data/llx/dashllm/dashllm/core/backend/engine/_disaggregated_prefilling.py:92
return self._prefill_engine.generate(
    ...
    multi_modal_data=multi_modal_data,
    mm_processor_kwargs=mm_processor_kwargs,
    ...
)
```

引擎层强制把 `max_tokens` 改成 1（只跑 1 步 prefill）并打上 `do_remote_decode`：

```python
# /mnt/data/llx/vllm/vllm/engine/omni3_5_thinker_engine.py:2512
if self.kv_role == "kv_producer":
    params.max_tokens = 1
    extra_args["kv_transfer_params"] = {"do_remote_decode": True}
    logger.info(
        "Thinker disagg P: request %s max_tokens=1, do_remote_decode=True",
        request_id,
    )
```

跑完一步 prefill，MM tower 的 embedding 已经在 transformer 每层产生了对应的 K/V，**全部以 token 维度连续写在 P 的 GPU KV 缓冲里**。

### 4.2 D 节点：StreamInfer 调用，**不传 raw MM**

`_call_decode()` 给 P→D handshake、KV transfer、生成都只发 `input_ids + sampling_params + extra_params`：

```python
# /mnt/data/llx/dashllm/dashllm/core/backend/engine/_disaggregated_prefilling.py:166
decode_iter = cli.StreamInfer(input_ids=new_input_ids, sampling_params=sampling_params, extra_params=new_extra_params)
```

omni 的 PD `extra_params` 里只有控制信息：

```python
# /mnt/data/llx/dashllm/dashllm/core/backend/engine/_disaggregated_prefilling.py:131
if self._is_omni:
    new_extra_params["omni_prefill_remote_endpoint"] = self._prefill_engine._thinker_engine.get_remote_endpoint()
    new_extra_params["thinker_only"] = int(sampling_params.get("thinker_only", False))
```

D 侧到了 model runner 后：

- `scheduled_encoder_inputs` 是空（KV 已覆盖 MM 段）
- `_gather_mm_embeddings` 进入 `continue` 分支
- D 上 **MM tower 完全不会跑**

### 4.3 跨节点真正传输的是什么

| 字段 | 是否跨 P→D 传输 | 通道 | 说明 |
|------|----------------|------|------|
| `prompt_token_ids`（含 pad token） | 部分 | StreamInfer 主包 | external_router 路径会 pop 完整 ids，只传 `prompt_token_ids_len` 让 D 用 pad token 补齐（omni 不走这套） |
| `mrope_position_delta` | 是 | handshake `pd_context` | D 计算 mrope 用 |
| `mm_prompt_len` | 是 | extra_params / handshake | talker 用 |
| `thinker_only` | 是 | extra_params | D 是否启 talker / code2wav |
| `multi_modal_data`（raw PIL/np） | **否** | — | 完全不传 |
| Vision/audio encoder 输出 | **否（单独传）** | 隐式融在 KV 里 | 经 `kvtbackend.py` 的 KVT |
| K/V blocks | 是 | KVT 数据面（RDMA / NCCL） | `_dash_prefill_rpc` / `_step_dinfoq` |
| Block 列表 + token 计数 | 是 | KVT 控制面 RPC | `KVTDInfo` / `PReqMeta` |

handshake 里塞 mm 元数据：

```python
# /mnt/data/llx/dashllm/dashllm/core/backend/engine/_disaggregated_prefilling.py:842
multimodal_decode_request = {}
processed_prompt_len = process_extra.get("processed_prompt_token_ids_len")
if processed_prompt_len is not None:
    multimodal_decode_request["prompt_token_ids_len"] = int(processed_prompt_len)
if "mrope_position_delta" in process_extra:
    multimodal_decode_request["mrope_position_delta"] = int(process_extra["mrope_position_delta"])
if multimodal_decode_request:
    handshake_extra["pd_context"] = json.dumps({"multimodal_decode_request": multimodal_decode_request}, ...)
```

### 4.4 KVT 控制面里也确实没有 MM 字段

```python
# /mnt/data/llx/vllm/vllm/v1/hybrid_connector/kvtbackend.py:335
class KVTDInfo(...):
    instid: str
    blkids: list[list[int]]
    cached_tokens: int
    max_tokens: int
    d_workers_info: list[str]
```

```python
# /mnt/data/llx/vllm/vllm/v1/hybrid_connector/kvtbackend.py:521
class PReqMeta:
    reqid: str
    d_inst_id: str
    p_block_ids: list[list[int]]
    d_block_ids: list[list[int]]
    new_tokens: int
    has_last_token: bool
    seen_tokens: int
```

仅 block id + token 计数；MM 语义完全在 P prefill KV 里。

---

## 五、三层 MM 缓存

| 层 | 位置 | Key | Value | 容量 | TTL |
|----|------|-----|-------|------|-----|
| dashscope `global_lru_cache` | `multimodal_data_preprocessor.py:465` | OSS 私链 URL | `MultimodalResult`（已处理 raw） | **8** | 无 |
| chat-serving v6d encoder cache | mm_key 内 | `mm_key`（md5 of attrs + md5_hashes） | encoder 结果 path handle | GB 级 | 由 v6d 管 |
| vLLM `MultiModalProcessorCache` | `vllm/multimodal/cache.py` | `mm_hash`（path or blake3） | `MultiModalKwargsItem + prompt_updates` | GB 级 LRU | 无 |
| vLLM `ImageLoader` LRU | `image_loader.py` | 路径 or PIL hash | `PIL.Image` / `MediaWithPath` | 4 GiB | 无 |
| vLLM worker `MMMetaInfoCache` | `worker_base.py:230` | `mm_feature.identifier` (= mm_hash) | mm metainfo（grid_thw 等，非 raw） | **10000** 项 | 无 |
| vLLM GPU `encoder_cache` | model runner 内 | `mm_hash` | encoder 输出 tensor（vision/audio embedding） | 调度器管 | scheduler-evict |

---

## 六、五个深入问题

### 6.1 `VLLM_OMNI_ENABLE_ENCODER_BATCH` 的 sort 与还原

**排序 key 是 `(modality_str, original_index)`**。同 modality 内保留原序，目的不是优先 image/audio/video 谁先跑，而是为了让 `group_mm_kwargs_by_modality` 能用 `itertools.groupby` 把同 modality 的 item 沿 batch 维拼成一次 `embed_multimodal` 调用。

```python
# /mnt/data/llx/vllm/vllm/v1/worker/gpu_model_runner.py:2786
if envs.VLLM_OMNI_ENABLE_ENCODER_BATCH:
    logger.info("resort mm input to enable encoder batch input")

    indexed_kwargs = [(i, item) for i, item in enumerate(mm_kwargs)]
    sorted_indexed = sorted(indexed_kwargs, key=lambda x: (x[1].modality, x[0]))
    sorted_mm_kwargs = [item for _, item in sorted_indexed]
    sorted_indices = [idx for idx, _ in sorted_indexed]

    for modality, num_items, mm_kwargs_group in group_mm_kwargs_by_modality(
        sorted_mm_kwargs,
        device=self.device,
        pin_memory=self.pin_memory,
    ):
        ...
            curr_group_outputs = model.embed_multimodal(**mm_kwargs_group)
        ...
        encoder_outputs_sorted.extend(curr_group_outputs)

    encoder_outputs = [None] * len(mm_kwargs)
    for out_idx, original_idx in enumerate(sorted_indices):
        encoder_outputs[original_idx] = encoder_outputs_sorted[out_idx]
```

**Batch 内部**：

```python
# /mnt/data/llx/vllm/vllm/multimodal/utils.py:410
def group_mm_kwargs_by_modality(...):
    for modality, items in groupby(mm_kwargs, key=lambda item: item.modality):
        items_lst = list(items)
        mm_kwargs_items = MultiModalKwargsItems.from_seq(items_lst)
        mm_kwargs_data = mm_kwargs_items.get_data(
            device=device,
            pin_memory=pin_memory,
        )
        yield modality, len(items_lst), mm_kwargs_data
```

**还原**：`sorted_indices[k] = 排序后第 k 个对应原始的哪个位置`，逆映射写回，无额外搜索成本。

**例子**：5 张图 + 3 段音频，不开 batch 跑 8 次 encoder forward；开 batch 跑 2 次（image group 1 次 batch=5，audio group 1 次 batch=3）。Video EVS 开启 + `num_items > 1` 是例外，会被强制拆成 micro-batch=1。

### 6.2 mp3/mp4/jpg 的解码与"放进请求"

**Image**：

```python
# /mnt/data/llx/dashscope-serving/multimodal_serving/preprocessor/multimodal_data/image/image_data_preprocessor.py:77
def image_data_load_bgr_image(image_data: ImageData) -> np.ndarray:
    nparr = np.frombuffer(image_data.raw, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    return img
```

resize 用 Qwen-VL 的 `smart_resize`（H/W 是 16 的倍数）：

```python
# /mnt/data/llx/dashscope-serving/multimodal_serving/preprocessor/multimodal_data/image/image_data_preprocessor.py:101
resized_height, resized_width = smart_resize(h, w,
    factor=mm_conf.QWEN_VL_SIZE_FACTOR,
    min_pixels=image_data.min_pixels,
    max_pixels=image_data.max_pixels)
resize_img = cv2.resize(img, (resized_width, resized_height))
rgb_image = cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB)
```

**Video**：自研 `sniper_codec`（非 decord/OpenCV），帧率 clamp：

```python
# /mnt/data/llx/dashscope-serving/multimodal_serving/preprocessor/multimodal_data/video/video_file_data_preprocessor.py:293
frames = sniper_decode(input_video_file=video_file,
                       input_video_info=input_video_info,
                       output_fps=video_data.sampled_fps,
                       output_size=output_size,
                       enable_chunked_decode=True,
                       ...)
```

视频里的音频抽出来用 ffmpeg subprocess：

```python
# /mnt/data/llx/dashscope-serving/multimodal_serving/utils/ffmpeg.py:27
def extract_audio(video_path: str, audio_output_path: str) -> bool:
    command = ['ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a', audio_output_path]
    subprocess.run(command, check=True)
```

**Audio**：ffmpeg pipe 解到 s16le 再 / 32768 转 float32 16kHz mono：

```python
# /mnt/data/llx/dashscope-serving/multimodal_serving/utils/ffmpeg.py:125
def load_bytesio_audio(audio_file: str, sr: int = 16000) -> np.ndarray:
    cmd = [
        'ffmpeg', '-y', '-nostdin', '-threads', '0', '-i', audio_file, '-f', 's16le', '-ac', '1', '-acodec', 'pcm_s16le',
        '-ar', str(sr), 'pipe:'
    ]
    ...
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
```

### 6.3 D 端如何知道 MM token 数（不跑 input_preprocessor）

三层机制：

1. **D 收到的 input_ids 是 P 展开后的完整序列**（含 N 个 pad token），来源于 P 通过 `extra_params["prompt_token_ids"]` 推过来。
2. **KV slot 分配**：调度器只看 `prompt_token_ids` 长度，不区分 pad/普通 token。
3. **M-RoPE 位置**：P 算出 `mrope_position_delta` 并传给 D，D 不重算 MM 段位置：

```python
# /mnt/data/llx/vllm/vllm/v1/worker/gpu_model_runner.py:1234
if self.uses_mrope:
    remote_mrope_position_delta = (
        sampling_params.extra_args.get("mrope_position_delta")
        ...
    )
    if remote_mrope_position_delta is not None:
        input_len = len(self.requests[req_id].prompt_token_ids)
        req_state.mrope_position_delta = remote_mrope_position_delta
        req_state.mrope_positions = (
            torch.arange(input_len) + remote_mrope_position_delta
        ).expand(3, input_len)
    else:
        self._init_mrope_positions(req_state)
```

非 omni 的 v1 PD path 才有 `prompt_token_ids_len + pad token prepend` 那套，omni PD 直接传完整 ids。

### 6.4 Placeholder → embedding 位置映射

**Token id**：

| Token | ID |
|-------|-----|
| `<\|vision_start\|>` | 248053 |
| `<\|vision_end\|>` | 248054 |
| `<\|image_pad\|>` | 248056 |
| `<\|video_pad\|>` | 248057 |
| `<\|audio_start\|>` | 248070 |
| `<\|audio_end\|>` | 248071 |
| `<\|audio_pad\|>` | 248076 |

**Placeholder 展开（pad 数量计算）**：

```python
# /mnt/data/llx/vllm/vllm/model_executor/models/qwen3_omni_next_thinker.py:604
audio_tokens_per_second = math.ceil(
    feature_extractor.sampling_rate
    / feature_extractor.hop_length
    / 2**downsample_times
)
...
def get_audio_replacement_qwen35omni(item_idx: int):
    ...
    num_features = _get_feat_extract_output_lengths(
        out_item["feature_attention_mask"].data.sum()
    )
    audio_token_str = hf_processor._get_audio_tokens(
        num_features,
        audio_tokens_per_second,
        timestamp_interval,
    ).replace("<|audio_placeholder|>", audio_token)
    return PromptUpdateDetails.select_token_id(audio_token_str, audio_token_id)
```

`audio_tokens_per_second = ceil(sr / hop_length / 2^downsample)`，对 Qwen3.5 典型配置 = **7**。

**Scatter encoder embedding 到 pad 位置**（三步）：

```python
# /mnt/data/llx/vllm/vllm/v1/worker/gpu_model_runner.py:2972
for mm_feature in req_state.mm_features:
    pos_info = mm_feature.mm_position
    start_pos = pos_info.offset
    num_encoder_tokens = pos_info.length
    ...
    encoder_output = self.encoder_cache.get(mm_hash, None)
    ...
    is_mm_embed[req_start_pos + start_idx : req_start_pos + end_idx] = (
        True if is_embed is None else is_embed
    )
```

```python
# /mnt/data/llx/vllm/vllm/model_executor/models/qwen3_omni_next_thinker.py:1255
is_multimodal = (
    (input_ids == video_token_id)
    | (input_ids == audio_token_id)
    | (input_ids == image_token_id)
)
inputs_embeds = self._embed_text_input_ids(
    input_ids,
    self.language_model.embed_input_ids,
    is_multimodal=is_multimodal,
    ...
)
...
inputs_embeds = _merge_multimodal_embeddings(
    inputs_embeds=inputs_embeds,
    multimodal_embeddings=multimodal_embeddings,
    is_multimodal=is_multimodal,
)
```

```python
# /mnt/data/llx/vllm/vllm/model_executor/models/utils.py:455
def _merge_multimodal_embeddings(
    inputs_embeds: torch.Tensor,
    multimodal_embeddings: NestedTensors,
    is_multimodal: torch.Tensor,
) -> torch.Tensor:
    ...
    inputs_embeds.masked_scatter_(
        is_multimodal.unsqueeze(-1), mm_embeds_flat.to(dtype=input_dtype)
    )
```

### 6.5 `omni_prefill_remote_endpoint` 在 talker 侧

**Endpoint 是 ZMQ RPC + Vineyard 共享内存控制面地址**：

```python
# /mnt/data/llx/dashllm/dashllm/core/backend/engine/_vllm_omni_rpc.py:384
def get_remote_endpoint(self) -> str:
    if self._ip is None or self._send_port is None:
        return None
    return self._ip + ":" + str(self._send_port)
```

**PD 模式下分布**：

| 组件 | P (kv_producer) | D (kv_consumer) |
|------|-----------------|-----------------|
| Thinker MM encoder + prefill（产 KV） | ✓ | ✗ |
| Thinker decode（出 text token 流） | ✗ | ✓ |
| Talker（出 audio token 流） | ✗ | ✓ |
| Code2wav（audio token → 波形） | ✗ | ✓ |

Talker 需要两路 t2t 流：

1. `t2t_controller_prefill` → 连 P 的 thinker（拿 prefill 阶段 hidden states）
2. `t2t_controller` → 连 D 的 thinker（拿后续 text token）

```python
# /mnt/data/llx/vllm/vllm/engine/omni3_5_talker_engine.py:2185
embed_endpoint = prefill_remote_endpoint or remote_endpoint
...
if prefill_remote_endpoint and self.t2t_controller_prefill is not None:
    asyncio.run_coroutine_threadsafe(
        self.t2t_controller_prefill.prepare_request_transfer(
            request_id, prefill_remote_endpoint, mm_prompt_len
        ),
        self._recv_loop,
    ).result()
...
asyncio.run_coroutine_threadsafe(
    self.t2t_controller.prepare_request_transfer(
        request_id, t2t_endpoint, mm_prompt_len
    ),
```

**一句话**：`omni_prefill_remote_endpoint` 是 P thinker 的 t2t RPC 地址，让 D 上的 talker 额外开一条 t2t 流去 P 拿 prefill 阶段算出来的 hidden states。普通 `omni_remote_endpoint` 仍然指 D 自己的 thinker。两条流缺一不可。

---

## 七、omni 实时场景特别注意点

1. **D 永远没机会重建 MM 段**：raw `multi_modal_data` 在 P→D 的 StreamInfer 包里被故意去掉，KVT 控制结构里也没有 MM 字段。所以 D 一旦 KV 没拉到（P abort 或网络问题），**唯一出路是放弃这个 request**。这就是引入 `CODE_REQGONE` 的根本动机。
2. **`prompt_token_ids` 在 external_router 路径被剪枝**：D 端依赖 `prompt_token_ids_len + pad token` 补齐 — omni 路径不走这套（直接传完整 ids），但要确认每次 dashllm 路径选择正确。
3. **三层 hash 要对齐**：dashscope `mm_key`（md5）→ vLLM `MultiModalHasher`（path 或 blake3）→ encoder_cache `mm_hash`。`VLLM_USE_FILE_PATH_MMD_CACHE=True` 时同 path 不同内容会错命中。
4. **omni 视频显式禁用 OSS cache**，高并发下每个 video URL 都要走 ffmpeg 抽音频 + 重新预处理。
5. **`MMMetaInfoCache` capacity 10000 项**：实时多用户场景容易撑满，命中失败时触发 SHM recovery 读，相对慢。

---

## 八、KV transfer 修复（CODE_REQGONE）回顾

> 这一节单独成文，记录前面 PD 高并发问题的修复方案。

### 8.1 问题

并发高 + 实时 omni 场景下：
- P 端：`503 ServiceUnavailable: Too many requests`
- D 端：`KVTResp(code=404, cached=0, computed=-1, output_token_ids=[])` 重试到几十次，`HybridScheduler` 卡在 `loading=4`

### 8.2 根因

时间窗错配 + 状态错配：
- P 上 prefill 完成后 `_dash_done[reqid]` 等 D 来取 KV
- 实时业务在新音频段到达时主动 abort 前段的 prefill（但保留 decode 请求）→ P 清掉 `_dash_done[reqid]` + KV
- D 还在 retry `TRANSFER_KV_REQ`，P 返回 404（普通的 "not found"）
- D 收到 404 默认 retry，loading slot 一直占着不释放 → 占满 KV → 触发 D 端 503 → 又触发更多 abort → 雪崩

### 8.3 修复方案

引入 `CODE_REQGONE=410`（P 显式告诉 D「这个 reqid 确实没了」）+ TTL LRU：

```python
# /mnt/data/llx/vllm/vllm/v1/hybrid_connector/kvtbackend.py:706
self._gone_reqs: OrderedDict[str, float] = OrderedDict()
```

```python
# /mnt/data/llx/vllm/vllm/v1/hybrid_connector/kvtbackend.py:1146
self._dash_done.pop(areq.request_id, None)
# Record the abort for future TRANSFER_KV_REQ for this reqid
self._mark_gone(areq.request_id, "abort_save")
```

```python
# /mnt/data/llx/vllm/vllm/v1/hybrid_connector/kvtbackend.py:1012
if self._is_gone(rdinfo.reqid):
    code = CODE_REQGONE
    reason = "req gone"
else:
    code = CODE_REQNOTFOUND
    reason = "req not found"
```

```python
# /mnt/data/llx/vllm/vllm/v1/hybrid_connector/kvtbackend.py:2247
if kvtresp.code == CODE_REQGONE:
    logger.warning(
        "disagg kv gone on P. reqid=%s peer=%s retry=%s",
        reqid, peer_hint, retry,
    )
    raise RuntimeError(
        f"kv gone on P "
        f"(reqid={reqid}, peer={peer_hint}, retry={retry})")
```

环境变量 `VLLM_KVT_GONE_REQ_TTL_S`（默认 0 关闭，omni 默认 60s）：

```python
# /mnt/data/llx/vllm/vllm/engine/omni3_5_llm_engine.py:3673
os.environ.setdefault("VLLM_KVT_MAX_DELAY_MS", "30000")
os.environ.setdefault("VLLM_KVT_GONE_REQ_TTL_S", "60")
```

### 8.4 验证修复是否生效

修复生效时，应该在日志里看到：

- **P 端**：每次 abort 都会打 `mark gone reqid. reqid=... reason=abort_save gone_size=N ttl_s=60`
- **D 端**：再来的请求收到 410（不再是 404），打 `disagg kv gone on P. reqid=... peer=... retry=N`，**retry 不应再涨到 20+**
- **D 端 `kvtresp`** 字段从 `code=404` 变成 `code=410`

如果 P 上根本看不到 `mark gone reqid` 这条 log，说明要么 `VLLM_KVT_GONE_REQ_TTL_S` 没生效（启动时被覆盖成 0），要么实际清空 `_dash_done` 走的不是 `_step_aborting` 路径。
