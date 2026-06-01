# Mooncake 项目功能与关键概念笔记

## 1. 项目整体功能

Mooncake 是一个面向大模型推理的分布式 KVCache / 对象缓存系统。它的核心目标是把 LLM 推理过程中产生的 KVCache、Tensor、临时对象等数据放到跨节点的 DRAM、VRAM、SSD 资源池中，并通过高性能传输引擎完成跨机器、跨设备的数据读写。

典型使用场景包括：

- vLLM / SGLang 的 prefill-decode 分离。
- 多实例之间复用 KVCache，减少重复 prefill 计算。
- 分层缓存：GPU 显存、CPU 内存、远端内存、SSD / 文件后端。
- Tensor、checkpoint、KVCache 等大对象的高速传输。

一句话理解：Mooncake 不是普通 Redis 类缓存，而是为 LLM 推理中的 KVCache 和 Tensor 数据流专门设计的分布式高速对象存储与传输系统。

## 2. 仓库主要模块

### 2.1 Transfer Engine

路径：`/mnt/data/llx/mooncake/mooncake-transfer-engine`

Transfer Engine 是底层数据传输层，负责在不同机器、不同设备之间搬运数据。它支持 TCP、RDMA、CXL、NVLink、NVMe-oF、Ascend 等协议或设备路径。

核心概念：

- `Segment`：一个进程或节点暴露出来的可访问地址空间。
- `Buffer`：Segment 中具体注册给传输引擎的内存范围。
- `BatchTransfer`：批量读写请求，可以一次提交多个读写任务。
- `TransferEngine`：统一管理传输协议、内存注册、远端 segment 打开、批量传输状态。

### 2.2 Mooncake Store

路径：`/mnt/data/llx/mooncake/mooncake-store`

Mooncake Store 是分布式 KVCache 存储层，基于 Transfer Engine 实现对象级别的 `Put`、`Get`、`Remove`、`BatchPut`、`BatchGet` 等操作。

主要角色：

- `MasterService`：管理元数据、对象到 segment 的映射、空间分配、lease、soft pin、驱逐、任务调度。
- `Client`：执行真正的数据读写，通过 Transfer Engine 把数据写到目标 segment 或从目标 segment 读出。
- `RealClient`：完整客户端 / 存储节点实现，通常作为常驻进程运行。
- `DummyClient`：轻量代理客户端，通常嵌在 Python 推理进程中，通过 RPC 和共享内存连接 RealClient。

### 2.3 Python 绑定

路径：`/mnt/data/llx/mooncake/mooncake-integration/store/store_py.cpp`

这里通过 pybind11 暴露 Python API，例如：

- `MooncakeDistributedStore`
- `ReplicateConfig`
- `put`
- `get`
- `put_parts`
- `put_from`
- `get_into`
- `get_buffer`
- `put_tensor`
- `get_tensor`

Python 用户主要通过：

```python
from mooncake.store import MooncakeDistributedStore, ReplicateConfig
```

来使用 Mooncake Store。

## 3. 全局 Segment 的含义

`global_segment_size` 指的是单个 RealClient / Client 在本机贡献出来的存储 segment 大小。

它不是一块天然跨机器的连续内存。物理上，每个 segment 都属于某台机器上的某个 RealClient 进程。多个机器上的 RealClient 各自挂载自己的 segment 后，Master 会把这些 segment 统一管理成一个逻辑上的分布式资源池。

所以可以这样理解：

- 物理层面：segment 在单机本地。
- 逻辑层面：Master 把所有机器的 segment 组合成跨机器资源池。
- 写对象时：`PutStart` 可以在任意可用 segment 上分配空间，因此对象副本可以被放到远端机器。
- 如果 `global_segment_size = 0`：该 client 是纯客户端，不贡献存储空间。

结论：全局 segment 不是单块跨机器共享内存，而是“单机贡献、多机汇总”的分布式存储池。

## 4. RealClient 与 DummyClient 的关系

### 4.1 RealClient

核心文件：

- `/mnt/data/llx/mooncake/mooncake-store/include/real_client.h`
- `/mnt/data/llx/mooncake/mooncake-store/src/real_client.cpp`
- `/mnt/data/llx/mooncake/mooncake-store/src/real_client_main.cpp`

RealClient 是完整客户端和存储节点实现。它负责：

- 初始化底层 `Client`。
- 挂载本机 global segment。
- 注册 Transfer Engine。
- 连接 MasterService。
- 维护本机内存池、共享内存、hot cache。
- 启动 RPC server，给 DummyClient 调用。
- 在 local master 模式下，同进程内启动 MasterService。

独立进程 `mooncake_client` 本质上就是一个 RealClient 服务。

### 4.2 DummyClient

核心文件：

- `/mnt/data/llx/mooncake/mooncake-store/include/dummy_client.h`
- `/mnt/data/llx/mooncake/mooncake-store/src/dummy_client.cpp`

DummyClient 是业务进程中的轻量代理。它通常运行在 Python / vLLM / SGLang 进程里，不自己完整管理存储节点，而是连接本机 RealClient。

它负责：

- 通过 RPC 调用 RealClient。
- 通过 Unix domain socket + SCM_RIGHTS 获取共享内存 fd。
- mmap RealClient 暴露的共享内存区域。
- 在 local master 模式下，直接通过共享内存池 `G` 执行 `put_parts` / `get_buffer` 快路径。
- 必要时把写入 fallback 到 offload storage。

### 4.3 多个 DummyClient 连接一个 RealClient

多个 DummyClient 可以连接同一个 RealClient。每个 DummyClient 有自己的 `client_id` 和共享内存上下文。

这些 DummyClient 提交的任务不是在 RealClient 上全局按顺序排队执行。RealClient 的 RPC server 是多线程 / 协程服务，不同 DummyClient、不同 key 的请求可以并发执行。

正确性不是靠“串行执行所有任务”保证，而是靠以下机制保证：

- MasterService 的元数据锁。
- 对象状态机：`PutStart`、`PutEnd`、`Remove`、`GetReplicaList`。
- lease / refcount。
- allocator 的并发控制。
- 同 key 写入冲突检测，例如对象已存在、正在写入、PutStart 未完成等。

因此：

- 不同 key 的请求通常可以并发。
- 同一个 key 的冲突由 MasterService 元数据逻辑处理。
- local master 模式下，`DummyClient::put_parts()` 的快路径甚至会在 DummyClient 进程中通过 mmap 的统一内存池写入，并不是所有数据都进入 RealClient 排队执行。

## 5. Soft Pin

### 5.1 概念

soft pin 是“软固定”对象的机制。它表示某个对象应该尽量保留在内存中，不要优先被驱逐。

配置入口是：

```cpp
ReplicateConfig config;
config.with_soft_pin = true;
```

Python 侧也可以设置：

```python
config = ReplicateConfig()
config.with_soft_pin = True
store.put("key", b"value", config)
```

### 5.2 它不是 hard pin

soft pin 不等于永远不能驱逐。它更像一个驱逐优先级 hint：

- 普通对象优先被驱逐。
- soft pinned 对象尽量延后驱逐。
- 如果配置允许驱逐 soft pinned 对象，空间非常紧张时仍可能被驱逐。

默认 TTL 在 `DEFAULT_KV_SOFT_PIN_TTL_MS` 中定义，目前是 30 分钟。

### 5.3 Master 层作用

在 `MasterService::DoPutStart` 中，如果 `config.with_soft_pin` 为 true，会设置 `soft_pin_timeout`。

驱逐时大致逻辑是：

- 第一轮优先驱逐非 soft pin 对象。
- 如果空间仍不够，并且允许驱逐 soft pin 对象，再驱逐 soft pin 对象。

### 5.4 Offload 层作用

在 tiered offload 中，`with_soft_pin` 还有额外含义。

例如 `TieredOffloadStorage` 里，如果 primary backend 写入失败，只有当 `with_soft_pin = true` 时，才会尝试 fallback backend。

也就是说 soft pin 在这里还控制“主后端失败时是否允许走 fallback”。

## 6. put_parts

### 6.1 API 语义

`put_parts(key, part1, part2, ...)` 用来把多个不连续的 buffer 当作同一个对象写入。

它适合以下场景：

- 上层数据天然分成多段。
- 不想在 Python 里先拼成一个大 bytes。
- Tensor metadata 和 payload 分开存储。
- local master 模式下希望走共享内存快路径。

Python 绑定在 `store_py.cpp` 中会把多个 bytes-like 参数转换成 `std::span<const char>`，再调用 C++ 的 `put_parts`。

### 6.2 RealClient 路径

`RealClient::put_parts_internal` 的大致流程：

1. 计算所有 part 的总大小。
2. 从 client buffer allocator 中分配连续 buffer。
3. 把每个 part 依次 memcpy 到连续 buffer 中。
4. 调用 `split_into_slices` 切分为 Mooncake Store 的 slice。
5. 调用 `client_->Put(key, slices, config)` 完成存储。

相关函数：

- `RealClient::put_parts`
- `RealClient::put_parts_internal`
- `RealClient::put_parts_dummy_helper`

### 6.3 DummyClient local master 路径

local master 模式下，`DummyClient::put_parts` 会优先走快路径：

1. 把每个 part 转成 `Slice`。
2. 调用 `Client::DummyPut`。
3. `DummyPut` 向 Master 执行 `PutStart`。
4. 将 RealClient 侧统一内存池 `G` 的地址映射成 DummyClient 进程里的 mmap 地址。
5. 通过 `TransferWrite` 写入。
6. 调用 `LocalPutEnd` 标记写入完成。

如果这条路径失败，并且配置了 offload storage，则会走：

```cpp
DummyClient::put_parts_offload(...)
```

最终调用：

```cpp
offload_storage_->PutParts(...)
```

### 6.4 get_buffer 配合

`get_buffer` 返回的是 `BufferHandle`，Python 可以通过 buffer protocol 直接读取其中内容。

DummyClient 的 `get_buffer` 会优先尝试：

- hot cache 共享内存路径。
- local master 共享内存池 `G` 路径。
- allocator-backed fallback。

它和 `put_parts` 一起构成 DummyClient local master 模式下最关键的读写路径。

## 7. 主要函数速查

### Python API

- `MooncakeDistributedStore.setup()`：初始化 real client，连接 Master / metadata server。
- `MooncakeDistributedStore.setup_dummy()`：初始化 dummy client，连接 RealClient。
- `put()`：写入单个 bytes-like 对象。
- `get()`：读取对象并返回 bytes。
- `put_parts()`：把多个 part 作为一个对象写入。
- `put_from()`：从已注册 buffer 直接写入。
- `get_into()`：直接读入用户提供的 buffer。
- `register_buffer()` / `unregister_buffer()`：注册 / 反注册可被 Transfer Engine 访问的内存。
- `get_buffer()`：返回可直接读取的 `BufferHandle`。
- `put_tensor()` / `get_tensor()`：存取 PyTorch tensor。

### Store / Client 层

- `MasterService::PutStart()`：为对象分配副本位置，创建 processing 状态元数据。
- `MasterService::PutEnd()`：标记对象写入完成，使对象变为可读。
- `MasterService::GetReplicaList()`：查询对象副本列表，并授予读取 lease。
- `MasterService::AcquireReplica()`：local master 模式下占用对象副本，防止读取期间被回收。
- `Client::Put()`：常规对象写入路径。
- `Client::DummyPut()`：DummyClient local master 模式下的共享内存写入路径。
- `RealClient::put_parts_internal()`：RealClient 侧多 part 写入实现。
- `DummyClient::put_parts()`：DummyClient 侧多 part 写入入口。
- `DummyClient::put_parts_offload()`：local put 失败后的 offload 写入路径。

### Transfer Engine 层

- `registerLocalMemory()`：注册本地内存。
- `unregisterLocalMemory()`：反注册本地内存。
- `openSegment()`：打开远端 segment。
- `submitTransfer()`：提交批量读写请求。
- `getTransferStatus()`：查询异步传输状态。

## 8. 两个问题的回答整理

### 问题 1：全局 segment 是单台机器上的吗？还是可以跨机器申请？

回答：全局 segment 物理上是单台机器上的。每个 RealClient 根据 `global_segment_size` 在本机贡献一段存储空间。多个 RealClient 的 segment 会被 Master 汇总成逻辑上的分布式资源池。

所以，不能把它理解成“一块跨机器连续共享内存”。更准确的理解是：

- 每个 segment 属于某台机器。
- Master 统一管理所有 segment。
- 对象写入时可以被分配到任意机器的 segment。
- 多副本时，不同副本也可以分布在不同 segment / 不同机器上。

### 问题 2：DummyClient 和 RealClient 的关系是什么？多个 DummyClient 连接一个 RealClient 时，任务是否按顺序执行？

回答：DummyClient 是业务进程里的轻量代理，RealClient 是常驻的完整客户端 / 存储节点。多个 DummyClient 可以连接同一个 RealClient，并通过 RPC 与共享内存提交请求。

这些任务不是全局按顺序执行的。RealClient 的 RPC server 支持并发处理，不同 DummyClient 的请求可以并发进入。系统通过 MasterService 元数据锁、对象状态机、lease、allocator 锁和冲突检测来保证正确性。

对同一个 key：

- 如果对象已经写入完成，再写同 key 会触发冲突。
- 如果对象正在 PutStart 但还没 PutEnd，其他写入会看到 `REPLICA_IS_NOT_READY` 等状态。
- 读取期间通过 lease / refcount 防止对象被错误回收。

因此，Mooncake 的一致性依赖元数据状态机，而不是依赖所有 DummyClient 请求串行化。

## 9. 一句话总结

Mooncake 的核心架构是：Transfer Engine 负责高速数据搬运，MasterService 负责元数据和空间分配，RealClient 负责本机存储节点和 RPC 服务，DummyClient 作为业务进程内的轻量代理通过 RPC 和共享内存连接 RealClient；`put_parts` / `get_buffer` 是 DummyClient local master 模式下最关键的高性能读写路径，soft pin 则影响对象的内存驻留优先级和 tiered offload fallback 策略。
