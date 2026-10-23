# 设计：FlyDSL intranode DeepEP 支持 MI455X（gfx1250）

> 交给 MI455 机器上的 session 执行。全程用中文和用户交流。先完整读完本文再动手。
> 文中所有路径都相对于仓库根目录。

## 0. 背景与现状

- 分支：本文所在的 `feat/flydsl-deepep-gfx1250`，基于 `feat/flydsl-deepep-intranode`（`86fc567f`，gfx950 版 FlyDSL 后端，待开 PR）。在本分支上开发，不要动 `feat/flydsl-deepep-intranode` 和 main。
- 启用方式：`Buffer(..., intranode_backend="cpp"|"flydsl")`，或环境变量 `PRIMUS_TURBO_DEEPEP_INTRANODE_BACKEND`，默认 `cpp`。
- 已完成的工作（gfx950 / MI355X ×8）：
  - DeepEP intranode 路径（`num_rdma_ranks == 1`）的 FlyDSL 后端，支持 2/4/8 卡。
  - 与 C++ 后端逐位一致（layout、dispatch 输出、handle、combine），两个后端的 handle 可以互用。注意 `rank_prefix_matrix` 只有第 `rank` 列有意义。
  - 性能基线（gfx950，供对比）：
    - 按 Primus 方式调用（每 rank 8192 token，hidden 7168，top-8/256，一次 dispatch + combine 的前向加反向）：80 CU 时 FlyDSL 7.43 ms/step、C++ 7.61；20 CU 时 11.92 vs 12.16。
    - DSV3 4 层 proxy 端到端（TP1/PP1/EP8，mbs 2，GBS 512，seq 4096，`turbo_deepep_num_cu=80`）：BF16 9537 vs 9529 ms/iter，FP8 9103 vs 9105 ms/iter（FlyDSL vs C++），与 C++ 持平。
    - XGMI：理论 504 GB/s/方向（7 × 72），实测纯远端写上限约 388 GB/s，dispatch/combine 约 330 GB/s（实测上限的 85%）。
- 相关代码：

  | 文件 | 内容 |
  |---|---|
  | `primus_turbo/flydsl/deep_ep/common.py` | `WARP_SIZE`、`load_index`、`load_i32`、`tensor_addr`、LDS helper（`lds_base` / `ld_lds` / `st_lds`） |
  | `primus_turbo/flydsl/utils/launch_cache.py` | `run_launcher`：`flyc.compile` 缓存，constexpr 从 launcher 签名自动识别 |
  | `primus_turbo/flydsl/deep_ep/layout_kernel.py` | `get_dispatch_layout` |
  | `primus_turbo/flydsl/deep_ep/intranode_kernel.py` | notify / cached notify / dispatch / cached_notify_combine / combine，以及 barrier、轮询等 device helper |
  | `primus_turbo/flydsl/utils/prims.py` | `copy_warp`、`read_clock`、`_readlane_i32`、`ld`/`st`/`atomic_add`、`memory_fence`。**Mega MoE 也在用，改动必须向后兼容** |
  | `primus_turbo/flydsl/utils/buffer_ops.py` | buffer resource / load / store、`_get_buffer_flags` |
  | `primus_turbo/pytorch/deep_ep/flydsl_intranode.py` | host 运行时 `FlyDSLIntranodeRuntime` |
  | `primus_turbo/pytorch/deep_ep/buffer.py` | 后端切换（`intranode_backend` / `PRIMUS_TURBO_DEEPEP_INTRANODE_BACKEND`） |
  | `primus_turbo/pytorch/core/symm_mem.py` | `SymmetricMemory(uncached_buffer=...)` |
  | `tests/pytorch/deep_ep/test_intranode.py` | 双后端测试和逐位 parity 测试 |

## 1. 目标与非目标

**目标**
1. FlyDSL intranode DeepEP 在 MI455X（gfx1250）上功能正确：layout / dispatch（bf16、fp8 + scales、top-k、`num_worst_tokens`、cached）/ combine（bf16、fp32、bias）。
2. gfx950 行为不变：生成的代码和性能不回退，与 C++ 仍逐位一致。
3. MI455 上性能不差于同机的 C++ DeepEP（如果 C++ 能用）。至少给出 micro-bench 和端到端数据。
4. FlyDSL 后端在 gfx1250 上**不依赖 C++ `deep_ep` 模块**，这样即使 C++ DeepEP 在 gfx1250 上没编进来也能用。

**非目标**
- internode / rocSHMEM / low-latency。
- 和 GEMM 融合（Mega MoE）。
- gfx1250 上的 C++ DeepEP 本身。它有独立分支，见 §8。

## 2. gfx950 实现依赖的硬件假设（逐条对照 gfx1250）

| # | 现在的假设（gfx950） | 代码位置 | gfx1250 情况 | 处理 |
|---|---|---|---|---|
| A1 | wave64 | `common.WARP_SIZE`；`prims._WARP`；`intranode_kernel` 里所有 `lane_id/warp_id`、`NUM_WARPS`、`warps_per_rank`；notify 的 `shuffle_xor`（32→1）；`cached_notify_combine` 的 `range_constexpr(WARP_SIZE)` readlane 展开和 `num_threads = max(128, WARP_SIZE*R)`；combine 的 `num_steps`、`lane < num_ranks`/`num_topk` 断言；scales 循环 | **wave32** | traits 化（§3），全部改用 `traits.wave_size` |
| A2 | ballot 返回 i64 | combine reducer 的 `_wave_any`（来自 `utils/attn_helper.py`，内部写死 i64 ballot；attention 也在用，不要直接改） | wave32 下为 i32 | 在 deep_ep 内按 wave size 选 ballot 宽度（或给 `_wave_any` 加可选参数，默认 i64） |
| A3 | 统一 `s_waitcnt` | `group_barrier(publish)`、`wait_new_tail`、combine head-updater 的 `fx.rocdl.s_waitcnt(0)`；`prims.read_clock` 的内联汇编 `s_memrealtime; s_waitcnt lgkmcnt(0)` | gfx12 **去掉了**统一 waitcnt，拆成 loadcnt / storecnt / dscnt / kmcnt（C++ 的 `PRIMUS_TURBO_WAIT_ALL_STR` 就是这么写的）；`s_memrealtime` 不可用 | `traits.wait_all()`；`read_clock` 按架构实现（`llvm.readsteadycounter` 之类，只用于超时打印） |
| A4 | cache policy 用 gfx94x 编码：`SYS_LOAD = 0x13` = sc0 \| sc1 \| nt | dispatch receiver 的行拷贝、combine reducer 的行读取 | gfx12 是 **TH（3 bit）+ SCOPE（2 bit）**。参考 `primus_turbo/flydsl/grouped_gemm/grouped_gemm_bf16_kernel_gfx1250.py` 中 `th = value & 0x7, scope = (value >> 3) & 0x3` 的解码 | `traits.cpol_sys_load` / `traits.cpol_sys_store` |
| A5 | **cheap fence**：堆用 `hipDeviceMallocUncached`；发送端 `s_waitcnt` 等自己的 store 完成后，用 relaxed store 发布 tail；接收端用 system scope 的 load 读 payload | `group_barrier(..., publish=True)` + `_st_sys(tail)`；`wait_new_tail`；`SYS_LOAD`；`_ld_sys` 读 src_idx / topk / scales | C++ 在 gfx1250 上**不用** cheap fence（`kHasCheapFence = false`），走完整 release/acquire | 见 §4：先保守（release/acquire），再尝试 storecnt 版快路径 |
| A6 | gfx9 的 vmcnt 也计 store，所以行拷贝要分小块（按 channel 数 1/2/8 KB） | `dispatch_copy_piece_bytes`、`combine_copy_piece_bytes` | loadcnt 和 storecnt 分开，前提不成立 | 分块策略放进 traits，在 MI455 上重新扫描 |
| A7 | buffer resource 越界语义：越界读返回 0、越界写丢弃 | combine reducer 对没路由到的 rank 用 `num_records_bytes = select(routed, hidden_bytes, 0)` 的空 resource；`copy_warp` 的不整齐尾部（hidden 2880 → 5760 B）；行拷贝的 tail lane | `_get_buffer_flags`：`is_rdna_arch('gfx1250')` 为 **False**，走 CDNA flags。**越界语义未验证** | probe P1；不一致就改成显式 mask（`arith.select` + 偏移） |
| A8 | 组内同步用 LDS 计数器加自旋（gfx9 没有硬件子组 barrier） | `group_barrier` | gfx1250 有硬件 named barrier（`s_barrier_signal_var` / `s_barrier_wait`，FlyDSL rocdl 已有） | 第二阶段可选替换（C++ 分支 `a55d3815` 就是这么做的） |
| A9 | 用标量 load 读 `is_token_in_rank`（避免 vmcnt 等远端 store） | `_sload_u8` | 原因（A6）不成立，但标量 load 本身仍可用 | 保留；性能阶段对比向量 load |
| A10 | 宿主依赖 C++ `deep_ep`：`EventHandle`、`Config`（需要 `bindings_pytorch.cpp` 的 `def_readonly`）、一个 0 字节堆的 C++ `Buffer`（拿 comm stream 和 rank 信息） | `flydsl_intranode.py`、`buffer.py` | gfx1250 上 C++ DeepEP 的状态不确定（适配在独立分支上） | §5：flydsl 模式下纯 Python |
| A11 | rank 数 ∈ {2,4,8}，`NUM_MAX_NVL_PEERS = 8`，barrier signal pad 8 个 int | `SUPPORTED_NUM_RANKS`；`FlyDSLIntranodeRuntime.__init__`；kernel 断言 `NUM_WARPS % R == 0` | 取决于 MI455 节点内能 IPC 互访的 GPU 数。wave32 下 1024 线程 = 32 个 warp，kernel 最多支持 32 rank | probe P5 确认拓扑后再决定是否放开 |
| A12 | 每个 block 1024 线程，最多 128 个 VGPR | 所有 dispatch / combine kernel | gfx1250 的寄存器文件和占用模型不同 | 编译后看 ISA 的 vgpr_count / spill（§7） |
| A13 | `Buffer.get_dispatch_config` / `get_combine_config` 的 chunk 大小，以及 Primus 的 `turbo_deepep_num_cu=80`，都是按 MI300/MI355 定的 | `buffer.py`；Primus yaml | MI455 的 CU 数和 XGMI 带宽不同 | 性能阶段重新扫描 |

## 3. 设计：`ArchTraits`

新增 `primus_turbo/flydsl/deep_ep/arch.py`：

```python
@dataclass(frozen=True)
class ArchTraits:
    name: str                    # "gfx950" / "gfx942" / "gfx1250"
    wave_size: int               # 64 / 32
    cpol_sys_load: int           # system-scope 流式读的 cache policy
    cpol_sys_store: int          # system-scope 写（gfx950 = 0，保持现状）
    cheap_fence: bool            # gfx942/950 True；gfx1250 先 False
    has_named_barrier: bool      # gfx1250 True（第二阶段用）
    def wait_all(self): ...      # gfx9: s_waitcnt(0)；gfx12: s_wait_loadcnt/storecnt/dscnt/kmcnt(0)
    def wait_stores(self): ...   # gfx9: s_waitcnt vmcnt(0)；gfx12: s_wait_storecnt(0)
    def ballot_type(self): ...   # i64 / i32
    def copy_piece_bytes(self, op, num_channels): ...  # gfx950 = 现在的表；gfx1250 实测后填
def get_arch_traits(arch: Optional[str] = None) -> ArchTraits:  # 从 FLYDSL_GPU_ARCH 或当前设备推断
```

要点：
- **traits 只在 kernel builder（Python 侧、编译期）里读**，生成不同 IR。所有 `@flyc.jit` launcher 在签名里加一个 `arch: fx.Constexpr[str]` 参数，调用时传 `traits.name`。`run_launcher` 会从签名自动把它算进缓存 key，同时它也进入 FlyDSL 的编译 key。
- `prims.copy_warp` / `_readlane_i32` / `read_clock` 增加可选参数（`wave_size=None`、`arch=None`），默认值保持现在的行为，**不影响 Mega MoE**。
- gfx950 的 traits 必须让生成代码与现在一致。验收：gfx950 上 parity 测试仍逐位一致，micro-bench 不回退。

## 4. 内存序设计（gfx1250）

现在的协议（同 C++ intranode）：发送端写 payload，然后发布 tail；接收端读到新 tail 后读 payload，用完发布 head；发送端读 head 判断空槽。双方都在接收端的堆上。

**阶段 1：保守（先求正确）**
- 发送端发布：`group_barrier` 前各 warp `wait_stores()`，然后 tail 用 `st(..., order="release", scope="sys")`（prims 已支持 `order`）。
- 接收端：tail 用 `ld(..., order="acquire", scope="sys")`；payload 用普通 load 加 `traits.cpol_sys_load`。
- head 同理：接收端用 release store，发送端用 acquire load。
- 堆仍然 `uncached_buffer=True`，先不变，以 probe P3 的结论为准。
- notify / cached notify 的 `barrier_block` 本来就是 release fence + system scope 原子 + acquire fence，按 A3 替换 wait 后应可直接用。

**阶段 2：快路径（可选，需要压测证明）**
- gfx12 的 storecnt 独立计数。payload store 带 SCOPE_SYS cpol，`s_wait_storecnt 0` 之后 store 已在 system scope 可见，再用 relaxed store 发布 tail；接收端用 SCOPE_SYS 读。这相当于 gfx12 版 cheap fence，而且不依赖 uncached 分配。
- 正确性论证要写进代码注释，参照 `utils.cuh` 里 cheap fence 那段的写法。并做 §6 的压力测试：多轮、随机路由、不同 `num_sms` 和 chunk 配置，对比逐位结果或参考结果。

## 5. 宿主侧：flydsl 模式不依赖 C++ `deep_ep`

1. `EventHandle`：在 `flydsl_intranode.py` 里实现一个 Python 版，包装 `torch.cuda.Event`，接口为 `current_stream_wait()`、在指定 stream 上 record。`Buffer.capture()`、`previous_event` 在 flydsl 模式下也用它。注意 `EventOverlap` 只调用 `.current_stream_wait()`。
2. `Config`：Python 兼容类，字段与 C++ 一致，并实现 `get_nvl_buffer_size_hint`（照抄 `csrc/include/primus_turbo/deep_ep/config.hpp` 的公式），供 `moe_dispatch_combine_impl.init_buffer` 计算 buffer 大小。C++ 可用时仍兼容 C++ `Config`（字段已暴露）。
3. `Buffer.__init__`：当 `intranode_backend == "flydsl"` 且是 intranode 场景时，不构造 `deep_ep_cpp.Buffer`。需要处理的点：
   - comm stream 自己用 `torch.cuda.Stream()` 创建；
   - 遵守 `PRIMUS_TURBO_EP_FORCE_CURRENT_STREAM`；
   - `get_num_rdma_ranks()` 返回 1；
   - `get_local_buffer_tensor` 走 FlyDSL 的堆；
   - `destroy` 只释放 FlyDSL 的堆；
   - `runtime` 的其他 C++ 方法（internode、low-latency）调用时直接报错。
4. `import primus_turbo.pytorch._C.deep_ep` 改成懒加载或 try/except：gfx1250 上 `_C` 里可能没有 deep_ep。
5. 验收：在 gfx950 上设 `PRIMUS_TURBO_DEEPEP_INTRANODE_BACKEND=flydsl`，pytest 和 parity 全过。可以临时 mock 掉 `_C.deep_ep` 验证确实没用到。

这一步完全可以在 gfx950 上做完并回归，建议在拿到 MI455 机器前或刚开始时先做。

## 6. 硬件 probe（在 MI455 上先跑，结论写回本文）

probe 脚本放在仓库外的临时目录，不进仓库。

- **P1 越界语义**：buffer resource 的 `num_records = 0` 时，`buffer_load` 返回 0、`buffer_store` 丢弃；`num_records = N` 时越界访问同样处理。覆盖 b32 / b128、读写。
- **P2 cache policy**：用 system scope 的 cpol 读远端 GPU 写入的数据，确认读到的是新值，不是陈旧缓存。写一个两卡乒乓：A 写 payload 后发 flag，B 读 flag 后读 payload，循环 1e5 次检查。
- **P3 uncached 内存**：`hipExtMallocWithFlags(hipDeviceMallocUncached)` 在 MI455 上是否可用、IPC 能否共享、带宽如何（远端写和本地读）。对比普通 `hipMalloc` 加 SCOPE_SYS cpol。决定堆用哪种分配。
- **P4 原子和 fence 的 lowering**：`ld/st` 在 order acquire/release、scope sys 下，`atomic_add` 在 sys 下，看生成的 ISA 指令是否合理（无多余的全局 cache 刷新等）。
- **P5 拓扑**：`torch.cuda.device_count()`、两两 `hipDeviceCanAccessPeer`、XGMI 链路（`amd-smi topology` 或 `rocm-smi --showtopo`），以及 IPC 跨卡带宽。决定 A11。
- **P6 FlyDSL 工具链**：用 `FLYDSL_DEBUG_DUMP_ASM=1 FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=/tmp/fdump` 编出所有 kernel，检查：ballot、shuffle、readlane、`s_buffer_load`（`is_scalar=True`）是否正确降级，vgpr_count 和 spill，有没有 `readfirstlane` waterfall 循环（`s_cbranch_execnz` 加 `v_readfirstlane` 包着 buffer 指令）。

## 7. 分阶段计划和验收

| 阶段 | 内容 | 机器 | 验收 |
|---|---|---|---|
| S0 | 新建分支；在 MI455 容器里确认 `_C` 能 import（不能也没关系，S2 会去掉依赖）、FlyDSL 版本、GPU 数和拓扑 | MI455 | 环境说明写进本文 §9 |
| S1 | `arch.py` + traits 化（A1–A4、A6 的接口化），gfx950 traits 等于现状 | gfx950 或 MI455 | gfx950：`test_intranode.py` 在 2/4/8 卡全过（含逐位对比 `test_flydsl_matches_cpp`）；§9 的 Primus 式 bench（`num_cu` 20/80）不回退 |
| S2 | 宿主去掉 C++ 依赖（§5） | gfx950 | 同上；mock 掉 `_C.deep_ep` 后 flydsl 路径仍通过 |
| S3 | 跑 probe P1–P6，结论写进本文 | MI455 | 本文更新 |
| S4 | gfx1250 traits，阶段 1 内存序；按 P1 结果处理 A7；按 P5 决定 rank 支持范围 | MI455 | `test_intranode.py`（`test_intranode` 用的是 DeepEP 参考流程，不依赖 C++）在 2/4/8 卡（按拓扑）全过；如果 C++ 后端在 gfx1250 可用，加跑 parity |
| S5 | 性能：扫拷贝分块、`num_sms`、chunk 配置；尝试阶段 2 快路径和 named barrier（A8） | MI455 | §9 两个 bench 的数据；快路径通过压力测试 |
| S6 | 可选：TDM 异步拷贝（`global_load_async_to_lds` / `global_store_async_from_lds` + `s_wait_asynccnt`，参考 C++ `a55d3815`） | MI455 | 相对 S5 有收益才保留 |
| S7 | 端到端：MI455 上的 DSV3 4 层 proxy（或用户指定模型），cpp 对比 flydsl（C++ 可用时）；整理数据，push，PR 描述放数据 | MI455 | PR 描述含 micro-bench 和端到端数据、正确性结果、未覆盖项 |

## 8. 参考资料

- C++ intranode：`csrc/kernels/deep_ep/intranode.cu`、`utils.cuh`（cheap fence 注释、`PRIMUS_TURBO_WAIT_ALL_STR`、`kHasCheapFence`）、`csrc/include/primus_turbo/deep_ep/configs.h`。
- C++ gfx1250 相关分支（只作参考，不合入）：
  - `origin/dev/mi455_v2_deepep_boyang`：`114df410`，C++ DeepEP 在 wave32 gfx1250 上用设备的 wave size（`configs.h` 里 `kWarpSize` / mask 按 `__gfx1250__` 区分；host 侧用 `primus_turbo::warp_size()`）。
  - `origin/feat/gfx1250-mi455-build-deepep`：`a55d3815`，intranode 的 TDM 异步拷贝路径和硬件 named barrier（`__builtin_amdgcn_s_barrier_signal_var` / `s_barrier_wait`）。
- FlyDSL gfx1250 示例：`primus_turbo/flydsl/grouped_gemm/grouped_gemm_bf16_kernel_gfx1250.py`（cache policy 的 TH/SCOPE 解码、`s_wait_*` 的用法）、`primus_turbo/flydsl/attention/gfx1250_mla/`。
- `csrc/include/primus_turbo/arch.h`：`is_gfx1250()`，以及 wave size 的 host 侧判断。

## 9. 测试和 benchmark 方法

**正确性**
- `python -m pytest tests/pytorch/deep_ep/test_intranode.py -q`，用 `HIP_VISIBLE_DEVICES` 控制 2/4/8 卡。它包含三类测试：
  - `test_intranode`：DeepEP 参考流程，两个后端各跑一遍，不需要 C++ 后端可用以外的东西；
  - `test_dispatch_layout`：layout 对照 torch 参考；
  - `test_flydsl_matches_cpp`：与 C++ 逐位对比，需要 C++ 后端可用。
- gfx1250 早期 bring-up（C++ 不可用、`Config` 没有 Python 字段时）可以直接驱动 runtime，用一个 `SimpleNamespace` 代替 `Config`：

  ```python
  # torchrun --nproc_per_node=2 smoke.py
  import os, torch, torch.distributed as dist
  from types import SimpleNamespace
  from primus_turbo.pytorch.deep_ep.flydsl_intranode import FlyDSLIntranodeRuntime

  rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
  torch.cuda.set_device(rank); dist.init_process_group("nccl")
  os.environ["PRIMUS_TURBO_EP_FORCE_CURRENT_STREAM"] = "1"
  rt = FlyDSLIntranodeRuntime(dist.group.WORLD, int(1e9), torch.cuda.current_stream())
  T, H, K, E = 1024, 7168, 8, 256
  x = torch.ones((T, H), dtype=torch.bfloat16, device="cuda") * (rank + 1)
  topk = torch.topk(torch.rand(T, E, device="cuda"), K, dim=-1)[1]
  w = torch.rand(T, K, device="cuda")
  cfg = SimpleNamespace(num_sms=8, num_max_nvl_chunked_send_tokens=8, num_max_nvl_chunked_recv_tokens=256)
  ntr, _, nte, itr, _ = rt.get_dispatch_layout(topk, E, None, False, False)
  recv_x, _, _, rtw, _, rpm, _, rcpm, src, sh, _ = rt.intranode_dispatch(
      x, None, topk, w, ntr, itr, nte, 0, None, None, 1, 0, cfg, None, False, False)
  start = 0
  for i in range(world):  # rows from rank i carry the value i + 1
      end = rpm[i][rank].item(); assert (recv_x[start:end].float() == i + 1).all(); start = end
  out, _, _ = rt.intranode_combine(recv_x, rtw, None, None, src, rpm, rcpm, sh, cfg, None, False, False)
  assert torch.allclose(out.float(), x.float() * itr.sum(1, keepdim=True))
  ```

**性能**
- 单个 op：`PRIMUS_TURBO_DEEPEP_INTRANODE_BACKEND=cpp|flydsl python -m benchmark.ops.training.deep_ep.test_intranode`。要用 `-m`；按 benchmark README 直接跑脚本会因相对导入报错。
- 按 Primus 的方式调用：`turbo.ops.moe_dispatch` 加 `moe_combine`，前向和反向一起，`async_finish=True`、`allocate_on_comm_stream=True`，force-current-stream，每 rank 8192 token。§0 的基线就是这么测的：

  ```python
  # torchrun --nproc_per_node=8 e2e.py；后端用 PRIMUS_TURBO_DEEPEP_INTRANODE_BACKEND 选
  import os, time, torch, torch.distributed as dist
  os.environ.setdefault("PRIMUS_TURBO_EP_FORCE_CURRENT_STREAM", "1")
  import primus_turbo.pytorch as turbo
  from primus_turbo.pytorch.kernels.moe.moe_dispatch_combine_impl import set_buffer_global_config

  rank = int(os.environ["RANK"]); torch.cuda.set_device(rank); dist.init_process_group("nccl")
  g = dist.group.WORLD
  T, H, K, E = 8192, 7168, 8, 256
  set_buffer_global_config(num_use_cu=int(os.environ.get("NUM_CU", 80)))
  x = torch.randn((T, H), dtype=torch.bfloat16, device="cuda", requires_grad=True)
  w, idx = torch.topk(torch.rand((T, E), device="cuda"), K, dim=-1)

  def step():
      recv, _, _, _, handle = turbo.ops.moe_dispatch(x, idx, w, E, g, async_finish=True, allocate_on_comm_stream=True)
      out = turbo.ops.moe_combine(recv * 1.0, g, handle, async_finish=True, allocate_on_comm_stream=True)
      out.float().sum().backward()

  for _ in range(5): step()
  torch.cuda.synchronize(); dist.barrier(); t0 = time.perf_counter()
  for _ in range(30): step()
  torch.cuda.synchronize()
  if rank == 0: print(f"{(time.perf_counter() - t0) / 30 * 1e3:.3f} ms/step")
  ```

  需要拆开 GPU kernel 时间和 host 开销时，用 `torch.profiler` 包一段 `step()`。
- 端到端：Primus 的 DSV3 4 层 proxy，两个后端各跑一遍（gfx950 上的参数如下，MI455 上 `turbo_deepep_num_cu` 等要按该机器调整）：

  ```bash
  PRIMUS_TURBO_DEEPEP_INTRANODE_BACKEND=flydsl bash ./primus-cli direct --numa \
    -- train pretrain --config examples/megatron/configs/MI355X/deepseek_v3-BF16-pretrain.yaml \
    --num_layers 4 --micro_batch_size 2 --global_batch_size 512 --seq_length 4096 \
    --max_position_embeddings 4096 --train_iters 20 \
    --tensor_model_parallel_size 1 --pipeline_model_parallel_size 1 --expert_model_parallel_size 8 \
    --moe_layer_freq 1 --moe_shared_expert_intermediate_size None --moe_shared_expert_overlap False \
    --pipeline_model_parallel_layout null --num_virtual_stages_per_pipeline_rank null \
    --virtual_pipeline_model_parallel_size null --decoder_first_pipeline_num_layers null \
    --decoder_last_pipeline_num_layers null --recompute_granularity null --recompute_num_layers 0 \
    --recompute_layer_ids null --lr_warmup_iters 0 --lr_decay_iters 20 \
    --moe_router_force_load_balancing_type uniform --enable_primus_turbo True \
    --use_turbo_deepep True --use_turbo_mega_moe False --mock_data True
  ```

  在 Primus 仓库根目录执行，取第 3–20 次迭代的 ms/iter 均值，同时对比 lm loss 曲线。

## 10. 风险和坑

- **FlyDSL 磁盘缓存不会因为 helper 函数改动而失效**（kernel 定义在 builder 闭包里）。改了 `primus_turbo/flydsl/deep_ep/*.py` 或 `primus_turbo/flydsl/utils/prims.py` 后，要么清掉 `~/.flydsl/cache/launch_*` 和 `~/.flydsl/cache/_launch_dispatch_layout_*`，要么每次实验设一个新的 `FLYDSL_RUNTIME_CACHE_DIR`。否则跑的是旧二进制，表现为“改了没效果”，在 gfx950 上踩过。
- **FlyDSL：while 循环用 LDS 原子返回值做循环变量会挂起**（在 gfx950 上复现过：用原子计数器领任务的循环）。同样结构改成按固定步长自增就正常。需要动态分配任务时要注意。
- **uniform 值要固定到 SGPR**：从向量 load 来的地址、flag 必须 `readfirstlane`（`prims._readfirstlane_i32` / `intranode_kernel._uniform_i64`），否则每个 buffer 访问都会变成 waterfall 循环。gfx1250 同样适用，§6 P6 要检查。
- **内存序是最大风险**：错误通常表现为偶发数据错或 hang，不是每次都复现。阶段 2 的快路径没有压力测试证明之前不要默认开启。
- **`rank_prefix_matrix` 只有第 `rank` 列有意义**，其余列是堆的残留，C++ 也一样。比对时只比这一列。
- **`num_worst_tokens` 下超过实际接收数的行**是未初始化内存（只有 `recv_topk_idx` 填了 -1），比对时要截断。
- **Mega MoE 共享 `prims.py`**：改它必须保持默认行为不变；可以跑 Mega MoE 的测试回归。
- **gfx1250 上 `_C` 的编译**：仓库 `setup.py` 对 gfx1250 有特殊处理（turbo / CK 后端被跳过）。`deep_ep` 能否编进来要看当时 main 的状态，所以 §5 很重要。
