# A0 / A1 / A8：零 GPU 编译链路通了，并且当场坐实了 descriptor 的问题

2026-09-21 · 分支 `dev/lhz/flydsl-attn` · **全程零 GPU 发射，零 KFD 占用，没有碰卡**

---

## 结论先说

| | 结果 |
|---|---|
| A0 零 GPU 编译门 | **通过** |
| A1 对生产内核编译 | **通过**，三个内核全部出 ISA |
| A8 descriptor 现状 | **坐实**：18 个 buffer descriptor 全部是 CDNA 形态 |
| 挂卡账目 | 0 次发射、0 次故障、0 次 AC-cycle |

计划里写的「A0 是唯一允许的人工确认点，三条绑定路全不通才算门失败」——
**第一条路（torch CPU 张量）就通了，meta 张量也通**，所以这个确认点不需要你介入。

---

## 1. 零 GPU 编译怎么做到的

FlyDSL 自带一个「只编译不执行」的模式。**这两个变量没有 `FLYDSL_` 前缀**——
`flydsl/utils/env.py` 给它们显式写了 `env_var`，而元类只给留空的那些加前缀：

| 变量 | 值 | 作用 |
|---|---|---|
| `COMPILE_ONLY` | `1` | 只编译不执行。官方描述原文就是 "useful for verifying compilation without a GPU" |
| `ARCH` | `gfx1250` | 覆盖编译后端的目标 arch |
| `FLYDSL_GPU_ARCH` | `gfx1250` | 覆盖 `runtime/device.py` 的 arch 解析（buffer descriptor 走这条） |
| `FLYDSL_DUMP_IR` / `FLYDSL_DUMP_DIR` | `1` / 目录 | dump 出 21 个阶段，末尾是 `21_final_isa.s` |
| `FLYDSL_RUNTIME_ENABLE_CACHE` | `0` | 保证真的重编 |

**两个 arch 变量必须同时设。** `ARCH` 管编译后端，`FLYDSL_GPU_ARCH` 管 descriptor 选择。
只设一个会得到一份目标混合的 ISA，**而且会安静地编译成功**。

`ARCH` 是个极其常见的变量名，外层任何脚本为自己的目的设了它都会静默改掉编译目标。
所以驱动器的第一件事是**断言解析出的 target 等于 gfx1250**，并把 arch 写进每一条记录，
就像每个时间数字都要带 sclk 见证一样。

容器**不挂 `/dev/kfd`、不挂 `/dev/dri`**，物理上碰不到卡，也不占用 chcai 那边的资源。
容器内 `torch.cuda.is_available()` 实测为 `False`。

### 路上踩到的三个坑，都已解决

1. **`stream=None` 不是合法的 jit 参数**（`NoneType is neither a JitArgument nor has a
   registered constructor`）。内核自己的 `launch_*` 包装器签名里有 `stream: fx.Stream`，
   而无卡环境造不出 `torch.cuda.Stream`。
   → 驱动器自己声明**不带 stream 的 `@flyc.jit` 包装器**，包住同样的 `@flyc.kernel` 构建器。
   被筛的是内核体，不是发射包装器。
2. **`import aiter` 在无卡容器里走不通**。`chip_info.get_gfx_runtime()` 直接调 `_detect_native()`，
   **绕过了 `GPU_ARCHS` 覆盖**，去跑 `rocminfo`；就算用 shim 顶过去，它接着走到一条无关的
   Triton/gluon 路径去 import `jax`，而镜像里没有 jax。
   → `kernels.py` 其实只要两个符号：`create_llvm_ptr`（8 行 flydsl 指针转换）和
   `_to_raw`（3 行 ir.Value 强制）。驱动器**按路径加载这两个 vendor 文件**，
   预置空的父包，跳过 `aiter/__init__.py`。**用的是 vendor 自己的源码，不是重新实现。**
3. `flydsl.utils.env` 的 `OptBool` 没有 `.get()`，探针里那一处写错了，已改成读 `env_var` 名。

---

## 2. 三个生产内核的静态 ISA（prod 形状 b=4 sq=skv=8192 hq=32 hkv=8 d=128）

| kernel | vgpr | sgpr | LDS | **spill** | instr | wmma | tr16 | exp | vgpr_msb | dscnt | 判定 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|
| `k_delta_bshd` | 40 | 19 | 0 | **0** | 256 | 0 | 0 | 0 | 0 | 0 | pass |
| `k_dkdv` | 255 | 45 | 9216 | **0** | 898 | **24** | 18 | 8 | 0 | 17 | pass |
| `k_dq` | 269 | 40 | 8192 | **0** | 830 | **24** | 16 | 16 | 32 | 8 | pass |

全部 `.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx1250"`，全部 `.amdhsa_wavefront_size32 1`。

### 这张表的第一个用途是校准尺子，不是看数

`k_dkdv` 的 WMMA 计数是 **24**，与 round 1 文档里记的「循环体 24 个 WMMA 里有 16 个在乘零」
**精确对上**。这是计划里写的那条验收：对不上就说明我们在看一份不是这份内核的 ISA，要立刻停。
**对上了，所以后面的静态定价可以信这条链路。**

三个内核 `spill` 全为 0，所以硬门（`spill > 0` 直接淘汰）现在一个都不杀。
`vgpr` 255 / 269 离 1024 的天花板还远。

> **静态筛只负责判死，不负责判活。** spill=0 但 LDS bank 冲突更重、global 合并更差的候选
> 会被放行。**筛选通过不构成任何性能预测。**

---

## 3. A8：descriptor 的现状被坐实了，而且比预想的更弱

从 `10_convert_scf_to_cf...convert_gpu_to_rocdl.mlir` 里直接读出来，**一次发射都没有**：

| kernel | `make.buffer.rsrc` 个数 | flags 常量 |
|---|--:|---|
| `k_dkdv` | 8 | **159744 = 0x00027000** |
| `k_dq` | 7 | **159744 = 0x00027000** |
| `k_delta_bshd` | 3 | **159744 = 0x00027000** |

`(7<<12)|(4<<15) = 0x00027000` 正是 **CDNA 形态**：bit 24 未置、`OOB_SELECT = 0`。
RDNA 形态应当是 `0x21027000`。**18 个 descriptor 一个例外都没有。**

> 顺带纠正：vendored `buffer_ops.py:57` 的注释把 RDNA 形态写成 `0x21020070`，是数字转置的笔误。
> **代码是对的，注释是错的**，两边（flydsl 原包和 turbo 的 vendored 拷贝）都带着这个笔误。

### 一个让「靠 descriptor 兜底」更站不住的观察

`k_dkdv` 和 `k_dq` 传进去的 `num_records` 是常量 **1073741824 = 1 GiB**。
也就是说，即便在当前的 CDNA 形态下，硬件边界检查的上界也是一个 1 GiB 的天花板，
**本来就不是按张量实际 extent 在保护**。

**所以计划 §六 的处置不变，而且理由更强了**：
causal tile 跳过 / 尾块 / varlen 的边界安全性，**一律靠显式谓词与钳位实现，
不依赖任何 descriptor 语义**。在 `OOB_SELECT` 在 gfx12 上该取 2 还是 3 弄清楚之前，**不改 descriptor**。

---

## 4. 复现

```bash
R=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo
B=/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-flydsl-attn-bwd-20260917-115934/job_context/op/baseline

docker run --rm --network none -v /home/lihuzhan:/home/lihuzhan \
  -e COMPILE_ONLY=1 -e ARCH=gfx1250 -e FLYDSL_GPU_ARCH=gfx1250 \
  -e FLYDSL_ROCM_AGENT_TIMEOUT=10 -e TORCH_BLAS_PREFER_HIPBLASLT=0 \
  --entrypoint python3 fa-tune:deps \
  $R/output/0921__flydsl/bin/compile_only_driver.py \
  --impl $B --dump-dir $R/output/0921__flydsl/isa/baseline \
  --json $R/output/0921__flydsl/isa/baseline.json
```

注意命令里**没有** `--device /dev/kfd`、**没有** `--device /dev/dri`。这是有意的，不要加回去。

产物：`output/0921__flydsl/isa/baseline.json`（含 arch 与 flydsl 版本见证）、
`output/0921__flydsl/isa/baseline/<kernel>/21_final_isa.s`。

`bin/rocminfo_shim.sh` 是给 aiter 硬件探测准备的备用 shim，**这次最终没有用到**
（按路径加载两个 vendor 文件就绕开了整条 import 链）。留着是因为将来要 import 更多 aiter
时可能需要它。**绝不要把它挂进一个有 `/dev/kfd` 的容器**——在真卡上给出一个假的 arch 答案，
正是审计对着错误目标却报成功的那条路。
