# 阶段 2 · S0 探针结果（2026-09-23）

三件事同时被推翻：profiling 的禁令、hipBLASLt 的诊断、round 10 的 icache 假说。

---

## S0-a　轮数

`op-evolve tune --max-rounds 40 --fast-rounds 40`，两者同时抬。
`spec_version` 仍为 `v000`（未 bump，`op/` 未重建，refcache 与手工补丁完好）。
round 13..24 全部为 `fast`，不会触发 DEEP 轮的 rocprofv3/rocprof-compute 模块。

## S0-b　hipBLASLt：原诊断是错的

v000 记录的结论是「这个镜像缺 hipBLASLt 的 gfx1250 Tensile 库」。**不成立。**
容器里有**三个** hipBLASLt 库目录，只有一个是完整的：

| 目录 | 内容 | 结果 |
|---|---|---|
| `/opt/rocm/lib/hipblaslt/library` | 空 | **诱饵**——这个容器的 ROCm 根本不在 `/opt/rocm` |
| `_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library` | 缺 gfx1250 载荷 | hipBLASLt 的**默认搜索路径**。这才是 `HIPBLAS_STATUS_INVALID_VALUE` 的真因 |
| `_rocm_sdk_devel/lib/hipblaslt/library/gfx1250` | 326 文件，但**残缺**（无 `Kernels.so-000`、无 extop/transform，`.dat` 未压缩） | 指向它 → **SIGSEGV，exit 139** |

完整的一份是宿主机 ROCm 10.1.0 的 328 文件库。拷到 `$HOME`（唯一 bind mount）下：

```
HIPBLASLT_TENSILE_LIBPATH=/home/lihuzhan/.local/hipblaslt-gfx1250/gfx1250
TORCH_BLAS_PREFER_HIPBLASLT=1
```

**实测干净**：bf16 与 fp32 GEMM 于 512 / 2048 / 4096，以及精度门当年 fault 的
`fp32 QK^T [2,16,4096,128]`。零 `rocblaslt error`，零 dmesg 故障签名。
**没有**跑旧 `'0'` 配置做反向对照——那等于故意重现一次挂卡。

### 为什么十二轮没人发现

`/usr/lib/python3.12/sitecustomize.py:18` 在**每个解释器启动时**
`setdefault("TORCH_BLAS_PREFER_HIPBLASLT","0")`，早于任何项目代码。
`op/*/_env.py` 用的也是 `setdefault`，因此**一直是空操作**。
`docker exec` 走 `bash -c` 不读 profile，所以 `/etc/profile.d/zz-gfx1250.sh` 里的同一行不参与；
真正钉死它的是 sitecustomize。现已改为显式赋值，留 `OPEVOLVE_KEEP_BLAS_ENV=1` 作逃生口。

## S0-c　profiling 不会挂卡

六次 rocprofv3 运行，**全部 rc=0，KFD 干净，零 aperture / page fault / INVALIDATE_TLBS**。
禁令来源的那次 rc=134 里，故障回调点名的是 eager 的 Tensile GEMM，不是 profiler。**禁令解除。**

但工具本身有一个真实缺陷：

| 模式 | 结果 |
|---|---|
| `--kernel-trace` | **零 dispatch**。`Number of services generating output: 0`，`rocpd_kernel_dispatch` 表 0 行（kernel *符号* 倒是登记了 458 个） |
| `--runtime-trace` / `--hip-trace` | 有 .db 产出，但同样 0 条 dispatch |
| **`--pmc`** | **可用**。prod 形状两组计数器各 93 行 |

→ **每轮 profiling 采用 `--pmc`，约 2 分钟/组。**
控制量已校验：fast 形状 `k_dkdv` 读到 64 wave = (1024/32 KV 块) × (hkv=2) × 1 wave/WG，与静态网格完全吻合。

---

## prod 形状计数器（round 12 出货，b=4 sq=skv=8192 hq=32 hkv=8 d=128）

| kernel | grid | wg | VGPR | LDS | SQ_WAVES | SQ_BUSY/SQ_CYCLES | ICACHE miss |
|---|--:|--:|--:|--:|--:|--:|--:|
| `k_dkdv_0` | 262144 | 32 | **376** | **70656** | 8192 | 0.9977 | **0.000%** |
| `k_dq_0` | 524288 | 32 | **480** | 8704 | 16384 | 0.9976 | **0.000%** |
| `at::native::…` | 524288 | 256 | 32 | 0 | 16725 | 1.0000 | 0.250% |
| `vectorized_elementwise` | 16777216 | 256 | 16 | 0 | 264485 | 0.9981 | 0.038% |
| `k_delta_bshd_0` | 8388608 | 256 | 24 | 0 | 262144 | 0.9970 | 0.298% |

### 读出来的四件事

**① round 10 的 icache 假说死了。**
`k_dkdv` 8654 万次取指请求中只有 **171 次 miss**（0.0002%）。
「指令数与时间反相关是 icache 压力造成的」是唯一没检验过的解释，现在检验了，**不成立**。

**② 缺口是 stall，不是 wave 不够。**
两个热 kernel 的 `SQ_BUSY_CYCLES/SQ_CYCLES` 都是 **0.998**——SQ 上几乎永远有 wave 驻留。
这正是 η 在结构上分不开、而探针要分开的那件事：wave 在那儿，它们在**等**。

**③ 两个 kernel 的占用率被不同的东西卡住——这是新的、从未定价过的杠杆。**

- `k_dkdv`：LDS **70656 B/WG**。320 KB ÷ 70656 = **4 WG/CU**。
  而 VGPR 376 允许 ⌊1024/376⌋ = 2 wave/SIMD = **8 wave/CU**。
  → **LDS 卡住了一半**。降到 ≤65536 B 得 5 WG/CU（+25%），降到 ≤40960 B 得 8（+100%）。
- `k_dq`：LDS 仅 8704 B（允许 37 WG/CU），**VGPR 480** 才是限制，⌊1024/480⌋ = 2 wave/SIMD。
  → 降到 ≤341 得 3 wave/SIMD（+50%）。

**④ 5.8% 的时间花在 torch 的 elementwise kernel 上。**
按 `GRBM_GUI_ACTIVE` 分解每步时间：`k_dkdv` 62.9%、`k_dq` 30.7%、
torch `at::native` 两个 kernel 合计 **5.8%**、`k_delta` 0.6%。
那 5.8% 不做任何注意力数学。

---

## 仍然未钉死

**真峰值**（S0-d）。现有一切「距屋顶多远」的说法都建立在一条 Triton bf16 GEMM 的成绩
（1002.7 TF/s）上，而 aiter 的 ASM 前向已经跑到 1401.5，比它高 1.398×。
在真峰值被实测钉死之前，**不引用任何绝对天花板**。
