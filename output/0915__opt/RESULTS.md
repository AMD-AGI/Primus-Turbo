# 0915 优化日 — c07-1（单卡 gfx1250，VR 限频 1100 MHz）

形状 `b=4 s=8192 hq=32 hkv=8 d=128 bf16 causal`（`llama31-8b`），
harness / 计时 / SQNR 门沿用 `tools/gfx1250/tune_attention.py` 默认
（iters 20 / warmup 5 / sqnr-min 50，CUDA event 取中位数，每 rep 冲刷 256 MiB L2）。

## 阶梯

| 版本 | n | fwd | bwd | **total ms** | 离散 | 备注 |
|---|--:|--:|--:|--:|--:|---|
| turbo 出厂 @`1cb2e183` | 3 | 10.651 | 45.134 | **55.785** | 0.93% | 事故前 |
| 强制 Triton 前向 + 融合反向 | 2 | 4.166 | 17.835 | **22.001** | 0.06% | 事故前，Rank 0 对照 |
| 强制 Triton 前向 + ASM 反向 | 3 | 4.249 | 10.418 | **14.666** | 10.84% | 隔离出 ASM 前向值 2.94 ms |
| ASM 前向 + 融合反向（晨间冠军） | 5 | 1.561 | 17.686 | **19.239** | 0.53% | 与事故前 19.233 一致 |
| **ASM 前向 + ASM 反向（产品路径）** | 5 | 1.572 | **10.160** | **11.726** | **0.57%** | **今日冠军** |

**出厂 → 今日冠军 = 4.757×**（55.785 → 11.726），全部为本机同口径实测，不经跨机换算。
晨间冠军 → 今日冠军：总时间 **1.641×**、反向 **1.741×**。

上表全部是 **AC-cycle 恢复之后**用 n=5 重测的。晨间冠军那行测出 19.239，与事故前的 19.233
几乎逐位一致，证明机器回到了同一性能状态（rocBLAS GEMM 也复现：27.63 vs 27.64）。

`强制 Triton 前向 + ASM 反向` 那行离散 10.84%，远高于其余各行的 0.5–0.9%，原因未查，
引用时注意。它把 ASM 前向的价值隔离出来：14.666 − 11.726 = **2.94 ms**。

## 今天做成的三件事

### 1. ASM 前向的开关（Rank 0，前置条件）

分发层自己会选 ASM 前向，导致 `--impl turbo` / `fused` / `asm` **三者走同一条路**
（19.222 / 19.298 / 19.285 ms，差 0.4% 是噪声底）。当前 HEAD 上**测不到纯 Triton 前向**，
任何关于 ASM 前向价值的归因都做不了 —— 所以这从"待办第 5 项"变成了 T2 的前置条件。

`PRIMUS_TURBO_ATTN_DISABLE_ASM_FWD=1` 让 gate 走既有的 `_no()` 路径拒绝，
harness 暴露为 `--asm-fwd {auto,off}`。行里新增 `asm_fwd` 字段供审计
（`impl_note` 不能用：它是在任何东西跑起来之前按 `--impl` 赋的硬编码字符串）。

验收的硬证据不是计时而是**SQNR 指纹**：关掉后 out 从 53.6197 变成 53.6697，
而 `HANDOFF.md` 记录的正是"树内 Triton 前向给 53.67、ASM 给 53.62"。
这独立证明了开关换掉的是内核，不只是数字。

ASM 前向本身值 **2.74×**（4.253 → 1.549 ms）。注意它**不是** 1.93 的时钟比乘 B0 的 2.7
（那会预测 5.2 ms）—— Triton 前向在本机的缩放是 1.57×，而其余各行都是 1.88–1.98×。

### 2. T2：aiter 预编译 ASM 反向跑通（主线）

`asm_bwd_launcher.py` 此前**一行都没在 GPU 上跑过**（写它的时候卡是挂的）。
分阶段做，因为 `T2-ASM-BACKWARD-SPEC.md` 列的四个风险并不等价。

**`odo` 是头号嫌疑** —— 三个 kernarg 布局里唯一没被反汇编确证的（用 kernarg 预加载，
`s_load` 为 0，所以问不出它读哪些偏移）。但 `delta = rowsum(dO*O)` 在 torch 里是三行，
所以 `--stage odo` 能在**根本不发射主内核**的情况下判它。
结果第一次就对：2048/2048 元素全写（缓冲区预填 NaN，所以"没写"和"写了零"分得开），
**SQNR 147.41 dB**，0.21 秒。

**真正的 bug 是 GQA，而且不在 spec 的清单上。**
ratio=4 下 dq 正确（52.24 dB）而 dk/dv 塌到约 −0.3 dB。dq 对这件事本身就是定位线索：
dq 依赖按组读 k/v，说明内核的分组已经和参考一致，错的只在 dk/dv 的**写出**路径。

主内核 grid 是 `(kv_tiles, nhead_q, batch)`，GQA 下 `ratio` 个 workgroup 拥有同一块
dk/dv tile 且无同步 —— −0.3 dB 正是 4 路竞争的样子。三个实验钉死它：

| | dq | dk | dv | |
|---|--:|--:|--:|---|
| ratio=1，无 GQA | 52.26 | 52.27 | 52.76 | 全对 |
| ratio=4，dk/dv 按 kv head | 52.24 | −0.94 | −0.65 | |
| ratio=4，dk/dv 按 q head + host 规约 | 52.24 | 50.55 | 51.09 | 全对 |

预清零 dk/dv 无改善，排除了"累加 vs 赋值"。
`asm_backward` 因此接受 `dkdv_heads={"kv","q"}`。

独立对拍（生产形状，同进程同卡，两个独立实现）：
ASM 反向 **8.661 ms** vs 冠军 **17.407 ms** = **2.010×**，
dq 59.50 / dk 51.28 / dv 51.42 dB。

### 3. 接进 harness：`--impl asmbwd`

见上表。独立 bring-up 测反向 8.68 ms，harness 里是 10.098 ms —— 差的 1.4 ms 是 autograd
管路（ctx 存取、`do` 的 `contiguous()`），该算进去，所以对外引用 harness 那个数。

## 边界与未决

**`_perf` 变体不是 drop-in。** 那两个 `.co` 没有任何 campaign 文档提到过。
在 gqa2k 上它返回的 dk/dv 与 shipped 版**逐位相同**
（50.577884674072266 / 50.86431884765625，每一位），但 dq 是 **5.84 dB** 而非 52.21。
dk/dv 逐位相同说明主计算一致、只有 dq 的原子累加不同；5.84 dB 是"部分正确"而非垃圾，
符合贡献缺失 —— 大概率是 dq pass 需要不同的 grid 或 split。它的计时行被 `needs` 门挡着。

**"seqlen 256 会 fault" —— 这条我先前判错了，已分诊，它和 GQA 那个 bug 是同一个。**
二分扫描下 s128 / s256 / s384 / s512 / s768 在 `--dkdv-heads q` 下**全部通过**
（dq 52.0–52.2、dk 50.3–50.6 dB）。而最初那次 fault 用的是默认 `--dkdv-heads kv`。

所以只有一个 bug：内核按 q head 索引写进 kv 尺寸的缓冲区，**越界写**。
它表现成两种样子，取决于越界落在哪里 —— s=256 时跨了页边界所以 fault，
s=1024 时落进别的分配里所以只是静默损坏（dk/dv 约 −0.3 dB 而 dq 仍然正确）。
**不存在"小形状边界问题"，资格门也不需要 seqlen 下界。**
（注意 fault 杀的是进程不是卡：dmesg 零记录，`gpu_health.sh` 全程 HEALTHY。）

**存在下界，但判据是 seqlen 不是并行度 —— 我在这里错了两次，记下来。**

第一次：看到 `b1 s1024 hq8` 输 3 倍，就推出 `b*hq >= 32` 的门槛，还在注释里写"这个数是独立测出来的"，
其实只有一个样本。第二次：一组数据显示 b·hq=8 全线赢，我差点把门槛整个拿掉 —— 而那组数据是**被污染的**
（见下）。

干净对拍（参照臂直接点名调 `dense_fused_backward`，比值 = 融合/ASM，>1 表示 ASM 赢）：

| seqlen | b·hq=8 | =12 | =16 | =24 | =32 | =128 |
|--:|--:|--:|--:|--:|--:|--:|
| 1024 | **0.660×** | | | | **0.632×** | **0.537×** |
| 1536 | **0.869×** | | | | | |
| 2048 | 1.748× | | | | 1.549× | |
| 4096 | 2.812× | 2.025× | 2.571× | 2.235× | 2.110× | |
| 8192 | **3.954×** | | | | | **2.002×** |

**seqlen 1024 在任何并行度下都输，包括生产级的 b·hq=128（0.537×）；
seqlen 8192 在 b·hq=8 下赢 3.95×。** 所以旧门槛两个方向都错：会拒绝后者、放行前者。

机制能从原始时间直接看出来，不用推断：ASM 反向耗时 0.88 / 0.91 / 1.00 / 1.34 ms
（seqlen 1024→8192，工作量差 16 倍）**几乎是平的**，说明有约 0.85 ms 的固定地板；
而融合反向在 seqlen=1024 上是 0.5806 / 0.5893 / 0.5785 ms（b·hq 8/32/128）**也是平的**，
说明那个规模下它延迟受限、加并行度没用。两条平线相交 —— 这是固定开销问题，不是并行度问题。

**门槛定为 `seqlen >= 2048`**，由 9 个点支撑。非因果已验证并放行
（mask=0 对象、grid 不折半，dq 52.48–52.52 / dk 50.77–50.82 / dv 50.68–50.91 dB）。

### 我自己造成的一次测量污染

把 ASM 反向接进分发层之后，`--stage time` 的参照臂（走 `flash_attn_func`）**自己也变成了 ASM 反向** ——
在门触发的地方，测量变成了拿新实现和新实现比。

它的签名是**比值向 1.0 收敛而不是报错**：`b·hq=32` 那行当时报 1.362×，而门槛下所有行报 1.8–3.1×。
看起来像"结果偏弱"，不像"测量坏了"，这正是它危险的地方。

修法是参照臂点名调 `dense_fused_backward` —— 任何 gate 都捕获不到它。

这与本次 campaign 记录在案的另外两个同族：镜像 editable 安装遮蔽 checkout、`--impl fused` 被分发层接管。
文档自己的统计是：七次"改动应该有效果却没有"的复盘，**七次根因在测量链路，零次在改动本身**。

**已接进分发层**：`--impl fused`（不带 `--impl asmbwd`）实测 **11.7156 ms**，说明资格门在产品路径上正常触发。

**尚未做**：非因果路径、varlen。

## e2e：跑通了，但这个配置测不了 attention

`converters: []` 此前关掉了 primus_turbo 的 model converter，而 converter 才是真正装上
TurboAttention 的东西 —— 所以**今天之前所有 e2e 数字测的都是原版 attention**。
关它的理由是 torchtitan 0.2.2 传 `enable_gqa` 而 `TurboAttention.forward` 不接受 → TypeError。
该参数已接受（刻意不转发：kernel 从 nhead_q/nhead_k 原生分组，形状已带这个信息）。

打开 converter 后三步跑通，`loss 12.05` 紧贴 `ln(128256) ≈ 11.76` 的理论初值。

### 两臂对照（唯一差别是 `PRIMUS_TURBO_ATTN_DISABLE_ASM_BWD`）

| step | A loss | B loss | A grad_norm | B grad_norm | A tps | B tps |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 12.052 | 12.138 | 42.98 | 42.65 | 240 | 240 |
| 2 | 27.942 | 28.421 | 72.10 | 78.53 | 243 | 243 |
| 3 | 24.361 | 16.812 | 206.39 | 117.23 | 244 | 244 |

**loss 上涨与 ASM 反向无关** —— 两臂同样 12 → 28 再回落，grad_norm 都在爆，是 mock 随机数据
加 lr warmup 的行为。但**这个配置也证明不了梯度正确**：第 3 步两臂差 45%，随机数据下轨迹本就混沌。
它只能排除"明显破坏训练"。

**tps 三个数字逐一相同，显存也相同**（313.72 → 378.00 GiB，说明 ASM 反向那套 4 倍 dk/dv
在 e2e 峰值里不占位，峰值由激活主导）。

### 为什么测不出来：e2e 是 GEMM-bound，而且 bound 在一条坏掉的路径上

单步 134 秒，而 attention 全部 32 层是 `11.7 ms × 32 ≈ 374 ms` —— **占 0.28%**。
`output/0914__campaign/bin/gemm_roof.py` 同进程测两个半边：

| | TFLOP/s | 8192³ 耗时 |
|---|--:|--:|
| `torch.mm`（rocBLAS） | **27.64** | 39.78 ms |
| Triton GEMM（最佳配置 BM128/BN256/BK32/w8/s3） | **897.53** | 1.225 ms |
| **gap** | **32.5×** | |

算术吻合：llama-3.1-8B 每步约 `6 × 8e9 × 32768 = 1.57e15` FLOPs，按 27.6 TF/s 要 57 秒。

**为什么走 rocBLAS**：hipBLASLt 在这个镜像里整个不可用 —— 缺
`_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/TensileLibrary_lazy_gfx1250.dat`，
任何 matmul 都抛 `HIPBLAS_STATUS_INVALID_VALUE`，只能用
`TORCH_BLAS_PREFER_HIPBLASLT=0` 退回 rocBLAS。

**这是 e2e 的真正阻塞点，不是 attention。** 在 BLAS 路径修好之前，
attention 上的任何改进在 e2e 里都不可见。文档给的方向是把 GEMM 钉到 Triton
（`TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1` + `..._BACKENDS=TRITON`），但那需要
`compile.enable=true`，而配置自己的注释记载 inductor 对 TransformerBlock 做 autotune 时
会抛 `hipErrorLaunchFailure` **并打死 GPU**。这两者的矛盾是明天要解的。

### 一个反复出现的坑，值得单列

`TORCH_BLAS_PREFER_HIPBLASLT=0` **今天咬了我三次**：T2 bring-up 的 fp32 参考、
第一次 e2e、以及 `gemm_roof.py`。根因是 `tune_attention.py` 在 import torch 之前自己设了它，
所以**整个 campaign 的 op 级测量从来没见过这个问题**，而任何不走那个 harness 的新路径都会撞上。
它必须在进程启动前进入环境（torch 在选后端时读一次），从脚本内部设来不及。

## 与文档界线的关系

`PLAN-4GPU-TOMORROW.md:203` 给的 1100 MHz 界线是：7-GEMM MFMA 下界 **7.68 ms**、
"真实墙" **9.2–9.9 ms**。今天的反向 10.098 ms（harness 口径）落在真实墙略上方，
独立 bring-up 的 8.661 ms 落在墙**下面**。这不矛盾 —— 那些界线描述的是**我们这个 7-GEMM 结构**，
而 aiter 的手写 ASM 是另一个结构：`v_wmma` 864 vs 448、LDS 320 KB vs 64 KB、
零 spill vs 113 store / 245 load。它达到的正是"对我们的结构不可达"的区间。

**不要**据此论证已接近极限 —— 文档里那句"调度良好的 Triton 落在墙的 1.3–1.5×"
已被我们自己的两个内核证伪（实测 2.0× 和 2.2–2.4×）。

## 复现

```bash
# 容器（镜像自带 editable 安装必须先卸掉，否则测的是镜像不是 checkout）
docker exec fa-repro bash -lc 'pip uninstall -y primus_turbo; pip install psutil'

# 今日冠军
timeout 210 docker exec -e GPU=0 -e PYTHONPATH=/home/lihuzhan/code/aiter-src \
  -e TRITON_CACHE_DIR=/tmp/triton_cache_g0 fa-repro bash -lc \
  'cd "$0" && exec timeout -k 15 -s TERM 180 python3 tools/gfx1250/tune_attention.py "$@"' \
  /home/lihuzhan/code/2026_0903__turbo/Primus-Turbo --shape llama31-8b --impl asmbwd

# T2 分阶段 bring-up（odo 单内核 -> 全链 -> 对拍计时）
python3 tools/gfx1250/t2_bringup.py --shape tiny  --stage odo
python3 tools/gfx1250/t2_bringup.py --shape gqa2k --stage full --dkdv-heads q
python3 tools/gfx1250/t2_bringup.py --shape llama31-8b --stage time
```

无人值守队列：`output/0915__opt/`，`cat STATUS.md` 看当前状态，
`>> queue.jsonl` 可在运行中追加实验，`touch STOP` 暂停，
`kill $(cat sched.pid)` 停止（**不要** `pkill -f`，那个 pattern 会匹配到杀手自己）。
