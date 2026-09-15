# 0915 优化日 — c07-1（单卡 gfx1250，VR 限频 1100 MHz）

形状 `b=4 s=8192 hq=32 hkv=8 d=128 bf16 causal`（`llama31-8b`），
harness / 计时 / SQNR 门沿用 `tools/gfx1250/tune_attention.py` 默认
（iters 20 / warmup 5 / sqnr-min 50，CUDA event 取中位数，每 rep 冲刷 256 MiB L2）。

## 阶梯

| 版本 | n | fwd | bwd | **total ms** | 离散 | 备注 |
|---|--:|--:|--:|--:|--:|---|
| turbo 出厂 @`1cb2e183` | 3 | 10.651 | 45.134 | **55.785** | 0.93% | 今晨实测 |
| 强制 Triton 前向 + 融合反向 | 2 | 4.166 | 17.835 | **22.001** | 0.06% | Rank 0 对照 |
| ASM 前向 + 融合反向（晨间冠军） | 3 | 1.549 | 17.675 | **19.233** | 0.22% | |
| **ASM 前向 + ASM 反向** | 3 | 1.559 | **10.098** | **11.657** | **0.07%** | 今日冠军 |

**出厂 → 今日冠军 = 4.786×**（55.785 → 11.657），全部为本机同口径实测，不经跨机换算。
晨间冠军 → 今日冠军：总时间 **1.651×**，反向 **1.750×**。

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

**但存在并行度下界。** 反向加速比随规模递减，到足够小就翻负：

| 形状 | b·hq | 融合反向 | ASM 反向 | 比值 |
|---|--:|--:|--:|--:|
| b4 s8192 hq32 | 128 | 17.675 | 10.098 | 1.750× |
| b2 s8192 hq32 | 64 | 9.007 | 6.141 | 1.467× |
| b4 s4096 hq32 | 128 | 5.538 | 4.405 | 1.257× |
| b1 s4096 hq32 | 32 | 3.201 | 2.497 | 1.282× |
| b4 s4096 hq8 | 32 | 3.232 | 2.518 | 1.284× |
| **b1 s1024 hq8** | **8** | 0.743 | 2.218 | **0.335×（慢 3 倍）** |

固定开销（fp32 `dq_acc` 的分配与清零、host 规约、三次发射）在小规模下压倒收益。
所有获胜形状的 `b·hq ≥ 32`，唯一失败的是 8 —— 与产品代码里 `_MIN_PARALLEL_WORK` 的阈值重合，
但资格门里那个数是**独立测出来的**，不是照抄。

**已接进分发层**：`--impl fused`（不带 `--impl asmbwd`）实测 **11.7156 ms**，说明资格门在产品路径上正常触发。

**尚未做**：非因果路径、varlen、torchtitan e2e。

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
