# 待测实验 — 代码写好了，但一次都没测过

**这些补丁不在主干上。** 它们是为了让 GPU 恢复后直接进入测量，而不是先写代码。

**为什么不直接提交**：今天的纪律是「测量之后才相信」。一个没有测量的性能改动不该落地——
今天已经有两次，我差点基于「只有一种测法能产生的数字」去动代码（前向 0.9 ms 的假差距），
以及基于「静默失效的 sweep」去下结论。合理的做法是把代码准备好，然后**测**。

每个实验都必须过：四张量 SQNR 门 + 生产形状计时 + 与冠军的 A/B。
**接受规则：总时间改善 > 2%（超过实测的 0.87% 重复离散度）且 SQNR 不变。否则丢弃。**

用法：

```bash
cd /home/lihuzhan/code/2026_0903__turbo/Primus-Turbo
patch -p0 < output/0913__opt_plan__claude/phase2/prepared-experiments/E1-hoist-Di.patch
tools/gfx1250/exclusive.sh python3 tools/gfx1250/tune_attention.py \
    --shape llama31-8b --impl fused --tune "fwd:num_stages=2"
# 对照冠军：24.435 ms / 315.0 TFLOP/s，SQNR 53.67/52.24/52.31/52.71
```

---

## E1 — 把 Di 的加载提前（已写成补丁）

**改动**：`Di = tl.load(D + offs_m * stride_deltam, ...)` 从 dv 的点积之后，移到和 `m`
的加载放在一起。

**依据**：上游自己在 `m` 那里写了注释 *"Load m before computing qk to reduce pipeline
stall"*，然后**没有对 Di 应用同一条道理**。两者是同一个基址、同样的 `[BLOCK_M]` 形状、
几行之后就要用。留在循环中间等于在两个点积之间插了一个依赖标量加载。

**风险**：极低。同一个值，只是取得更早，不可能改变数值。
**预期**：小（1-3%）。列在这里是因为它几乎免费。

---

## E2 — 消掉四次 `tl.trans`（未写补丁，需要更大改动）

**位置**：`dpT = tl.dot(v, tl.trans(do))`（do 刚在上面非转置地加载过，为了 dv 的点积）、
`dk = tl.dot(dsT, tl.trans(qT))`（qT 是转置加载的，这里又转回去）、
`dq = tl.dot(ds, tl.trans(kT))`（K 同样的来回）。

每次迭代三到四个转置，每个是一次 LDS 往返或整套 VGPR permute 网络，**在关键路径上**，
每个 (b,hkv) 约 4224 次迭代。

**改法**：维护第二套指针（`do_ptrs` 和 `doT_ptrs`、`qT_ptrs` 和 `q_ptrs`），两种布局都从
内存来。多出的 L2 流量可忽略（一个 `[32,128]` bf16 的 q tile 是 8 KiB，在 4 深的 hqid
循环里常驻 L2）。

**先做一个 15 分钟的检查再动手**：gfx1250 **有** `ds_load_tr16_b128`（gfx950 只有 b64）。
dump 一次迭代的 ISA，看 Triton 的 AMD 后端是发这条指令还是退回 permute。
**这一个检查决定 E2 值 2% 还是 10%。**

---

## E3 — 折掉 exp 之前的逐元素 VALU（**补丁已就绪**：`E3-fold-log2-scale-dq.patch`）

**已写的部分**（dq 那一趟，低风险）：上游算 `qk * sm_scale`，然后在 `exp2` 里又把
**整个 tile** 乘一次 `RCP_LN2`。两者都是 `tl.constexpr`，乘积在编译期折成一个常量，
所以 tile 只需要缩放**一次**。`m` 是 `[BLOCK_M2]`，缩放它是行操作不是 tile 操作。

在 `BLOCK_M1=32, BLOCK_N1=256` 下，每次迭代省掉一次 8192 元素的乘法。

**⚠ 这个改动不是逐位中性的。** `(qk * s) * r` 和 `qk * (s * r)` 在最后一个 ulp 上不同，
因为 fp32 乘法不满足结合律。**验收只能用 SQNR，不能用哈希比对**——这点和 E1 不同，
E1 只是把一个加载提前，值完全相同。

ALIBI 路径也一并处理了（上游的 alibi 是自然对数单位，tile 现在是 log2 单位，所以要同步缩放）。
这条路在我们的形状上没启用，但改了就得改对。

### 未写的部分（累加器初值法，更激进）

把 per-row 的 LSE 偏移折进 **GEMM 累加器初值**：`tl.dot(q, kT, acc=neg_m)`，
然后 `p = exp2(qk)`，exp 之前**零** VALU。

**仅 dq 那趟合法**——那里 `m` 不随 n-tile 变化。**dkdv 那趟不合法**，`m` 随 `curr_m` 前进。

**必须实测而非假设**：acc tile 是 `[BLOCK_M2, BLOCK_N2]` fp32，每次迭代都要重新初始化，
所以收益是 3 个操作 → ~1（那个 v_mov），不是 3 → 0，除非编译器把初值折进 WMMA 的 D 操作数写回。
先测已写好的那半，再决定值不值得做这半。

**现状**：`qk_scaled = qk * sm_scale`，然后 `p = exp2(qk_scaled * RCP_LN2 - m * RCP_LN2)`
——每个元素 mul、mul、sub。

**改法**：
- `sm_scale * RCP_LN2` 是常量，可以在**加载时折进 q**（q 每个 pid 只加载一次，
  `[256,128]` = 32768 次标量乘，对比循环内的 3460 万次）。
- `m * RCP_LN2` 是循环不变量，应当提出循环（Triton 的 LICM 可能做也可能不做——
  **必须看生成的 IR，不能假设**）。

**更进一步（仅 dq 那一趟合法）**：把 per-row 的 LSE 偏移折进 **GEMM 累加器初值**——
`tl.dot(q_prescaled, kT, acc=neg_m)`，然后 `p = exp2(qk)`，exp 之前零 VALU。
dq 那趟 `m` 不随 n-tile 变化所以是循环不变的；**dkdv 那趟不合法**，那里 `m` 随 curr_m 前进。

**必须实测而非假设的地方**：acc tile 是 `[256,32]` fp32 = 64 VGPR/lane，每次迭代都要重新
初始化，所以收益是 3 个操作 → ~1（那个 v_mov），不是 3 → 0，除非编译器把初值折进 WMMA 的
D 操作数写回。

---

## E4 — `num_warps=8`（不需要补丁，一个环境变量）

**这是最高价值的未测项，而且零代码改动。**

```bash
PRIMUS_TURBO_FUSED_MHA_BWD_TUNE=num_warps=8 \
tools/gfx1250/exclusive.sh python3 tools/gfx1250/tune_attention.py \
    --shape llama31-8b --impl fused --tune "fwd:num_stages=2"
```

**依据**：`BLOCK_N1=256` + `HEAD_DIM=128` + wave32 下，dk/dv 的 fp32 累加器就占
**512 VGPR/lane**，加上跨 hqid 循环常驻的 k/v tile 再 256 —— **1024 个里用掉 768，
还没算任何 qT/pT/dsT tile**。这正是 gfx1250 独有的 `s_set_vgpr_msb` bank 切换停顿区，
而且是 1 wave/SIMD + `num_stages=1`：**占用率和流水都不提供延迟隐藏**。

`num_warps` 是从 aiter 原样继承的 4，**在融合内核上从没扫过**（本次活动扫的 num_warps
是树内非融合内核的）。8 把每 lane VGPR 砍半到 384。

同时扫 `num_warps` × `num_stages`，因为如果寄存器压力真是瓶颈，
`num_stages=2` 在 `num_warps=8` 下可能才第一次变得可行。
