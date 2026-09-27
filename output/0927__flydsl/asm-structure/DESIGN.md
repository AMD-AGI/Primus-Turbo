# gfx1250 fwd 设计研究：ASM 为何到 1399 TF/s，FlyDSL 0.3.4.1 能表达哪些结构

日期 2026-09-27。本研究全部为 compile-only（容器 `fa-repro`，flydsl 0.3.4.1，`COMPILE_ONLY=1 ARCH=gfx1250`，GPU 隐藏），没有在卡上运行任何东西。
champion = round 11（`OE/artifacts/gfx1250-flydsl-attn-fwd-20260925-114644/job_context/op/current`，只读，以下所有改动都在拷贝上完成）。

## 0. 结论先行

1. **gap 的构成**（换算到每 SIMD、每 256 个 WMMA，即一个 wave 处理一个 256-KV tile 的工作量）：ASM 3197 cycles，我们 4020 cycles，差 823 cycles（20.5%）。
   按测量值做一阶分解：**softmax 的 trans/VALU 没有被 WMMA 盖住，约 390-540 cycles（占 gap 的 50-65%）**；
   **barrier 约 140-180 cycles（17-22%）**；**每 tile 的固定开销约 110-240 cycles（13-29%）**，包括 TDM drain 到 0、SALU 重建 descriptor、permlane、rescale 分支。
   纯吞吐量不是差距来源：两边每 256 KV 的 WMMA、LDS、trans 条数相同（REPORT.md）。
2. **1 wave/SIMD + 1024 VGPR 本身不带来收益，它是前提条件**。只有在这个配置下，S 双缓冲的 wave 内流水（ASM 用来盖住 softmax 的做法）和每步 256 KV 才能放进寄存器。
   在 8 waves（2 waves/SIMD，上限 512 VGPR）下，n_block=128 会 spill（A4：512 VGPR + 642 spill），r5 在 512 VGPR 下做的流水也输了 15%。
3. **FlyDSL 0.3.4.1 可以表达这些结构，compile-only 已验证**：
   - **>512 VGPR 可用**：4 waves、n_block=128 的变体（A2）编译为 **688 VGPR、0 spill、0 scratch**，`max_flat_workgroup_size 128`，LLVM 自动插入 `s_set_vgpr_msb`（1334 条）。不需要手写 VGPR bank，也不需要 `maxnreg`；只要 block=128 线程加上 `waves_per_eu=1` 即可。
   - **4-wave roadblock 只是 5 个 guard 加 1 个硬编码**。真正的依赖在 `_tdm_load_views` 里的 `make_tdm_atom(num_warps=_DEFAULT_NUM_WAVES)`：这是一个 collective TDM atom，lowering（`FlyDSL lib/Dialect/FlyROCDL/GFX1250/CopyAtom.cpp:361-389`）按 `ROCDL::WaveId % num_warps` 把 K/V tile 的行切给各个 wave。
     4 个 wave 却用 `num_warps=8` 时，行 [n/2, n) 永远不会被加载，结果会**静默出错**，不会报错。Q 和 O V3 用的是 `num_warps=1`（每 wave 独立），它们的 guard 纯属多余。
     修复只需约 8 行（`A1.patch`），prod、thd、win_sink 三个入口都能编译，VGPR 456，0 spill。
   - **单体 n_block=256 不可行**：A3 为 1024 VGPR + 476 spill + 592 B scratch，按规则 spill 就是 kill。因此每步 256 KV 必须做成 **2 个 128 子 tile 共享一次 barrier**。
4. **推荐的原型 P2**（§4）：4 waves、每 wave 32 行（R=2）、BLOCK_M=128（GQA 打包，32 seq × 4 heads）、1 wave/SIMD、每个 barrier 步进 256 KV（2×128 子 tile，编译期 unroll）、S 双缓冲，QK(j) 与 softmax(j-1) 交织，每 256 KV 只有 1 个 split barrier，LDS 用 4-slot ring。
   VGPR 预算约 840-940，LDS 321536 B（≤ 327680）。**预测 prod +8-20%（1200-1330 TF/s）**，风险主要来自两条先例（§5）。
5. **上卡第一步不是 P2**，而是 M0-M3（§5）。先测 A1（纯 4-wave 移植）和 A2（再加 n_block=128）相对 champion 的价格，再在 A2 上做两个 timing-only 探针：去掉 exp2、去掉 barrier。
   这两个探针定出 P2 两大收益项在 1-wave 机体里的上限。bwd 的 4-wave 机体曾经无解释地输 34%（h45，0.664x），fwd 必须先确认有没有复现。

## 1. 校准数据（全部为已测量值）

| 量 | 值 | 来源 |
|---|---|---|
| prod FLOP | 2.199292e12 (b4 s8192 hq32 hkv8 d128 causal) | `op_flops.py` |
| ASM | 1.574 ms / 1398.7 TF/s | baselines |
| ours (r11) | 1.979 ms / 1111.2 TF/s | fwd.md r11 |
| 负载时钟 | ~1.04 GHz（A/B 见证 1038-1049 MHz） | `fwd341/ab_prod.log` |
| 设备 | 256 CU × 4 SIMD = 1024 SIMD，1 WG/CU（两边 LDS 都是 327680） | r11 opt.md |
| 去掉 softmax exp2 | prod 时间 -13.4% | fwd.md r3 |
| 去掉每 tile barrier（timing-only） | prod +4.5-5.2% 上限 | fwd.md r7 |
| barrier 减半原型（ring bookkeeping） | -3.5..-6% | fwd.md r7 |
| QK/softmax 流水 @2 waves/SIMD, 512 VGPR | -15.2% | fwd.md r5 |
| lock_simd / branch-free rescale | -3.5% / -8.3% | fwd.md r7, r9 |

## 2. 一阶 cycle 模型

**工作单位**：WMMA 16x16x32 bf16 = 16384 flop。prod 共 1.342e8 个 WMMA，每 SIMD 131088 个，相当于 **每 SIMD 512 个"单位"，每单位 256 个 WMMA**（= 一个 wave 的 32 行 Q × 256 KV，含 QK 和 PV）。两个 kernel 每单位的 WMMA 数相同。
ASM 的 causal 对角线浪费大约多 2.3%（128 行 × 256 KV 的 tile，我们是 64 seq × 64 KV），模型里忽略不计，这一项对 ASM 不利。

| | ASM | ours r11 |
|---|---|---|
| 总 cycles（1.04 GHz） | 1.637e6 | 2.058e6 |
| **每单位 cycles** | **3197** | **4020** |

**WMMA 发射周期 W 没有测过，这是唯一的自由参数**。由于 ASM 不可能超过 roofline，W ≤ 12.5；取 W=8（每 SIMD 2048 flop/cycle，1.04 GHz 下峰值 2.18 PF），此时 ASM 达到 64%，我们 51%。
下面的 **gap 分解只用到测量值，与 W 无关**；W 只影响"残差"这一项有多大。

| 每单位 cycles | ours r11 | ASM（推断） | gap | 依据 |
|---|---|---|---|---|
| WMMA 管线（W=8） | 2048 | 2048 | 0 | 两边都是 256 WMMA |
| 暴露的 trans（v_exp） | **539** | 0-150 | 390-540 | ours：0.134×4020，即 264 条 v_exp，每条暴露约 2 cycles。ASM：QK(j) 与 exp(j-1) 在同一 wave 内交织（REPORT 的 ISA 证据） |
| barrier | **190-210**（4 个/单位，每个约 50） | 30-60（约 2 个 split，gap 里有 WMMA） | 140-180 | ours 是测得的上限；ASM 按 split + gap 做功推断 |
| 残差（LDS 延迟、非 trans VALU、SALU、msb、nop、TDM drain、rescale 分支） | ~1230 | ~990-1120 | 110-240 | 差值 |
| **合计** | 4020 | 3197 | **823** | |

补充说明：

- **为什么 2 waves/SIMD 没有把 softmax 盖住**：每 64 KV 一次 WG 级 `s_barrier`，把同一 SIMD 上的两个 wave 锁在同一相位（REPORT #1）。两个 wave 同时做 softmax，同时做 WMMA，第二个 wave 的延迟隐藏基本失效，exp2 于是 100% 暴露。
- ASM 走另一条路：1 wave/SIMD，靠 **wave 内** 的 S 双缓冲把 trans 和 WMMA 交织。WMMA 占满矩阵管线的 W cycles 里，同一 wave 可以继续发射无依赖的 VALU/trans。每单位 256×8=2048 个 WMMA 周期内，大约有 1790 个空闲发射槽，正好容纳 ASM 内部 tile 的约 750 VALU + 260 trans + 256 LDS + SALU。
- 静态 ISA 统计（每 256 KV，clean loop，`loops.py`）：

| | A0 = champion（8w, n64） | A2（4w, n128, 无流水） | ASM half（含 mask 块） |
|---|---|---|---|
| 指令 | 1996 | 2114 | 3198 |
| WMMA / ds / trans | 256 / 256 / 264 | 256 / 256 / 260 | 256 / 256 / 260 |
| VALU（非 trans） | 744 | 632 | ~1776（内部 tile 约 750） |
| SALU | 184 | 92 | 58 |
| `s_set_vgpr_msb` | 160 | **470** | **540** |
| v_nop | 60 | 100 | 13 |
| barrier / tensorcnt wait / dscnt wait | 4 / 4 / 40 | 2 / 2 / 32 | ~2 / 2 / 17 |
| VGPR | 456 | 688 | 1024 |

- 超过 256 VGPR 以后，bank-switch 前缀（msb）必然增加。A2 每 256 KV 有 470 条，ASM 有 540 条，所以 **ASM 在同等 msb 密度下也能跑到 1399**，msb 不是否决项（bwd r11 的结论"源码层面够不到这个变量"仍然成立，但它不构成 gap）。
- 按规则 8，静态计数只用来筛选，不用来排序。

## 3. 结构性杠杆排序（预测的 prod 增益，相对 r11 的 1111 TF/s）

| # | 杠杆 | 对应的模型项 | 预测 prod | FlyDSL 0.3.4.1 能否表达 | 备注 |
|---|---|---|---|---|---|
| S1 | **1 wave/SIMD、4 waves/WG、最多 1024 VGPR** | 前提条件（单独使用时 0 到负） | 0.80-1.00x（单独） | ✅ A1/A2 已编译 | 失去第二个 wave 的延迟隐藏，所有延迟都得靠 wave 内调度盖住 |
| S2 | **S 双缓冲 wave 内流水**：QK(j) ∥ softmax(j-1)，然后 PV(j-1) | trans 390-540 | **+6-13%**（上限 13.4% 为实测） | ✅ 需要把 loop-carried S 放进 `scf.for` iter_args（每组 R×NKV 个 v8f32），去掉 `sched_barrier(0)` 栅栏 | r5 在 512 VGPR、2 waves/SIMD 下输了 15%；1 wave/SIMD 时这是唯一的重叠来源，寄存器也足够 |
| S3 | **每个 barrier 256 KV**（2×128 子 tile，编译期 unroll，1 个 split barrier） | barrier 140-180，外加部分残差 | **+3-5%** | ✅ `s_barrier_signal/_wait` 从 `flydsl._mlir.dialects.rocdl` 取（bwd h52 已用过）；编译期选 slot 已有先例（kernel :884 注释） | 结构上 barrier 从 4 个降到 1 个，而且不需要 r7 那种运行期 `tile%G` 的 bookkeeping |
| S4 | 每 tile 固定开销摊薄（n_block 128：SALU 184→92，tensor wait 4→2，permlane 和 rescale ballot 减半） | 残差 110-240 | **+1-3%** | ✅ A2 | 与 msb 的 +310 条互相抵消，净值要上卡测 |
| S5 | TDM 深度（`s_wait_tensorcnt` > 0） | 残差的一部分 | 0-1% | ✅（`tdm_ops.tensor_wait(n)`） | 每步 256 KV 时，下一步的 TDM 提前约一整步（约 6000 cycles）发出，depth-2 ring 已经够；bwd 增加深度全输 |
| S6 | Q-tile 配对（t, N-1-t 在同一个 WG 内） | 尾部不均 + epilogue 重叠 | +0.5-2% | ✅（grid_x 减半，WG 内循环两次） | LPT 已拿到 +8.6%，剩余不均很小。放 P3 |
| S7 | 手写 VGPR bank 布局 / msb 最小化 | — | 0 | ❌（无寄存器钉住，bwd r11 已关闭） | 不做 |

**累计**：S1+S2+S3+S4 ≈ 1.10-1.20x，即 1220-1330 TF/s；加上 S6 可达 1240-1355。ASM 本 session 为 1399-1408。
这些增益能否兑现取决于 S1 的代价：S1 在 A1/A2 上的实测值直接从这个区间里扣掉。

## 4. 原型规格（P2 = `fwd-4w-kv256p`）

### 4.1 几何

| 项 | 值 |
|---|---|
| waves/WG, threads | 4, 128（`NUM_WAVES=4`，`known_block_size=[128]`） |
| waves/SIMD | 1（`waves_per_eu=1`；LDS 占满，每 CU 1 个 WG） |
| Q 行/WG | 128 个 GQA 打包行（32 seq × 4 heads），每 wave 32 行（R=`WMMA_ROW_PER_WAVE`=2，沿用现有 softmax 的 R=2 交织） |
| grid | x = ceil(8192×4/128) = 256，y = 8，z = 4，共 8192 个 WG；LPT 顺序保留（`_lpt_block_id`） |
| KV 子 tile | 128（`n_block=128`，K/V manager 已支持） |
| KV 每步（每个 barrier） | 256 = 2 个子 tile，编译期 unroll（子 tile a/b 用常量 slot offset） |
| causal 尾部 | 三段式分割保留；clean 段按 256 步进，masked 段按 128 步进（避免 256 粒度把对角浪费从 ~2.5% 放大到 ~5%） |
| rescale | 保留 deferred（`ENABLE_DEFER_RESCALE`，branch-free 已测过，-8.3%） |
| LO/HI warp type | **合并成一种**：1 wave/SIMD 没有 SIMD mate，stagger 没有意义，合并后代码体积减半（I$ 在 r6/r9 被证实是真实开销） |

### 4.2 VGPR 预算（每 lane，目标 ≤ 960，给 1024 留余量）

| 项 | VGPR |
|---|---|
| O 累加器 R×d_tiles×8 = 2×8×8 | 128 |
| S 两组 2×(R×NKV×8) = 2×(2×8×8) | 256 |
| P(j-1) bf16：32 行 × 128 KV | 64 |
| Q 片段 R×4×v16bf16 | 64 |
| K 在途（流式 16-20 条 `ds_load_b128`，ASM 用 dscnt 0x10） | 64-80 |
| V 在途（流式 `ds_load_tr16`，同上） | 64-80 |
| m/d/corr/行 max 部分值、mask、地址、计数器 | ~64 |
| 调度余量 | 100-200 |
| **合计** | **~840-940** |

依据：A2（单组 S、K/V 突发加载、无流水）编译为 688。**K/V 必须改成按子块流式加载**：`load_k_to_reg` 目前一次突发全部 n_block/16×4×2 个片段，n_block=128 时是 256 VGPR，A3 正是因此在 256 时 spill。
可以改成按 kv 子块（16 行）分批返回，并与 WMMA 交织。

### 4.3 LDS 布局（327680 B 内）

| 区域 | 偏移 | 大小 |
|---|---|---|
| slot 0-3，每个 = K 128×(128+8)×2 = 34816 + V 128×(128+16)×2 = 36864 = 71680 | 0 / 71680 / 143360 / 215040 | 4 × 71680 = 286720 |
| Q staging（序言期间）/ O staging（V3，4×32×272 = 34816） | 286720 | 34816 |
| 空闲 | 321536 | 6144 |

- 步 i 使用 slot 对 p = i%2，即 {2p, 2p+1}，对应子 tile a 和 b；步 i+1 预取到另一对。
- 要把 `MIN_KV_BLK_BYTES` 设为 0（或 ≤ 34816），当前 64 KB 的下限在 4 个 slot 下放不进去。
- Q 和 O 使用独立区域，所以 epilogue 不需要复用 non-current slot，也为 S6 的 epilogue/下一 Q tile 重叠留出空间：O TDM store 完成（`s_wait_tensorcnt`）后才发下一个 Q 的 load。

### 4.4 barrier / wait 计划（每 256 KV，每个 wave）

```
top of step i (slot pair p holds step i, issued at top of step i-1):
  [V(i-1,b) ds_load_tr already issued and drained -- last read of pair 1-p]
  tensor_wait(0)                 # own quarter of step i's K/V; issued ~1 step earlier, normally free
  s_barrier_signal               # "I have finished reading pair 1-p, and my TDM part of pair p has landed"
  -- gap: register-only work --
  softmax(i-1,b) on S set B: max/exp/rowsum (trans+VALU), rescale O
  PV(i-1,b) WMMA (V already in registers)
  s_barrier_wait                 # everyone past the signal: pair p is complete, pair 1-p is writable
  TDM issue step i+1 -> pair 1-p (collective, num_warps=4)
  sub-tile a:  stream ds_load K(a); QK(a) WMMA into S set A  ||  (nothing left to overlap: softmax(i-1,b) already done in the gap)
  sub-tile b:  stream ds_load K(b); QK(b) WMMA into S set B  ||  softmax(a) on S set A
               stream ds_load_tr V(a); PV(a) WMMA           ||  (rowsum tail)
               ds_load_tr V(b) into registers, s_wait_dscnt -> ready for the next signal
```

- 每 256 KV 只有 **1 个 barrier**（今天是 4 个），split 形式，gap 里有约 400-600 条寄存器内指令（ASM 的 gap 中位数约 18 条）。
- **tensor wait 每步 1 次，深度 0**：数据提前一整步发出，不需要 `tensorcnt>0`。
- dscnt 交给 mode-2 编译器自动插入，不手写 drain，graded waits 已经被 bwd g46 判死。
- 手写的 `sched_barrier(0)` 栅栏全部去掉，交织由数据依赖加上源码顺序自然产生。只有 ISA 显示 LLVM 把 trans 聚成一堆时，才加 `sched_group_barrier`（bwd 0 胜 5 负，只能作为 S2 的一部分，不单独用）。

### 4.5 最小代码改动集（按阶段）

**阶段 A1：4-wave 移植（已完成，`A1.patch`，compile-only 通过 prod/thd/win_sink）**
1. `fmha_b16_buffer_managers.py`：
   - 删掉 5 个 `num_waves != 8` 的 raise（Q/K/V V2 共 3 个，O V2、O V3 各 1 个）。
   - `_tdm_load_views(..., num_warps=...)` 增加参数，K/V 的 `load_views` 传入 `num_warps=self.num_waves`。
2. `fmha_fwd_prefill_a16w16_m32x8.py`：`NUM_WAVES = 4`，两处 `waves_per_eu = 2 → 1`。
   BLOCK_M（128）、BLOCK_SIZE（128）、grid、`_zero_fill_attention` 的 LSE 路径（要求 BLOCK_SIZE == BLOCK_M，128 = 128 成立）都会自动跟随。
   `_lane_id` 使用 mbcnt，所以不会踩 bwd 的 `thread_idx.x` 陷阱。
3. `impl.py`：m16x8 的门槛用的是 `_kern.BLOCK_M`。BLOCK_M 变成 128 后，grid 翻倍，门槛语义也随之改变；fast shape 需要重新决定是否仍走 m16x8。

**阶段 A2：n_block=128（已完成，`A2.patch`）**：只加 `DEFAULT_N_BLOCK = 128`，结果 688 VGPR，0 spill。

**阶段 P2（待写，预计 300-400 行，改动集中在 `_core_attention`/`main_loop`）：**
1. 合并 LO/HI 分派。
2. 主循环按 256 KV 步进，编译期 unroll 2 个子 tile；slot 对编号为 `(local_iter % 2) * 2 + {0,1}`；同时补上 `N_KV_PP` 和 `% 2` 的泛化。
3. iter_args 增加一组 S（子 tile b），并加入 P/m/d 的延迟一拍状态；序言先单独做一次 QK(start) 来填充流水，结尾做一次 softmax+PV 来排空。
4. `load_k_to_reg`/`load_v_to_reg` 增加按 kv 子块切片的接口，`imm` 偏移公式不变。
5. `_drain_barrier` 改为 `tensor_wait(0)` + `s_barrier_signal`，接 gap 工作，再接 `s_barrier_wait`。
6. `MIN_KV_BLK_BYTES=0`；Q/O 使用独立区域（§4.3）；O 的 `assert <= slot_bytes` 改为对独立区域检查。
7. masked 尾段沿用 128 步进的现有 `main_loop`，这样 masked 路径不需要流水化，只是正确性回退路径。

## 5. 上卡前必须先测的（预注册，按顺序；每个 shape 单独一个进程，同 session palindromic，不和 beat 同进程，带时钟见证）

| 步 | 内容 | 预注册预测 | kill / 分支判据 |
|---|---|---|---|
| **M0** | A1、A2 的 UT（prod + 7 个 edge case，SQNR ≥ 49 dB，o/lse 按位确定），并用"故意 scale×2"检查缓存是否陈旧 | 通过 | 失败说明 collective TDM 切分之外还有隐藏的 8-wave 假设，先修正确性 |
| **M1** | A1 与 champion 对比（prod/proxy/fast） | 0.85-1.00x | **< 0.75x**：bwd 的 4-wave barrier 病理（h45 0.664x）在 fwd 复现。这时先在 A1 上做 h51 式的 barrier-free timing 探针（错误结果、工作量相同），不继续构建 |
| **M2** | A2 与 A1、champion 对比 | A2/A1 = 1.00-1.06 | A2 < A1 说明 msb 税和 1-wave 延迟超过了 barrier 减半的收益，S3/S4 需要重新定价（P2 仍可能成立，但 +3-5% 要下调） |
| **M3a** | A2 去掉 exp2（timing-only，同 r3 方法） | ≥ 15%（1-wave 下暴露得更多） | < 8% 时，S2 的上限不足以抵消 S1 的代价，**P2 不建** |
| **M3b** | A2 去掉 barrier（timing-only） | ≤ 3% | > 8% 时说明 1-wave 下 barrier 远比 2-wave 贵，这是 P2 的 split barrier 必须优先证明的项 |
| M4（可选） | WMMA 发射率微基准：1 wave/SIMD，独立 WMMA 链，无内存访问 | W = 4-8 cycles | 用来确定模型的地板；只影响残差项 |

完成 M0-M3 后：如果 M1 ≥ 0.85 且 M3a ≥ 10%，就构建 P2，先过 compile-only 门槛（0 spill，VGPR ≤ 1000，`s_barrier_wait` = 每 256 KV 1 个）再上卡。
可以作为 op-evolve 的一个 deep round 下发（hint hN：先跑 M0-M3，再做 P2），但不能与正在运行的 job 抢卡。

## 6. 产物

| 路径 | 内容 |
|---|---|
| `output/0927__flydsl/asm-structure/mkvariant.py` | 从 champion 拷贝并打补丁：`<label> <waves> <n_block> <wpe> [min_kv_kb]` |
| `output/0927__flydsl/asm-structure/loops.py` | 定位 ISA 中的回边循环并按指令类分类计数 |
| `output/0927__flydsl/asm-structure/A1.patch`, `A2.patch` | 相对 champion 的完整 diff |
| `output/0927__flydsl/asm-structure/work/{A0..A4}` | 变体树（A0 = champion 原样，用作对照） |
| `output/0927__flydsl/asm-structure/isa/*.s` | final ISA：A0、A1、A1_thd、A1_win_sink、A2、A2_thd、A2_win_sink、A3、A4 |

编译结果汇总（prod，bshd causal lse）：

| 变体 | waves | n_block | wpe | VGPR | spill | scratch B | 指令 | msb | 结论 |
|---|---|---|---|---|---|---|---|---|---|
| A0 champion | 8 | 64 | 2 | 456 | 0 | 0 | 3935 | 240 | 对照 |
| A1 | 4 | 64 | 1 | 456 | 0 | 0 | 3935 | 240 | ✅ 4-wave 移植（`max_flat_workgroup_size 128`，TDM 按 4 切分：`s_lshr 2→1`） |
| A2 | 4 | 128 | 1 | 688 | 0 | 0 | 6808 | 1334 | ✅ >512 VGPR 可行 |
| A3 | 4 | 256 | 1 | 1024 | 476 | 592 | 13234 | 3740 | ❌ spill（禁止上卡） |
| A4 | 8 | 128 | 2 | 512 | 642 | 960 | 7818 | 1440 | ❌ 复现 L10（8 waves 下 n_block=128 不可行） |
