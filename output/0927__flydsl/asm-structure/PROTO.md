# P2 原型（fwd-4w-kv256p）：compile-only 构建报告

日期 2026-09-27。全部为 compile-only：容器 `fa-repro`，flydsl 0.3.4.1，`COMPILE_ONLY=1 ARCH=gfx1250 FLYDSL_GPU_ARCH=gfx1250`，GPU 隐藏（`HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=`）。
卡上没有运行任何东西。champion（round 11）只读，没有改动。改动都在拷贝 `asm-structure/op` 上，`asm-structure/base` 没有碰过（目录不存在，也没有新建）。

## 0. 结论

1. **DESIGN.md §4 的 P2 已经完整构建，而且编译通过**：4 waves/WG，1 wave/SIMD，BLOCK_M=128（GQA 打包），KV 子 tile 为 128，4-slot K|V LDS 环，Q/O 使用独立区域。
   clean 段按 256 KV 步进，一步两个编译期子 tile，**每 256 KV 只有 1 个 split barrier 和 1 个 `s_wait_tensorcnt`**。S 有两组寄存器。LO/HI 两种 warp 合并成一种。masked 边界段保留 128 步进的普通循环。
   没有任何一项因为 API、编译器或资源的限制做不了。与设计不一致的地方都来自 LLVM 的调度，见 §4。
2. **资源（prod，bshd causal lse gqa4 d128 bf16）**：
   - VGPR 800（4 个 bank：v0-255、v256-511、v512-767、v768-799），**VGPR spill 0，SGPR spill 0，scratch 0**，SGPR 70；
   - LDS 按容量申请 327680 B，布局实际使用 321536 B；
   - `max_flat_workgroup_size 128`，4 waves/WG；占用率为每 CU 1 个 WG（受 LDS 限制），每 SIMD 1 个 wave（`waves_per_eu=1`，VGPR > 512）。
   - VGPR 在设计预算 840-940 之下。
   - fast（b1 s1024 hq8 hkv2）编译出的 ISA 与 prod **逐字节相同**：shape 只是运行期参数。
3. **gate 关闭时与 champion 逐字节相同**：`ASM_STRUCT=False` 下，prod、fast、thd、win_sink 四个入口的 final ISA 与 champion 拷贝的 md5 完全一致。impl.py 在 fast 上走的 m16x8 kernel（8 waves，没有改动）在 gate 打开时也与 champion 逐字节相同。
4. **每 256 KV 的静态画像**（§3）：WMMA 256，v_exp 260，ds_load 256，TDM 4，barrier signal/wait 各 1，tensorcnt wait 1，dscnt wait 26，非 trans VALU 562（外加 rescale 分支内 130），SALU 85，msb 626，nop 120。
   champion 对应的数字是 barrier 4/4，tensorcnt 4，dscnt 40，SALU 168，msb 164，nop 64。
5. **S 双缓冲的交织在 ISA 里确实出现了**：pipe 循环中 260 条 v_exp 里有 110 条落在某个 WMMA 之后 8 条指令以内，champion 是 0/66。
   softmax(a) 的 129 条 exp 分散在 QK(b) 的 128 条 WMMA 之间，最长连续段 21 条。softmax(b) 的 129 条 exp 与 PV(a) 的 64 条 WMMA 交织得差，有一段 79 条连续 exp。QK(a) 和 PV(b) 旁边没有可以交织的工作。
   这些只是静态筛选，**不能用来排序**（规则 8）。gain 预测仍然要等 DESIGN §5 的 M0-M3 在卡上测过才能确认。
6. **op/ 目录的默认值是 `ASM_STRUCT = True`，即原型开启**。改成 False 就回到 champion，且逐字节相同。

## 1. 产物

| 路径 | 内容 |
|---|---|
| `op/` | champion 拷贝 + P2（`flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py` 和 `fmha_b16_buffer_managers.py` 两个文件有改动） |
| `P2.patch` | `diff -ru champion op`（723 行 +/-） |
| `proto/build.sh <on\|off\|champ> <tag> [prod\|thd\|win_sink]` | 生成 `proto/variants/<gate>` 并编译；支持环境变量 SHAPE=prod/fast、GQA、CAUSAL、D、DT |
| `proto/mkknob.sh <name> 'sed' ...` | 在 gate 打开的基础上改 knob，生成变体 |
| `proto/rc.py`、`rc16.py`、`build16.sh` | compile 驱动（来自 `proto/pipeline/review/rc.py`，rc16 编译 m16x8） |
| `proto/dscnt_cfg.py` | 在 CFG 上做数据流，统计每个 `s_barrier_signal` 处还有几条未完成的 LDS 操作 |
| `proto/tools/report.py`、`hist.py`、`overlap.py`、`seq.py` | 按循环汇总、每 256 KV 直方图、exp/WMMA 交织度、指令类别序列 |
| `proto/isa/<tag>.s` | final ISA：`on_{prod,fast,thd,ws,g1c1,g2c1,g4c0,g1c0,fp16}`、`off_*`、`champ_*`、`m16_*`、`k_*`（knob 变体）；`isa/on_prod/` 保留了完整的 IR dump |
| `proto/logs/<tag>.log` | 编译日志 |

## 2. 实现（全部在 `ASM_STRUCT` 后面）

**模块常量**（`fmha_fwd_prefill_a16w16_m32x8.py:112-156`）：
```python
ASM_STRUCT = True
NUM_WAVES = 4 if ASM_STRUCT else 8          # BLOCK_SIZE 128, BLOCK_M 128 自动跟随
WAVES_PER_EU = 1 if ASM_STRUCT else 2       # 两个 launcher 的 compile_hints 都改成用它
SUPPORTED_QK_HDIM = (128,) if ASM_STRUCT else (128, 192, 256)
DEFAULT_N_BLOCK = 128 if ASM_STRUCT else 64
```
两个 kernel 入口：`if const_expr(ASM_STRUCT): _core_attention_p2(...)`（只 trace 一次，没有 warp type），否则走原来的 LO/HI 分派。
**buffer managers**：
- 5 个 `num_waves != 8` 的 raise 改成 `not in (4, 8)`；
- `_tdm_load_views(num_warps=...)`，K/V 的 `load_views` 传入 `self.num_waves`（这就是 A1.patch 的修复。collective TDM 按 wave id 切分行，写死 8 时，行 [n/2, n) 会静默地永远不被加载）。
- 8 waves 时传入的值仍然是 8，这一点已由逐字节相同验证。

**`_core_attention_p2`** 及其 helper（`:1430-2082`，约 650 行，含 knob；champion 的 `_core_attention` 一个字节都没改）：

| 设计项（DESIGN §4） | 实现 |
|---|---|
| 4 waves，每 wave 32 行（R=2），BLOCK_M=128 | ✅ `_packed_tile_indices`、Q V2 per-wave TDM 和 grid 都跟随 BLOCK_M |
| 1 wave/SIMD，≤1024 VGPR | ✅ 800 VGPR，LLVM 自动插入 `s_set_vgpr_msb` |
| LDS：4 个 slot × 71680 B + Q/O 独立区域 34816 B，位于 286720 | ✅ 共 321536 B，有编译期 assert。不再使用 `MIN_KV_BLK_BYTES`，P2 直接用 K/V 的真实大小 |
| clean 段每个 barrier 走 256 KV（2×128，编译期 unroll） | ✅ `pipe_step`，`scf.for` 步长为 2 |
| 两组 S：QK(b) ∥ softmax(a)，然后 PV(a) | ✅ S_A 和 S_B 在同一个 basic block 内，没有 `sched_barrier` 栅栏 |
| K/V 按块流式加载，不一次性突发 | ✅ `_p2_qk` 每组 16 kv 的 8 条 `ds_load_b128` 紧挨着各自的 WMMA；PV(a) 的 `ds_load_tr16` 按 d-tile 流式加载 |
| 每 256 KV 1 次 TDM wait + 1 个 split barrier，gap 里做 softmax(b)+PV(b) | ✅ 源码顺序是：V(b)→VGPR，`s_wait_dscnt 0`，`tensor_wait(0)`，`s_barrier_signal -1`，softmax(b)，rescale，PV(b)，`s_barrier_wait -1`，然后发 TDM。LLVM 实际的排布见 §4 |
| 合并 LO/HI | ✅ |
| masked 尾段保留 128 步进 | ✅ `plain_step`（K 流式加载，V 突发加载）。右侧 plain 循环也会吸收 clean 段剩下的奇数个子 tile（mask 精确，只多花 mask 的 VALU） |
| deferred rescale | ✅ 保留（每个 q-tile 一个 `scf.if`） |
| S6（Q-tile 配对） | ❌ 未做，这是设计中 P3 的内容 |

**环的不变式**：
- 子 tile u（从 start_tile 起的局部编号）放在 slot u%4。
- 进入子 tile u 时，u..u+3 已经发出（只发 < n_tiles 的），且 u 和 u+1 已经落地，并对所有 wave 可见。
- 每步的做法：
  - **prologue**：发 start 和 start+1，`tensor_wait(0)`，barrier，再发 start+2 和 start+3。
  - **plain**：计算，然后 `tensor_wait(0)`，barrier，把 u+4 发进 slot u%4。
  - **pipe**：计算两个子 tile，把 V(u+1) 读进 VGPR，`s_wait_dscnt 0`，`tensor_wait(0)`，signal，做 gap 工作，wait，再把 u+4 和 u+5 发进 slot u%4 和 (u+1)%4。
- 在 pipe 和 plain 之间来回切换，这个不变式都成立。
- 环的 ds 基址作为 8 个 iter_args 携带，每步旋转（plain 旋转 1 个，pipe 旋转 2 个），在 ISA 里表现为每步几条 `v_dual_mov`（循环头 2 条，共 4 次 mov；全循环 8 条）。
- TDM 的 LDS 目标 slot 由运行期 `(t-start)%4` 算出，走 SALU。

**knob**（只在 ASM_STRUCT 打开时生效，默认值都是 LLVM 自由调度）：`P2_GAP_FENCE`、`P2_INTERLEAVE`、`P2_SGB_ROUNDS/VALU/TRANS`，见 §5。

## 3. 资源和指令直方图

### 3.1 编译矩阵（gate on，全部 0 VGPR spill，0 scratch）

| 配置 | VGPR | SGPR | SGPR spill | scratch B | LDS | WG | 备注 |
|---|---|---|---|---|---|---|---|
| **prod** bshd causal lse gqa4 bf16 | **800** | 70 | 0 | 0 | 327680 | 128 | 与 fast 逐字节相同 |
| **fast** b1 s1024 hq8 hkv2 | **800** | 70 | 0 | 0 | 327680 | 128 | 同上（但 impl.py 在 fast 上会走 m16x8，见 §6） |
| thd causal lse | 800 | 76 | 0 | 0 | 327680 | 128 | |
| win_sink（mask_left + sink） | 808 | 107 | **1** | 0 | 327680 | 128 | SGPR 溢出到 VGPR lane，不是 scratch；job 不测 window |
| gqa1 causal / gqa2 causal | 800 / 800 | 77 / 70 | 0 | 0 | 327680 | 128 | |
| gqa4 non-causal / gqa1 non-causal | 800 / 800 | 82 / 83 | 0 | 0 | 327680 | 128 | |
| fp16 causal gqa4 | 801 | 70 | 0 | 0 | 327680 | 128 | |
| 对照 champion prod（= gate off） | 456 | 100 | 0 | 0 | 327680 | 256 | 8 waves，2 waves/SIMD |

- 整个 kernel 共 5404 条指令（champion 为 3935），其中 1252 条 `s_set_vgpr_msb`。
- VGPR bank：用到了 4 个，最高物理寄存器是 v799。

### 3.2 每 256 KV 指令直方图（clean 段主循环，prod）

"main" 是 fall-through 路径（含受保护的 TDM 发射块）。"resc" 是 deferred-rescale 的 `scf.if` 体，只在 ballot 触发时执行。
champion 按每 64 KV 一次迭代 × 4 计算。ASM、A2 的数据取自 DESIGN.md §2。

| 类别 | **P2 pipe**（L593-2979，1 次迭代 = 256 KV） | P2 plain masked（×2） | champion clean（×4） | A2（4w n128，×2） | ASM half |
|---|---|---|---|---|---|
| WMMA | **256** | 256 | 256 | 256 | 256 |
| v_exp | **260** | 260 | 264 | 260 | 260 |
| VALU（非 trans，不含 permlane） | **562** + resc 130 | 1178 + 128 | 732 + 128 | 626 + 64 | 内部 tile 约 750 |
| v_permlane | 8 | 8 | 16 | 8 | |
| ds_load（b128 + tr16） | **256** | 256 | 256 | 256 | 256 |
| TDM（tensor_load_to_lds） | **4** | 4 | 8 | 4 | |
| SALU | **85** | 80 | 168 | 84 | 58 |
| s_barrier_signal / s_barrier_wait | **1 / 1** | 2 / 2 | 4 / 4 | 2 / 2 | 约 2（1.5 split） |
| s_wait_tensorcnt | **1** | 2 | 4 | 2 | 2 |
| s_wait_dscnt | **26** | 84 | 40 | 32 | 17 |
| s_wait_alu | 29 | 52 | 12 | 6 | |
| v_nop / s_nop | 120 | 126 | 64 | 102 | 13 |
| s_set_vgpr_msb | **626** | 1064 | 164 | 472 | 540 |
| 合计（main） | 2243 | 3384 | 2012 | 2122 | 3198（含 mask 块） |

- VALU 比 champion 少约 170 条，因为每个 tile 的固定开销减半了：permlane、rescale ballot 的 SALU/VALU、ring bookkeeping。
- msb 比 A2 多约 150 条。800 VGPR 下有 4 个 bank，A2 只有 3 个。msb 是 LLVM 自动插入的，源码层面控制不了（S7）。
- nop 多出来的部分主要是 trans 之后的 `v_nop`，由 mode-2 hazard 插入。

## 4. 主循环 ISA 摘录（`proto/isa/on_prod.s`，已省略 `s_set_vgpr_msb` 和 `s_wait_alu`）

```asm
.LBB0_8:                                    ; ---- pipe 循环回边 / 头（L593）
  v_pk_fma_f32 ... v_pk_add_f32 ...         ; 上一步 d_new = corr*d + rowsum（sum-tree 尾，被 LLVM 推到回边）
  v_dual_mov_b32 v8, v204 :: v_dual_mov_b32 v9, v201   ; 环形 ds 基址旋转 2（4 条 dual mov/步）
  s_add_nc_u64 s[42:43], s[42:43], 2        ; tile += 2（每步 256 KV）
  s_cbranch_vccz .LBB0_22                   ; 出循环
.LBB0_9:                                    ; ---- QK(a) + QK(b) + softmax(a)（一个 BB，L617-1560）
  ds_load_b128 v[192:195], v202             ; K(a) 流式源码被 LLVM 聚成 ~84 条的突发（寄存器够用）
  ds_load_b128 v[196:199], v202 offset:32
  ...
  s_wait_dscnt 0x38                         ; 第一条 WMMA 只等最早的 28 条 K 回来
  v_wmma_f32_16x16x32_bf16 v[120:127], v[248:255], v[72:79], 0      ; QK(a) 开始：步顶 LDS 延迟无可交织（设计已知）
  ...
  ; ~L1085：softmax(a) 与 QK(b) 交织 —— S 双缓冲在 ISA 里的样子
  v_pk_fma_f32 v[26:27], v[26:27], s[36:37], v[6:7]   ; exp2 参数 s*log2e - m（packed）
  v_exp_f32_e32 v184, v24                   ; softmax(a) exp
  v_exp_f32_e32 v186, v25
  v_pk_fma_f32 v[24:25], v[30:31], s[36:37], v[6:7]
  v_exp_f32_e32 v188, v26
  v_wmma_f32_16x16x32_bf16 v[240:247], v[154:161], v[160:167], 0   ; QK(b) 新 kv-tile（C=0）
  v_exp_f32_e32 v190, v27
  v_exp_f32_e32 v194, v24
  v_exp_f32_e32 v196, v25
  v_exp_f32_e32 v192, v28
  v_exp_f32_e32 v152, v29
  v_wmma_f32_16x16x32_bf16 v[216:223], v[76:83], v[152:159], v[216:223]  ; QK(b) 累加
  v_exp_f32_e32 v154, v26
  ...                                       ; 本段合计 WMMA 128、v_exp 129，exp 分布在 25 个 WMMA 间隙（最长 21）
  s_cbranch_vccz .LBB0_11                   ; rescale(a, qt0) —— wave-uniform ballot 分支
  v_pk_mul_f32 ... (32 条)                  ;   O *= corr（只在 max 移动 > 8 时执行）
.LBB0_13:                                   ; ---- PV(a) + V(b)→VGPR + softmax(b)（L1644-2590）
  ds_load_tr16_b128 v[20:23], v204 offset:4704        ; V(a) 按 d-tile 流式
  v_wmma_f32_16x16x32_bf16 v[88:95], v[208:215], v[80:87], ...   ; PV(a)
  v_exp_f32_e32 ...                         ; softmax(b) 被 LLVM 提到 signal 之前，与 PV(a) 交织（但有 79 条的连续 exp 段）
  ds_load_tr16_b128 v[232:235], v204 offset:9408      ; V(b) 最后几条 → VGPR
  s_wait_dscnt 0x0                          ; 本 wave 对当前 slot 对的最后一次读已完成
  s_wait_tensorcnt 0x0                      ; 本 wave 那一份 t+2/t+3 已落地
  s_barrier_signal -1                       ; ==== 每 256 KV 唯一的 barrier（L2590）====
  v_wmma_f32_16x16x32_bf16 v[88:95], v[216:223], v[72:79], ...   ; gap：PV(a) 尾 12 条 WMMA（纯寄存器）
  ...
  s_cbranch_vccz .LBB0_15                   ; gap：rescale(b) 两个 ballot 分支
  v_pk_mul_f32 ...
.LBB0_17:
  v_cvt_pk_bf16_f32 v247, v178, v182        ; gap：P(b) → bf16
  v_wmma_f32_16x16x32_bf16 v[40:47], ...    ; gap：PV(b) 头
  s_barrier_wait -1                         ; ==== L2773 ====（signal→wait 主路径约 80 条，含 18 WMMA）
  v_wmma_f32_16x16x32_bf16 v[112:119], v[16:23], v[240:247], v[112:119]  ; PV(b) 剩余 ~46 条 WMMA（V(b) 已在 VGPR）
  ...
  s_cbranch_scc1 .LBB0_19                   ; t+4 < n_tiles ?
  s_mul_i32 s21, s5, 0x11800                ; slot=(t-start)%4 → *71680（SALU 描述符）
  tensor_load_to_lds s[4:7], s[12:19]       ; K(t+4) → slot t%4（collective，num_warps=4）
  tensor_load_to_lds s[4:7], s[20:27]       ; V(t+4)
.LBB0_19:
  s_cbranch_scc1 .LBB0_8                    ; t+5 < n_tiles ?（否则直接回边）
  tensor_load_to_lds ... ×2                 ; K/V(t+5) → slot (t+1)%4
  s_branch .LBB0_8
```

**与设计 §4.4 的差异（全部来自 LLVM 调度，源码顺序与设计一致）**：
1. softmax(b) 没有留在 barrier gap 里，而是被提前到 signal 之前，与 PV(a) 交织：它只依赖寄存器，s_barrier_signal 挡不住纯 ALU。
   结果是 gap 变短：主路径约 80 条，含 18 条 WMMA，外加 rescale 分支 64 条 pk_mul。设计预计 400-600 条，ASM 中位数约 18 条。
   `P2_GAP_FENCE=True` 可以强制按设计的字面意思来：gap 变成 547 条（25 WMMA），VGPR 786，但交织度降到 94/260。**哪种更好只能上卡决定**。
2. K(a) 的流式源码被 LLVM 重新聚成步顶的一次突发（约 84 条 ds_load）。因为 VGPR 足够（800），这不影响正确性，也不会 spill。
   但步顶 QK(a) 的 LDS 延迟仍然暴露，这是设计里就已知的"QK(a) 无可交织"。
3. PV(b) 的大部分 WMMA 排在 `s_barrier_wait` 之后，所以下一对 tile 的 TDM 发射晚了约 46 条 WMMA，预取距离仍接近一整步（约 2300 条指令）。

## 5. knob 筛选（静态，仅供上卡 A/B 参考，不能用来排序）

| 变体 | VGPR | spill | v_exp 落在 WMMA 后 ≤8 条内 | signal→wait（含 rescale 块） |
|---|---|---|---|---|
| **默认**（LLVM 自由调度） | 800 | 0 | **110/260** | 147（18 WMMA） |
| `P2_GAP_FENCE=True`（softmax(b) 钉在 gap 里） | 786 | 0 | 94/260 | 547（25 WMMA） |
| `P2_INTERLEAVE=True`（softmax(a) 分阶段编织进 QK(b) 源码） | 800 | 0 | 93/260 | 160 |
| INTERLEAVE + GAP_FENCE | 788 | 0 | 64/260 | 531 |
| `P2_SGB_ROUNDS=16`（1 WMMA / 6 VALU / 2 TRANS） | 808 | 0 | 73/260 | 147 |
| `P2_SGB_ROUNDS=32`，VALU=4 | 808 | 0 | 70/260 | 160 |

- 源码层面的编织和 `sched_group_barrier` 都**没有**超过 LLVM 自己的调度，与 bwd 上 sgb 0 胜 5 负的结论一致。所以默认两者都不开。
- 所有变体都可以用 `proto/mkknob.sh` 重建。

## 6. 正确性核查（静态）

- **gate off 与 champion 逐字节相同**：`proto/isa/off_{prod,fast,thd,ws}.s` 与 `champ_*.s` 的 md5 一致；gate 打开时 `m16_on_fast.s` 也与 `m16_champ_fast.s` 相同。
- **barrier 处没有未完成的 LDS 读**（`dscnt_cfg.py`）：
  - prod、ws、g1c0、fp16 的所有 `s_barrier_signal` 处未完成数都是 0。
  - thd 的 prologue barrier 报 16，这是误报：那里是 `s_wait_storecnt_dscnt 0x0`，脚本只认 `s_wait_dscnt`。而且那 16 条是本 wave 对自己的 Q 区域的读，其它 wave 不会写这块区域。
- **WAR/RAW**：
  - pipe 的 TDM 只在 `s_barrier_wait` 之后发出（L2929/2970 在 L2773 之后）。
  - 目标 slot 最后一次被读，是在所有 wave 各自的 signal 之前：K(t)、K(t+1)、V(t) 由 WMMA 消耗；V(t+1) 读进 VGPR 后有 `s_wait_dscnt 0`。
  - 读 slot 之前，先经过 `tensor_wait(0)`（每个 wave 各自的那一份），再经过下一个 barrier（所有 wave 的份都齐了）。
- **每个 wave 执行的 barrier 次数相同**：所有循环边界都是 WG-uniform。prologue 1 次 + 每个 pipe 步 1 次 + 每个 plain 子 tile 1 次。
- **O epilogue**：写到 Q/O 独立区域中本 wave 自己的 8704 B（与本 wave 的 Q 行完全重合）。不存在跨 wave 竞争，也不会有 TDM 在飞：所有发射都受 `< n_tiles` 保护，而且都会被消耗。
- **尚未验证（M0 必须覆盖）**：数值正确性、按位确定性，以及 n_tiles=1/2/3 等短序列、奇数个 clean 子 tile、unequal seqlen、kv_len%128≠0、mask_left 窗口。
  - 这个原型没有和 champion 逐位对齐的承诺：softmax tile 从 64 KV 变成了 128 KV，树形归约的舍入不同，所以用 SQNR 判定。

## 7. 没做 / 限制（都不是阻塞）

| 项 | 状态 | 原因 |
|---|---|---|
| S6 Q-tile 配对 | 未做 | 设计里属于 P3 |
| QK(a) 与 softmax(b_{i-1}) 跨 barrier 交织（真正的 ASM 式流水） | 未做 | 需要把 S_B 和 PV(b) 延到下一步：QK(a_i) ∥ softmax(b_{i-1})。设计版把 V(b) 读进 VGPR，但 V(b_{i-1}) 如果留在 LDS，4-slot 环不够。**资源上可行的方案**：K 环 4 slot（139264）+ V 环 5 slot（184320）= 323584 B ≤ 327680；Q 放在 prologue 尚未装载的 K slot 2/3，O 放在最后一个 plain barrier 之后的 K slot，VGPR 估计约 800-850。这可以作为 P2b，前提是 M3a 证明 exp 暴露值得这么做 |
| qk_hdim 192/256 | 关闭 | `SUPPORTED_QK_HDIM=(128,)`，P2 有 assert。192/256 的 K 行更宽，没有做预算 |
| impl.py | 未改 | m16 的门槛 `grid_m32 < _NUM_CU` 用的是 `_kern.BLOCK_M`（现在是 128）：prod 的 grid 为 8192 ≥ 256，走 P2；fast 的 grid 为 64 < 256，**仍然走 8-wave 的 m16x8**。要在卡上测 P2 的 fast，得临时绕过这个门槛 |
| win_sink | 1 个 SGPR spill（VGPR lane，不是 scratch） | job 不测 window，上卡前可以留意 |

## 8. 上卡顺序建议（沿用 DESIGN §5，并补上 P2）

1. M0-M3 不变（A1、A2、去掉 exp2、去掉 barrier）。
2. 只有满足 M1 ≥ 0.85 且 M3a ≥ 10% 时才测 P2：
   - 先跑 UT：prod 加 7 个 edge case，SQNR ≥ 49 dB，o/lse 按位确定，同一 shape 重复 80 次，并用 scale×2 检查缓存是否陈旧。edge case 必须包括 n_tiles ≤ 3 和奇数个 clean 子 tile。
   - 然后做 palindromic A/B：P2 对 A2、P2 对 champion。
3. knob 的 A/B：默认值对 `P2_GAP_FENCE=True`。这是唯一一个结构上说得通、且只有卡能裁决的开关。

## 9. 复现

```bash
cd output/0927__flydsl/asm-structure/proto
./build.sh on on_prod prod                  # gate on，prod
SHAPE=fast ./build.sh on on_fast prod       # fast（ISA 与 prod 相同）
./build.sh off off_prod prod && cmp isa/off_prod.s isa/champ_prod.s   # gate off 与 champion 逐字节相同
python3 tools/report.py isa/on_prod.s       # 循环汇总（WMMA/exp/barrier/gap）
python3 tools/hist.py isa/on_prod.s 593 2979 256   # 每 256 KV 直方图
python3 dscnt_cfg.py isa/on_prod.s          # barrier 处未完成的 ds 操作
```
容器里 root 写出的 dump 会导致 `rm` 权限报错：先执行 `docker exec fa-repro chown -R <uid>:<gid> <dir>`。
