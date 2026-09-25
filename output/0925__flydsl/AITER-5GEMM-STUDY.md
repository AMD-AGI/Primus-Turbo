# aiter 反向的 5-GEMM 方法：是否成立、能否照抄

结论先行：**方法在数值上成立，精度代价可忽略（6.0e-6 dB，树内受控对实测）；但"照抄"有一个此前四个角度都没抓到的硬阻塞 —— FlyDSL 已验证能发出的那条 fp32 atomic 是 `SCOPE_CU`，aiter 发的是 `SCOPE_DEV`，而这张卡有 8 个 XCD。这是静默错答级别，不是性能问题。绕路存在且我已把编码表打穿。**另外，真正该先建的不是融合版，而是不动契约的 BLOCK_KV=128 + 4 波版。

---

## 1. 算法：aiter 的 5 个 GEMM 是什么，靠什么免去重算

一个 workgroup = 128 线程 = 4 个 wave32（`mha_bwd.cu:710` `int bdx = (arch_id == "gfx1250") ? 128 : 256;`），拥有 **128 个 kv 行**（`fmha_bwd_dqdkdv.csv:3` 的 `ts_qo=32, ts=128`；causal 下 `mha_bwd.cu:715-718` 把 gdx 折半，所以一个 workgroup 吃一对 kv block）。wave *w* 拥有 kv 列 `[32w, 32w+32)`。q 以 **32 行为一个 substep**，4 个 substep 组成一次 128q 的循环体。

每个 substep 每个 wave 发 80 条 `v_wmma_f32_16x16x32_bf16`，分成 5 组（我复核过 body A `[0x42E8,0x89AC)` 的 320 条，按 dst 分组恰好落成 5×64）：

| GEMM | A 操作数 | B 操作数 | 输出 | 累加器 |
|---|---|---|---|---|
| S = Q·Kᵀ | v[340:403] (Q, ds_load_tr16) | v[644:707] (Kᵀ 常驻) | 32q×32kv | v[8:39] |
| dP = dO·Vᵀ | v[72:135] (dO) | v[512:575] (Vᵀ 常驻) | 32q×32kv | v[140:171] |
| dVᵀ = dOᵀ·P | v[276:339] (dOᵀ) | v[188:203] (bf16 P) | 128d×32kv | **v[896:1023] 跨循环常驻** |
| dKᵀ = Qᵀ·dS | v[340:403] | v[260:275] (bf16 dS) | 128d×32kv | **v[768:895] 跨循环常驻** |
| dQ = dS·K | v[72:135] (dS 从 LDS offset 8704 重载) | v[576:639] (K 的第二个朝向) | 32q×32d | v[204:235] → atomic |

**零重算的三件事，按可实现性排序：**

1. **dKᵀ/dVᵀ 的 fp32 累加器常驻 v[768:1023]**，每 wave 256 VGPR = 2×(32kv×128d)/32 lane，横跨整个 q 循环不落地。这就是 BLOCK_KV=128 的独立证明：4 wave × 32kv = 128；若是 256 则每 wave 要 512 VGPR 累加器，`.vgpr_count: 1024` 放不下。
2. **dS 经 LDS 在 4 个 wave 间交换**。dQ 的收缩维是整个 128 kv，单个 wave 只有 32 列，所以每 substep 4 条 `ds_store_b128` 写出本 wave 的 bf16 dS，`s_barrier_signal`/`s_barrier_wait` 之后各 wave 读回全 128 kv。body A 内 16 条 ds_store + 8 个 barrier，body B 逐项相同。
3. **K 以两种 layout 同时常驻**（v[644:707] 给 S，v[576:639] 给 dQ），Vᵀ 常驻 v[512:575]，共 192 VGPR 的 kv-block 不变量。

写回：dQ 是唯一的 atomic；dK/dV 走 `v_cvt_pk_bf16_f32` → 32 条 `ds_store_b128` → 2 条 `tensor_store_from_lds`（`.long 0xd0314000`，在 0xF818/0xF824）。全 kernel `buffer_load`/`buffer_store`/`global_store` 均为 **0** —— 所有取数是 38 条 TDM `tensor_load_to_lds`。

**纠正一条本会话早先的错误断言：** drain 段 `[0xD070,0xF84C)` 的 256 条 atomic **不是** dK/dV。我对全部 514 条解析了 SRD 和数据寄存器：**514/514 用 `s[52:55]`，数据寄存器 100% 落在 v[204:235]（32 个 distinct）**。`s[52:53]` 全 kernel 只被写一次，来自反汇编第 3 行 `s_load_b64 s[52:53], s[0:1], 0x0` = kernarg offset 0 = `ptr_dq`（`csrc/include/mha_bwd.h:166` `void* ptr_dq; // 0x00: dq or dq_acc`）。drain 的 256 条是流水尾部的 dQ 冲刷。BLOCK_KV=128 的结论不受影响（它由 csv 的 ts 列、gdx 折半、和上面的寄存器预算三路独立支撑）。

静态计数复核（`/opt/rocm/llvm/bin/llvm-objdump -d --mcpu=gfx1250`）：v_wmma **864** = 32(prologue) + 320(bodyA) + 320(bodyB) + 192(drain)；atomic **514** = 2 + 128 + 128 + 256；`s_barrier_signal` 26；`.long 0xd031` 40 = 38 load + 2 store；buffer_load 0。metadata：`.vgpr_count 1024`、`.group_segment_fixed_size 327680`、`.wavefront_size 32`、`.max_flat_workgroup_size 1024`。

---

## 2. 正确性：514 条 fp32 atomic 的数值账

**成立，而且比我们现在的做法更准。**

算术（prod b=4 sq=skv=8192 d=128 causal）：
- 一个 dq_acc 元素收到的 atomic 数 n = ⌊i/128⌋+1 ∈ [1,64]，均值 32.5，全次发射 **4,362,076,160** 条（我独立枚举 8320 tile × 4096 dword × 128 个 (b,hq)，与反汇编的 514 静态条数自洽）。
- fp32 u = 2⁻²⁴ = 5.96e-8。任意求和次序的差 ≤ 2·63·u = 7.51e-6 → **102.5 dB 最坏**；随机化次序 √63·u = 4.73e-7 → **126.5 dB 期望**。
- **我们今天的确定性链更差**：`kernels.py:645 KV_STEP = 32`，k_dq 的 dQ 累加链深 8192/32 = **256**，最坏 2·255·u = 3.04e-5 → **90.3 dB**。aiter 的两级树（寄存器内 4 步 + 64 次 atomic）最坏界紧 **12.1 dB**。

**对照的天花板不是 59 dB，是 52.9 dB。**（此前两个角度在这里差了一个 binade：bf16 有 8 位有效位、7 位存储，ulp 相对间距 2⁻⁷，u = 2⁻⁸ = 3.906e-3，RTNE 的 RMS 相对误差 u/√3 = 2.2553e-3 → **52.94 dB**。）这个数的可信度由树内实测背书：我们自己的 dq/dk/dv 实测 52.56/52.60/52.71 dB —— **正好坐在 bf16 量化地板上**。也就是说这三个数已经被输出 dtype 钉死，累加方式在其中的份额可以忽略。

**50 dB 闸门：过，且差距是四个数量级。** 树内有一个完美受控对，我逐字节复核过：

```
output/0915__opt/sweeps/r1_twokernel_bwd.jsonl:1  bwd_path=twokernel(forced)  确定性两内核
  out 53.73858945795214  dq 52.20623686628632  dk 52.29367998505609  dv 52.69978426098354
output/0915__opt/sweeps/r2_fwd.jsonl:1            bwd_path=auto               aiter ASM
  out 53.73858945795214  dq 52.20623084183665  dk 50.55807096477905  dv 50.83765533233197
```

同 shape（b=4 s=4096 causal d=128）、同 session、forward 逐位相同到 14 位。**Δdq = −6.02e-6 dB。Δdk = −1.736 dB。Δdv = −1.862 dB。**

这把"精度风险在哪"彻底翻了个面：**atomic 是免费的；aiter 真正掉精度的是 dK/dV，机制是 grid.y = nhead_q 展开到 Hq 后 host 侧求和（每个 g=4 的 partial 先被舍成 bf16 再相加），代价 1.7–1.9 dB。** `beat/kernel.yaml:149` 记为 50.2–51.0 dB 对 50 dB 闸门（该行是散文，树内无对应 gate log —— 但受控对的 50.558/50.838 独立佐证了同一量级）。

**run-to-run（唯一真正让掉的东西）：** `output/0925__flydsl/gate-change/atomic_spread.py` 本机 6 次实测 —— aiter 的 dk/dv **逐位相同**（SQNR inf，最大相对偏差 0.0），dq 是 fast 113.0 dB / prod 98.0 dB。`op/validation.py:199` 的 `DQ_STABILITY_DB = 70.0` 坐在 98 dB 下 28 dB、50 dB 上 20 dB，位置是对的。

**必须对外说清的两件事：**
- 买到的是"精度不变"，不是"结果不变"。按 98 dB 反推，两次运行间约 1e-4 量级的元素会差 1 个 bf16 ulp。不要说"通常逐位相同"。
- 语料 `hd128.md:255` 那条 "45.45 → 46.97 dB" 的 1.5 dB 不能搬过来。`hd128.md:878-884` 明写那条路径用的是 **`buffer_atomic_pk_add_bf16`**、且**目标不是 dQ**。bf16 atomic 在 n=64 下 RMS ≈ √64·2⁻⁸/√12 ≈ 9.0e-3 ≈ 40.9 dB，那才是 1.5 dB 的来源。aiter gfx1250 强制 fp32（`mha_bwd.cu:465-467` "gfx1250 only support atomic32=1"），噪声低约 500 倍。**照抄 packed-bf16 那一版会直接掉穿 50 dB 闸门。**

顺带：aiter 自己把这条路与确定性对立 —— `mha_bwd.cu:399-404`，只要调用方要 `is_deterministic` 就 `return -1` 退回 CK。它从未声称 bitwise 可复现。

---

## 3. 能不能建：阻塞项排序

### B0（新，最重）：FlyDSL 能发的那条 atomic 是 CU 作用域，aiter 的是 DEV 作用域

本会话已经验证过"FlyDSL 0.3.2 能发 fp32 buffer atomic"（`hint.md:3256`，`output/0925__flydsl/gate-change/atomic_probe2.py`，COMPILE_ONLY）。但那份 ISA 的原文是：

```
atomic_probe_isa.s:16    buffer_atomic_add_f32 v1, v0, s[0:3], null offen
```

aiter 的原文是：

```
buffer_atomic_add_f32 v204, v20, s[52:55], null offen scope:SCOPE_DEV
   // 编码 C415807C 408868CC 00000014 -> byte[6] = 0x88
```

我用 `llvm-mc -mcpu=gfx1250 -show-encoding` 打穿了这一位：**无修饰符 ≡ `scope:SCOPE_CU`，byte[6] = 0x80；SCOPE_SE = 0x84；SCOPE_DEV = 0x88；SCOPE_SYS = 0x8c。** 也就是说探针发出的和 aiter 发出的**不是同一条指令**。

为什么这是硬阻塞：`kernels.py:681-683` 记录 `/sys/class/kfd/kfd/topology/nodes/2/properties` 报这张 gfx1250 `num_xcc 8`。dQ 的同一个地址由分布在不同 XCD 上的 workgroup 累加。一条 CU 作用域的 atomic 在语义上不保证跨 CU/XCD 可见 —— 失效模式是**静默丢更新**，不是变慢。语料里这一类已经有前科：`hd128.md:888-890` 对同族 atomic 写着 "an `sc0` cache-scope bit on this atomic is a **silent wrong answer**, not a slowdown"。

而 `expr/rocdl/universal.py:139-146` 的 `BufferAtomic(atomic_op, val_type)` **签名里根本没有 scope 参数**（它构造的是 `CopyOpCDNA3BufferAtomicType`），所以 copy-atom 这条已验证可用的路子没有旋钮。

**绕路存在，两条，都要在开工前用 COMPILE_ONLY 验：**

1. `rocdl.raw_ptr_buffer_atomic_fadd(vdata, rsrc, offset, soffset, aux=16)`。我把 aux 枚举了 0..31 编译，映射是确定的：`aux=0 → 无修饰(CU)`，`aux=8 → SCOPE_SE`，**`aux=16 → scope:SCOPE_DEV`**（无 TH 位，与 aiter 逐位一致），`aux=24 → SCOPE_SYS`；低 3 位是 TH（`aux=1` 会变成 `TH_ATOMIC_RETURN`，**绝对不要**）。`expr/rocdl/__init__.py:663-674` 确实把 int 型 `aux` 转成 IntegerAttr 传下去。**代价：** `hint.md:3250-3253` 记录这条路"cost one failed build" —— 它要 addrspace(8) 指针，拒绝 `fx.get_iter(make_buffer_tensor(...))` 产出的 `fly.ptr<f32, #fly_rocdl.buffer_desc>`。这是必须先解决的类型问题。
2. `fx.UniversalAtomicAdd(fx.Float32, SyncScope.Agent)`（`expr/primitive.py:208-218`，`expr/rocdl/enum.py:25` `Agent = "agent"`）。我用 ROCm clang 23 验过 IR 层：`atomicrmw fadd ptr addrspace(1) ... syncscope("agent") monotonic` 在 gfx1250 上选出 **`global_atomic_add_f32 v1, v0, s[4:5] scope:SCOPE_DEV`**，无 RETURN、**零条 cmpswap**（system scope 亦然）。代价是 global 而非 buffer：每 lane 要 64 位地址，多花 VGPR，且失去 descriptor。

**杀死闸门：** 如果两条路都拿不到 `scope:SCOPE_DEV`，融合 dQ 这条路线就此终止 —— 不要用 CU 作用域的 atomic 去测性能，那个数字是没有意义的，而且会在闸门前静默通过。

### B1：4 波是强制的，且我们树内没有先例

单 wave32 做 BLOCK_KV=128 需要 2×128×128/32 = **1024 VGPR** 只放 dK/dV 累加器，整个寄存器文件。必须 4 波、每波 32 kv。我们现有的 attention kernel 全是 `known_block_size=[32,1,1]`（`kernels.py:602/611/997/1006`）。三个具体的改动点，其中一个不报错：

- **静默损坏：** `kernels.py:173` 与 `:668` 都是 `lane = fx.Int32(fx.thread_idx.x)`。block 变 (128,1,1) 后它是 0..127，而下游全部的 `row = lane % 16` / `half = lane // 16`（`:189/:190/:709/:710`）是 WMMA fragment 寻址和 LDS offset。不 fault、不报错、结果错。必须换成 `fx.lane_id()`（`expr/gpu.py:67`）加一个独立的 `wave` 变量。
- **两组 barrier 必须加回：** `kernels.py:453` 和 `:492` 的删除理由原文是 "block=(32,1,1): the workgroup is ONE 32-lane wave, so s_barrier is semantically a no-op"，这个前提被 4 波证伪。:453 守 LDS 的 P/dS 写对 ds_load_tr16 读的跨 wave RAW，:492 守下一轮覆盖的 WAR。代价是 r8.i1.g23+g24 拿到的那部分收益要重新调（注释说删它的收益来自去掉保守的 ALL-COUNTER waitcnt）。aiter 用的是 26 组 `s_barrier_signal`/`s_barrier_wait` 分离 barrier，FlyDSL 有（`expr/rocdl/cluster.py`、`_mlir/dialects/_rocdl_ops_gen.py`）。
- **LDS 推导整段作废：** `S_ROW_B = BLOCK_KV*2 + 16`（`kernels.py:54`）在 BLOCK_KV=128 时变成 272，与 `X_ROW_B = D*2+16 = 272`（`:56`）数值相撞，`:226-243` 那整段关于 `LDS_SEG=65536`、"adding exactly 65536 flips bit 16"、以及 70656 B / 4-workgroup 占用率档位的手推全部需要重做，不是改几个常数。新的 LDS = 65536 + 2×32×272 = **82944 B**，`smem_allocator.py:248` 的 gfx1250 上限是 327680（正好等于 aiter 的 `group_segment_fixed_size`），所以容量不是阻塞。

### B2：BLOCK_KV 必须一起抬到 128，否则 atomic 流量 4 倍

dQ atomic 流量只跟 BLOCK_KV 走，与 BLOCK_Q 无关。我枚举 prod：

| BLOCK_KV | atomic dword | 流量 |
|---|--:|--:|
| 128（aiter） | 4,362,076,160 | **16.25 GiB / 17.45 GB** |
| 64 | 8,657,043,456 | 32.25 GiB |
| 32（我们今天的分块） | 17,246,978,048 | **64.25 GiB / 68.99 GB** |

aiter 的 17.45 GB 在 7.236 ms 里吞下去了（= 2.41 TB/s payload，还没算 RMW 读），这是一个存在性证明。68.99 GB 要 9.28 TB/s，对着树内记的 6.46 TB/s HBM 顶（`rounds/002/findings_snapshot/facts.md:115`）—— 强烈指示不可行，但因为 L2 会吸收一部分，这是 ARGUED 不是证明。**结论：BLOCK_KV=32 的融合虽然 WMMA 更省（1.0038× 算法当量，优于 aiter 的 1.0155×），但走不通。BLOCK_KV=128 是可行性前提，不是调优参数。**

另外：dQ 在 4 个波之间必须**切分**而不是复制。aiter 按 d 切（514 条 atomic 只用 4 个 voffset 寄存器 v56/v60/v64/v68，各 128 次，`v0*4` 跨 512 B = 完整的 d=128 行；32 instr × 32 lane × 4 wave = 4096 = 32q×128d，每地址恰好一次）。若四个波各打各的，流量 ×4 直接回到 B2 的死区。

### B3：不是阻塞但必须记账

新增 `dq_acc` fp32 [b,hq,sq,d] = **512 MiB** 常驻显存，且每次调用必须清零（`mha_bwd.cu:617-618` `workspace_alloc(..., zero_init=true)`），外加一个 fp32→bf16 转换 kernel。按 aiter 自己的实测份额：`beat/kernel.yaml:191` dq-acc-zero 1.22% + `:180` dq-convert 1.18% = **2.40% = 0.184 ms**，在任何收益之前先付掉。清零可以折进 k_delta（今天 0.064 ms，`rounds/017/1-profiling/kernel.yaml:113`）省一次 dispatch。

### B4：TDM 先不要碰

aiter 的 buffer_load=0 / 38 条 TDM 是真的，FlyDSL 0.3.2 有全套（`tdm_ops.py:1118/1149/1176`）。但 `~/.claude/skills/gfx1250-card-safety` 里 TDM 有两条挂卡路径（错误 extent 的 descriptor 等一个永不退休的计数器；gather 不做 addr64 carry-safe 更新在大张量上硬挂），恢复要人工 AC 循环。**不要和 4 波改动捆在一起**，否则挂了归因不了。

---

## 4. 值多少、代价是什么；GQA 寄存器归约能不能保住

**能保住，而且它是这次移植唯一的余量来源。**

GQA 的寄存器内归约是 q 循环的性质，与 kv 分块和波数正交。我们的 `_body` 已经带 `gh`（q-head）参数在循环内遍历 G=4 个 q head（`kernels.py:346`），grid.y = hkv（`impl.py:153` `_wgs = (skv // _k.BLOCK_KV) * hkv * b`，`:170` `dk_o = torch.empty((b, skv, hkv, d))`）。4 个波沿 kv 切分与之不冲突。**不要连 aiter 的 grid.y = nhead_q 一起抄** —— 那同时是它 0.2225 ms 的 gqa-reduce 成本（`beat/kernel.yaml:128` 111269 ns × 2 call）和它 1.7–1.9 dB 的 dk/dv 精度损失，两件事同源。

**平价条件（与内核份额无关的形式）：**

```
bar 716 TF/s = 7.6791 ms
  − 新增 dq-acc-zero + dq-convert (2.40%)   0.1843 ms
  − k_delta（我们已有）                      0.0641 ms
  = 融合主内核预算                           7.4308 ms
  → 需要 issued 751.4 TF/s（BLOCK_KV=128 口径）
  = aiter 主内核自己 771.6 TF/s 的 97.4%
```

aiter 主内核的 771.6 是**直接读出来的**（`beat/kernel.yaml:89` `ns_per_call: {prod: 7236356}`，issued 5.5835e12），不是从份额反推的。

**但这个 97.4% 的余量（2.6%）小于它自身输入的 session 漂移。** 同一批预编译 .co：`rounds/006/gate.log` 记 716.92 TF/s / 7.669 ms，`beat/kernel.yaml:24` 记 710.54 TF/s / 7.738 ms —— **相差 0.90%**。按后者重算，预算 7.4884 ms，需要 745.6 TF/s = aiter 的 96.6%，余量 3.4%。而本 operator 自测的同 session 底噪是 0.40–0.66%。**所以余量是 2.6%–3.4%，不是一个可以放心的数。**

**落点区间（按融合体的 issued 速率外推，BLOCK_KV=32 口径 5.519e12 issued）：**

| 融合体达到的 issued 速率 | 总时间 | 对 bar |
|---|--:|--:|
| 624.1 TF/s（今天 k_dkdv 的速率，零吸收） | 9.09 ms | **0.845×** |
| 719.9 TF/s（我们整 op 的速率） | 7.91 ms | 0.970× |
| 771.6 TF/s（aiter 主内核的速率） | 7.40 ms | 1.038× |
| dQ GEMM 完全被气泡吸收（时间 == 今天的 k_dkdv） | 7.32 ms | **1.049×** |

（今天是 0.714×。k_dkdv/k_dq 的份额 62.93%/32.13% 来自 `rounds/017/1-profiling/kernel.yaml:40,:56`，那次 profile 的 vgpr 列记 368/480，而冠军是 736/960 —— **这个份额需要重测一次**，落点区间对它极敏感，平价条件本身不敏感。）

**关键洞察：**"我们的调度效率已等于手写 ASM（719.9 vs 727.1）"是整 op 混合下的说法，它掩盖了决定成败的那个数。分内核看：**k_dkdv 624.1 TF/s、k_dq 920.3 TF/s、aiter pssk 771.6 TF/s**。融合体的 body 继承的是 k_dkdv，那一侧对 ASM 有 **19% 的逆差**。这既是风险，也是下面第 6 节那个更便宜方案的全部上限。

---

## 5. 有序计划（每步独立可测，含零卡时筛与早杀闸门）

**G0 — 零卡时，COMPILE_ONLY，一票否决。** 让 FlyDSL 发出一条 `scope:SCOPE_DEV` 的 fp32 atomic。基于 `output/0925__flydsl/gate-change/atomic_probe2.py` 改两版：(a) `raw_ptr_buffer_atomic_fadd(..., aux=16)`，先解决 addrspace(8) 的类型问题；(b) `fx.UniversalAtomicAdd(fx.Float32, fx.rocdl.SyncScope.Agent)`。判读：`21_final_isa.s` 里必须出现 `scope:SCOPE_DEV`，且**不得**出现 `TH_ATOMIC_RETURN`、不得出现任何 `cmpswap`。**两条都失败 → 融合 dQ 路线终止，去做第 6 节。** 必须在任何性能测量之前做：CU 作用域的 atomic 会静默通过所有闸门。

**G1 — 零卡时，COMPILE_ONLY ×2。** (a) 把 `kernels.py:453/:492` 的 barrier 加回来，保持 block=(32,1,1)，diff ISA、重算 cover distance —— 在不改波数的前提下给"barrier 砍 prefetch cover"这个风险定价。(b) 一个最小 4 波 k_dkdv（BLOCK_KV=128、dkdv-only、无 dQ、无 atomic），读 `.vgpr_spill_count` 和 `group_segment_fixed_size`。判读：spill == 0、VGPR ≤ 1024、LDS ≈ 82944（若 ≈ 4×70656 说明 Q/dO 被复制了四份、共享没发生）。约 4 分钟一次编译。

**G2 — 1 个 benchmark 槽位。** 建第 6 节那个**不改契约的 BLOCK_KV=128 + 4 波 k_dkdv**。这一步本身就有 +13.6% 的上限，同时是整个 4 波骨架的可行性证明。闸门：dk/dv 200 次自洽逐位（注意：**不会**与冠军逐位相同，kv 收缩顺序变了，别把闸门写成对冠军逐位）+ 50 dB；时间落在 0.40–0.66% 同 session 底噪内即算通过（工作量本身涨了 1.17%）。

**G3 — 1 个槽位，唯一真正未定价的量。** dQ atomic 流量的可承受性。写一个只做 dQ atomic、不含任何 WMMA 的 micro-kernel，按 BLOCK_KV ∈ {128, 64, 32} 发出 16.25 / 32.25 / 64.25 GiB 的 `scope:SCOPE_DEV` fp32 atomic，测 wall time。不需要正确性，风险极低，可无人值守。判据不是"会不会成为瓶颈"，而是"BLOCK_KV=128 一档的 atomic 时间是否吃掉 7.43 ms 预算的一大块"。

**G4 — 1 个槽位。** 融合 + atomic。只有 G0/G2/G3 都通过才做。设计约束写死：grid = (kv_tile, **Hkv**, B)；BLOCK_KV=128；dQ 按 d 在 4 个波之间**切分**不复制；照 aiter 做近/远 kv block 配对，两个 partial 先在寄存器里合再打一次 atomic（可省 25.4% 流量）。闸门用树里已改好的 `op/validation.py`：dk/dv 逐位 ×200、dq run-to-run ≥ 70 dB、三者 ≥ 50 dB。

**并行零卡时：** dq_acc 清零折进 k_delta；修正 `beat/kernel.yaml:113` 的 "BLOCK_KV = 256 at both"（真值 128）。

**闸门的一个缺口，必须在任何 atomic dQ 出货前补上：** `op/validation.py:287` 调的是 `check_determinism(impl_dir)`，用默认 `shape="fast"` 的单一配置。而 docstring 自己承认"What caught it was that the check ran **across configurations**"（dQ 曾静默跌到 8–14 dB）。70 dB 地板保住了幅度那一半，没保住 sweep 那一半 —— 而且 fast 跑的是最浅的累加（aiter 自己 fast 113.0 dB vs prod 98.0 dB，15 dB 的深度效应 fast-only 闸门看不见）。改成扫 (q_split, BLOCK_KV) × {fast, prod}。

**另一个 atomic 特有的新失效模式，两个闸门都看不见：** `kernels.py:911-913` 记录 k_dq 的 descriptor 带平坦的 1 GiB `num_records`（"no g07 clamp here"），越界地址被夹到一个**活的**界内元素，理由是"the loaded value is dead"。这个习惯对读是安全的，对 **atomic add 是灾难** —— 副作用落在目的地址上，会永久累加进一个正确的元素。它是确定性的（逐位可复现），所以 70 dB 的 run-to-run 地板读不到；稀疏的话也吃得进 52.56→50 dB 的 2.56 dB 余量。**在任何 atomic 发进 k_dq 的 descriptor 之前，先给它真实 extent（g07）。** 补充理由：`expr/rocdl/universal.py:242-248` 的 OOB_SELECT 只在 `is_rdna_arch(arch)` 时设置，而 `runtime/device.py:95-97` 只匹配 gfx10*/gfx11*/gfx120* —— "gfx1250".startswith("gfx120") 为 False，gfx1250 拿不到。不要指望 num_records 兜底，在 kernel 里谓词化每一条 atomic。

---

## 6. 更便宜的部分方案：BLOCK_KV=128 + 4 波，**不融合、不改契约**

**有，而且应该先建这个。**

只把 k_dkdv 从「单波 / BLOCK_KV=32」改成「4 波 / BLOCK_KV=128」，k_dq 原样不动，dQ 还是确定性的、没有 atomic、没有 dq_acc、没有转换 kernel、没有 512 MiB 显存、契约一个字不改。

WMMA 会**略微变多**（causal 粒度变粗）：8320 tile × 256 = 2,129,920 对今天的 32896 × 64 = 2,105,344，**+1.17%**。收益不在 WMMA，在 operand 流量 —— Q/dO 每 128 个 kv 行 staging 一次而不是每 32 行一次，全局取数降 4 倍，这正是 aiter 的 buffer_load=0 想解决的同一件事。

落点：

| k_dkdv 达到的 issued 速率 | 总时间 | 对 bar |
|---|--:|--:|
| 624.1（今天，零收益） | 10.83 ms | 0.709×（略退） |
| 700 | 10.06 ms | 0.764× |
| 771.6（aiter 主内核的速率） | 9.47 ms | **0.811×** |

今天是 0.714×。**上限 +13.6%，不动契约、不碰 atomic、不碰 TDM。**

它同时是融合版的先决条件：4 波骨架、lane/wave 拆分、两组 barrier 回归、LDS 段分离重推 —— 这些成本融合版无论如何都要付。先单独付一次、单独测一次，比和 atomic + dq_acc + 512 MiB workspace + scope 问题捆在一起测要便宜得多，且失败时归因清晰。

---

## 未验证清单（不要当已知）

1. **gfx1250 上 CU 作用域 atomic 到底会不会丢更新。** 我 ENUMERATE 的是编码差异（0x80 vs 0x88）、FlyDSL 的 `BufferAtomic` 无 scope 参数、KFD 报 num_xcc 8、以及 `hd128.md:888-890` 的同类前科。"CU 作用域在 8-XCD 上不足以跨 workgroup 累加"是 ARGUED，我没查 gfx1250 ISA 手册。但方向只有一个安全解：发 SCOPE_DEV，与 aiter 一致。
2. **FlyDSL 能否真的发出 scope:SCOPE_DEV。** 两条绕路我只验到 LLVM IR 层（ROCm clang 23）和 API 签名层，**没有跑过 FlyDSL 的 COMPILE_ONLY**。这是 G0。
3. **`UniversalAtomicAdd` 在 FlyDSL 里落到哪个 address space。** 我的 clang 探针用的是 addrspace(1)；FlyDSL 的 lowering 未验。
4. **16.25 GiB device-scope atomic 在这张卡上的实际时间。** aiter 吞下去了是存在性证明，我们在 BLOCK_KV=128 上的需求与它完全相同，但这是推理不是测量。BLOCK_KV < 128 纯属未知。
5. **FlyDSL 能否生成 4 波 + barrier + 跨 wave LDS 交换 + 每波近 740 VGPR 常驻累加器的代码，且 spill=0。** aiter 是手写 ASM 做到 scratch=0，编译器路径未必。G1。
6. **融合 body 能吸收多少新增 WMMA。** `hint.md` 里 h28 的假设 1 从写下那天起就标着 "not measured"。落点区间 0.845×–1.049× 全靠它。
7. **k_dkdv/k_dq 的 62.93%/32.13% 份额来自第 17 轮、vgpr 368 的 build**，冠军是 736。落点区间对它极敏感（份额 60% → 容易，70% → 不可能），平价条件本身不敏感。重测一次 rocprofv3 dispatch 即可，不占 benchmark 槽位。
8. **aiter dk/dv 的 50.2–51.0 dB** 只在 `beat/kernel.yaml:149` 出现一次，树内无 gate log。但受控对的 50.558/50.838 是独立实测，所以"aiter dk/dv 比我们差 1.7–1.9 dB"这个结论不依赖那行散文；"只剩 0.2–1.0 dB 余量"依赖。
9. **fast shape 的落点没算。** 融合后 fast 的 grid = (1024/128/2)×2×1 = **8 个 workgroup** 对 256 CU（`pool.md:72` 已记"fast 形状饿死机器：128 个 workgroup 对 256 CU"，从 128 掉到 8 是再塌 16 倍）。fast 今天 1.099× 正在贡献 geomean，split-K 兄弟不是"build surface 翻倍"的问题，是 fast 能不能活的问题。
10. **proxy shape 没算。**

---

## 一句话给 operator

aiter 的方法是对的、精度是免费的（6e-6 dB，受控对实测）、而且它的两级求和树比我们现在的 256 深链更准 12 dB；**但"照抄"目前卡在一个字节上** —— FlyDSL 发的 atomic 是 `SCOPE_CU`，aiter 发的是 `SCOPE_DEV`，在 8-XCD 上这是静默错答而不是变慢。先花零卡时把这个字节拿下（G0），然后建那个不改契约的 BLOCK_KV=128 + 4 波版本（上限 +13.6%，是融合版的先决条件），再谈融合 —— 融合的诚实落点是 0.845×–1.049×，余量 2.6%–3.4%，而同一批 .co 的 session 漂移就有 0.90%。

**相关绝对路径：**
- `/tmp/aiter_dis/bwd_hd128_bf16_causal_br_a32_pssk.s`
- `/home/lihuzhan/code/aiter-src/csrc/cpp_itfs/mha_bwd.cu`、`csrc/include/mha_bwd.h`、`hsa/gfx1250/fmha_v3_bwd/fmha_bwd_dqdkdv.csv`
- `/home/lihuzhan/.local/flydsl032/flydsl/expr/rocdl/universal.py`（:139-146 无 scope 参数）、`expr/rocdl/__init__.py`（:663 aux）、`expr/primitive.py`（:208-218）、`expr/rocdl/enum.py`（:25）、`runtime/device.py`（:95-97, :102-114）、`utils/smem_allocator.py`（:248）
- `/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0925__flydsl/gate-change/atomic_probe2.py`、`atomic_probe_isa.s`、`atomic_spread.py`
- `/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0915__opt/sweeps/r1_twokernel_bwd.jsonl`、`r2_fwd.jsonl`
- `/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-flydsl-attn-bwd-20260917-115934/job_context/op/current/kernels.py`、`impl.py`、`op/validation.py`、`profiling/beat/kernel.yaml`、`rounds/017/1-profiling/kernel.yaml`
- `/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/knowledge/backends/flydsl/attention/recipes/hd128.md`