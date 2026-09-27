# FUSED5：k_dkdv 内融合 dQ（fp32 SCOPE_DEV 原子），去掉 k_dq

2026-09-27。全程 CPU + COMPILE_ONLY（容器 fa-repro，flydsl 0.3.4.1，`ARCH=FLYDSL_GPU_ARCH=gfx1250`，
`HIP_VISIBLE_DEVICES=` 置空）。**零卡时，没有任何 kernel 上过卡。**
工作目录 `output/0927__flydsl/bwd/fused5/`，下文简写 `F5/`。

## 0. 结论

1. **能编出来，没有 spill。** `k_dkdv_f5`（K^T 每轮从 LDS 读，下称 f5-lds）：**VGPR 904 → 936，spill 0，scratch 0，
   LDS 70656 不变**（K 的拷贝放进段 0 的空闲区），1 wave/SIMD、4 WG/CU 这一档不变。
   全部 256 条 atomic 都是 `global_atomic_add_f32 ... scope:SCOPE_DEV`，没有 `TH_ATOMIC_RETURN`，也没有 cmpswap。
   K^T 常驻 VGPR 的版本（f5-res）**VGPR 1024，spill 29，scratch 120 B → 淘汰**（规则 4：有 spill 就会挂卡）。
2. **每个 tile 多出的开销**：每个 (32q × 32kv × 1 个 q head) 迭代多 **16 条 v_wmma**（64 → 80，+25%），
   多 **128 条 `global_atomic_add_f32`**（分两个 64 条的 `s_clause`，每条覆盖 2 行 q × 64 B 连续 d），
   多 32 条 `ds_load_tr16_b128`，其余是地址算术、`s_set_vgpr_msb` 和 wait。
   full body 从 778 条指令涨到 **1095 条（+40.7%）**。按 prefetch 落点到下一轮 `s_wait_loadcnt 0x0` 的距离算，
   cover 为 603 → 602 条（静态指标，只用来筛，不能拿来排名）。
3. **时间估算**：matrix 部分按实测的 issued-matrix 速率（~720 TF/s）算是 **7.665 ms**。
   加上 k_delta 0.064 ms 和 dq_acc 清零 + convert 0.184 ms，总计 **7.913 ms + E_atom**。
   - E_atom = 0（原子全部被计算盖住）：**0.970× bar，1.36× 冠军**。
   - 与冠军打平的条件：E_atom ≤ **2.84 ms**。若按 k_dkdv 今天自身的 624 TF/s 算，这个上限只有 1.66 ms。
   - **在这个 1-wave、BLOCK_KV=32 的形态下达不到平价**：即使 E=0，也要求 issued 速率到 743 TF/s。
4. **真正决定这件事的未知量是原子吞吐。** BLOCK_KV=32 的原子载荷是 **68.99 GB**（aiter 的 3.95 倍）。
   要被计算完全盖住，需要在 7.67 ms 内并发跑到 **≥ 9.0 TB/s**；aiter 实证过的只有 ≥ 2.41 TB/s（17.45 GB）。
   如果原子与计算串行，**R < ~25 TB/s 就会输给冠军**。所以上卡的第一件事是 P2 原子吞吐微基准，已经写好并通过编译。
5. **go/no-go 见 §6**。一句话：P1（算力代价）和 P2（原子吞吐）各占一个槽位，两者之和 ≤ 10.0 ms 就直接 GO；
   两者取大 > 10.0 ms 就判 1-wave 融合死亡，改走 4-wave BLOCK_KV=128（原子流量降到 1/4 = aiter 形态），
   用 P2 的 KVG=4 档给它把关。

## 1. 设计

### 1.1 形态（相对冠军 r20 的最小改动）

- 不改 grid、BLOCK_KV（32）、波数（1）和循环结构（mask/full 拆分、g62 两级 prefetch、GQA 在寄存器内归约）。
- `_body` 每个 16-query 半块 `hh` 算完两个 kv 子块 `kh` 的 dS（bf16）以后，追加：
  - **A 操作数白给**。kh=0 的 8 个 dS 在 lane 内对应 kv `half*8+si`，kh=1 对应 `16+half*8+si`。
    两者直接拼接，正好是 16×32 A-fragment 的列顺序（和 k_dq 的 `a_ds` 是同一个 idiom）。
  - **B 操作数 = K^T**。prologue 用 kf 已有的寄存器把 K 的 32 行写进 LDS `lds_k = lds_q + 32*272`（偏移 17408–26112，
    落在段 0，总分配仍是 70656 B）。body 里每个 dtile 用两条 `ds_load_tr16_b128` 读出 K^T，这是 k_dq 的 idiom。
    这里只有一个 wave，LDS 按 wave 顺序执行，不需要 barrier。
  - **8 条独立的 WMMA**（每个 d tile 一条，K=32 一次收缩完，C=0），得到 dQ[16q × 128d] 的一个 partial。
  - **64 次 fp32 原子/lane**。WMMA D 的布局是 lane%16 = d 列、half*8+si = q 行，所以一条原子指令覆盖 2 行 q × 16 个连续 d（2 × 64 B）。
    反过来的朝向 dQ^T = K^T·dS^T 会让 32 个 lane 落在 16 个不同的行上，所以没选它。
- dq_acc 为 fp32 `[B, Sq, Hq, D]`，由 host 端清零。convert 直接复用 **`k_redsp_q`，nsp=1**，它本身就是 fp32→bf16 的扁平 pass。
  编译结果：41 VGPR，spill 0。
- 原子写法：`fx.UniversalAtomicAdd(fx.Float32, rocdl.SyncScope.Agent)` + 普通 global 指针的 view
  （`make_view(get_iter(DQA), (1<<30,1))`）。**不走 `make_buffer_tensor`**：`BufferAtomicAdd` 是 SCOPE_CU，
  buffer 版的 UniversalAtomic 会 abort（h42）。
  ISA 为 `global_atomic_add_f32 vOff, vData, s[44:45] offset:64k scale_offset scope:SCOPE_DEV`。
  8 个地址 VGPR（每个 si 一个）被 8 个 dtile 复用，dtile 的偏移走立即数 `offset:64*dtile`。
- fast shape 走 `k_dkdv_sp_f5`（q 轴 split-K）。各个 split 切分的是 q 范围，所以原子累加天然正确。
  编译结果：936 VGPR，spill 0。
- `impl.py`：分配 dq_acc（`torch.zeros`），发 `launch_dkdv_f5` / `launch_dkdv_sp_f5`，再发 `cvt_dq`。**k_dq / k_dq_sp 不再被调用。**

### 1.2 精度与契约

- **dk/dv 的算术一条没动**：同一个 IR 序列，只多出旁路。所以预期对冠军**逐位相同**，而且 200 次运行逐位自洽。
  **这一点要上卡验证**。
- **dq**：每个元素收到的原子次数 n = 覆盖它的 kv tile 数，prod 最多 256（aiter 是 64）。
  fp32 任意求和次序的最坏界是 2·255·u → **90.3 dB**，期望值 √255·u → ~120 dB，都远高于 70 dB 的 run-to-run 地板。
  和 k_dq 相比：dS 同样先舍入到 bf16 再进 GEMM。区别在于 S 的 fp32 收缩顺序用的是 g61 的 chain-split，
  所以 dq 对冠军**不逐位相同**，但属于同一精度档（预期约 52.5 dB vs eager）。
- 规则 5 的界证明：`F5/bounds_proof.py` 在 CPU 上逐条复现 `_dkdv_impl` 的控制流。
  覆盖 prod、proxy、fast（nsp=1/2/16）以及 Sq≠Skv 的 UT 形状，全部 OK：
  - 索引上界 = size−1。prod 为 134,217,727，int32 安全。
  - 每个因果上有效的 (b, qh, q pair, kv tile) **恰好访问一次**，没有访问任何死 pair。
  - prod 每个 (b,qh) 有 32,896 次访问，与 h40 一致。

  原子全部位于真实迭代内。defect c 的越界预取只影响 load，不触发原子。
  非因果路径（causal=0）没有做证明，scored shapes 用不到它。

## 2. ISA 证据（COMPILE_ONLY，prod 形状）

| 构建 | VGPR | spill | scratch B | LDS B | full body 指令 | v_wmma | atomic | tr16 | s_wait_dscnt | 判定 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|
| 冠军 k_dkdv（`F5/base`） | **904** | 0 | 0 | 70656 | 778 | 64 | 0 | 40 | 3 | 参照 |
| 冠军 k_dq | 960 | 0 | 0 | 8704 | 741/810 | 96 | 0 | 16 | 7–8 | 被删除 |
| **f5-lds k_dkdv_f5** | **936** | **0** | 0 | 70656 | **1095** | 80 | 128 | 72 | 33 | **可建** |
| f5-lds k_dkdv_sp_f5（fast） | 936 | 0 | 0 | 70656 | 1093 | 80 | 128 | 72 | 33 | 可建 |
| f5-res（K^T 常驻） | 1024 | **29** | 120 | 70656 | 1455 | 80 | 128 | 40 | 9 | ❌ spill |
| P1 探针 f5-lds, F5_ATOM=0 | 996 | 0 | 0 | 70656 | 945 | 80 | 0 | 72 | 12 | 可建（探针） |
| P1 探针 f5-res, F5_ATOM=0 | 1024 | 24 | 100 | 70656 | 1186 | 80 | 0 | 40 | 3 | ❌ |
| cvt_dq（k_redsp_q nsp=1） | 41 | 0 | 0 | 0 | — | — | — | — | — | 可建 |

注：任务书里写的"740 VGPR"已经过时。g62（r20）之后冠军是 904，本次重编确认了这个数（`F5/isa/base/dkdv`）。
预算是 1024，f5-lds 还剩 88。

**full body 的差分（f5-lds 相对冠军）**：+128 `global_atomic_add_f32`，+57 `s_set_vgpr_msb`，+32 `ds_load_tr16_b128`，
+30 `s_wait_dscnt`，+16 `v_wmma`，其余约 +54 条标量/整数地址算术和 wait。
32 条 buffer_load_b128、4 条 b32 prefetch、`s_wait_loadcnt` 的数量（2 条）不变；body 内 `s_wait_storecnt` 为 0，也就是说原子不回等。
Masked body 从 659 条涨到 987 条。

两个静态风险点，都要靠卡上的数据回答，不能靠推理：
- **K^T 的 tr16 读是用前才发**。每条 dQ WMMA 前都有 `s_wait_dscnt`，第一个 hh 的 load→use 只有约 10 条指令，LDS 延迟是暴露的。
  调优旋钮：把 16 条 K^T 读提到 body 开头，代价是 +64 VGPR 的生命周期，936+64 已经贴着 1024，大概率会 spill。
- 每个 64 原子的 clause 之后都跟着 **`s_wait_xcnt 0x4` / `0x0`**，原因是紧接着的 tr16 读要复用原子的数据寄存器 v[386:393]。
  wave 会在这里等 64 条原子把数据交给内存管线（不是等原子完成）。这个等待有多贵，完全取决于原子发射背压，
  **正是 P2 要量的东西**。

**P1 探针的偏差**：F5_ATOM=0 把 16 个 WMMA 结果折进 dV 累加器 0（所以 dV 是故意算错的），
为此多出 64 条 `v_pk_add_f32`，VGPR 也比真实版多 60。它测的是"+16 WMMA +32 tr16 +64 pk_add"的代价，对真实融合 body 的算力部分略偏悲观。

## 3. 时间估算

FLOP 按 issued 计。每 (b,hq) 32,896 次迭代 × 128 = 4,210,688 次迭代，× 80 WMMA × 16384 = **5.519e12**
（算法当量 1.0038×，比 aiter 的 1.0155× 还省）。

| 项 | ms | 来源 |
|---|--:|---|
| k_dkdv_f5 matrix 部分 @ 720 TF/s | **7.665** | h40：整个 op 的 issued 速率 719.9 |
| 同上 @ 624 TF/s（今天 k_dkdv 自身的速率） | 8.843 | AITER-5GEMM-STUDY §4 |
| k_delta | 0.064 | r17 profile |
| dq_acc 清零 + convert | 0.184 | aiter 的实测份额 2.40%；按带宽上界算是 0.27 |
| **原子未被盖住的部分 E_atom** | **未知** | P2 |

原子载荷 R = 有效 payload 吞吐（fp32 lane-op × 4 B）：

| R (TB/s) | 68.99 GB 所需时间 | 全重叠时的总时间 | 全串行时的总时间 | 对比冠军 10.751 ms |
|--:|--:|--:|--:|---|
| 4 | 17.25 | 17.50 | 25.16 | 两种情况都输 |
| 6.46（HBM 顶） | 10.68 | 10.93 | 18.59 | 两种情况都输 |
| 9 | 7.67 | **7.91** | 15.58 | 只有重叠时能赢 |
| 12 | 5.75 | 7.91 | 13.66 | 只有重叠时能赢 |
| 20 | 3.45 | 7.91 | 11.36 | 只有重叠时能赢 |
| 30 | 2.30 | 7.91 | 10.21 | 两种情况都赢 |

几点读法：
- dq_acc 是 512 MiB，放不进 L2。原子在 L2 或内存侧执行的速率、以及 1024 个并发 WG 在 q 轴上错开时的命中率，全树都没有记录。
- 融合 kernel 的原子是 fire-and-forget（不带 RETURN，body 内不等 storecnt），**架构上允许重叠**。
  限制重叠的是内存系统的吞吐，以及 xcnt 和发射队列的背压。
- 4-wave BLOCK_KV=128 的形态（aiter）把载荷降到 **17.45 GB**，同样 7.67 ms 内只要 2.28 TB/s，
  而 aiter 已经在卡上跑到 ≥ 2.41 TB/s。**1-wave 这条路需要 aiter 实证速率的 3.7 倍。**

## 4. 需要先上卡的探针（按顺序，一个槽位一个进程一个 shape）

前置条件：卡空闲（`ls /sys/class/kfd/kfd/proc`、`rocm-smi --showpids` 只看到预期的 PID），fwd job 已经停下。
每个进程只跑一个 shape、一个模式（规则 3）。

### P2：fp32 SCOPE_DEV 原子吞吐微基准（**最先跑**，风险最低）

- 代码：`F5/atomprobe/k_atom.py`，驱动 `run_probe.py`，编译检查 `compile_probe.py`。三种模式都已经 COMPILE_OK。
- 形态：**完全复制 k_dkdv_f5 的地址流**，但不含 WMMA、不含 load。
  grid 同为 (hkv, kv tile, b)，longest-first；q-pair 在外、q-head 在内；每次迭代 2 个 64 原子的 clause；
  索引公式逐字相同（界证明共用 `bounds_proof.py`）；用 `ATOM_LDS=70656` 把占用率钉在 1 wave/SIMD，与融合 kernel 一致。
- 扫描点（prod：b4 s8192 hq32 hkv8）：

| 点 | ATOM_MODE | KVG | 载荷 | 用途 |
|---|---|--:|--:|---|
| a | atom | 1 | 68.99 GB（5.39e8 条 wave 指令） | 1-wave 融合所需 |
| b | atom | 4 | 17.45 GB（= aiter 逐字节） | 4-wave BLOCK_KV=128 所需 |
| c | atom | 2 | 34.63 GB | 中间点，看是否线性 |
| d | store | 1 | 同 a，`buffer_store_b32` | 写带宽对照：原子/普通写的比值 |
| e | none | 1 | 0 | 循环地板（约 10 条指令/迭代） |
| f | atom, `ATOM_LDS=0` | 1 | 同 a | 占用率不受限时的上限 |

- 输出：中位 ms、payload TB/s（`run_probe.py` 会打印）。每点 20 次，每次先 `zero_()`，palindromic 顺序，同一 session 里带时钟见证。
- 安全：没有 TDM、没有 barrier，也没有多 wave。原子地址已经在 CPU 上证明全部在界内（没有 descriptor 兜底，`global_atomic_*` 不做 clamp）。
  预计可以无人值守。它仍然是新的内存路径，所以按 `gfx1250-card-safety` 的规程先跑 a 点一次，确认返回后再扫其余点。

### P1：dQ-GEMM 代价（即 PARITY §5 预注册的那个探针）

- 构建：`F5_KT=lds F5_ATOM=0` 的 `k_dkdv_f5`（996 VGPR，spill 0）。PARITY 原来设想的是 K 常驻寄存器，但那个版本会 spill，只能用 LDS 版。
- 与冠军 k_dkdv 在同一 session 里做 palindromic A/B，只计 kernel 时间，得到 x = T_P1/T_dkdv0 − 1。
- 沿用预注册区间：**≤ +3.0% / +3.0–7.6% / > +7.6%**。本探针多出的 64 条 pk_add 会让 x 略偏高。

### P3：融合 kernel 本体计时（只有 §6 判为"再测"或"GO"时才跑）

- `F5_KT=lds F5_ATOM=1`：k_dkdv_f5 + 清零 + cvt，计时，不做正确性判定。回答的是原子和计算实际重叠了多少：
  E_atom = T_P3 − T_P1。

### P4：闸门（在 P3 之后）

- dk/dv：200 次运行逐位自洽，**并且对冠军逐位相同**（这是本设计的额外预言，不成立就说明哪里动了 dk/dv 的 IR）。
- dq：run-to-run ≥ 70 dB，三个输出对 eager 都 ≥ 50 dB。
- 按 AITER-5GEMM §5 的缺口，把 check_determinism 扩到 fast（sp_f5，nsp=16）和 prod 两个配置，不能只跑 fast。

## 5. 与冠军 k_dq 的账（为什么值得先测）

k_dq 今天约占 op 的 32%（r17 的份额，需要重测），约 3.4 ms。融合的净收益是
T_dq0 − [T_dkdv0·x + E_atom + 0.184]。按份额 63/32 估算：
- x = 25%（新增 WMMA 完全没被吸收）时，要求 E_atom < 1.6 ms；
- x = 10% 时，要求 E_atom < 2.6 ms。

## 6. go/no-go 判据（预注册，上卡前 commit）

记号：T_c = P1 实测的 k_dkdv_f5（无原子）时间；T_a = P2 a 点（KVG=1，钉住占用率）的原子时间；
冠军总时间按同一 session 实测，缺省 10.75 ms；10.0 ms = 冠军 − 0.25（delta+清零+cvt）− 0.5 余量。
0.5 ms 约 5%，是 0.66% 噪声地板的 7 倍以上。

| 条件 | 判定 | 动作 |
|---|---|---|
| **T_c + T_a ≤ 10.0 ms** | **GO** | 即使原子完全串行也能赢冠军 ≥ 5%。直接跑 P3、P4，按 1-wave 融合出一轮 arm |
| max(T_c, T_a) ≤ 10.0 < T_c + T_a | **再测** | 跑 P3 这一个槽位。T_P3 + 0.25 ≤ 10.0 就 GO，否则 NO-GO |
| **max(T_c, T_a) > 10.0 ms** | **NO-GO（1-wave）** | 1-wave BLOCK_KV=32 融合宣告死亡，写进 dead_ends。转去 4-wave BLOCK_KV=128，用下一行把关 |
| 4-wave 路线把关：P2 b 点（KVG=4）T_a,128 ≤ 3.0 ms，且 P1 的 x ≤ 7.6% | 4-wave 可继续 | 进入 AITER-5GEMM §5 的 G2/G4。barrier 那 45% 仍是独立风险（h51/h53） |
| T_a,128 > 5.0 ms | 整条融合路线 NO-GO | 17.45 GB 在这张卡上都盖不住，说明 aiter 的吞吐靠的是 FlyDSL 发不出来的东西（TDM/buffer 原子/调度）。上报 0.72× 为契约下的天花板 |

另有一条硬规则：任何上卡构建只要 `.vgpr_spill_count > 0` 或 scratch > 0，就不上卡。f5-res 已经按这条被杀。

## 7. 文件

| 文件 | 内容 |
|---|---|
| `F5/op/` | 冠军（`bwd341/op0341`，flydsl 0.3.4.1 pin）的副本加融合改动；`kernels.py` 有 `F5_KT`、`F5_ATOM` 两个环境变量旋钮 |
| `F5/fused5.patch` | 相对冠军的完整 diff（kernels.py + impl.py） |
| `F5/base/` | 未改动的冠军副本（ISA 对照用） |
| `F5/compile_f5.py`, `F5/run_variant.sh` | COMPILE_ONLY 驱动（dkdv_f5 / dkdv_sp_f5 / cvt_dq 以及冠军的各个 kernel） |
| `F5/body_stats.py` | 按循环体统计 ISA 指令 |
| `F5/bounds_proof.py` | CPU 上的原子索引界与覆盖证明 |
| `F5/isa/{base,lds,lds_sp,res,probe_lds,probe_res}/**/22_final_isa.s` | 本报告所有数字的来源 |
| `F5/atomprobe/{k_atom.py,run_probe.py,compile_probe.py,isa3_*}` | P2 微基准，已通过编译；`run_probe.py` 只能在卡空闲时运行 |

没有碰 op-evolve 的 artifacts，没有 git commit，也没有启动任何 GPU 进程。
