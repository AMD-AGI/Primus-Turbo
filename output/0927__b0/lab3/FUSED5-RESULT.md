# FUSED5 结果：P1 / P3 / 全 op A/B（h57 第 3 步，h58 判据）

2026-09-27，B0，GPU 3（fa-g3，`/tmp/b0-gpu3.lock`），**flydsl 0.3.2**（与 job 的 `current` 同一 pin，同一进程内 A/B）。
P2 的结果见同目录 `P2.md`。

## 0. 结论：**LOSS，两种形态都 NO-GO**

| 量 | 实测（prod，3 个进程，每个 n=101） | h58 门槛 | 判定 |
|---|---|---|---|
| P1：T_c（`k_dkdv_f5`，F5_ATOM=0） | **7.00 ms**，是冠军 k_dkdv（5.27–5.39 ms）的 **1.297 / 1.327 / 1.327 倍**，x = **+30~33%** | 1-wave：T_c ≤ 7.97 → "再测"；4-wave：x ≤ 7.6% | 1-wave 进入 P3；**4-wave 失败** |
| P3：T_P3（清零 + `k_dkdv_f5` 带原子 + cvt） | **21.82 / 21.84 / 21.70 ms**（只算 kernel 是 21.70 / 21.54 / 21.66） | T_P3 ≤ 7.72 ms | **1-wave NO-GO**（2.8 倍于门槛） |
| 全 op（job 的 `benchmark.py`，prod） | f5 **21.94 / 21.88 / 21.86 ms**，冠军 8.58 / 8.56 / 8.57 ms | 胜过冠军 | **0.391× / 0.391× / 0.392× 冠军**（慢 2.55 倍） |

- 按 h58 §6：1-wave BLOCK_KV=32 融合判死（P3 远超门槛）。4-wave BLOCK_KV=128 的把关条件是 "T_a,128 ≤ 3.0 **且** x ≤ 7.6%"，
  P2 那一半通过（2.46 ms），但 P1 的 x = 30–33%，**不通过**。P1 探针自带 +64 条 `v_pk_add_f32` / +60 VGPR 的悲观偏差，
  但是 30% 离 7.6% 差了 4 倍，这个偏差翻不过来。
- 因为输了，**没有 `fused5_tree/`，也不提议 must-hint**，只提议一条 dead-end hint（§5）。
- 正确性全部通过（§2）：设计是对的，输在速度。

## 1. 上卡前的检查

| 检查 | 结果 | 证据 |
|---|---|---|
| COMPILE_ONLY（0.3.2，fa-g3，不加 flock） | `k_dkdv_f5` **936 VGPR / spill 0 / scratch 0**，LDS 70656，wave32，256 条 `global_atomic_add_f32 scope:SCOPE_DEV`（两个 body 各 128），无 RETURN，无 cmpswap；`k_dkdv_sp_f5` 936/0/0；P1 探针 996/0/0；`cvt_dq` 41/0/0 | `isa032/{f5a1,f5a0,f5hc,f5p1hc}/**/21_final_isa.s`，`isa032/*.log` |
| 与 0.3.4.1 版本一致 | VGPR 和 spill 数与 FUSED5 §2 的数字一致 | 同上 |
| f5 模块中 `FUSE=False` 的 `k_dkdv` 对冠军 | **ISA 逐行相同**（0 行 diff） | `isa032/f5a1/dkdv` 对 `isa032/champ/dkdv` |
| 硬编码 knob 的树（`F5_KT="lds"`，`F5_ATOM=True/False`）对 env 版本 | ISA 逐行相同 | `isa032/f5hc`、`isa032/f5p1hc` |
| `bounds_proof.py` | 8 个原有形状加 toy（1,256,256,2,1，nsp=16/1）全部 OK；prod max_idx = 134,217,727 = size−1 | `bounds_proof.log`、`bounds_proof2.log` |

## 2. 正确性（每个形状一个进程；门槛与 job 的 `validation.py` 相同）

| 形状 | f5 对 eager（dq / dk / dv） | 冠军对 eager | dk、dv 对冠军 | dq 对冠军 | 运行间一致性 |
|---|---|---|---|---|---|
| toy256（首跑，`AMD_SERIALIZE_KERNEL=3`，CPU fp32 autograd 参考） | 52.09 / 52.18 / 52.37 | 相同 | **逐位相同** | 97.2 dB | 20 次：dk/dv 逐位相同，dq 最差 93.4 dB |
| fast | 52.61 / 52.64 / 52.84 | 相同 | **逐位相同** | 87.8 dB | **200 次**：dk/dv 逐位相同，dq 最差 98.6 dB（门槛 70） |
| proxy | 52.52 / 52.57 / 52.67 | 相同 | **逐位相同** | 85.0 dB | 50 次：dk/dv 逐位相同，dq 最差 98.1 dB |
| prod | 52.56 / 52.60 / 52.71 | 相同 | **逐位相同** | 83.3 dB | 30 次：dk/dv 逐位相同，dq 最差 97.6 dB |

- 覆盖率（先 poison 再检查 isfinite）全部满覆盖。h58 的预言"dk/dv 对冠军逐位相同"在四个形状上都成立。
- 脚本：`corr.py`，日志：`run/corr_{toy256,fast,proxy,prod}.log`。
- 注意：refcache 的 provenance 里 `common_sha` 与现在的 `ut/common.py`（09-25 改过）**对不上**，只有 dims、seed、eager_sha 对得上。
  `corr.py` 仍然使用 cache，并用冠军这一臂作对照：它复现了已知的 52.5–52.8 dB，说明 cache 的数值依然有效。见 §5 第 2 条。

## 3. P1 / P3：只计 kernel 时间（`ktime.py`；方法照抄 benchmark.py：每臂 3 s 连续 warmup，窗口外刷 L2，中位数，palindromic）

每个进程都跑 5 个臂，三个进程轮换顺序。时钟 1860–2191 MHz（host 端从 card24 采样，`run/kt_p*.sclk`）。

| 臂 | p1（C,P1,P3k,P3t,DQ） | p2（P3k,P3t,DQ,C,P1） | p3（DQ,C,P1,P3k,P3t） | 对 C 的比值（p1/p2/p3） |
|---|--:|--:|--:|---|
| C = 冠军 `k_dkdv` | 5.394 | 5.274 | 5.279 | 1 |
| **P1** = `k_dkdv_f5`，ATOM=0 | **6.997** | **6.999** | **7.008** | **1.297 / 1.327 / 1.327** |
| P3k = `k_dkdv_f5`，带原子，只算 kernel | 21.700 | 21.542 | 21.655 | 4.02 / 4.08 / 4.10 |
| **P3t** = 清零 + P3k + cvt_dq | **21.821** | **21.845** | **21.696** | 4.05 / 4.14 / 4.11 |
| DQ = 冠军 `k_dq`（融合要删掉的那个） | 3.009 | 3.016 | 3.241 | 0.56 / 0.57 / 0.61 |

见证：P1 的 dk 对 C 逐位相同，dv 不同（这是设计使然）；P3k、P3t 的 dk 和 dv 都对 C 逐位相同；P3t 的 dq 对冠军 k_dq 为 83.3 dB。
所以被计时的 kernel 确实跑了，而且算的是正确的东西。

**读法**
- **原子不仅没有被计算盖住，还和计算互相拖累，代价超过两者相加。** 原子带来的额外时间是 T_P3k − T_P1 ≈ **14.6 ms**，
  是 P2 中同一地址流单独跑时间（4.43 ms）的 **3.3 倍**。如果完全串行，只会多 4.43 ms。
  可能的原因有两个，都**没有测**：
  - body 里有 162 条 `s_wait_xcnt`，在每个 1 wave/SIMD 的 WG 上，原子 clause 的发射背压全部暴露；
  - 原子流量推迟了 prefetch 的 Q/dO load 的返回。
  这两条都只是假设，这张卡上静态 ISA 指标不能用来排名（规则 8）。
- 即使 E_atom = 0，融合也只能拿到 P1 的 7.00 + 0.064 + 0.18 ≈ 7.25 ms，而冠军的两个 kernel 合起来是 8.3 ms，
  胜出空间约 1 ms。实际拿到的是 21.7 ms。

## 4. 全 op A/B（job 的 harness，prod，`--iters 101`，三个进程轮换）

| 进程 | 顺序 | f5 ms（中位 / 最小） | current ms（中位 / 最小） | f5 / current 速度比 | sclk 开始→结束 |
|---|---|---|---|---|---|
| ab_p1 | f5, current | 21.944 / 21.703 | 8.579 / 8.383 | **0.391×** | 1797→2085 |
| ab_p2 | current, f5 | 21.875 / 21.683 | 8.562 / 8.401 | **0.391×** | 2190→2081 |
| ab_p3 | f5, current | 21.863 / 21.604 | 8.574 / 8.405 | **0.392×** | 1791→2079 |

冠军是 640.9 / 642.2 / 641.3 TF/s，f5 是 250.6 / 251.3 / 251.5 TF/s。三个进程互相吻合，差 < 0.4%。
没有测 `beat`（规则 4 允许排名时不带它），也没有打 dB（这不是可以应用的 arm）。
JSON：`run/ab/ab_p{1,2,3}.json`，日志：`run/ab_p*.log`。

## 5. 给 GPU-1 bwd job 的提议（由 operator 投递，本 lab 不写 hint.md）

1. **dead-end hint（不是 must）**：
   > FUSED5（在 1-wave BLOCK_KV=32 的 k_dkdv 内融合 dQ，用 fp32 SCOPE_DEV 原子，删除 k_dq）已经在卡上测过（B0 lab3，2026-09-27）：
   > 正确性通过，dk/dv 对冠军逐位相同，dq 为 52.6 dB，运行间最低 97.6 dB。但全 op 是 **0.391× 冠军**（21.9 对 8.57 ms，3 个进程）。
   > 原子的代价超过串行相加：k_dkdv_f5 从 7.00 ms 涨到 21.7 ms，而同一地址流单独跑只要 4.43 ms。
   > 仅 dQ GEMM 一项就让 k_dkdv 慢 30–33%（P1），所以 4-wave BLOCK_KV=128 的闸门（x ≤ 7.6%）也不通过。
   > 在出现新证据之前（例如 1 wave/SIMD 下的原子发射背压，或 xcnt 等待被消除），不要再提在 k_dkdv 内融合 dQ 的 arm。
   > 证据在 `output/0927__b0/lab3/FUSED5-RESULT.md`。
2. **与这次实验无关、但对 job 很重要的发现**：job 的 `validation.py` 现在**拒用自己的 refcache**
   （`ut/common.py` 在 09-25 改过以后，`common_sha` 就对不上了）。
   rounds/023 的日志里有 `refcache fast/proxy/prod: IGNORED, provenance differs on ['common_sha'] -- recomputing`，
   也就是说每一轮的 gate 都会在卡上重算 prod 的 fp32 参考。而 refcache 存在的原因正是这个路径在 09-22 造成过 MES 挂死、花掉一次断电重启。
   建议 operator 核对 common.py 在 09-25 的改动是否影响 `make_inputs` 或 `forward_reference`。
   从本次冠军对照的 52.5–52.8 dB 看，cache 的数值仍然有效。如果确认无影响，可以只更新 provenance 里的 `common_sha`，不必重建 cache。

## 6. 卡的安全

- 本 lab 在 GPU 3 上共跑了 13 个进程：4 个正确性进程（toy、fast、proxy、prod），3 个 ktime，3 个全 op A/B，另有 3 个在发出任何 FlyDSL kernel 之前就停在 host 端 assert 上（toy 的 `o` 不连续一次，refcache sha 检查两次）。
  **全部 RC=0，dmesg 新增 0 行**：session 结束时共 2750 行，与 P2 开始时相同。每个 log 末尾都附有 dmesg 差分和 KFD 快照。
- 只用了 fa-g3 和 `/tmp/b0-gpu3.lock`。op-evolve 的目录只读，没有 git 操作。

## 7. 文件

| 路径 | 内容 |
|---|---|
| `f5/` | FUSED5 树：冠军 + fused5.patch，`_env.py` 取自冠军（0.3.2），knob 硬编码为 lds/原子 |
| `f5p1/` | P1 探针树（`F5_ATOM=False` 硬编码；dV 故意算错） |
| `champ/` | 冠军 `current` 的逐字节副本（只用于编译） |
| `cc.sh`、`compile_f5_032.py` | 0.3.2 下的 COMPILE_ONLY 驱动 |
| `rc.sh` | 在卡上跑一个进程的包装：flock、dmesg 差分、sclk/KFD 采样 |
| `corr.py` | 正确性和运行间一致性检查 |
| `ktime.py` | P1/P3 只计 kernel 时间的 A/B |
| `run/corr_*.log`、`run/kt_p*.log`、`run/kt/*.json`、`run/ab_p*.log`、`run/ab/*.json` | 原始证据 |
