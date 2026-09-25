# 4-wave BLOCK_KV=128 转换 —— 可直接施工的有序编辑清单

全部行号已对当前文件逐字节复核（`kernels.py` 1170 行、`impl.py` 233 行）。引文中的空格数与文件一致。

绝对路径：
- `/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-flydsl-attn-bwd-20260917-115934/job_context/op/current/kernels.py`
- `/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-flydsl-attn-bwd-20260917-115934/job_context/op/current/impl.py`
- `/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-flydsl-attn-bwd-20260917-115934/job_context/op/ut/common.py`
- `/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-flydsl-attn-bwd-20260917-115934/job_context/op/validation.py`

---

## 0. 施工顺序的硬规则

**按行号降序施工**（E1 → E26 已排好）。E17（:191 之后插入两行）和 E18（:173 一行变两行）都会把下方所有行号推移；降序施工可以让每一条的引文都在它自己的原始行号上命中。

**绝对不要用全局 sed。** 四段文本在文件里不唯一：

| 文本 | k_dkdv（要改） | k_dq / 其他（禁止改） |
|---|---|---|
| `@flyc.kernel(known_block_size=[32, 1, 1])` | 602, 611 | **997, 1006** |
| `grid=..., block=(32, 1, 1), stream=stream)` | 628, 641 | **1023, 1035** |
| `lane = fx.Int32(fx.thread_idx.x)` | 173 | **668** |
| `# r8.i1.g23+g24 -- fx.barrier() DELETED. ...` | 453, 492 | **855, 877** |

另外 `tid = fx.Int32(fx.thread_idx.x)` 在 115 / 1072 / 1132（k_delta_bshd、k_redsp、k_redsp_q，均 256 线程）——文本不同，但任何把 `fx.thread_idx.x` 换成 `fx.lane_id()` 的模糊替换都会让这三个内核的 8 个 wave 全算同一个 tile，静默塌掉。

---

## 1. 编辑清单（降序）

### E1 — `kernels.py:641` 【mechanical】
```
        grid=(nxs, nblk, nb), block=(32, 1, 1), stream=stream)
```
→
```
        grid=(nxs, nblk, nb), block=(DKDV_THREADS, 1, 1), stream=stream)
```

### E2 — `kernels.py:628` 【mechanical】
```
        grid=(nhkv, nblk, nb), block=(32, 1, 1), stream=stream)
```
→
```
        grid=(nhkv, nblk, nb), block=(DKDV_THREADS, 1, 1), stream=stream)
```

### E3 — `kernels.py:611` 【silent-corruption-risk】（k_dkdv_sp 的装饰器）
```
@flyc.kernel(known_block_size=[32, 1, 1])
```
→
```
@flyc.kernel(known_block_size=[DKDV_THREADS, 1, 1])
```
E3/E4 与 E1/E2 必须同批落地。`known_block_size` 落进 ISA 的 `.reqd_workgroup_size`：声明 32 而以 128 启动是 UB；声明 128 而以 32 启动则线程 32..127 不存在，3/4 的 kv 行永不计算。两者都不报错。

### E4 — `kernels.py:602` 【silent-corruption-risk】（k_dkdv 的装饰器）
同 E3 的替换文本。**按行号定位，不要按文本。**

### E5 — `kernels.py:592` 【silent-corruption-risk】
```
                kv = kv0 + fx.Int32(kh * 16) + half * fx.Int32(8) + fx.Int32(si)
```
→
```
                kv = kvw + fx.Int32(kh * 16) + half * fx.Int32(8) + fx.Int32(si)
```
漏改 = 四个 wave 用不同的值竞写同一批 32 个输出行 → 结果 run-to-run 不确定，是唯一会被 determinism 门抓到的失效模式。

### E6 — `kernels.py:557-558` 【comment】
```
        # Exactly ONE query pair per q head is masked and it is the FIRST one
        # (qp_start), so it stays with split 0 and the mask-split of g19/g20 survives
```
→
```
        # FOUR query pairs per q head are masked now (BLOCK_KV=128 spans 4 pairs) and
        # they are the FIRST ones (qp_start..), so they stay with split 0 and the
        # mask-split of g19/g20 survives
```
只是注释，但 :568 的 `_mk = (sp != fx.Int32(0)).select(fx.Int32(0), nmaskp)` 现在把 4 个 masked pair 全压给 split 0（原来 1 个）。纯性能，不是错答案。

### E7 — `kernels.py:492-499` 【redesign，整块 8 行替换】
当前 492-499（第 499 行是 `        # instructions instead of a full iteration.`，第 500 行是 `        if const_expr(carry):`）：
```
        # r8.i1.g23+g24 -- fx.barrier() DELETED. block=(32,1,1): the workgroup is ONE
        # 32-lane wave, so s_barrier is semantically a no-op and LLVM already removes
        # it (0 `s_barrier` in the shipped ISA). What survives is the conservative
        # ALL-COUNTER waitcnt the backend inserts FOR the barrier before removing it --
        # `s_wait_loadcnt_dscnt 0x0`. Its dscnt half is the real LDS RAW and the backend
        # re-derives it from the memory dependence; its loadcnt half has no dependence
        # at all here and is exactly what truncates r7.i1.g21's prefetch cover to 299
        # instructions instead of a full iteration.
```
→
```
        # r23 -- BARRIER RESTORED, and THIS one is load-bearing. WAR on the shared
        # Q/dO image: wave w is still issuing its 32 ds_load_tr16_b128 of query pair
        # qt above while another wave has already passed the loop back-edge and is
        # storing query pair qt+1 into the same lds_do/lds_q bytes. Different pair,
        # different values -- this is a real cross-wave race, not a same-value one.
        # Bare fx.barrier(): rocdl.s_waitcnt raises ValueError on gfx1250
        # (flydsl/expr/rocdl/universal.py:47-55); gfx1250 has SPLIT counters and the
        # backend derives the dscnt wait from the LDS memory dependence itself.
        fx.barrier()
```

### E8 — `kernels.py:482` 【silent-corruption-risk】
```
            col = lane_c * fx.Int32(2) + fx.Int32(kh * 32)
```
→
```
            col = lane_c * fx.Int32(2) + wcol + fx.Int32(kh * 32)
```
**E8 和 E10 必须同时落地。** 只改一处 = 写入 band w、读出 band 0（或反之）→ dK/dV 数值饱满、量级合理、完全错误。

### E9 — `kernels.py:453-460` 【redesign，整块 8 行替换】
当前 453-460（第 460 行是 `        # instructions instead of a full iteration.`，第 461 行是空行）：与 E7 引文逐字相同的 8 行。
→
```
        # r23 -- BARRIER RESTORED. In THIS build the Q/dO staging is replicated: all
        # four waves run _ldqd on wave-uniform (qt, gh) and each writes the ENTIRE
        # [32 q][D] image with identical bytes, then reads back only bytes it wrote.
        # So this barrier is a same-value redundancy today, not a live dependence --
        # it is restored because G1a measured it free at one wave, and because it
        # becomes mandatory the moment the staging store is split across waves.
        # Bare fx.barrier(): see the note at the second barrier below.
        fx.barrier()
```

### E10 — `kernels.py:446` 【silent-corruption-risk】
```
                       + fx.Int32(kh * 32) + half * fx.Int32(16))
```
→
```
                       + wcol + fx.Int32(kh * 32) + half * fx.Int32(16))
```
（:445 不动，它是同一个表达式的首行，列出来只为让施工者确认上下文：`                off = ((fx.Int32(hh * 16) + row) * fx.Int32(S_ROW_B)`）

### E11 — `kernels.py:431` 【silent-corruption-risk，全局最高危】
```
                kvb = kv0 + fx.Int32(kh * 16) + half * fx.Int32(8)
```
→
```
                kvb = kvw + fx.Int32(kh * 16) + half * fx.Int32(8)
```
漏改而其他四处都改了：wave 1-3 用低 32/64/96 的 key 索引做因果比较 → **少 mask**。非因果测试全过，因果 SQNR 只下降一点，容易被当成 bf16 噪声。全套改动里可见度最低、后果最重。

### E12 — `kernels.py:338-340` 【comment】
```
    # (b) MASK-SPLIT. Under bottom-right causality exactly ONE query pair per q head
    #     straddles this workgroup's kv tile; every later pair has kv0+BLOCK_KV-1 <= q0 +
    #     cshift, so the predicate is provably false for all 8 elements of all 4 tiles.
```
→
```
    # (b) MASK-SPLIT. Under bottom-right causality FOUR query pairs per q head straddle
    #     this workgroup's 128-row kv tile (nmaskp = 4 at cshift = 0); every later pair
    #     has kv0+BLOCK_KV-1 <= q0 + cshift, so the predicate is provably false for all
    #     8 elements of all 4 tiles. The bound is the workgroup's LARGEST kv row, so it
    #     is a superset of what any single wave needs -- conservative and correct.
```
连带 `kernels.py:353` 的 `# runs G*nmaskp = 4 iterations at prod against the full body's ~508, so a` 里的 `4` 现在是 `16`（可选改）。

### E13 — `kernels.py:315-316` 【comment，可选但推荐】
```
    # q >= j - cshift, so this workgroup's kv tile [kv0, kv0+16) is untouched by every
    # query tile below qt_start = max(0, (kv0 - cshift) // 16). Tiles below it contributed
```
→
```
    # q >= j - cshift, so this workgroup's kv tile [kv0, kv0+BLOCK_KV) is untouched by
    # every query pair below qp_start = max(0, (kv0 - cshift) // 32). kv0 is the
    # WORKGROUP's smallest kv row, so this skip is valid for all four waves at once.
    # Pairs below it contributed
```
这段已经在 BLOCK_KV=32 时代就陈旧（写 16，代码做 32）。四波之后它正是「哪些量必须保持 workgroup 级」的唯一成文依据，不能继续错着。

### E14 — `kernels.py:309` 【silent-corruption-risk】
```
    vf = [[gfrag(g_v, base_kv, rs_kv, kv0 + fx.Int32(kh * 16), dt) for dt in range(NDT)]
```
→
```
    vf = [[gfrag(g_v, base_kv, rs_kv, kvw + fx.Int32(kh * 16), dt) for dt in range(NDT)]
```

### E15 — `kernels.py:307` 【silent-corruption-risk】
```
    kf = [[gfrag(g_k, base_kv, rs_kv, kv0 + fx.Int32(kh * 16), dt) for dt in range(NDT)]
```
→
```
    kf = [[gfrag(g_k, base_kv, rs_kv, kvw + fx.Int32(kh * 16), dt) for dt in range(NDT)]
```
E14/E15 最容易被跳过：它们各自的 `for kh in range(NKV)]` 续行在下一行（308 / 310），看起来像同一条语句的一部分。漏掉两条 = 四个 wave 载入同一批 K/V → 行 [kv0+32, kv0+128) 的 dK/dV 永不计算，保留 `torch.empty` 残留。

### E16 — `kernels.py:226-242` 【redesign，整块 17 行替换】
当前 226-242（226 起 `    # r1.i6.g06. The workgroup owns BLOCK_KV key rows as NKV 16-row WMMA accumulator`，242 止 `    # total under the 4-workgroup rung.`，243 是 `    LDS_SEG = 65536`）→
```
    # r23 -- FOUR WAVE32 PER WORKGROUP. The workgroup owns BLOCK_KV = 128 key rows,
    # split KV_PER_WAVE = 32 rows per wave; each wave still carries NKV = 2 16-row WMMA
    # accumulator sets, i.e. the per-wave register working set is bit-for-bit what it
    # was. Q/dO staging stays 32 query rows and is now SHARED by the four waves -- that
    # sharing is the whole point of the change.
    #   B ring (dO, Q): [32 q][D] bf16, X_ROW_B = D*2+16 = 272 B  ->  2 x 8704 = 17408 B
    #   A ring (P, dS): [32 q][BLOCK_KV] bf16, S_ROW_B = 272 B    ->  2 x 8704 = 17408 B
    # The A ring is COLUMN-PARTITIONED: wave w owns kv columns [32w, 32w+32), i.e. byte
    # columns [64w, 64w+64) of every row (`wcol`). The B ring is NOT partitioned.
    # S_ROW_B == X_ROW_B == 272 is a numerical COINCIDENCE at D = BLOCK_KV = 128. They
    # are different quantities (head-dim columns vs kv columns); do not merge them.
    # r12.i1.g39 -- LDS SEGMENT SEPARATION, unchanged and still exact. gfx1250's LDS is
    # organised in 64 KB segments served by TWO 256 B/cycle read ports; two reads in
    # different segments are served in the same cycle, two in the same segment are not
    # (`optimization/techniques/6-gfx1250-cdna5-mechanisms.md:252-268`). The B ring sits
    # at [0, 17408) and the A ring one whole segment up at [65536, 82944): adding
    # exactly 65536 flips bit 16, so the two rings land in ADJACENT segments whatever
    # the workgroup's physical LDS base is -- no alignment assumption. 17408 < 65536, so
    # the argument survives the new sizes verbatim.
    # OCCUPANCY INVERTS. Total is now 65536 + 2*32*272 = 82944 B of the 327680 B gfx1250
    # allows (flydsl/utils/smem_allocator.py:249). LDS would permit
    # floor(327680/82944) = 3 workgroups per CU, but at 904 VGPR the measured rung is
    # 1 wave/SIMD (r12.i0.g38), so 4 SIMDs hold exactly ONE four-wave workgroup. VGPR is
    # the sole binding constraint and per-CU LDS pressure DROPS from 4*70656 = 282624 B
    # to 82944 B. The old rationale -- "the SMALL ring is the one placed high, which
    # keeps the total under the 4-workgroup rung" -- is void: the two rings are now the
    # same size and the rung no longer binds. Waves per CU is unchanged at 4.
```

### E17 — `kernels.py:191` 【redesign，保留原行 + 其后插入两行】
当前：
```
    kv0 = bid * fx.Int32(BLOCK_KV)
```
→
```
    kv0 = bid * fx.Int32(BLOCK_KV)
    # kv0 stays the WORKGROUP base (128-aligned) and must remain wave-uniform: the
    # causal bounds at :320-322 and :544-550 derive the two scf.for trip counts from
    # it, and _body now contains two fx.barrier(). Per-wave trip counts = hang.
    kvw = kv0 + wave * fx.Int32(KV_PER_WAVE)      # this WAVE's 32 kv rows
    wcol = wave * fx.Int32(KV_PER_WAVE * 2)       # its A-ring byte band: 0/64/128/192
```

### E18 — `kernels.py:173` 【silent-corruption-risk，任务书点名的头号项】
```
    lane = fx.Int32(fx.thread_idx.x)
```
→
```
    lane = fx.lane_id()                                  # 0..31 WITHIN this wave
    wave = fx.Int32(fx.thread_idx.x) // fx.Int32(WAVE)   # 0..3, which 32-kv slice
```
`fx.lane_id()` 在 `/home/lihuzhan/.local/flydsl032/flydsl/expr/gpu.py:67`，已在 `__all__`（同文件 :41），经 `flydsl/expr/__init__.py` 的 `from .gpu import *` 暴露为 `fx.lane_id`，**已返回 Int32，不要再包一层**。

不改的后果（已逐项算过，全部落在 82944 B 分配之内，**不 fault、不报错**）：`row = lane % 16`（:189）侥幸仍是 0..15；`half = lane // 16`（:190）变成 0..7，于是 `gfrag` 的 `t = base + (r+row)*rs + half + dt*4`（:254）多走 6 个 vec8 tile，读到错误的全局列；`lane_r = (lane//16)*8 + lane%8`（:311）最大到 59，`lane_r * 272 = 16048` 越过 8704 B 的 lds_do 打进 lds_q 的地盘。全是静默错答案。

`wave` 用 `thread_idx.x // 32` 而不是 `rocdl.wave_id()`：后者读 TTMP8[29:25]，本树里只在 `tdm_ops.py` 用过，我**无法确认 ROCm dispatch 路径会为一个普通 `@flyc.kernel` 初始化 TTMP8**；若不初始化，`wave` 是垃圾且被算术掩成小偏移 → 又是静默错答案。先按可证正确的写法建，标量化留到正确性确认之后（`rocdl.readfirstlane`，`flydsl/expr/rocdl/__init__.py:877`；`rocdl` 已在 kernels.py:38 导入）。

### E19 — `kernels.py:168-171` 【docstring】
```
    """One wave owns a 16-row kv tile of one kv head and streams every (q head, q tile).

    grid = (Skv/16, Hkv, B). The accumulators persist across the G q heads that share this
    kv head, so the GQA reduction happens in registers -- atomic-free, written once.
    """
```
→
```
    """One WORKGROUP of 4 wave32 owns a BLOCK_KV=128-row kv tile of one kv head and
    streams every (q head, q tile). Each wave owns KV_PER_WAVE = 32 of those rows.

    grid = (Hkv, Skv/128, B). The accumulators persist across the G q heads that share
    this kv head, so the GQA reduction happens in registers -- atomic-free, written once.
    The four waves share the Q/dO staging tile in LDS; they do NOT share the P/dS tile,
    which is column-partitioned by `wcol`.
    """
```

### E20 — `kernels.py:53-55` 【redesign，三行变六行】
```
BLOCK_KV = 32               # r1.i6.g06: kv rows one workgroup owns (was 16)
S_ROW_B = BLOCK_KV * 2 + 16  # r4.i1.g16: 64 -> 80 B. 16 dwords is a 4-way bank
                             # collision at 64 banks; 20 dwords walks all 64.
```
→
```
BLOCK_KV = 128              # r23: kv rows one WORKGROUP owns; 4 wave32 x 32 (was 32)
KV_PER_WAVE = 32            # kv rows ONE WAVE owns -- unchanged, this is what keeps
                            # NKV = 2 and the 256-VGPR accumulator set per wave
WAVES_DKDV = BLOCK_KV // KV_PER_WAVE   # 4
DKDV_THREADS = WAVES_DKDV * WAVE       # 128 -- k_dkdv / k_dkdv_sp block size
S_ROW_B = BLOCK_KV * 2 + 16  # r4.i1.g16 -> r23: 80 -> 272 B. Unpadded would be 256 B =
                             # 64 dwords, the EXACT full-collision stride; 272 B = 68
                             # dwords gives row*4 mod 64 -- 16 distinct start banks over
                             # the 16 rows a ds_load_tr16_b128 phase touches.
```
**`WAVE = 32`（:50）不动。** 它是 shuffle 宽度，唯一消费者是 :139 `fx.gpu.shuffle_xor(acc, 1 << sft, WAVE)`，在 256 线程的 k_delta_bshd 里做 warp 内归约；改成 128 会让那个无关内核静默坏掉。`NKV = 2`（:48）、`NST`（:49）、`NDT`、`NDO` 全是**每 wave**量，一律不动（:48 注释里的 "per workgroup" 可选改成 "per wave"）。

**这一条不能漏。** 漏掉 E20 而其余全做：`KV_PER_WAVE` 未定义 → NameError（响的，安全）；若施工者顺手把 `KV_PER_WAVE` 写成字面量 32 而 BLOCK_KV 留在 32，则 S_ROW_B 仍是 80，`wcol = wave*64` 越出 80 B 的行进入下一个 q 行的数据，仍在 70656 B 之内、不 fault —— 确定性的错答案。

### E21 — `kernels.py:20` 【docstring / 对外契约】
```
Divisibility is asserted, not handled: Sq % 16 == 0, Skv % 32 == 0. The bring-up files
```
→
```
Divisibility is asserted, not handled: Sq % 16 == 0, Skv % 128 == 0 (r23: k_dkdv's
workgroup is four wave32 covering BLOCK_KV = 128 kv rows). The bring-up files
```
**这是本次改动唯一真正的契约收窄**，必须成文。见 §5 关于 `toy` 的处置。

---

### E22 — `impl.py:167` 【redesign】
```
    while _wgs * nsp < 2048 and nsp < 16:
```
→
```
    while _wgs * nsp < 512 and nsp < 16:
```
`_wgs = (skv // BLOCK_KV) * hkv * b` 自动除以 4。2048 这个阈值编码的是「两个 dispatch wave」，而驻留量是 1024 个**单 wave** workgroup；四 wave workgroup 的驻留量是 256，所以目标值同步除以 4 → 512。验算（三个计分 shape 全部还原今天的 nsp）：

| shape | `_wgs` 今天 → 新 | nsp @2048 | nsp @512 | 今天的 nsp |
|---|---|---|---|---|
| fast (1,1024,1024,8,2) | 64 → 16 | 16（撞上限） | 16 | 16 ✓ |
| proxy (1,4096,4096,32,8) | 1024 → 256 | **8** | 2 | 2 ✓ |
| prod (4,8192,8192,32,8) | 8192 → 2048 | 1 | 1 | 1 ✓ |

不改 = proxy 静默从 nsp=2 跳到 nsp=8：`dkp`/`dvp` 各从 67 MB 涨到 268 MB，k_redsp 折叠 8 份而不是 2 份。结果仍然正确、determinism 仍然成立，但被测的已经不是同一个配置，+13.6% 无从归因。**这不是 bug，是一次会被误读成「4-wave 在 proxy 上没用」的错误归因。**

`impl.py:210` 的 `while _wgs_q * nsp_q < 2048 and nsp_q < _NSP_Q_CAP:` 属于 k_dq，**不动**。

### E23 — `impl.py:142-147` 【comment】
```
    # k_dkdv's workgroup count is (Skv/BLOCK_KV)*Hkv*B and one workgroup is one wave32,
    # so at 1 wave/SIMD (g38) the device holds 256*4 = 1024 of them at once. prod
    # launches 8192 -- eight dispatch waves, and the greedy hardware dispatcher balances
    # the 256:1 causal work skew to a modelled 100% (raw/census.txt). proxy launches
    # EXACTLY 1024: one wave, every workgroup resident from t=0, so the kernel ends when
    # its LONGEST workgroup ends and the modelled efficiency is 50.4%. fast launches 64
```
→
```
    # k_dkdv's workgroup count is (Skv/BLOCK_KV)*Hkv*B and one workgroup is FOUR wave32
    # (r23), so at 1 wave/SIMD (g38) a four-wave workgroup fills one CU and the device
    # holds 256 of them at once -- still 1024 waves, one quarter the workgroups. prod
    # launches 2048 -- eight dispatch waves, and the greedy hardware dispatcher balances
    # the 256:1 causal work skew to a modelled 100% (raw/census.txt). proxy launches
    # EXACTLY 256: one wave, every workgroup resident from t=0, so the kernel ends when
    # its LONGEST workgroup ends and the modelled efficiency is 50.4%. fast launches 16
```
下面 :148 的 `# onto 1024 slots: 3.2%.` 改成 `# onto 256 slots: 6.25%.`，:167 附近 :161-165 那段「两个 dispatch wave」的叙述与新的 512 一致，数字无需再改。

`impl.py:120-121`、`:153`、`:175`、`:182` 全部通过 `_k.BLOCK_KV` 读常量，**自动跟随，一律不改**。特别提醒：`:175` / `:182` 的 `skv // _k.BLOCK_KV` 是 grid.y（`nblk`），自动从 skv/32 变成 skv/128 —— 这是**正确的**（workgroup 数少 4 倍，每个宽 4 倍）。看到 grid.y 缩了 4 倍而手工改回 skv/32 的人，会得到每个 kv 行被算四遍、最后一个写者获胜的内核：不 fault、不报错、dK/dV 错。

### E24 — `ut/common.py:19` 【契约收窄的处置】
```
    "toy":               (1, 64,   64,   2,  1,   128),   # smallest legal, mha
```
→
```
    "toy":               (1, 128,  128,  2,  1,   128),   # smallest legal at BLOCK_KV=128
```
skv=64 会在 `impl.py:120` 的 `assert skv % _k.BLOCK_KV == 0` 直接抛 AssertionError，而 `toy` 是 `ut/test_correctness.py:36` `DEFAULT` 列表的**第一项** → UT 门在任何测量之前整体非零退出。这是本次改动唯一**响亮**的失败。

新值复核：sq=128 % 32 == 0 ✓；skv=128 % 128 == 0 ✓；`n_rows = 1*128*2 = 256`，256 % ROWS_DELTA(32) == 0 ✓；nblk = 1 ✓（仍然覆盖单块路径）。其余六个 edge shape 的 skv = 128/256/1024/2048/512 全部整除 128，三个计分 shape 1024/4096/8192 也是 —— **只有 toy 一个坏**。

替代方案（若不想动 shape 表）：把 `"toy"` 从 `ut/test_correctness.py:36` 的 `DEFAULT` 里摘掉，并在实验记录里写明「BLOCK_KV=128 把最小合法 Skv 从 32 抬到 128」。**必须二选一，不能不选。**

### E25 — `validation.py:105-106` 【comment，安全论证】
```
    found to fault the card. ... None of the kernels under test is implicated: k_dkdv is 32 threads /
    22528 B of LDS, k_dq 32 / 8704, k_delta 256 / 0.
```
实际引文（:105-106）：
```
    cost a power cycle. None of the kernels under test is implicated: k_dkdv is 32 threads /
    22528 B of LDS, k_dq 32 / 8704, k_delta 256 / 0.
```
→
```
    cost a power cycle. None of the kernels under test is implicated: k_dkdv is 128 threads /
    82944 B of LDS, k_dq 32 / 8704, k_delta 256 / 0.
```
（22528 这个数字在今天就已经陈旧 —— 实际是 70656。这段是「哪个内核可能弄挂卡」的论证，数字必须真实。）

---

## 2. 新的 LDS 推导（全文，含 bank 冲突论证）

### 2.1 常量

```
D        = 128
X_ROW_B  = D * 2 + 16        = 272 B   （不变）
S_ROW_B  = BLOCK_KV * 2 + 16 = 272 B   （原 80 B）
```
两者数值相等是 D = BLOCK_KV = 128 下的巧合。X_ROW_B 的 256 B 载荷是 128 个 head-dim 列，S_ROW_B 的 256 B 载荷是 128 个 kv 列。**保持两个独立符号**；任何把 `2 * 32 * S_ROW_B` 简化成字面量或与 X_ROW_B 合并的改动，会让日后任一侧的 D / BLOCK_KV 变化不再传播。

### 2.2 内存图（`kernels.py:243-249` 四行代码一字不改，自动求值到此）

```
segment 0  [base+0, base+65536)
  lds_do  @ +0       [32 q][128 d]  bf16, 行距 272 B   32*272 =  8704 B  -> [0,     8704)
  lds_q   @ +8704    同形                              32*272 =  8704 B  -> [8704, 17408)
                                              B ring 合计 17408 B（段 0 的 26.6%）
  -------- 48128 B 空洞 --------
segment 1  [base+65536, base+131072)
  lds_p   @ +65536   [32 q][128 kv] bf16, 行距 272 B   32*272 =  8704 B  -> [65536, 74240)
  lds_ds  @ +74240   同形                              32*272 =  8704 B  -> [74240, 82944)
                                              A ring 合计 17408 B
allocate(LDS_SEG + 2*32*S_ROW_B) = 65536 + 17408 = 82944 B
```
容量：gfx1250 = 327680 B（`/home/lihuzhan/.local/flydsl032/flydsl/utils/smem_allocator.py:249`）。82944 ≤ 327680，余 244736 B。**容量不是瓶颈。** 更硬的旁证：今天在跑的 70656 B 已经超过传统的 64 KiB per-workgroup 上限且工作正常。

`:244` `smem = fx.SharedAllocator().allocate(LDS_SEG + 2 * 32 * S_ROW_B)`、`:246-249` 的四个偏移全部已经参数化，**零文本改动**，自动得到上表。`:243` `LDS_SEG = 65536` 也不动 —— gfx1250 是 5 × 64 KB 段（327680 = 5×65536），65536 仍是段跨度。

### 2.3 两个 ring 的分区方式**不对称**，这是施工者最需要内化的一点

**B ring（Q/dO staging）—— 共享，地址里不出现 `wave`。**
- 写（:379-380 + :383-387）：`xo = (hh*16 + row)*272 + half*16`，`o = xo + dt*64 + u*32`。
  - `row = lane%16` ∈ 0..15，`hh` ∈ {0,1} → 行 0..31 全覆盖。
  - 列字节 = `half*16 + dt*64 + u*32`，half∈{0,1}、dt∈0..3、u∈{0,1} → {0,16,32,...,240}，即每行 256 B 的 16 个 16 B 块**全部覆盖**，272 B 行末的 16 B padding 不碰。
  - 最大地址：`31*272 + 240 + 16 = 8688 ≤ 8704` ✓
  - 四个 wave 写**逐字节相同**的内容（`_ldqd` 只依赖 lane/row/half 和 wave-uniform 的 qt/gh/hkv/bat）→ 良性同值竞争。
- 读（:477-479）：`tr(lds_do + lane_r*272 + c, 272)`，`c = (lane_c + dtile*16)*2`。`tr` 读 `base` 和 `base + 16*272` 两个 16 B 块。最大：`15*272 + (8+112)*2 + 16*272 + 16 = 31*272 + 256 = 8688 ≤ 8704` ✓
- **地址里绝不能加 `wave` 项。** 加了 = wave 1-3 去读 32 行 tile 的第 32..127 行，地址仍在 82944 B 之内、不 fault，只是输出 GEMM 拿到垃圾 dO/Q 操作数。

**A ring（P/dS）—— 按列分区，每个 wave 一条 64 B 带。**
- wave w 拥有 kv 列 [32w, 32w+32)，即每行的字节带 [64w, 64w+64)。
- 写（:445-446 加 `wcol`）：`off = (hh*16+row)*272 + wcol + kh*32 + half*16`，`kh*32 + half*16` ∈ {0,16,32,48}，各存 16 B → 恰好铺满 64 B，无重叠无空隙。
  - 最大：`31*272 + 192 + 48 + 16 = 8688 ≤ 8704` ✓
- 读（:482 加 `wcol`）：`col = lane_c*2 + wcol + kh*32`，`lane_c*2` ∈ {0,16}、`kh*32` ∈ {0,32} → {0,16,32,48}，同一条 64 B 带。
  - 最大：`15*272 + 48 + 16 + 192 + 16*272 = 31*272 + 256 = 8688 ≤ 8704` ✓
- **每个 wave 既写又只读自己的带，A ring 完全没有跨 wave 依赖。** 两个 barrier 只为 B ring 存在。

### 2.4 Bank 冲突论证

gfx1250 LDS = 64 bank × 4 B，`bank(addr) = (addr / 4) mod 64`。一次 `ds_load_tr16_b128` 相位用 `lane_r = (lane//16)*8 + lane%8`：lane 0..31 → lane_r ∈ 0..15，**16 个不同的行**、同一列，每个 lane 取 16 B（跨 4 个连续 bank）。要无冲突，需要 `r ↦ r*(P/4) mod 64` 在 r ∈ 0..15 上单射。

因为所有 payload 都是 16 B 对齐，`P/4` 必是 4 的倍数，于是**单射 ⟺ gcd(P/4, 64) = 4 ⟺ P ≡ 16 (mod 32)**。逐个代入：

| 行距 P | dword = P/4 | gcd(·,64) | 16 行落在几个 bank 组 | 结论 |
|---|---|---|---|---|
| 256 B（X 若不填充） | 64 | 64 | 1 | 64 路全冲突 |
| 64 B（S @ BLOCK_KV=32 不填充） | 16 | 16 | 4 | 4 路冲突 |
| **80 B（今天的 S）** | 20 | 4 | 16 | 无冲突 |
| **272 B（X 与新的 S）** | 68 | 4 | 16 | 无冲突 |

P = 272 时行 r 起始 bank = `4r mod 64`，r = 0..15 给出 {0,4,8,...,60}，每个 16 B 访问跨 4 个连续 bank → **16 行恰好把 64 个 bank 各铺一次，零冲突**。这与今天 X_ROW_B 的论证同构，也与今天 S_ROW_B=80 同等干净。注意：若 S 不填充，BLOCK_KV=128 下行距会是 256 B = 64 dword = **单 bank 64 路全冲突** —— 比今天不填充时的 4 路严重得多，所以 `BLOCK_KV*2 + 16` 这个表达式现在比以前更有价值。

`wcol = 64w` 加的是常数 16 dword，只把整个 bank 集合旋转 `16w mod 64`；因为 16w ≡ 0 (mod 4)，集合仍是 {0,4,...,60} 的平移，**不塌缩**。

写侧（:445-446）：32 lane，`row = lane%16`（16 行）、`half = lane//16`（2 个 16 B 列）。`bank = (4row + 16w + 8kh + 4half) mod 64`，{4row} 与 {4row+4} 重合 → **2 路冲突**；但今天在 S_ROW_B=80 下是**完全相同的 2 路冲突**（{20row mod 64} 同时含 0 和 4），而且 32 lane × 16 B = 512 B 的 wave 级 b128 访问本来就要 2 个周期（端口 256 B/cycle）。**无退化。**

### 2.5 段分离与占用率

B ring [0, 17408)，A ring [65536, 82944)。17408 < 65536，所以「加 65536 翻转 bit 16 → 两个 ring 落在相邻的 64 KB 段，与物理基址无关」这条 r12.i1.g39 的推理**逐字成立**，不需要重推。

占用率（这一条相对旧注释是**反转**的）：
- 每 wave VGPR = 904（`rounds/022/1-opt/raw/isa_cur_a_k_dkdv.s:3353`），其对应的 1 wave/SIMD 是 r12.i0.g38 的**实测结论**（注释 :240 里写的 740 与构建物的 904 本来就不一致 —— 我沿用结论，不重推）。
- 4 SIMD/CU × 1 wave = 4 wave = **恰好一个四 wave workgroup / CU**。
- LDS 允许 `floor(327680/82944) = 3` workgroup/CU ≥ 1 → **LDS 不再是约束**，每 CU 的 LDS 压力从 4×70656 = 282624 B 降到 82944 B（3.4 倍余量）。
- **每 CU 的 wave 数完全不变：4。** 今天是 4 个互相独立的单 wave workgroup，改后是 1 个 barrier 耦合的四 wave workgroup。占用率按 wave 计一模一样；变的是 workgroup 粒度和耦合结构（见 §6）。

---

## 3. 每 wave 的因果谓词

### 3.1 必须保持 workgroup-uniform 的两个量（改了就挂死）

```
kernels.py:320    _c = kv0 - cshift
kernels.py:321    qp_start = ((_c < fx.Int32(0)).select(fx.Int32(0), _c)) // fx.Int32(32)
kernels.py:544    _u = kv0 + fx.Int32(BLOCK_KV - 1) - cshift
```
- `qp_start` 用 `kv0` —— workgroup 的**最小** kv 行。查询对 qt（32 个 query，最大索引 qt*32+31）对本 workgroup 有贡献 ⟺ 它对最小 kv 行有贡献：`qt*32+31 >= kv0 - cshift` ⟺ `qt >= floor((kv0-cshift)/32)`（kv0-cshift ≥ 0 时）。写法里根本不含 BLOCK_KV，**BLOCK_KV=128 下自动正确**，而且它是四个 wave 的**下确界** → 只跳过对每个 wave 都为零的查询对。保守且精确。
- `_qsf`（:545）用 `kv0 + BLOCK_KV - 1 = kv0 + 127` —— workgroup 的**最大** kv 行。查询对完全无需掩码 ⟺ 最大 kv 行被该对的最小 query 关注。它是四个 wave 的**上确界** → 只要任何一个 wave 需要掩码，整个 workgroup 就走 masked body。**BLOCK_KV 已在表达式里，自动跟随。**

这两个量喂给两个 `scf.for` 的 trip count：`G * nmaskp`（qloop_mask）和 `G * (nqp_eff - nmaskp)`（qloop_full）。**`_body` 现在含两个 `fx.barrier()`。** `kv0`、`cshift`、`causal`、`nqt2`、`G`、`sp`、`nsp` 全部 wave-uniform（`sp` 来自 `block_idx.x`），所以四个 wave 的 trip count 相同、执行的 barrier 次数相同。

**任何把 `:320` 或 `:544` 的 `kv0` 换成 `kvw` 的「优化」都会让 trip count 按 wave 分叉 → 挂死或 UB。** 它不是错答案：它表现为一个永不 retire 的 dispatch，在 gfx1250 上代价是一次断电重启。源码和构建都不会警告这件事。

`nmaskp` 的新值：cshift = 0、kv0 = 128m 时，`_u = 128m+127`，`_qsf = (128m+127+31)//32 = 4m+4`，`qp_start = 4m` → **nmaskp = 4**（原来是 1）。masked body 每 q head 跑 4 趟而不是 1 趟，且每趟 4 个 wave 宽。设备级 masked wave-iteration 从 Skv/32 涨到 Skv/8（4 倍），但 prod 下占比只从 0.78% 升到 3.1%。这是既定成本的一部分。

### 3.2 唯一的每 wave 因果编辑：`kernels.py:431`

```
kvw = kv0 + wave * 32
kvb = kvw + kh*16 + half*8
masked(si) ⟺ (causal != 0) AND
             ( kv0 + 32*wave + 16*kh + 8*half + si  >  qt*32 + 16*hh + row + cshift )
```
其中 `row = lane%16` 是 **query 行**索引，`half = lane//16` 是 kv 子索引，`si` ∈ 0..7 是片段内的 kv 元素。这就是 :434 的表达式原样，只把 `kv0` 换成 `kvw`。**整个 kernel 里没有第二处需要动的因果逻辑。**

### 3.3 `kv0` 六处读取的分派（这是本次改动的核心表）

| 行 | 语义 | 用 |
|---|---|---|
| 307 | K 片段基址 | **kvw** |
| 309 | V 片段基址 | **kvw** |
| 320 | qp_start（循环边界） | **kv0**（必须） |
| 431 | 因果掩码 key 索引 | **kvw** |
| 544 | nmaskp（循环边界） | **kv0**（必须） |
| 592 | dK/dV 全局写行号 | **kvw** |

另注：`kernels.py:593` 的 `idx = base_o + kv * Hkv * fx.Int32(D) + fx.Int32(dtile * 16) + row` 里的 `row` 是 **d 列**索引（输出片段的 lane 映射），**不加任何 wave 偏移**，保持原样。

---

## 4. ISA 检查清单

基线取自 `/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-flydsl-attn-bwd-20260917-115934/rounds/022/1-opt/raw/isa_cur_a_k_dkdv.s`（全文件计数，不是单个 body）。

| # | grep | 今天 | 改后期望 | 失败意味着 |
|---|---|---|---|---|
| C1 | `s_barrier`（含 `s_barrier_signal` / `s_barrier_wait`，按前缀 grep） | **0** | **非 0**，对应 4 个 barrier 站点（2 个 body × 2 处） | **0 = block=(128,1,1) 根本没到编译器手里，后面所有数字都不用看。** 这是唯一能区分「真四 wave」和「编译器仍当单 wave」的观测量 |
| C2 | `.max_flat_workgroup_size`（:3341） | 32 | **128**（k_dkdv 与 k_dkdv_sp 都要） | known_block_size 没改 |
| C3 | `.reqd_workgroup_size`（:3344-3347） | 32/1/1 | **128/1/1** | 这才是 `known_block_size` 真改了的证据；C2 单独也可能由 launch bound 满足 |
| C4 | `.amdhsa_group_segment_fixed_size`（:3187）+ `.group_segment_fixed_size`（:3338） | 70656 | **82944**（两处同步） | 70656 = BLOCK_KV 没改；69632 / 279552 / 205312 = 有人做了「复制 staging」或「四段分离」变体 |
| C5 | `v_and_b32 v?, 31, v0` 或 `v_mbcnt_lo_u32_b32 v?, -1, 0`（wave32 下**不应有** `v_mbcnt_hi`） | `v_and_b32` = **0**，`v_mbcnt` = **0** | **至少一个出现** | 两者皆无 = `lane` 还是裸 v0 = E18 没做 = 每个 WMMA 片段地址都是错的。**单条最重要的检查** |
| C6 | `v_lshrrev_b32 v?, 5, v0` | **0** | **出现**，且只在 prologue（`.LBB0_4` 之前），不在任一循环体内 | 出现在体内 = `wave` 被反复重算；根本不出现 = `wave` 没建出来 |
| C7 | `.vgpr_count`（:3353）/ `.vgpr_spill_count`（:3354）/ `scratch_` | 904 / 0 / 0 | **~904 / 0 / 0（不变）** | 每 wave 的寄存器工作集完全没变（NST=32 累加器 ×8 = 256 VGPR，kf+vf = 128，深度 2 的预取元组）。**任何 spill > 0 直接杀死本次改动** —— 904 VGPR × 1 wave/SIMD 已经是最低一档，spill 会在新 barrier 之上再叠 LDS 流量 |
| C8 | `.wavefront_size32`（:3198）/ `.wavefront_size`（:3355） | 1 / 32 | **不变** | 读出 64 = 编译目标被改掉了 |
| C9 | `v_wmma_f32_16x16x32_bf16` | **128**（两个 body 各 64） | **128，不变** | 变了 = 链拆分或 NKV=2 子块结构被扰动 |
| C10 | `v_cmp_gt_i32` / `v_cndmask_b32` | **28 / 32** | **28 / 32，不变** | 掩码的**形状**没变，只有 kv 基址移了；涨了说明有人动了谓词结构 |
| C11 | k_dq / k_dq_sp 的 `.max_flat_workgroup_size` 与 `.reqd_workgroup_size` | 32 / [32,1,1] | **仍是 32 / [32,1,1]**；k_delta_bshd / k_redsp / k_redsp_q 仍是 256 | 任何一个读出 128 = 装饰器改错了内核（四行字面量完全同形）。**这是过度替换的回归守卫，必须和 C2/C3 在同一份 dump 里一起看** |

### C12 —— 如何区分「共享 staging」与「复制 staging」

这是最容易误判的一条，因为**本次第一版故意是复制的**：

| grep | 今天 | 本次第一版（复制，期望值） | 若做了 staging 拆分（本步**不**做） |
|---|---|---|---|
| `buffer_load_b128` | 160 | **160，不变** | ~40 |
| `buffer_load_b32` | 16 | **16，不变** | 16 |
| `ds_store_b128` | 80 | **80，不变** | ~20，且新增约 32 条普通 `ds_load_b128`（qfr/dfr 从 LDS 回读） |
| `ds_load_tr16_b128` | 80 | **80，不变** | 80 |

**本步看到 160 / 80 / 80 就是对的。** 若 `buffer_load_b128` 掉到 40 而**没有**新增 `ds_load_b128`，说明 S/P GEMM 在用这个 wave 从未载入过的操作数 —— 静默错答案。

### C13 —— host 侧 dispatch 维度（不必读 ISA，打印即可）

- prod（非 split，nsp=1）：`grid = (nhkv, nblk, nb) = (8, 64, 4) = 2048` workgroup（今天 `(8, 256, 4) = 8192`）。`nblk = skv//128 = 64`；仍看到 256 说明 impl 传的还是 skv//32。
- proxy（nsp=2）：`grid = (nxs, nblk, nb) = (16, 32, 1) = 512`。
- fast（nsp=16）：`grid = (32, 8, 1) = 256`。

三个 shape 的**总 wave 数**：1024 / 2048 / 8192 —— 与今天逐一相等。dispatch wave 数 0.0625 / 1.00 / 8.00 —— 也与今天逐一相等。**这正是这一步被隔离成「同样的 wave 数，重新分组成四 wave workgroup」这一件事的证据。**

### C14 —— prefetch 的 waitcnt 没被 barrier 毁掉

消费那 4 条 lse/delta `buffer_load_b32` 的等待必须仍是**部分**等待（`s_wait_loadcnt 0x20` 一类），不是 `s_wait_loadcnt 0x0`。`kernels.py:297` 的 `rocdl.sched_barrier(0)` 必须在两个 barrier 重新插入后依然把 4 条 b32 钉在 32 条 b128 之前。若 barrier 的保守 waitcnt 把 32 条 b128 沉到 b32 的消费点之上，这里会塌成 `0x0`，就是 r10.i1.g27 那个 −14.6% 的死法。今天全文件 `s_wait_loadcnt` = 14、`s_wait_dscnt` = 14，可作对照基线。

### C15 —— K/V 片段地址里的 wave 项只出现一次

`kf` / `vf`（:307/:309）在两个循环之外被 hoist，所以 `kvw` 缩放过的基址应当在 prologue（`.LBB0_4` 之前）出现**一次**，不在循环体内。出现在体内 = hoist 断了，有一笔可观的指令数回归要解释。

---

## 5. `fast` 怎么办

**结论：`fast` 的 grid 不会塌，前提是 split-K 还在 —— 而它确实在，且 nsp 不变。**

`fast` = (1, 1024, 1024, 8, 2)。`_wgs = (1024/128)*2*1 = 16`（今天 64）。nsp 循环撞上限：`16*16 = 256 < 512`，`nsp < 16` 为假 → **nsp = 16，与今天完全相同**（今天 `_wgs=64` 也是撞 16 的上限）。两种阈值下都是 16，所以 E22 对 fast 无影响。

于是 `fast` 的实际 dispatch：

| | 今天 | 改后 |
|---|---|---|
| workgroup 数 | `nxs*nblk*nb = 32*32*1 = 1024` | `32*8*1 = 256` |
| 每 workgroup wave 数 | 1 | 4 |
| **总 wave 数** | **1024** | **1024** |
| 被触及的 CU 数（VGPR 限：今天 4 wg/CU，改后 1 wg/CU） | 256 | 256 |

**wave 数、CU 覆盖、机器铺满程度全部一致。** `fast` 的测量不会因为 workgroup 变粗而失真。

split 粒度也没退化：block m 的 `_fn = nqp_eff - nmaskp = (32-4m) - 4 = 28-4m`，nsp=16 时 `_ch = ceil(28/16) = 2`（m=0）—— 与今天 `_ch = ceil(31/16) = 2` 同量级。

**必须写进实验记录的警告：`fast` 从不走非 split 路径。** 若有人为了「简化这一步」关掉 split-K 只测 `k_dkdv`，`fast` 会掉到 16 个 workgroup = 16 个 CU = 全机的 6.25% 占用，测出来的数字与本次改动的真实收益毫无关系。**A/B 必须在 nsp = 16 / 2 / 1 三个今天的值上做。**

`prod` 的 nsp = 1 在新阈值下是 `2048 < 512` 为假得到的，离边界 4 倍远（旧阈值下是 `2048 < 2048` 恰好为假，**踩在边界上**）。E22 顺带修掉了这个脆弱点。

---

## 6. 全部编辑落地之后，还可能静默出错的地方，以及最便宜的检查

按可见度从低到高：

**R1 — `kvw` 只漏了 `:431` 一处。**
非因果测试全过；因果 SQNR 降一点点，落在 bf16 噪声里。
→ 最便宜的检查：把 `ut/test_correctness.py` 的因果 case 跑在 `sq_gt_skv` **之外**的某个 GQA shape 上，并把 SQNR 与今天的值逐 shape 对比（不是只看过不过 50 dB 门）。因果 SQNR 相对今天掉 3 dB 以上 = 这条。

**R2 — 四个 wave 全部的 Q/dO 全局载入并没有被 L1 吸收。**
这是**整步的承重假设**，我没有也无法静态验证。四个 wave 在 barrier 锁步下发同样的地址，指令数与今天逐条相同；省下来的只有 L2/HBM 往返。如果 CU 的 TCP 不把它们合成 ~1 次 L2 请求，本步的流量收益**为零**，+13.6% 无从谈起。
→ 最便宜的检查：一次 profile，看 `k_dkdv` 的 L2 读请求数。应当降约 4 倍；若持平，**立刻停手**，这一步需要换成「把全局载入按 wave 切分 + qfr/dfr 从 LDS 回读」的更大改造（那要第三道 barrier、要重写 `_ldqd`、要动 `_NP = 4 + 32` 的携带元组，是另一个 change）。

**R3 — B ring 的 `ds_store` 流量翻了 4 倍。**
每迭代每 CU 多出 96 条 `ds_store_b128`（这是复制 staging 的既定代价，C12 期望值 80 就是它）。我没有 gfx1250 的 LDS 写带宽模型，说不准它会不会吃掉收益。
→ 最便宜的检查：若 R2 确认 L2 流量降了 4 倍而端到端收益明显低于 +13.6%，这是第一嫌疑人。下一个 arm 就是把 staging store 拆到每 wave 只写自己的 1/4（利用 `NDT == 4 == wave 数`，用 `(wave == dt).select(real_ptr, dump_ptr)` 做无分支门控，`dump` 取 `_lds0 + LDS_SEG + 2*32*S_ROW_B`，`:244` 相应加宽 32 B → 82976）。**本步不要做。**

**R4 — barrier 耦合本身的回归。**
今天一个 CU 上是 4 个**互相独立**的单 wave workgroup：4 条不耦合的指令流，没有会合点，一条流等 waitcnt 时另外三条照跑。改后是 1 个 workgroup 的 4 个 **barrier 耦合**的 wave：每个 barrier 是全 CU 会合，而且**没有第二个 workgroup 可以切过去掩盖它**。G1a 那次测量（多一条 `s_wait_dscnt 0x0`、`.LBB0_8` 短 6 条、VGPR/spill/LDS 不变）是在**单 wave**下取的，那里 LLVM 随后就把 barrier 删了 —— 它只界定了「保守 waitcnt」那一半风险，**会合那一半是无界的，除非有构建**。这不会产生错答案，但它是 +13.6% 落空最合理的解释。
→ 最便宜的检查：C1 确认 barrier 真的在 ISA 里；然后直接看端到端时间。若 C12 的四个计数全部符合「复制」期望、C7 的 VGPR/spill 不变、R2 的 L2 流量确实降了 4 倍，而时间不降反升 —— 就是这一条。

**R5 — `wave` 落在 VGPR，把 `kvw` / `wcol` / `off` / `col` 的全部地址算术都变成 per-lane VALU。**
结果正确，只是浪费，而且会顶高 VGPR（C7 会抓到）。
→ 检查：C6，看 `v_lshrrev_b32 v?, 5, v0` 后面有没有 `v_readfirstlane_b32 s?, v?`。没有就是停在 VGPR 里。修法是 `rocdl.readfirstlane`，不是 `rocdl.wave_id()`（TTMP8 初始化未验证，见 E18）。

**R6 — masked body 变重。**
`nmaskp` 从 1 变 4，masked body 是 `carry=False`（无预取）且带 93 条掩码指令。PARTIAL 路径下 `:568` 把这 4 个 masked pair 全压给 split 0，split 1..nsp-1 一个也不摊；proxy 在 nsp=2 下 split 0 的额外负担从 1 对变 4 对。
→ 检查：纯性能，不会错。若 proxy 相对 prod 异常落后，先看这一条。

**R7 — 契约收窄被遗忘。**
`Skv % 32 == 0` 变成 `Skv % 128 == 0`。E21（docstring）+ E24（toy shape）+ 实验记录三处必须一起写。`validation.py` 只跑 SPEC_SHAPES，**抓不到**；只有 `ut/test_correctness.py` 的 edge-case 套件会抓到，而且是响亮的 AssertionError。
→ 检查：先跑 `python3 ut/test_correctness.py`，**在任何 benchmark 之前**。

---

## 7. 确认**不需要改**的地方（供施工者放心跳过）

- `kernels.py:50` `WAVE = 32` —— shuffle 宽度，唯一消费者是 :139 的 k_delta_bshd 的 warp 内归约。改成 128 会静默坏掉一个无关内核。
- `kernels.py:48-49` `NKV = 2` / `NST` —— **每 wave**量，不变（注释里的 "per workgroup" 可选改成 "per wave"）。
- `kernels.py:243-249` —— LDS 分配与四个偏移已全参数化，**零改动**自动得到 82944 / +0 / +8704 / +65536 / +74240。
- `kernels.py:320-323`、`:544-550` —— 必须保持 workgroup 级，见 §3.1。
- `kernels.py:552` `init`、`:532` `_NP = 4 + 32`、`:533-537`、`:571-574`、`:583-584` —— 携带元组宽度由 `_ldqd` 的返回长度决定，`_ldqd` 不动，全部不变。
- `kernels.py:593` 的 `row` 是 d 列索引，不加 wave 偏移。
- `kernels.py:477-479` B ring 读，不加 wave 偏移。
- `kernels.py:379-390` B ring 写，**本步不动**（见 R3）。
- `kernels.py:645-648` 与整个 `_dq_impl`（:655-995）、`:997`、`:1006`、`:1023`、`:1035`、`:855`、`:877` —— k_dq 全链路与 BLOCK_KV 完全解耦（`grep -n BLOCK_KV kernels.py` 在 645 行之后零命中），**一个字都不能动**。`:668` 的 `lane = fx.Int32(fx.thread_idx.x)` 在 block=(32,1,1) 下 `thread_idx.x` 就是 lane，改它今天无害，正因如此才危险 —— 它违反「k_dq 不动」的明文要求，且会在有人改 k_dq block size 的那天静默出错。
- `kernels.py:111`、`:154`、`:1068`、`:1118`、`:1129`、`:1170` —— k_delta_bshd / k_redsp / k_redsp_q 的启动几何，不变。
- `impl.py:115-119`、`:153`、`:174`/`:181` 的 `sq // 16`、`:175`/`:182` 的 `skv // _k.BLOCK_KV`、`:192-193`、`:208-211` —— 全部自动跟随或与 BLOCK_KV 无关。

---

## 8. 未验证事项（诚实清单）

没有编译、没有上卡，全部是读码与算术推导。

1. `fx.lane_id()` 在 gfx1250 ROCm 后端的降级路径 —— 我确认了 Python 导出链（`expr/gpu.py:67` + `__all__:41` + `expr/__init__.py` 的 `from .gpu import *`）和它已返回 Int32，但本树里**没有一个 gfx1250 内核用过它**。C5 就是为这条准备的。
2. `rocdl.wave_id()` 依赖的 TTMP8[29:25] 是否被普通 `@flyc.kernel` 的 dispatch 路径初始化 —— 未验证，因此它被标为 defer-only。
3. 四个 wave 的相同 Q/dO 全局载入是否真的被 L1 吸收（R2）—— **整步的承重假设**，只有 profile 能回答。
4. 四 wave 下 barrier 的会合代价（R4）—— G1a 的测量在单 wave 下取得，只界定了保守 waitcnt 那一半。
5. 「904 VGPR → 1 wave/SIMD」沿用 r12.i0.g38 的结论；我没有独立查证 gfx1250 wave32 的 SIMD 寄存器文件大小，也没验证「一个 workgroup 的 4 个 wave 必须同 CU 共驻」（这是 LDS 为 per-CU 资源的标准推论）。注意 `kernels.py:240` 的注释写 740 而构建物是 904 —— 注释与构建物本来就不一致。
6. bank 判据 `gcd(P/4, 64) = 4` 是我自己从 `(addr/4) mod 64` 推导并用脚本枚举验证 16 行单射性的；它假设 bank 函数就是 `(addr/4) mod 64`，且一次 `ds_load_tr16_b128` 相位恰取 16 个不同行（后者从 `lane_r` 的定义读出，前者未独立验证）。跨 wave 的 bank 仲裁行为（四个 wave 在 64 B 相隔的列上并发命中同两个 ring）是新的，未建模。
7. `impl.py:167` 阈值 512 对**非计分** shape 的影响未逐一验算（三个计分 shape 已复算，全部还原今天的 nsp）。
8. 我没有检查 `kernels.py` / `impl.py` / `ut/` / `validation.py` 之外是否还有别处硬编码了 `BLOCK_KV=32`、`block=(32,1,1)` 或 `Skv % 32` 的假设。