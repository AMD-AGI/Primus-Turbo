# Round 14 opt (fast round, GPU 2 / fa-g2) -- route design

op/current = round 13 (g51). Same-session r13 (rounds/013/1-opt/act.yaml): fast 162.3 vs beat 153.75, proxy 1606
vs 1582.5, prod 1502.8 vs 1556.6 (0.9655).

## What I read
- findings: facts.md, dead_ends.md (headings + r12.i3.g49), pool.md, route.md (h2, h4, h18 index, operator tables,
  route); rounds/013/1-opt/opt.md; rounds/010/1-profiling/profiling_summary.md (round 10, before rounds 11 and 13.
  Neither touched the m32x8 prod kernel, whose ISA has been byte-identical since, so its prod counters still apply).

## The one reading that decides this round
The score is the capped mean of min(x/beat, 1) (r13 act acceptance line). fast (1.055) and proxy (1.015) are both
capped at 1.0. **Only prod (0.9655) can move the score.** So every fast-only lever is worth 0 now, including
r5.i3.g15 (split-KV, still blocked on h16), g51 follow-ons and any fast chain work. This round's candidates must
move prod.

Prod is power/clock-bound (round 10 g24: -6.6% cycles came back as x0.901 clock, wall x0.993). The r13 census puts
the clean tile's VALU at the algorithm's floor, and removing issue-only work is dead on this card.
routes/1-metrics-to-techniques.md (power-wall paragraph) says what pays on the power wall: "fewer bytes at full value,
fewer instructions at 20-45%". Nobody in this job has priced the BYTES term at prod.

## Bytes at prod (arithmetic)
m32x8: BLOCK_M 256 packed rows = 64 seq positions x gqa 4. Every WG TDM-loads each KV tile it needs,
64 keys x 128 d x bf16, K and V = 32 KB/tile. Prod has 4 x 8 kv heads x 128 q tiles = 4096 WGs, with ~64.5 tiles
each on average (causal) -> ~2.1 MB/WG -> **~8.5 GB L2->LDS per call**, ~5.8 TB/s at 1.46 ms. Priced at 5-10 pJ/B
that is ~40-85 mJ of a call's ~1.9 J (1.46 ms x ~1.3 kW), i.e. 2-4.5%. That is the same size as the prod gap (3.5%).
The number is uncertain by ~3x, so it is worth a probe.

## Instrument r14.i1.g52 -- throwaway probe: skip half the K/V TDM loads (prices the bytes term at prod)
Arm `arms/halfkv` = op/current with the m32x8 prefetch issued only for tile pairs (nxt-start)//2 even (tiles 0,1
loaded, 2,3 skipped, 4,5 loaded, ...). Each skipped tile reuses the slot's previous (real, random) tile, so the
compute, the data statistics, the masking and the speculation path are all unchanged. L2->LDS bytes halve. The output
is wrong. The m32x2 fast module is untouched.
Prediction before running: prod +2..4% (x1.02-1.04) if the bytes energy is what the power model says; <= +1% means
bytes are not a prod term and multicast is dead unbuilt. Proxy: small, because it is not power-bound the same way.
What would change my mind: a gain at prod that comes with a clock RISE (sclk_start/end) confirms the energy
reading. A gain at flat clock means TDM latency, which multicast would also buy but for a different reason.
Protocol: rounds/014/1-opt/raw/halfkv/run.sh, 3 rotated sessions {cur, halfkv, champ2 (A/A byte copy)}, proxy+prod,
no beat in process, PYTHONHASHSEED=0, GPU 2 idle before (0%, only the standing UNKNOWN KFD pid with 0 VRAM).
Launched 05:58:11Z.

## FlyDSL cluster support (static check, no card)
flydsl 0.3.4.1 in the container has: a `cluster=` launch kwarg (kernel_function.py:420, lowered to gpu.launch_func
cluster_size); `expr/rocdl/cluster.py` (cluster_barrier = WG barrier + one-wave cluster signal + wait, cluster
workgroup ids); and `tdm_ops` descriptors with a `workgroup_mask` (the MCAST mask in GROUP1 [15:0]). So the build is
expressible without touching FlyDSL. The package ships no in-package kernel that uses it, so the API is unproven here.

### g52 result (raw/halfkv/, raw/halfkv_summary.txt) -- the bytes term at prod is ~4% per half of the fetch
TF/s per session (blocked ruler, medians of 101):

| session order | cur | halfkv | champ2 (A/A) | halfkv/cur | champ2/cur |
|---|---|---|---|---|---|
| prod s1 cur halfkv champ2 | 1501.90 | 1564.12 | 1505.77 | **1.0414** | 1.0026 |
| prod s2 halfkv champ2 cur | 1498.66 | 1561.20 | 1497.70 | **1.0417** | 0.9994 |
| prod s3 champ2 cur halfkv | 1504.12 | 1561.91 | 1501.00 | **1.0384** | 0.9979 |
| proxy s1 / s2 / s3 | 1621.0 / 1619.9 / 1631.8 | 1613.0 / 1615.7 / 1626.4 | 1617.9 / 1607.3 / 1634.1 | 0.995 / 0.997 / 0.997 | 0.998 / 0.992 / 1.001 |

- **Prod +3.8..4.2% in 3/3 sessions, A/A within 0.3%.** Halfkv's prod medians (1561-1564) sit above beat's 1556.6
  (r13). Proxy is null (-0.3..-0.5%, inside its +-0.8% A/A).
- My prediction was +2..4%. It held, at the top of the range. My doubt, that beat moves 2x our bytes per FLOP and
  still uses less energy, did not hold as a veto. Beat's energy is lower for other reasons, and bytes are a real
  term for us too.
- Only prod pays, and prod is the power-bound shape. That fits energy (bytes cost power, and the power cap turns
  them into clock), not latency: proxy is latency-exposed at 2 WGs/CU and gains nothing. The per-arm clock cannot
  be split, because benchmark.py records sclk per shape, not per arm.
- The probe does not yet say WHICH bytes: the L2->LDS transfer, or L2 misses to MALL/HBM. Prod's KV working set is
  32 (b,kvh) x 4 MB = 128 MB, far more than proxy's 16 MB. That is the next instrument.
- GPU 2: 0% before; 6% after (the monitor-read tick). No new amdgpu dmesg line.

## Instrument r14.i1.g52, part 2 -- the same loads, pointed at a 2-tile window (raw/l2hit/, raw/l2hit_summary.txt)
Arm `arms/l2hit` = op/current with every m32x8 prefetch's `kv_row0 = kv_start + (nxt_row0 % 128)`. Every TDM op,
L2->LDS byte and LDS write is kept, but each WG re-reads tiles 0/1 of its own (b,kvh), so the KV working set
collapses to 64 KB per (b,kvh). Wrong output.
Question: is halfkv's +4% the transfer (then only multicast helps) or L2 misses past L2 (then order/locality helps)?
Prediction: I leaned toward the transfer, expecting l2hit <= +1%.

| session | cur | l2hit | halfkv | champ2 | l2hit/cur | halfkv/cur | A/A |
|---|---|---|---|---|---|---|---|
| s1 cur l2hit halfkv champ2 | 1503.51 | 1599.99 | 1564.76 | 1499.15 | **1.0642** | 1.0408 | 0.9971 |
| s2 l2hit champ2 cur halfkv | 1505.13 | 1600.02 | 1561.67 | 1507.20 | **1.0630** | 1.0376 | 1.0014 |
| s3 champ2 halfkv l2hit cur | 1500.44 | 1603.00 | 1563.52 | 1502.78 | **1.0684** | 1.0420 | 1.0016 |

**My prediction was wrong.** l2hit, which moves exactly as many bytes, pays MORE than halfkv: +6.3..6.8% in 3/3.
Prod's cost is **L2 misses** (traffic from MALL/HBM), not the L2->LDS transfer. Halving requests (halfkv) removed
part of the misses. Making all of them hit removed ~6.5%. That is ~2x the prod gap to beat (3.5%). No counter can
confirm this on gfx1250 (profiling-surface.md: 0 TCC, 13 live counters, no byte counter at any level), so the
two probes ARE the instrument.

### Why prod misses and proxy does not (static reading of `_lpt_block_id`)
Dispatch round-robins over the 8 XCDs by lin % 8. The r1 g01 remap is `rank = lin // (gy*gz)`, `y = rem % gy`,
`z = rem // gy`. At prod (gx 128, gy 8, gz 4) XCD = lin % 8 = y, so **each XCD streams one kv head for all 4
batches at once**. That is 4 KV streams x 4 MB = 16 MB live per XCD, re-read by WGs that started at different times.
At proxy (gz 1) an XCD holds one 2 MB stream, and halfkv is null there. That fits.

## Idea r14.i2.g53 -- zmajor dispatch (one (b,kvh) KV stream per XCD at a time) (raw/zmajor/, raw/zmajor_summary.txt)
Arm `arms/zmajor` = op/current with `_lpt_block_id` nesting y fastest, then the LPT rank, then z slowest. XCD = y is
kept, but an XCD now works through its 4 batches one after another instead of all at once. Bijection checked at prod.
At proxy (gz 1) the mapping is identical to current. **Bitwise equal to op/current** (raw/zmajor/check.txt: 5 shapes x
causal/non-causal, ALL_BITWISE_EQUAL True), because the math per WG is unchanged and only the dispatch order moved.
Prediction: recovers a large part of l2hit's +6.5%.

| session order | cur | zmajor | l2hit | champ2 | zmajor/cur | l2hit/cur | A/A |
|---|---|---|---|---|---|---|---|
| s1 cur zmajor l2hit champ2 | 1501.74 | 1500.88 | 1595.44 | 1506.23 | **0.9994** | 1.0624 | 1.0030 |
| s2 zmajor champ2 cur l2hit | 1505.63 | 1501.35 | 1601.53 | 1504.31 | **0.9972** | 1.0637 | 0.9991 |
| s3 champ2 l2hit zmajor cur | 1504.33 | 1503.28 | 1600.32 | 1501.90 | **0.9993** | 1.0638 | 0.9984 |

**Null, and my prediction was wrong again.** zmajor x0.997-0.999 is inside A/A. l2hit reproduced at +6.2..6.4% in a
second, independent run. So concurrency across the 4 batches on one XCD is not what costs. What remains is inside ONE
(b,kvh) stream of 4 MB: its re-reads by the ~32-64 co-resident WGs of that XCD do not hit well enough, or the cost is
address-footprint-related (UTCL2/TLB, MALL), which l2hit also collapses and zmajor does not. Without byte or TLB
counters these two readings cannot be told apart. g53 is closed and goes to dead_ends. It is not taken to the merge
step: it tied rather than won, so a "merge" would just be g52's wrong-output probe.
GPU 2: 0% before, 6% after (monitor tick), no KFD pid. No new amdgpu dmesg line (the last ones are 01:44 MES and
03:01 ifoe, both before this round).

## Decisions
- The prod lever that is measured and still open is fewer K/V **requests** into L2. halfkv (+4%) is its upper bound:
  it halves L2 requests AND LDS writes with zero synchronization. The build that gets it with correct output is
  **2-WG cluster TDM multicast** of K/V (r14.i3.g54). The pair is two adjacent q tiles of the same (b,kvh). Each WG
  issues half of every tile with `workgroup_mask` = both, and a cluster barrier guards slot reuse. It halves
  L2->CU requests but NOT LDS writes, so it can expect at most ~+4% and loses whatever the per-tile cluster barrier
  costs. The gap to beat is 3.5%, so the margin is thin, and the route puts a cheap null-check first.
- r5.i3.g15 is worth 0 on the score now (fast is capped). It stays blocked on h16.
- Consulted (corpus): backends/hipkittens/attention/recipes/gqa_d128.md (power wall, sec. 9);
  optimization/routes/1-metrics-to-techniques.md (A5/A7, power paragraph); backends/flydsl/attention/README.md;
  arch/gfx1250/gfx1250.md (workgroup clusters); optimization/techniques/6-gfx1250-cdna5-mechanisms.md (sec. 2
  multicast, sec. 3 split barrier); backends/hipkittens/gemm/recipes/bf16_gfx1250_ladder.md (multicast +6.53%, <=16
  WG/cluster, >5-destination masks demoted); arch/gfx1250/profiling-surface.md (no TCC/byte counters).

# Build step -- route row 4, r14.i3.g54 (2-WG cluster TDM multicast of K/V)

## What was built (rounds/014/op/flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py only)
- `PAIR_CLUSTER` / `PAIR_MCAST` module constants. `flash_attn_batch_m32x8` (bshd) selects `pair` when the grid x is
  even, there is no left window and there is no sink. That covers proxy (gx 64) and prod (gx 128). Odd-gx and other
  shapes keep the old kernel, and the m32x2 fast path is untouched. `pair` is a compile-time builder arg and is part of
  the launch-fn key.
- Launch: `cluster=(2,1,1)`.
- `_lpt_block_id(axis, pair)`: LPT over pairs (lin//2). The two WGs of a hardware cluster (bx 2k, 2k+1) get x = 2p, 2p+1
  of the same (y, z).
- `n_shared = min(kv_len_wg(self), kv_len_wg(peer)) // n_block`, computed with the same formula the peer uses. Tiles
  [0, n_shared) are full for both WGs, and both run exactly those iterations (start_tile 0).
- `_drain_barrier`: a cluster barrier (flydsl `cluster_barrier` = WG barrier + wave-0 signal + wait) when t <
  n_shared, the WG barrier otherwise. After it the peer has drained (tensor_wait) its writes of tile t into our slot,
  and it has finished reading tile t-1 out of the slot that t+1 is multicast into.
- Prefetch of a shared tile (nxt < n_shared): WG s=0 issues K, WG s=1 issues V, each with `workgroup_mask=3` via
  `fx.atom_set_value`. Other tiles, and the prologue tile 0, load as before.
- Probe arm `arms/pairbar` = the same tree with `PAIR_MCAST=False`: cluster launch + paired remap + cluster barriers,
  with loads unchanged. This is route row 4's null check.
- Compile cache (`/tmp/flycache` in fa-g2, FLYDSL_RUNTIME_CACHE_DIR) cleared at the start of raw/pair/run_check.sh.

Expected (route row 4 / pool g54, written before measuring): prod +2..4% if the cluster barrier costs < ~1%/tile;
upper bound halfkv's +4%. proxy/fast unchanged. Output bitwise = op/current (same math per WG).
Correctness protocol: toy-first in its own process (AMD_SERIALIZE_KERNEL=3), direct m32x8 calls; then bitwise vs
op/current on 8 impl cases incl. prod and proxy (raw/pair/check.py). If the cluster did not form, the multicast arm's
WGs would each receive only K or only V, so the bitwise check also proves the cluster formed.

## Build failures
1. The multicast arm was finite but not bitwise equal on every paired (even grid x) case. The odd-grid case and the probe
   were bitwise. Mechanism: I built a split issue, where WG 0 issued K and WG 1 issued V, each with workgroup_mask=3. TDM
   multicast is a join: every WG named in the mask issues the identical load, and the hardware merges the fetch. A WG
   that does not issue receives nothing, so each WG held only half of K/V. Sources: FlyDSL
   tests/unit/test_tdm_mcast_add_gfx1250.py and kernels/gemm/gemm_bf16_gfx1250.py (every masked WG issues its full
   tile), and the corpus text on early_timeout ("returns the load to whichever requesters have already joined").
   Fix: both WGs issue K and V with the mask. The failed tree and its outputs are in raw/pair/attempt1.
   There were no further failures.

## Correctness (raw/pair, raw/gates, raw/measure/validation.txt)
- Toy first, in its own process with AMD_SERIALIZE_KERNEL=3, then 8 impl cases including proxy and prod: bitwise equal
  to op/current, for both the probe and g54. No new amdgpu dmesg line during any run.
- ut/test_correctness.py: RESULT PASS. The 22-case armcheck is bitwise (BAD 0). verify_r6 (A: 16 ut rows; C: 192
  adversarial cases) ran with BOTH arms forced through m32x8, so the pair kernel actually runs: G54_CHECK PASS.
- validation.py: every correctness row passes at 49.0 dB (min 49.82 on short_q full, the same as op/current);
  determinism 200/200. rc 2 comes from the speed bar only.
- ISA (prod-like pair kernel): 0 spill; VGPR 454 for champ2, probe and g54 alike; code 30 780 / 31 424 / 31 596 B
  (cap 32 640). workgroup_mask=3 is present in 4 TDM group-1 descriptors (0x7510003, 0xf510003) and absent from the
  probe. That proves the mask reaches the ISA, but not that the hardware merged the fetches. The timing below
  (g54/probe prod x1.023) is the evidence that something was merged.

## Measurement (raw/measure, fa-g2, launched 06:24Z after the gates)
3 rotated sessions, each one process: r13 (incumbent = op/current), r10, g54, pairbar (probe) and champ2 (A/A byte
copy). Beat ran in its own process afterwards. rocm-smi GPU use was 0% before and 2% after. Other users' processes were
on GPUs 0 and 3 throughout (shared board power), not on this card. Mean of the 3 session medians, TFLOP/s:

| shape | r10 | r13 (incumbent) | champ2 (A/A) | probe | **g54** | beat (own process) | g54/r13 per session | A/A champ2/r13 |
|---|---|---|---|---|---|---|---|---|
| fast | 142.98 | 163.59 | 163.26 | 163.09 | 163.26 | 156.67 | 0.994 / 1.003 / 0.997 | 1.000 / 1.000 / 0.994 |
| proxy | 1623.59 | 1619.86 | 1627.66 | 1537.62 | **1535.20** | 1497.88 | 0.948 / 0.950 / 0.945 | 1.009 / 1.002 / 1.003 |
| prod | 1491.82 | 1500.53 | 1500.31 | 1475.25 | **1509.65** | 1561.58 | 1.006 / 1.008 / 1.004 | 0.999 / 1.001 / 0.999 |

Capped score: g54 0.9889, r13 0.9870, champ2 0.9869, probe 0.9816, r10 0.9560.

## Verdict: delivered and correct, but it fails route row 4's gates. It is left in the working copy as measured.
- Probe gate (prod cost <= 1%): prod x0.983, proxy x0.950. **Fail.** The route said not to build the multicast in that
  case. I built both before the probe's number existed, to save card startups; the gate is applied here.
- prod >= x1.02: x1.006. **Fail.** The gain is consistent across all 3 sessions and sits just outside the A/A range,
  but it is small.
- proxy not below its A/A: x0.946 in all 3 sessions. **Fail.** The route's fallback, pairing only when gz > 1 (proxy
  has B=1), would restore proxy. I did not build it because prod would still be x1.006, below the x1.02 gate.
- Split into parts: the multicast over the probe is prod x1.023, proxy x1.000. Merging the pair's K/V fetch is worth
  about 2.3% of prod, about half of halfkv's +4% upper bound. The pairing and cluster barrier alone cost 1.7% on prod
  and 5% on proxy, so they eat most of it. Where the probe's cost comes from (lock-step skew between the pair's
  unequal causal lengths, or the cluster's co-scheduling constraint on dispatch) is not measured. It is a hypothesis.
- **The acceptance rule, read literally, may pass g54, and that would be wrong to rely on.** Beat's own-process proxy
  read 1497.88 this session, against 1582.50 in r13's session and 1678.48 in-process in this validation. g54's proxy
  (1535) therefore counts as "at target" only because the beat reading was low. The -5.4% proxy loss is real and
  repeats in every session. My recommendation: do not promote g54.
