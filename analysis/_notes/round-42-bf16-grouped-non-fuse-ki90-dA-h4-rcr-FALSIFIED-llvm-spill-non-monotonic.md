# Round 42 — BF16 grouped, non-fuse `KI=90` specialization for gpt_oss-GateUP dA H4 RCR — FALSIFIED

## Goal coming in
R41 wall-decomposition probe (`/tmp/probe_r41_wall_decomp.py`) showed bwd
is the bottleneck (64-71% of wall, HK / Triton ratio 1.01-1.09). R42
deepened that diagnostic with a direct dA / dB split via
`/tmp/probe_r42_dadb_split.py` (calls `grouped_gemm_impl` and
`grouped_gemm_variable_k_impl` directly, bypassing the autograd
`Function.backward` which always runs both).

Per-component ratios (5/3/2026, GPU 3, paired):

| shape | fwd_r | dA_r | dB_r | wall_r |
|---|---|---|---|---|
| DSV3-GateUP-B32-M2048 | 1.191 | 1.107 | **1.015** | 1.099 |
| **gpt_oss-GateUP-B32-M2048** | 1.138 | **0.977** ★ | 1.052 | 1.053 |
| **gpt_oss-Down-B32-M2048** | 1.138 | **0.942** ★ | 1.061 | 1.042 |
| gpt_oss-GateUP-B32-M4096 | 1.119 | 1.081 | 1.051 | 1.083 |
| gpt_oss-GateUP-B4-M2048 | 1.090 | 1.077 | 1.093 | 1.086 |
| Qwen3-Down-B32-M2048 | 1.165 | 1.105 | **1.027** | 1.100 |

The two metric-worst shapes (`gpt_oss-{GateUP,Down}-B32-M2048`) had
**dA_r < 1.0** (HK losing to Triton on dA). dA = 33-37% of wall, so
lifting dA from 0.94-0.98 to 1.05+ would bring wall ratio from 1.05 to
~1.08 — a real (~3pp) win, ~+10 score after weighting.

## What dA hits

For `grouped_gemm(a, b, trans_b=True)` (HK convention RCR):
- `grad_a = grouped_gemm_impl(grad_out, b, trans_b=False)`
- For gpt_oss-GateUP fwd `K=2880` (K%128==64): `b.shape=(B, 5760, 2880)`,
  `b.shape[-1]=2880` is **not a multiple of `BLOCK_SIZE=256`** ⇒ H4
  reroute fires (`grouped_gemm_impl.py`):
  - b is transposed in-flight to `(B, 2880, 5760)`, `trans_b → True`.
  - kernel sees `K_kernel = a.shape[1] = 5760` (was N_fwd), `N_kernel
    = 2880` (was K_fwd).
- After H4: `K_kernel=5760` is **K%128==0**, so `fuse_ktail_eligible
  = false`, **non-fuse path**.
- `K_STEP=64`, `K_TWO_TILE=128`, `fast_k = (5760/128)*128 = 5760`,
  `g.ki = fast_k / K_STEP = 90`.

R40 had added `INSTANTIATE_K_GRP(88)` and `case 88` thinking this was
the path — but the real value is 90, so case 88 was **dead code**.
R42 fixed the off-by-2.

## Hypothesis (R42)

Adding `INSTANTIATE_K_GRP(90)` and `case 90` will:
- compile clean (R40 R40 KI=88 was 0 spill on RCR/RRR/CRR — KI=90 is
  +2 iters, expected similar);
- give the LLVM scheduler full visibility into the K loop, enabling
  better s-pipelining + barrier removal;
- lift dA_r on gpt_oss-GateUP-B32-M2048 from 0.977 → ~1.05;
- improve wall ratio by ~2-3pp on the 4 gpt_oss-GateUP shapes (only
  GateUP triggers the H4 path with K_kernel=5760; gpt_oss-Down has
  K_kernel=2880 which falls into the fuse path KI=44 instead).

Expected score lift: +5-10 (gpt_oss has weight 3x, GateUP has 4
shapes × ~2pp wall ratio improvement).

## Evidence

### Build (KI=90 added)

```
RCR / KI=90 / fuse=0 :  VGPRs:256  Spill:19  Occupancy:2
RRR / KI=90 / fuse=0 :  VGPRs:256  Spill:13  Occupancy:2
CRR / KI=90 / fuse=0 :  VGPRs:256  Spill: 0  Occupancy:2
```

**Surprise — KI=90 RCR spills 19 VGPRs**. R40's KI=88 (just 2 iters
fewer) was 0/0/0 spill. The spill is **non-monotonic in KI** — LLVM's
register allocation is highly sensitive to small unroll changes for
this kernel.

Compare nearby cases (R40 / repo log):
| KI | RCR spill | RRR spill | CRR spill |
|---|---|---|---|
| 56 | 14 | 14 | 13 |
| 64 | 0 | 0 | 0 |
| 88 (R40) | 0 | 0 | 0 |
| **90 (R42)** | **19** | **13** | **0** |
| 112 | 19 | 0 | 0 |
| 128 | 24 | 0 | 0 |

KI=90 looks structurally closer to KI=112 than KI=88 from LLVM's
liveness-graph point of view. Two extra iterations is enough to push
the live-range graph past the threshold where spill kicks in for the
RCR variant (and partially the RRR variant).

### Metric

```
baseline (32b133a, R41 HEAD):   884
R42 v1 (KI=90 specialization):  886    Δ = +2  (sub-noise)
post-revert recheck:            884    revert clean
```

Per-shape Δ on the 4 gpt_oss-GateUP shapes (only ones touching KI=90):
| shape | baseline ratio | v1 ratio | Δ |
|---|---|---|---|
| GateUP-B32-M2048 | 1.055 | 1.051 | -0.004 |
| GateUP-B32-M4096 | 1.089 | 1.089 | 0.000 |
| GateUP-B4-M2048 | 1.113 | 1.110 | -0.003 |
| GateUP-B4-M4096 | 1.106 | 1.112 | +0.006 |

Mean Δ ≈ 0. gpt_oss family geomean: 1.0936 → 1.0964 (+0.0028,
within run-to-run noise of ~±0.005).

The Down family was unaffected (KI=90 doesn't fire — Down-dA hits the
fuse path with KI=44).

## Mechanism — why it didn't move

The 19 VGPR spill on RCR cancels the unroll benefit. RCR is the
exact variant that gpt_oss-GateUP dA H4 hits. Each spilled VGPR
becomes ~1 LDS R/W round-trip per iteration on the K loop hot path,
which costs roughly the same as an MFMA cycle when occupancy drops
from 2 → 2 (occupancy stayed at 2 — but 19 spill at occupancy 2 still
means scratch traffic, just not occupancy-bound).

Net effect: the unrolled-90 kernel has the same wall-clock as the
KI=0 dynamic kernel **on this specific shape**, give or take noise.

R40's case 88 was dead code, so R40's "KI=88 0-spill compile-clean"
result was correct but irrelevant. **The relevant case 90 is not
spill-clean**, ending the line of reasoning that "long-K KI
specializations are spill-free and just need to be added".

## Falsification consequence

The conclusion from R39 + R40 + R42 is now:
- compile-time `KI` specialization is **hyper-sensitive to LLVM
  register allocation**;
- there is no monotonic "longer KI ⇒ more unroll ⇒ more win"
  relationship — even +2 iters can flip a 0-spill case into 19
  spill;
- the values that compile clean (56/64/88/...) are **already in
  the table**;
- the values that *would* help the metric's bottleneck dA paths
  (KI=90 for gpt_oss-GateUP, KI=24/32/48 for Qwen3-Down /
  gpt_oss-Down dA fuse) all spill, and the unroll benefit ≤ spill
  cost.

**This closes out the KI-specialization attack surface for BF16
grouped backward.** Further KI work would require either:
1. Changing the kernel body (less live state at unroll-eligible
   points) — risk: it'll affect all KI variants, easy to break the
   already-winning paths.
2. Instead of `KI` specialization, do `K_TWO_TILE` (k_step) tuning —
   different K granularity may move the LLVM scheduling sweet-spot.
3. Skip non-fuse short-K and target the **var-K kernel directly**
   (dB path) — R41 said dB share is 33-37% of wall, and dB_r on
   most shapes is already 1.05+, so the headroom there is smaller
   than the dA headroom on B32-M2048, but nothing else has yielded.

## Action

- HK kernel reverted to 32b133a state (no diff).
- Primus-Turbo: this round-42 note added; no code change.
- HK: no commit (kernel reverted).
- Primus-Turbo: 1 commit (this falsification note + r42 dadb-probe
  result table embedded in commit body).

## R43 next-action surface

Diagnostic + decision tree (carry forward from R41 + R42):

1. **Var-K (dB) kernel kernel-body audit.** Read
   `/workspace/code/HipKittens/analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp`
   `grouped_var_k_kernel`. Look for:
   - Pre-pipeline structure (does it look like the main `grouped_kernel`
     load-mfma overlap or is it simpler?)
   - LDS swizzle for K-major (var-K transposes M and K)
   - Atomic / barrier patterns specific to the variable-K dispatch.
   This is a 30-min scout, no code change. If the var-K kernel looks
   under-pipelined relative to `grouped_kernel`, that's the new
   direction.

2. **Single-shape rocprofv3 capture.** With
   `gpt_oss-GateUP-B32-M2048` only, `rocprofv3 --kernel-trace --pmc
   {VALUUtilization,MfmaUtilization,LDSBankConflict,VGPRSpills}`. Get
   exact bottleneck signature for fwd / dA / dB triplet. This is
   what the task body's "Lever A3" / "Lever B1" suggests; we've
   never run it.

3. **gpt_oss-Down dA path.** dA_r = 0.942 (worst). This path is
   **fuse-eligible** (K_kernel=2880, K%128=64). The fuse-path
   `KI=44` was R39 falsified by VGPR spill. But maybe a smaller
   `KI=22` (half) or `KI=88` (double — full unroll of 2 iters per
   pair) compiles cleanly on the fuse path? Worth a probe-build.

4. **`select_default_config` (group_m, num_xcds) re-tuning** for
   gpt_oss-{GateUP,Down}-B32-M2048 specifically. R32-R34 explored
   gpt_oss-GateUP post-H4 dA RCR with `(gm=1, xcds=4)` and
   `(gm=2, xcds=4)` rules — both falsified. But the dA-specific
   data is now richer (dA_r=0.94-0.98), which may unlock a new
   predicate.

Recommend R43 start with (1) — the var-K kernel scout has not
been done in any of R32-R42, so it's the highest expected-value
unknown surface left.
