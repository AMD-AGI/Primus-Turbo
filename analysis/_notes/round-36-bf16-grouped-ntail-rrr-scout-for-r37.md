# Round-36 — BF16 grouped GEMM: `grouped_ntail_kernel_lds_rrr` scout (no code change, R37 hand-off)

## Status
**SCOUT-ONLY, docs hand-off**. Chat session at ~87 min of 90 min
window — only 3 min budget remaining at R36 start. Not enough to
write / compile / probe / metric / paired-5-run a kernel change. R36
uses the last 3 minutes to locate and document the attack surface
in HipKittens for R37 cold-start agent.

## Metric (baseline only, R36 has no change)
```
score = 878/1000  (HEAD e2749a7)
Lowest: gpt_oss-GateUP-B32-M2048 @ 1.048
```

## Attack-surface located — native RRR N-tail kernel (R31 slow path)

**File**: `/workspace/code/HipKittens/analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp`

**Functions**:
1. `grouped_ntail_kernel_lds<Layout, 64>` at line **3287**
   (RCR version — used by the H4-rerouted fast path).
2. `grouped_ntail_kernel_lds_rrr<64>` at line **3488**
   **(THIS IS THE R37 TARGET — the 0.6× RCR slow path per R31).**

**Dispatch-level launcher**: around line 4397 in the grouped dispatcher
body. Search `grouped_ntail_kernel_lds_rrr` to find the kernel launch.

**Key observed structure** (3488-3547 snippet):
```cpp
constexpr int TBM = TAIL_BLOCK_M;    // 16
constexpr int TBN = TAIL_BLOCK_N;    // 16
constexpr int NTHR = TBM * TBN;      // 256
constexpr int K_CHUNK_LDS = K_CHUNK + 4;  // pad for bank distribution
__shared__ bf16 A_lds[TBM * K_CHUNK_LDS];
__shared__ bf16 B_lds[TBN * K_CHUNK_LDS];
__shared__ int s_offs[MAX_G_PLUS_1];  // 65

// Per-block: 16×16 threads = 256 (1 wavefront × 4 on MI355X)
// Each thread computes 1 output cell.
// CROSS_BOUNDARY path (line 3522-3541): scalar full-K reduction
//   — per-row, per-col scalar multiply-add across g.k iterations.
//   This is the SLOW PATH when a group boundary splits a TBM=16 row
//   block.
```

### Candidate slow-path hypotheses for R37

1. **(a) Scalar cross-boundary path (line 3522-3541)**: when a tail
   tile straddles a group boundary, it degrades to `for (kk = 0; kk <
   g.k; ++kk) acc += a[r,kk] * b[row_group,kk,c]` — **fully scalar,
   no LDS, no vector loads**. On gpt_oss K=2880 with B=32 groups,
   every second TBM=16 block might cross a group boundary (depends
   on `group_lens`).
   → Fix: either (i) skip the cross-boundary path to per-group
   dispatch, or (ii) add an LDS-staged scalar accelerator for it.

2. **(b) TBM=TBN=16 under-utilization**: 256 threads per block is
   only **1 wavefront × 4 on MI355X** (wavefront=64). Much smaller
   than a full CU of compute (4 wavefronts per SIMD × 4 SIMDs = 16
   waves per CU). Tail blocks are naturally small (< 256 cols) so
   we can only partially saturate. But if TBM were larger (e.g. 32)
   we'd do fewer launches.

3. **(c) LDS conflict in pad**: `K_CHUNK_LDS = K_CHUNK + 4` is mod-32=2
   which is the RCR version's pad — but RRR has a different access
   pattern (B stride-N in K), so this pad may not actually distribute
   banks correctly for the RRR load pattern. Check `ds_read_b64`
   conflicts via `rocprof` `lds_bank_conflict` counter.

4. **(d) Per-row scalar B load on cross-boundary**: the load
   `load_bf16_scalar_grp(g.b, row_group, kk, col_s)` does a per-K
   scalar load. For a 2880-K inner loop, this is 2880 scalar 2-byte
   loads — HBM bandwidth-starved, not compute-bound. Fix: even for
   cross-boundary, load B tiles into LDS once per N-block (shared
   across rows within the block) and reuse.

### Cross-round discipline reminder (from R35 consolidation)
- BF16 dispatch surface exhausted — DO NOT try more `select_default_config` rules.
- H4 transpose is a fast-path switch, not overhead — DO NOT try to bypass it.
- R28 transpose block-shape heuristic non-reproducible — DO NOT re-test.
- **R37+ priority: fix ONE of (a)-(d) above** in the RRR ntail kernel.

## R37 cold-start checklist (agent is fresh, no file-reads visible)
1. Read `round-35` consolidation note + this R36 scout note for context.
2. Open `kernel_bf16_dynamic.cpp` at line 3488 (RRR n-tail kernel).
3. Pick hypothesis (a) — highest expected impact since cross-boundary is
   fully scalar. Draft a minimal fix (try LDS-staging B within TBN=16
   columns across the K loop).
4. Build: `cd /workspace/code/HipKittens/analysis/bf16_gemm/mi350x && make -j8 tk_bf16_layouts.so 2>&1 | tail -20`
5. Correctness: `python3 /tmp/probe_round29_correctness.py` — 9 shapes fwd+bwd vs Triton ref.
6. Metric: `python3 scripts/_metric_grouped_bf16_weighted_wall.py`
7. If promising, paired 5-run:
   ```
   # /tmp/r33_paired_5run.sh does stash-based self-control
   bash /tmp/r33_paired_5run.sh
   ```
8. If mean delta ≥ +5 AND correctness PASS: commit both repos.
9. Expected headroom: fixing RRR ntail slowness may let us drop H4
   gate for some shapes → +30-40 score ceiling.

## Commit
- Primus-Turbo HEAD: this docs-only commit.
- HipKittens: unchanged.
