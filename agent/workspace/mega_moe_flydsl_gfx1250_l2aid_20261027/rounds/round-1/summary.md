## Hypothesis

Baseline round. No optimization; record the EP4 starting point of 8bbd8c89 (async A gather fix) and measure the
facts the reference (`opt.md`, minimax-poc `Q&A/l2-cluster-aid-mapping`) says placement work needs: which XCC
each workgroup lands on, and where the step time goes.

## Single change

No kernel change. `kernel_snapshot/git_state.txt`: 8bbd8c89. The EP2 / EP1 / grouped-GEMM tests and the bench of
this round ran after round 2's switches (`MEGA_AID_SPLIT_DG`, `MEGA_AID_SPLIT_GC`) were added to the repo tree,
both off: `const_expr` branches not taken, same kernels. The A/B baseline of every later round is the pristine
copy of 8bbd8c89 at `mega_scratch/l2c/accepted`.

## Results

### Validation
- Time: 2026-10-27 05:51
- Validation level: full
- Test command: `run_round.sh 1 full`
- Raw artifacts: `artifacts/test_*.log`, `artifacts/quick_*.log`, `artifacts/bench.csv`, `artifacts/run_round_full.txt`,
  `artifacts/xcc_probe.txt`, `artifacts/kernel_trace_t8192_random.txt`

Tests: EP4 test_deepseek_v3 2/2, test_ragged_skewed 1/1, test_one_token 1/1, test_staged_bias_gpt_oss 1/1;
EP2 5/5; EP1 19/19; grouped GEMM gfx1250 31/31.

| Shape | Check | fwd ms | fwd+bwd ms | Fwd TF | Bwd TF | Step TF |
|---|---|---|---|---|---|---|
| dsv3_t8192_random | PASS | 4.239 | 11.973 | 1362 | 1493 | 1446.4 |
| dsv3_t8192_skew1 | PASS | 9.763 | 28.114 | 591 | 629 | 616.0 |
| dsv3_t4096_random | PASS | 2.312 | 6.865 | 1248 | 1268 | 1261.3 |

Aggregate: Combined Step 1039.651 TF; Fwd 1001.774; Bwd 1059.866.

### Workgroup -> XCC placement (`tools/xcc_probe.hip`)

256 threads, 280 KB dynamic LDS (one workgroup per WGP, as the Mega GEMM tile), lane 0 reads
`s_sendmsg_rtn_b32 MSG_RTN_GET_SE_AID_ID` (XCC in bits [18:16]):

- Every workgroup of every launch ran on XCC `(bid + 7) % 8`: grids 7 / 256 / 1000 / 1001 / 4160 / 4163, 2-6
  back-to-back launches each, GPUs 0-3, before and after the 05:33 reboot. 0 exceptions.
- No load balancing across XCCs: with every 8th workgroup 4x slower (all on XCC 7) the launch takes 1373 us
  instead of 355 us; the other XCCs finish their 520 workgroups at ~330 us and idle.
- So AID 0 (XCC 0-3) runs the workgroups with `bid % 8` in {1, 2, 3, 4}, AID 1 those in {5, 6, 7, 0}. A remap
  keyed on `bid` alone is a fixed permutation: always a bijection, only its speed depends on this mapping.

### Kernel trace, rank 0, T=8192 random, 8 steady-state iterations (`tools/prof_ss.py`, `--kernel-trace`)

| Kernel | us | GEMM TF (65536 routed rows) |
|---|---:|---:|
| dispatch prologue | 159 | |
| L1 fwd dispatch_gemm (4464 WGs: 64 comm + 275 x 16 tiles) | 2052 | 1876 |
| swiglu fwd | 112 | |
| L2 fwd gemm_combine (275 x 28 tiles + 256 combine) | 1468 | 1311 |
| L2 dgrad dispatch_gemm (64 + 275 x 8) | 1174 | 1637 |
| swiglu bwd | 186 | |
| dW2 variable-K | 1097 | 1755 |
| L1 dgrad gemm_combine | 2360 | 1631 |
| x re-dispatch (dispatch-only) | 281 | |
| dW1 variable-K | 2108 | 1826 |
| GPU busy per iteration | 11074 | |

The six GEMM launches are 10.26 ms, 93% of the busy time.

### Which operand blocks each AID reads (`tools/aid_model.py`, XCC = (bid + 7) % 8)

| Launch, tile order | per AID now | total now | with an AID N-half split |
|---|---|---|---|
| gemm_combine (L2 fwd, L1 dgrad), row major, 28 N blocks | every B block, every A block | 2A + 2B | 2A + B (B 1792 -> 901 blocks at T=8192 random) |
| dispatch_gemm (L1 fwd, L2 dgrad), 16-row super-tiles | every B block, half the A | A + 2B | 2A + B: better at random (A 275 < B 512 blocks), worse at skew (rank 0 A 624 blocks) |

28 is 4 mod 8, so in row-major order a pool block's N block alternates AIDs from one row to the next. In a
dispatch_gemm super-tile (m fastest, 16 rows) the AID is set by `m_local % 8`.

## Decision

`BASELINE`

## Attribution

rocprofv3 `--pmc` (FETCH_SIZE, then GL2C_HIT / GL2C_MISS) was tried to measure the GL2 fetches: rank 3 SIGSEGV
on the first pass; during the second, 3 GPUs logged `SMU: No response`, every amdgpu sensor read blocked (a
concurrent `rocm-smi` from another user held the SMU mutex) and the node needed a reboot. Placement rounds
are therefore judged by same-session alternating A/B only, kernel trace for attribution.

## Next direction

1. gemm_combine AID N-half split (never more GL2 fill than now, at any routing).
2. dispatch_gemm AID N-half split, possibly chosen per launch from the tile count (A vs B size).
3. variable-K wgrads: same analysis for their super-tile order.

## Real-training transfer check
- Bucket: — (baseline)
- Decision: BASELINE
