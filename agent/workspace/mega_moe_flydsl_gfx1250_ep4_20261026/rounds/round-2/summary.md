## Hypothesis

At EP4, 3/4 of the combine fold rows go to peers (1/2 at EP2), so the 16 lead combine workgroups placed at 25%
of the GEMM tiles (tuned at EP2) push too little during the GEMM; after the GEMM drains, the tail combine still
runs ~275-375 us per launch (L2 forward, T=8192 random, `artifacts/tail_phases_l2fwd.txt`). More / earlier lead
workgroups should move that push under the GEMM.

## Single change

Lead combine knobs of `gemm_combine_nt_gfx1250` at EP4: `num_lead_combine` / `lead_gemm_percent`, forward (L2)
and dgrad (L1, `b_nn=True`) separately, through the monkeypatched bench (`tools/sweep_dcu_bench.py`,
`tools/sweep.sh`, env `LEAD_{N,PCT}_{FWD,DGRAD}`). No source edit.

## Results

### Validation
- Time: 2026-10-26 03:00
- Validation level: quick screen (dsv3_t8192_random, 2 alternating reps per config)
- Benchmark command: `tools/sweep.sh TAG dsv3_t8192_random "LEAD_N_FWD=n LEAD_PCT_FWD=p"` (EP4 torchrun, GPUs 0-3)
- Raw artifacts: `artifacts/leadF_sweep.txt`, `artifacts/leadD_sweep.txt`, `artifacts/lead*_dsv3_t8192_random.log`

All 26 runs PASS (bench SNR check).

fwd+bwd ms (rep 1 / rep 2), the other side at 16 / 25:

| lead WGs / % | forward (L2) | dgrad (L1) |
|---|---|---|
| 16 / 25 (default) | 11.514 / 11.528 | 12.165 / 11.574 |
| 8 / 25 | 11.563 / 11.566 | 11.560 / 11.549 |
| 24 / 25 | 11.550 / 11.528 | — |
| 32 / 25 | 11.543 / 11.551 | 11.585 / 11.643 |
| 48 / 25 | 11.575 / 11.569 | — |
| 32 / 10 | 11.540 / 11.538 | — |
| 16 / 10 | — | 11.516 / 11.520 |
| 16 / 40 | — | 11.533 / 11.536 |
| 32 / 40 | 11.522 / 11.512 | 11.976 / 11.532 |

Best configs (forward 32 / 40: 11.517, dgrad 16 / 10: 11.518) are -0.03% vs the default's 11.521: noise. The
12.0-12.2 ms runs are the known bimodal slow mode (round-1 Attribution), seen on default and non-default alike.

### Aggregate
- Not run on the full set: no config clears noise on the screen shape.

## Decision

`REJECTED` — no gain; defaults unchanged (no code change to roll back).

## Attribution

Lead workgroup count is not what bounds the post-GEMM push. Per-workgroup stamps (`gc_timed2`, all 4 ranks,
`artifacts/tail_phases_l2fwd.txt`): GEMM done max 1019 us, tail combine push median done 1294 us, reduce
done 1396 us on rank 0; the reduce itself is ~70 us. A token's fold row waits for its highest local expert's
pool rows (the whole prefix up to that row), so fold rows become pushable late in the GEMM: with ~2 local
routes per token at EP4, P(pushable by GEMM fraction x) ~ x^2, i.e. half of them only after ~71% of the GEMM.
Lead workgroups are flag-starved early and the backlog lands after the GEMM regardless of their count.

Side findings:
- Residency (`artifacts/concurrency_l2fwd_r0.txt`): 256 workgroups resident during the GEMM; after it,
  192 resident (16 lead + 176 tail) while 37-66 tail workgroups start up to ~390 us late. Work is taken from
  counters, so the late ones find little left; they are not the critical path.
- Placement probe (`tools/probe_hwid.py`, `artifacts/hwid_probe.txt`, GPU 0): gfx1250 has no
  `HW_REG_XCC_ID`; `HW_ID1` varies only in WGP_ID (bits 13:10, 0-8) and SA_ID (bit 16), SE_ID constant, so the
  XCD of a workgroup is not readable. ~16 WGPs (32 CUs) per XCD-sized group. 256 spinning 8-wave workgroups:
  only 221 resident at once, the rest queue behind occupied slots (start at 300/600/900/1200 us) while CUs
  are idle; 1-wave workgroups: all 256 resident.
- Pre-round screen, `num_dispatch_cu` of the fused dispatch (`artifacts/screen_dcu*.log`): 32 -> 12.288,
  64 -> 11.519 / 11.540, 96 -> 11.540, 128 -> 11.572, 192 -> 11.638 ms: default 64 stays.

## Rollback Analysis
- Change made: lead combine count 8-48, lead start 10-40% of the GEMM tiles, forward and dgrad
- Expected effect: shorter post-GEMM combine tail at EP4
- Actual effect: -0.03% best, +0.4% worst (48 or 8 workgroups), i.e. noise to slightly worse
- Root cause: the post-GEMM push backlog is set by when fold rows become pushable (highest local expert's
  rows), not by how many workgroups push during the GEMM
- Lesson learned: shorten the backlog itself (push per-route partials early, or order the GEMM so fold rows
  complete earlier) or overlap it with something else, rather than adding pushers

## Next direction

1. Prologue host overlap: split `dispatch_prologue` at phase C, enqueue the count D2H with an event before the
   phase-D/E kernel (~100 us of GPU idle per step before dispatch_gemm).
2. L2-cluster / AID-aware workgroup placement once the reference doc (wenxie-amd/minimax-poc-20261006,
   `Q&A/l2-cluster-aid-mapping`) is readable: GEMM tile locality and combine role placement.
3. Prologue phase-D scatter: per-lane global stores instead of non-uniform buffer resources.

## Real-training transfer check
- Bucket: K1 (config)
- Decision: REJECTED (no gain)
