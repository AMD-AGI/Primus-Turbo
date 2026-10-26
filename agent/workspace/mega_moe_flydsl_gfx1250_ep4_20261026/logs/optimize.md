# mega_moe_bf16 FLYDSL Optimization Log (EP4)

## Basic Information
- Target operator: mega_moe_bf16 (fused_mega_moe fwd+bwd)
- Implementation language: FLYDSL
- Backend: FLYDSL
- Target GPU: gfx1250 (MI455X), EP4 on GPUs 0-3
- Campaign: agent/workspace/mega_moe_flydsl_gfx1250_ep4_20261026
- Predecessor: agent/workspace/mega_moe_flydsl_gfx1250_ep2_20261023 (EP2, rounds 1-9)
- Start time: 2026-10-26 02:18
- Current status: Optimizing (round-2)

## Baseline
- Time: 2026-10-26 02:26
- Backend: FLYDSL
- GPU: gfx1250
- Commit: 1149c02b
- Validation level: full

| Shape | fwd ms | fwd+bwd ms | Fwd TF | Bwd TF | Step TF | Check |
|---|---|---|---|---|---|---|
| dsv3_t8192_random | 4.119 | 11.521 | 1402 | 1560 | 1503.1 | PASS |
| dsv3_t8192_skew1 | 9.509 | 27.406 | 607 | 645 | 631.9 | PASS |
| dsv3_t4096_random | 2.311 | 6.770 | 1249 | 1295 | 1279.0 | PASS |

- Aggregate score (Combined Step TFLOPS geomean): 1066.987
- All Check: PASS (tests: EP4 5 one per process, EP2 5, EP1 19)
- EP2 info: 1279.377
- Detailed data: rounds/round-1/summary.md
- Quick baseline logs: rounds/round-1/artifacts/quick_<shape>.log

## Optimization History

### round-2 — lead combine knobs re-tuned at EP4
- Time: 2026-10-26 03:00 (UTC)
- Validation level: quick screen (dsv3_t8192_random, 2 alternating reps per config)
- Hypothesis: with 3/4 of the fold rows remote at EP4, more / earlier lead combine workgroups push more under
  the GEMM and shorten the ~275-375 us post-GEMM combine tail per launch
- Changes: none in source; `num_lead_combine` 8/16/24/32/48 and `lead_gemm_percent` 10/25/40 for the L2
  forward and L1 dgrad gemm_combine launches via the monkeypatched bench
- Result: fwd+bwd 11.512-11.575 ms vs default 11.514 / 11.528 ❌ (best -0.03%, 48 or 8 workgroups +0.4%)
- Test: PASS (bench SNR, 26 runs)
- Decision: rejected (defaults unchanged)
- Detailed data: rounds/round-2/summary.md
- Notes: a fold row waits for its token's highest local expert's rows, so most become pushable late in the
  GEMM (~x^2); the post-GEMM backlog does not depend on the pusher count. Side probes: gfx1250 has no
  HW_REG_XCC_ID (HW_ID1 shows WGP/SA only); 256 spinning 8-wave workgroups are not all co-resident (221, the
  rest queue behind occupied slots while CUs idle); num_dispatch_cu 64 stays (32-192 screened)

### round-3 — prologue publishes the pool rows to host memory after phase C
- Time: 2026-10-26 03:25 (UTC)
- Validation level: full + same-session A/B (3 alternating reps, T=8192 / T=4096 random)
- Hypothesis: the host learns the rows only after the whole prologue (`.item()`), leaving ~100-130 us of GPU idle
  before dispatch_gemm; a host-mapped rows notify at the end of phase C hides it under phases D-E
- Changes: dispatch_prologue_kernel.py (gfx1250: block 0 stores total_rows at system scope into coherent pinned
  host memory; host polls with a device-read fallback), pyhip_runtime_wrapper.py (hipHostMalloc,
  hipHostGetDevicePointer)
- Result: full run 1071.307 vs 1066.987 TF (+0.405%) but A/B median +0.14% (T=8192) / -0.09% (T=4096) ❌
- Test: PASS (EP4 5 one per process, EP2 5, EP1 19)
- Decision: rollback (back to 1149c02b); patch kept in rounds/round-3/artifacts
- Detailed data: rounds/round-3/summary.md
- Notes: rows seen before kernel end 25/25 with no fallback; steady-state GPU idle per iteration 101-124 -> 28-31 us,
  prologue end -> dispatch_gemm 104-129 -> 18-20 us, yet gemm_combine and the prologue's barriers grow by
  about as much (cross-rank waits)
