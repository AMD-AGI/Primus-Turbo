## Hypothesis

Round 3 showed the EP4 step does not shrink when per-rank GPU idle is removed. Power / clock samples during the
bench (`artifacts/pwr_samples.txt`, amd-smi every ~0.6 s, GPUs 0-3) show why: socket power climbs to 2100-2430 W
against the 2500 W PPT0 limit and the GFX clock falls from 2380 MHz (idle) to 2000-2170 MHz, 1650-1680 MHz at the
peak sample; hotspot stays at 60-64 C. The step is power-limited, so energy per step, not busy time, sets the
speed. The variable-K wgrads (dW2 + dW1, ~3.2 ms of the 11.5 ms T=8192 step) launch one workgroup per output
tile in blockIdx order; dealing contiguous tile bands to one XCD (`num_xcd` / `xcd_band`, off by default because
the MI455X XCD mapping is unconfirmed) should raise L2 reuse and cut the fabric / HBM traffic and its energy.

## Single change

`grouped_gemm_bf16_variable_k_flydsl_kernel(..., num_xcd=8, xcd_band=B)` for both wgrad calls of
`fused_mega_moe_gfx1250_impl.py`, B in 16 / 32 / 64 / 128, through the monkeypatched bench
(`tools/sweep_dcu_bench.py`, env `XCD_WGRAD=num_xcd:band`). No source edit.

## Results

### Validation
- Time: 2026-10-26 03:31
- Validation level: quick screen (dsv3_t8192_random, 2 alternating reps per config)
- Benchmark command: `tools/sweep.sh TAG dsv3_t8192_random "XCD_WGRAD=8:32"`
- Raw artifacts: `artifacts/xcdW_sweep.txt`, `artifacts/xcdW_*_dsv3_t8192_random.log`

All 10 runs PASS (bench SNR check).

| num_xcd : band | fwd+bwd ms (rep 1 / rep 2) |
|---|---|
| 1 : 32 (default, remap off) | 11.554 / 11.547 |
| 8 : 16 | 11.579 / 11.588 |
| 8 : 32 | 11.575 / 11.590 |
| 8 : 64 | 11.604 / 11.570 |
| 8 : 128 | 11.558 / 12.130 |

### Aggregate
- Not run on the full set: every remap is +0.1..+0.4% slower on the screen shape (12.130 is the bimodal slow mode).

## Decision

`REJECTED` — remap stays off (no code change to roll back).

## Attribution

A blockIdx-mod-8 band remap does not help the wgrads: either the hardware does not place workgroups on XCDs by
blockIdx mod 8, or the wgrad's group_m=16 super-tiles already keep the shared column slices warm in the shared
cache. Round 2's residency probe is consistent with 8 groups of 32 CUs per `bid % 8` during a full GEMM phase,
but `HW_ID1` cannot confirm the XCD (no XCC_ID on gfx1250), so the mapping is still the open question the
L2-cluster / AID reference is meant to answer.

## Rollback Analysis
- Change made: wgrad XCD band remap, num_xcd 8, band 16-128
- Expected effect: fewer L2 misses in dW2 / dW1, lower energy, higher clocks under the power cap
- Actual effect: +0.1..+0.4% fwd+bwd (slower)
- Root cause: the assumed blockIdx -> XCD mapping (round robin over 8) is unverified on MI455X; the band remap
  may break the super-tile co-residency that group_m provides
- Lesson learned: placement remaps need the real workgroup -> XCD / L2-cluster mapping first

## Next direction

1. Read the L2-cluster / AID mapping reference once access is granted; re-try placement with the real mapping.
2. Energy-per-step view: rocprofv3 counters for HBM / fabric traffic of the five big kernels at EP4.
3. Shorten the post-GEMM combine backlog (round 2 attribution).

## Real-training transfer check
- Bucket: K1 (config)
- Decision: REJECTED (no gain)
