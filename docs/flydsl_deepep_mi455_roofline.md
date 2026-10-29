# Roofline of the FlyDSL DeepEP kernels on MI455X

Where the `flydsl_v2` dispatch and combine stand against the hardware: compute or I/O bound, which I/O,
and how far from the limits. Branch `dev/boyang/flydsl_deepep_gfx1250_roofline`, from PR #583
(`dev/boyang/flydsl_deepep_gfx1250`). Status: first analysis, from the EP2 data measured for PR #583;
EP4 and the measurements listed at the end are still to run.

## Summary

- **Not compute bound.** The dispatch does no arithmetic on the data; the combine adds about 0.5 FLOP
  per byte that crosses the link (0.1 per byte of HBM traffic). At the 315 TFLOP/s vector peak that is
  0.2 µs of a 140 µs kernel. The ridge points are 175 FLOP/B against the link and 13.5 FLOP/B against
  HBM, so both kernels sit at least 100× on the I/O side of either.
- **The link is the first limit, HBM the second.** Per rank and kernel, at the DeepSeek-V3 proxy shape
  (EP2, 8192 tokens): 116 MB cross the link each way, at the 1.8 TB/s link peak 64 µs. HBM carries 5–7×
  as many bytes (812 MB dispatch, 580 MB combine), 35 / 25 µs at the 23.3 TB/s peak, 40–55% of the
  link time, so it matters wherever it does not overlap the transfer.
- **Distance to the limits** (EP2, T = 8192, 160 CUs; in parentheses 256 CUs):

  | | µs | of the link peak (1.8 TB/s) | of the best measured link copy (1.54 TB/s) | headroom to the link peak |
  |---|---:|---:|---:|---:|
  | dispatch, receive counts known | 122.7 (118.3) | 52% (54%) | 61% (64%) | 1.9× |
  | dispatch with notify and host wait | 179.4 (174.2) | 36% (37%) | 42% (43%) | 2.8× |
  | dispatch, `num_worst_tokens` | 145.5 (140.6) | 44% (46%) | 52% (53%) | 2.3× |
  | combine | 145.7 (137.2) | 44% (47%) | 52% (55%) | 2.3× |

  The best copy the hardware does over the link (copy engines, 470 MB) is 86% of the link peak; that
  is the practical ceiling, 75 µs for these bytes, 1.6–2.4× away.
- **Four regimes**, by where the time goes:
  1. *Many CUs, large batches* (≥ 96 CUs, T ≥ 8192): I/O bound in two serialized phases. The cached
     dispatch is, within −4 to +9 µs from 80 to 256 CUs, the send at the link ceiling of a bare copy kernel
     plus the receiver's copy of every received row from the heap into `recv_x` at the local copy
     rate (at 160 CUs: 87 + 29 µs of 123). The receive copy is the largest removable part.
  2. *Few CUs* (≤ 64): bound by what each CU issues. The TDM copy moves about 22 GB/s per CU, so 32
     CUs reach 706 GB/s, 39% of the link peak. Below 96 channels the combine pushes with every warp
     and reduces only afterwards, so push and reduce add up (32 CUs: 164 µs send, 182 µs more).
  3. *Small batches*: bound by fixed costs. Fitted at 160 CUs: 36 µs (cached dispatch; 14 µs for the
     bare copy kernel), 59 µs (combine), 89 µs (dispatch with notify and host wait). Half the time is
     fixed below 3.4 k / 5.2 k / 8.1 k tokens per rank: at the proxy shape the synchronous dispatch is
     as much latency as bandwidth.
  4. *The combine's own overhead*: at ≥ 96 CUs it is 40–63 µs above the send at the link ceiling while
     HBM runs at 17% of its peak; not a bandwidth limit (pushers are capped at 22 of 32 warps by LDS,
     per-row flags, the reduce tail). Needs the per-phase timing below.

## Hardware ceilings

| | Peak (AMD) | Measured, best | Measured where it matters |
|---|---|---|---|
| Link, per GPU and direction | 1.8 TB/s (3.6 TB/s bidirectional scale-up, 36 links × 400 Gbit/s each way) | 1539 GB/s, copy engines, 470 MB, both directions (86%) | TDM copy kernel at 8192 rows: 1323 GB/s at 160 blocks, 1364 at 256 |
| HBM4 | 23.3 TB/s (12 stacks; the driver reports 23,347 GB/s) | 17.0 TB/s traffic, local TDM copy at 256 blocks (73%) | 16 TB/s traffic at 160 blocks |
| Vector FP32 / FP16 | 315 TFLOP/s | – | – |
| Per CU | 256 CUs, 2.4 GHz, wave32 | TDM copy over the link: 24.8 GB/s per block at 8 blocks, 22 at 32, falling as the link fills | register copy: 9 GB/s per block |

- The four GPUs of a node are fully connected, one XGMI hop between any two (`amd-smi topology`);
  link speed and counters read `N/A` on this stack, so the link numbers are measured.
- EP2 already uses 1.54 of the GPU's 1.8 TB/s each way to a single peer. At EP4 the three peers share
  the 1.8 TB/s: about 0.6 TB/s per peer if the links share evenly. EP4 has not been measured.

## Bytes and arithmetic per rank

Rows of `RB = 2 × hidden` bytes (14,336 B at hidden 7168); T tokens per rank; `rows_in` / `rows_out`
rows from / to the peers; `rows_self` rows a rank keeps (EP2 proxy: 8071, 8071, about 8071).

| kernel | link (larger direction) | HBM | FLOPs |
|---|---|---|---|
| dispatch | `max(rows_in, rows_out)·RB` | read x once per token `T·RB`; heap writes of all received rows; the receiver's copy heap → `recv_x` (read + write): `(T + 3·(rows_in + rows_self))·RB` | 0 |
| combine | `max(rows_in, rows_out)·RB` | read the rows pushed back, heap writes of the returned rows, the reduce reading own and returned rows, the output: `(rows_in + 2·rows_out + rows_self + T)·RB` | one add per element beyond a token's first row: `(rows_self + rows_out − T)·hidden` |

Per-row metadata (top-k indices and weights, counts, flags) is under 1% and not counted. The HBM column
follows the code (`intranode_v2_kernel.py`), not counters; verifying it is on the list below.

## Where the time goes

EP2, T = 8192, ceilings measured with the same number of blocks (`roofline.py`, phase table):

| CUs | cached dispatch µs | send at the link ceiling | receive copy at the local copy rate | rest | combine µs | send at the link ceiling | rest |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 271.2 | 163.9 | 53.0 | 54.3 | 346.4 | 163.9 | 182.4 |
| 64 | 159.3 | 99.1 | 36.4 | 23.8 | 194.7 | 99.1 | 95.7 |
| 80 | 141.2 | 105.9 | 35.5 | −0.2 | 167.4 | 105.9 | 61.5 |
| 128 | 123.7 | 86.4 | 28.1 | 9.3 | 149.4 | 86.4 | 63.0 |
| 160 | 122.7 | 87.4 | 29.0 | 6.3 | 145.7 | 87.4 | 58.2 |
| 256 | 118.3 | 84.9 | 27.2 | 6.3 | 137.2 | 84.9 | 52.3 |

- The dispatch's receive phase starts in each block after that block has sent all its rows and
  published its counts (`dispatch_v2_kernel`: send loop, tails, then `_poll_tails` / `_copy_received`).
  The model "send, then copy" fits from 80 CUs up; below, the per-channel polling adds 24–54 µs.
- The send at the ceiling already contains the copy kernel's own fixed 14–19 µs (launch, ramp, tail).
- The synchronous dispatch adds 57 µs to the cached one (notify kernel and the host wait for the
  counts); `num_worst_tokens` removes the host wait and adds 23 µs.

## What this means for optimization

Upper bounds, EP2 at 160 CUs; the first two are models, not yet measured:

| change | removes | cached dispatch | combine |
|---|---|---:|---:|
| overlap or drop the receive copy (receive warps alongside the senders, or `recv_x` in the heap) | up to 29 µs | → about 94 µs | – |
| overlap push and reduce below 96 channels | up to the reduce, 182 µs at 32 CUs | – | 32 CUs: → about 170 µs |
| reach the copy-engine ceiling on the send | the gap from 1.32 to 1.54 TB/s, about 12 µs | → about 110 µs (82 with the first row) | → about 134 µs |
| the link peak | | 64 µs | 64 µs |

## How to reproduce

Inside `dev_mi455_deepep_yangbo`, from `benchmark/ops/deep_ep_roofline/` of this branch, GPUs claimed in
`cursor-work/STATUS.md` (`NCCL_CUMEM_ENABLE=0` only matters for tests that tear down the process group):

```bash
export PYTHONPATH=$PWD/../../.. PRIMUS_TURBO_EP_FORCE_CURRENT_STREAM=1 ROCR_VISIBLE_DEVICES=2,3
OUT_JSON=/tmp/ops.json torchrun --nproc_per_node=2 bench_ops.py                         # CU sweep at T = 8192
TOKENS_LIST=256,512,1024,2048,4096,8192,16384,32768 CU_LIST=80,160 BACKENDS=flydsl_v2,rccl \
  OUT_JSON=/tmp/tsweep.json torchrun --nproc_per_node=2 bench_ops.py                      # token sweep
OUT_JSON=/tmp/ceiling.json torchrun --nproc_per_node=2 bench_ceiling.py                 # link and HBM ceilings
python3 roofline.py --ops /tmp/ops.json --tsweep /tmp/tsweep.json --ceiling /tmp/ceiling.json
```

The numbers above come from the PR #583 run of 2026-10-27 (EP2, GPUs 2,3, commit `8b19e035`; data in
`cursor-work/pr583_bench_data/pr583_bench_full_ep2rerun_20261027_215435/`), rerun through
`roofline.py`. `bench_ops.py` now also records `rows_self`; older JSON gets `rows_in / (world − 1)`.

## Next measurements

1. EP2 on this branch's head with the scripts above, to confirm the numbers (about 2 minutes of 2 GPUs).
2. **EP4** (4 GPUs): `bench_ceiling.py` for the per-GPU aggregate over three peers against 1.8 TB/s
   (the `hipmemcpy` / `sdma` references issue the peers one after another on one stream and need a
   stream per peer first), and `bench_ops.py` at the CU counts training uses.
3. **Per-phase timing** inside the kernels (`s_memrealtime` per block: send start / end, counts
   published, receive copy start / end; combine push end and reduce end) to confirm the serialized
   receive copy and to split the combine's 40–60 µs.
4. **Counters** (rocprofv3, if gfx1250 exposes them): HBM read / write bytes per kernel against the
   byte table; link bytes.
5. The copy kernel with 22 warps per block (the combine's pushers) and with T = 8192's row count, to
   see whether the push alone reaches the link ceiling.
