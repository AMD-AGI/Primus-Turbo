# Roofline of the FlyDSL DeepEP kernels on MI455X

How we measure the roofline of the `flydsl_v2` dispatch and combine, and how far they are from the limits
of the hardware. Branch `dev/boyang/flydsl_deepep_gfx1250_roofline`, from PR #583
(`dev/boyang/flydsl_deepep_gfx1250`). The EP2 numbers are measured; the EP4 numbers are an estimate until
the EP4 run of [Next measurements](#next-measurements).

## Summary

- **Not compute bound.** The dispatch does no arithmetic on the data; the combine does one FP32 add per
  2 bytes of a returned row: 0.49 FLOP per byte that crosses the link, 0.1 per byte of HBM traffic. Even
  at the lowest FP32 peak (78.6 TFLOP/s for plain adds) that is 0.7 µs of a 140 µs kernel, and the
  kernels sit 30× or more on the I/O side of every ridge point.
- **The link is the limit, HBM the second one.** Per rank and kernel at the DeepSeek-V3 proxy shape (EP2,
  8192 tokens, hidden 7168 bf16, top-8 of 128): 116 MB each way over the link, 64 µs at its 1.8 TB/s
  peak; 580–812 MB of HBM traffic, 25–35 µs at 23.3 TB/s. HBM adds to the time where it does not overlap
  the link, as in the dispatch's receive copy.
- **How far from the limit** (EP2, T = 8192, 160 CUs):

  | | µs | of the link peak 1.8 TB/s | of the achievable link 1.61–1.65 TB/s | of a copy kernel moving the same bytes with the same CUs |
  |---|---:|---:|---:|---:|
  | dispatch, receive counts known | 122.7 | 52% | 57–59% | 74% |
  | dispatch, `num_worst_tokens` (no host wait) | 145.5 | 44% | 48–49% | 63% |
  | dispatch with notify and host wait | 179.4 | 36% | 39–40% | 51% |
  | combine | 145.7 | 44% | 48–49% | 63% |

  At 256 CUs the cached dispatch is 118.3 µs (54% of the peak) and the combine 137.2 µs (47%).
- **EP4 (estimate):** the PR #583 EP4 runs (160 CUs, uniform top-8 of 256 experts) take 0.36 ms
  (dispatch with host wait) and 0.34 ms (combine) for about 317 MB per GPU to its three peers. An RCCL
  alltoall of that volume is about 208 µs on AMD's achievable roofline (MAIDAS), 176 µs at 1.8 TB/s:
  58–61% of the achievable roofline, 49–52% of the peak. At EP4 the three peers share the GPU's 1.8 TB/s.
- **Where the gap is** (dispatch at 160 CUs: 123 µs = 87 µs send at the copy kernel's ceiling + 29 µs
  receive copy + 6 µs): the receiver's copy from the heap into `recv_x` runs after each block's send; the
  notify and host wait add 57 µs; below 96 channels the combine reduces only after pushing; with 64 CUs or
  fewer each CU's issue rate bounds the send.

## How the roofline is measured

The method follows the speed-of-light / arithmetic intensity / hierarchical roofline approach of the HIP
training course (course 017 §1, course 009 §3) and the busbw roofline of the MI455X RCCL study
(`minimax-poc-20261006/examples/06-rccl`), applied to a kernel whose work is moving bytes.

### 1. Three ceilings per resource

| Resource | Peak (speed of light) | Achievable (vendor microbenchmark on MI455X) | Our harness, same data path (`bench_ceiling.py`) |
|---|---|---|---|
| Link, per GPU and direction | 1.8 TB/s: 3.6 TB/s bidirectional UALoE scale-up, 36 links × 400 Gbit/s each way (AMD spec); RCCL reports `bw=1800000Mb/s` | **1.61–1.65 TB/s** one pair, one direction, 1 GB copies, launch excluded (TransferBench `p2p`, `minimax-poc/examples/04-transfer-bench-p2p`; TDM reaches it with 64 CUs). **4 GPUs**: alltoall busbw roofline 1.38 / 1.48 / 1.54 / 1.57 TB/s at 128 MB / 256 MB / 512 MB / 1 GB (MAIDAS `MAIDAS_20260813_MI455X_SPX.csv`); RCCL alltoall measured 1.49 TB/s at 1 GB (`06-rccl`) | EP2, both directions busy: copy engines 1539 GB/s at 470 MB; TDM copy kernel 1323 GB/s (160 blocks) / 1364 (256) at 117 MB, 1491 at 470 MB; fit over sizes at 160 blocks: 14.2 µs + bytes / 1505 GB/s |
| HBM4 | 23.3 TB/s (12 stacks, 24,576-bit bus; `amd-smi static`: 23,347 GB/s) | **20 TB/s read**, **15.0–15.6 TB/s copy** (read + write bytes), from 768 MB up (`minimax-poc/examples/03-global-bandwidth`) | local TDM copy: 16–17 TB/s of traffic at 117 MB, above the vendor copy rate, probably because part of it stays in the 192 MB L2 |
| FP32 | 315 TFLOP/s vector (AMD spec, counts packed FP32); 157.3 FMA; **78.6 plain add**: 256 CUs × 2.4 GHz × 4 SIMDs × 32 lanes × (2 or 1) | – | – |
| Per CU | 256 CUs at up to 2.4 GHz, wave32 | – | TDM copy over the link 24.8 GB/s per block at 8 blocks, 22 at 32; register copy 9 GB/s per block |
| Launch | – | 1.2–1.3 µs per kernel launch (`minimax-poc/examples/09-kernel-launch`) | – |

- The peak is the bound no software can beat; the achievable value is what the best vendor code reaches
  on this system, the honest target; our harness copies with the kernels' own data path (TDM into the
  uncached symmetric heap, `copy_kernel.py`), so the difference between a kernel and the harness at the
  same CU count is the kernel's protocol, not the hardware.
- State which peak an efficiency uses: FP32 has three (packed, FMA, add); the link has a per-GPU peak
  shared by all peers (UALoE), so EP4 does not get 3 × the EP2 bandwidth.

### 2. Bytes and FLOPs of each kernel

From the code (`intranode_v2_kernel.py`), per rank, rows of `RB = 2 × hidden` bytes, T tokens per rank,
`rows_in` / `rows_out` rows from / to the peers and `rows_self` rows a rank keeps (all three recorded by
`bench_ops.py`):

| kernel | link (larger direction) | HBM | FLOPs |
|---|---|---|---|
| dispatch | `max(rows_in, rows_out)·RB` | `(T + 3·(rows_in + rows_self))·RB`: x read once per token; every received row written into the heap; then read from the heap and written into `recv_x` | 0 |
| combine | `max(rows_in, rows_out)·RB` | `(rows_in + 2·rows_out + rows_self + T)·RB`: rows pushed back read from x; returned rows written into the heap; the reduce reading own and returned rows; the output | `(rows_self + rows_out − T)·hidden` adds |

Per-row metadata (top-k indices and weights, counts, flags) is under 1% and left out. At the proxy shape:
link 115.7 MB, HBM 812 MB (dispatch) / 580 MB (combine), 57 MFLOP (combine).

### 3. Time

`bench_ops.py` (one process per GPU): before every iteration a 512 MB write evicts L2; the slowest rank of
each iteration counts; a round is the median of 30 iterations; the result is the mean of 3 rounds;
outputs are checked against a reference once per configuration. Kernels run on the timed stream
(`PRIMUS_TURBO_EP_FORCE_CURRENT_STREAM=1`). The run-to-run spread is 0.2% median, 2.6% at most.

### 4. Placing the kernel

- **Classical roofline**: arithmetic intensity = FLOPs / bytes per level against each ridge point
  (peak FLOP/s / bandwidth). Dispatch 0; combine 0.49 FLOP/B on the link, 0.10 on HBM. Ridge points:
  link 44 / 87 / 175 FLOP/B, HBM 3.4 / 6.7 / 13.5 FLOP/B (add / FMA / packed peak). Both kernels are on the
  bandwidth slope; FLOP-bound is ruled out.
- **Communication roofline**: achieved bandwidth = link bytes / time, against the ceiling at the same
  message size and CU count, because a transfer of 116 MB does not reach the 1 GB asymptote (MAIDAS:
  1.38 TB/s at 128 MB against 1.57 at 1 GB; our copy kernel: 14 µs fixed + 1.5 TB/s). Below the
  ceiling at the same size is the third case of the roofline ("below both roofs"): the hardware has room,
  the kernel does not feed it.
- **Attributing the gap**: time = send at the harness ceiling + the serialized HBM phase + fixed costs,
  each term from the measured ceilings at the same CU count (`roofline.py`, phase table); the token
  sweep fits fixed + bytes / bandwidth to separate latency from streaming.
- **Verifying**: counters (`rocprofv3`, gfx1250 has `FETCH_SIZE`, `WRITE_SIZE`, `GL2C_EA_RDREQ_DRAM`,
  `GL2C_EA_WRREQ_DRAM`: DRAM against fabric traffic, `SQ_INSTS_VALU`, `GRBM_GUI_ACTIVE`) for the byte
  table, and in-kernel timestamps (`s_memrealtime`) for the phases. `rocprof-compute` (the hierarchical
  roofline of course 009) is not in this container.

### 5. Commands

Inside `dev_mi455_deepep_yangbo`, from `benchmark/ops/deep_ep_roofline/` of this branch, GPUs claimed in
`cursor-work/STATUS.md`:

```bash
export PYTHONPATH=$PWD/../../.. PRIMUS_TURBO_EP_FORCE_CURRENT_STREAM=1 ROCR_VISIBLE_DEVICES=2,3   # 0,1,2,3 for EP4
N=2
OUT_JSON=/tmp/ops.json torchrun --nproc_per_node=$N bench_ops.py                          # CU sweep, T = 8192
TOKENS_LIST=256,512,1024,2048,4096,8192,16384,32768 CU_LIST=80,160 BACKENDS=flydsl_v2,rccl \
  OUT_JSON=/tmp/tsweep.json torchrun --nproc_per_node=$N bench_ops.py                       # token sweep
OUT_JSON=/tmp/ceiling.json torchrun --nproc_per_node=$N bench_ceiling.py                  # ceilings
python3 roofline.py --ops /tmp/ops.json --tsweep /tmp/tsweep.json --ceiling /tmp/ceiling.json
# counters (bytes per kernel; not yet run on gfx1250, the counter names are from rocprofiler-sdk's config.yaml)
rocprofv3 --pmc FETCH_SIZE WRITE_SIZE GL2C_EA_RDREQ_DRAM GL2C_EA_WRREQ_DRAM --kernel-include-regex 'dispatch_v2|combine_v2' \
  -d /tmp/pmc -- torchrun --nproc_per_node=$N bench_ops.py
```

`roofline.py` needs no GPU: it prints the ceiling, roofline, phase and fixed-cost tables from the JSON.

## How far from the limits (EP2, measured)

From the PR #583 run of 2026-10-27 (GPUs 2,3, commit `8b19e035`,
`cursor-work/pr583_bench_data/pr583_bench_full_ep2rerun_20261027_215435/`) through `roofline.py`:

| CUs | op | µs | link GB/s | of the link peak | of the best measured link copy (1539) | of the harness at these CUs |
|---:|---|---:|---:|---:|---:|---:|
| 32 | dispatch, counts known | 271.2 | 427 | 24% | 28% | 60% |
| 32 | combine | 346.4 | 334 | 19% | 22% | 47% |
| 80 | dispatch, counts known | 141.2 | 819 | 46% | 53% | 75% |
| 80 | combine | 167.4 | 691 | 38% | 45% | 63% |
| 160 | dispatch, counts known | 122.7 | 943 | 52% | 61% | 71% |
| 160 | dispatch with host wait | 179.4 | 645 | 36% | 42% | 49% |
| 160 | combine | 145.7 | 794 | 44% | 52% | 60% |
| 256 | dispatch, counts known | 118.3 | 978 | 54% | 64% | 72% |
| 256 | combine | 137.2 | 844 | 47% | 55% | 62% |

("the harness at these CUs": the TDM copy kernel at the same block count and 8192 rows per peer; the
summary's 74% uses the size fit of the same kernel at 160 blocks, 91 µs for these bytes.)

Fixed cost and streaming rate, fit over T = 256 … 32768 at 160 CUs:

| op | fixed µs | streaming GB/s | of the link peak | half the time is fixed below T ≈ |
|---|---:|---:|---:|---:|
| dispatch, counts known | 36.3 | 1324 | 74% | 3,400 |
| dispatch, `num_worst_tokens` | 55.5 | 1293 | 72% | 5,100 |
| dispatch with host wait | 88.6 | 1290 | 72% | 8,100 |
| combine | 59.2 | 1251 | 70% | 5,200 |
| copy kernel (harness) | 14.2 | 1505 | 84% | – |

At large T the kernels stream at 70–74% of the link peak; at the proxy shape the fixed part is 30–50% of
the time.

## How far from the limits (EP4, estimate)

Not measured with this harness yet. From PR #583's EP4 table (4 × MI455X, T = 8192, hidden 7168 bf16,
random (uniform) top-8 of 256 experts, 160 CUs, `Buffer` interface, mean of 20 iterations on rank 0 and of 3
passes, a different harness from `bench_ops.py`):
dispatch with host wait 0.36 ms, combine 0.34 ms. A peer receives a token with probability
1 − C(192,8)/C(256,8) ≈ 0.90, so a GPU sends about 3 × 0.90 × 8192 rows = 317 MB:

| | µs | of 1.8 TB/s (176 µs) | of the MAIDAS alltoall roofline at this volume (≈ 208 µs) |
|---|---:|---:|---:|
| dispatch with host wait | 360 | 49% | 58% |
| combine | 340 | 52% | 61% |

(MAIDAS gives the time of a 4-GPU alltoall per rank buffer size; 317 MB to the peers is the alltoall of
a 403 MiB buffer, between its 256 MB and 512 MB points.)

## Where the time goes

EP2, T = 8192, every term from the ceilings measured with the same number of blocks:

| CUs | cached dispatch µs | send at the harness ceiling | receive copy at the local copy rate | rest | combine µs | send at the harness ceiling | rest |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 271.2 | 163.9 | 53.0 | 54.3 | 346.4 | 163.9 | 182.4 |
| 64 | 159.3 | 99.1 | 36.4 | 23.8 | 194.7 | 99.1 | 95.7 |
| 80 | 141.2 | 105.9 | 35.5 | −0.2 | 167.4 | 105.9 | 61.5 |
| 128 | 123.7 | 86.4 | 28.1 | 9.3 | 149.4 | 86.4 | 63.0 |
| 160 | 122.7 | 87.4 | 29.0 | 6.3 | 145.7 | 87.4 | 58.2 |
| 256 | 118.3 | 84.9 | 27.2 | 6.3 | 137.2 | 84.9 | 52.3 |

- **Dispatch**: each block sends all its rows, publishes its counts, then polls and copies what it
  received from the heap into `recv_x` (`dispatch_v2_kernel`). "Send, then copy" matches the measurement
  within −4 to +9 µs from 80 CUs up; with fewer CUs the per-channel polling adds 24–54 µs. The send at the
  harness ceiling contains the copy kernel's own 14–19 µs of launch, ramp and tail.
- **Notify and host wait**: +57 µs over the cached dispatch; `num_worst_tokens` (no host wait) +23 µs.
- **Combine**: below 96 channels every warp pushes first and reduces afterwards
  (`combine_v2_push_warps`), so push and reduce add up (32 CUs: 164 + 182 µs). From 96 channels 22 of 32
  warps push (LDS holds 22 row slots at hidden 7168) and 10 reduce from the start; the combine is then
  40–63 µs above the send while HBM runs at 17% of its peak: not a bandwidth limit (flag waits, the
  reduce tail, fewer pushers).
- **Few CUs**: the TDM send moves about 22 GB/s per CU, so 32 CUs give 706 GB/s, 39% of the link peak,
  whatever the protocol does.

## What could still be gained

EP2, 160 CUs; modelled from the terms above, not yet measured:

| change | removes | cached dispatch | combine |
|---|---|---:|---:|
| overlap or drop the receive copy (receive warps beside the senders, or `recv_x` in the heap) | up to 29 µs | → about 94 µs | – |
| overlap push and reduce below 96 channels | most of the 182 µs at 32 CUs | – | 32 CUs: toward the send alone, 164 µs |
| reach the achievable link rate on the send (1.32 → 1.6 TB/s) | about 15 µs | → about 108 µs (79 with the first row) | → about 131 µs |
| the link peak | | 64 µs | 64 µs |

## Next measurements

1. EP2 on this branch's head with the commands above, plus the counters, to confirm the byte table and
   the numbers (2 GPUs, about 5 minutes).
2. **EP4** (4 GPUs, about 15 minutes): `bench_ceiling.py` (per-GPU aggregate over three peers against
   1.8 TB/s and the MAIDAS points; its copy-engine references now copy to each peer on its own stream;
   not yet run) and `bench_ops.py`, with uniform and DeepSeek-V3 routing, at 32 / 64 / 80 / 160 / 256
   CUs.
3. **Per-phase timing** inside the kernels (`s_memrealtime` per block: send start and end, counts
   published, receive copy start and end; combine push end and reduce end) to confirm the serialized
   receive copy and to split the combine's 40–60 µs.
4. The copy kernel with 22 warps per block (the combine's pushers), to see whether the push alone reaches
   the harness ceiling.

## References

- AMD Instinct MI455X: [product page](https://www.amd.com/en/products/accelerators/instinct/mi400/mi455x.html),
  [CDNA 5 white paper](https://www.amd.com/content/dam/amd/en/documents/products/technologies/cdna/amd-cdna5-whitepaper.pdf)
  (36 UALoE links of 400 Gbit/s each way).
- MI455X microbenchmarks and the RCCL busbw roofline (MAIDAS data):
  [minimax-poc-20261006/examples](https://github.com/wenxie-amd/minimax-poc-20261006/tree/main/examples)
  (`03-global-bandwidth`, `04-transfer-bench-p2p`, `06-rccl`, `09-kernel-launch`).
- Roofline method: HIP training course
  [017 Arithmetic Optimizations](https://github.com/wenxie-amd/hip-training-course-tas/blob/main/hip-training-course/017_Arithmetic_Optimizations/note.md)
  §1 (speed of light, arithmetic intensity, ridge point) and 009 Profiling §3 (hierarchical roofline,
  "below both roofs").
- PR #583 document `docs/flydsl_deepep_mi455.md` (Performance) for the measurement setup.
