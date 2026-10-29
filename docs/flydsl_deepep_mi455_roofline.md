# Roofline of the FlyDSL DeepEP kernels on MI455X

How we measure the roofline of the `flydsl_v2` dispatch and combine, and how far they are from the limits
of the hardware. Branch `dev/boyang/flydsl_deepep_gfx1250_roofline`, from PR #583
(`dev/boyang/flydsl_deepep_gfx1250`). EP2 and EP4 are measured on this branch (2026-10-29).

## Summary

- **Not compute bound.** The dispatch does no arithmetic on the data; the combine does one FP32 add per
  2 bytes of a returned row: 0.49 FLOP per byte that crosses the link, 0.1 per byte of HBM traffic. Even
  at the lowest FP32 peak (78.6 TFLOP/s for plain adds) that is 0.7 µs of a 140 µs kernel, and the
  kernels sit 30× or more on the I/O side of every ridge point.
- **The link is the limit, HBM the second one.** Per rank and kernel at the DeepSeek-V3 proxy shape (EP2,
  8192 tokens, hidden 7168 bf16, top-8 of 128): 116 MB each way over the link, 64 µs at its 1.8 TB/s
  peak; 580 MB of HBM traffic in either kernel, 25 µs at 23.3 TB/s (counters confirm the byte model).
  HBM adds to the time where it does not overlap the link: the dispatch's own-row stores and receive copy.
- **How far from the limit** (EP2, T = 8192, 160 CUs; measured on this branch, 2026-10-29):

  | | µs | of the link peak 1.8 TB/s | of the achievable link 1.61–1.65 TB/s | of a copy kernel moving the same bytes with the same CUs |
  |---|---:|---:|---:|---:|
  | dispatch, receive counts known | 122.8 | 52% | 57–59% | 74% |
  | dispatch, `num_worst_tokens` (no host wait) | 145.8 | 44% | 48–49% | 62% |
  | dispatch with notify and host wait | 180.2 | 36% | 39–40% | 50% |
  | combine | 145.8 | 44% | 48–49% | 62% |

  At 256 CUs the cached dispatch is 118.2 µs (54% of the peak) and the combine 138.1 µs (47%). Kernel time
  alone (rocprofv3, as DeepEP V2 times): 108.9 µs dispatch kernel (plus a 9.9 µs notify kernel), 136.0 µs
  combine at 160 CUs.
- **EP4 (measured)**, 4 GPUs, DeepSeek-V3 routing, 160 CUs: 277.5 MB from each GPU to its three peers;
  cached dispatch 258.1 µs (60% of the 1.8 TB/s peak, 66% of 1.62 TB/s, 71% of the MAIDAS alltoall
  roofline at this volume), combine 302.5 µs (51% / 57% / 61%), dispatch with host wait 319.9 µs (48% /
  54% / 57%). Kernel time alone: 242.4 / 284.3 µs, 71% / 60% of 1.62 TB/s. EP4 is closer to the limit
  than EP2: the fixed costs spread over 2.4× the bytes. The three peers share the GPU's 1.8 TB/s; the
  copy kernel reaches 1.52–1.53 TB/s to all three (1.60 one way).
- **In DeepEP's own terms**: DeepEP V2's model puts the ceiling at 0.9 × the link speed, 1.62 TB/s here
  (TransferBench measures 1.61–1.65); the cached dispatch is at 58% of it, the combine at 49%. DeepEP V1
  on H800 reaches 84–86% of its stated link maximum at EP8 (its printed GB/s count a rank's own rows too,
  so compare link bytes, as nccl-tests' busbw does).
- **Where the gap is** (dispatch at 160 CUs: 123 µs = 88 µs send at the copy kernel's ceiling + 29 µs of
  local writes + 6 µs): the rank's own rows are stored by the sending warps one row at a time, and the
  peers' rows are copied out of the heap after each block's send, about half each; the notify and host wait
  add 57 µs; below 96 channels the combine reduces only after pushing; with 64 CUs or fewer each CU's
  issue rate bounds the send.

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
| dispatch | `max(rows_in, rows_out)·RB` | `(T + rows_self + 3·rows_in)·RB`: x read once per token; this rank's own rows stored straight into `recv_x`; the peers' rows written into the heap, then read from it and written into `recv_x` | 0 |
| combine | `max(rows_in, rows_out)·RB` | `(rows_in + 2·rows_out + rows_self + T)·RB`: rows pushed back read from x; returned rows written into the heap; the reduce reading own and returned rows; the output | `(rows_self + rows_out − T)·hidden` adds |

Per-row metadata (top-k indices and weights, counts, flags) is under 1% and left out. At the proxy shape:
link 115.7 MB, HBM 580 MB (either kernel), 57 MFLOP (combine). The counters below check it.

### 3. Time

`bench_ops.py` (one process per GPU): before every iteration a 512 MB write evicts L2; the slowest rank of
each iteration counts; a round is the median of 30 iterations; the result is the mean of 3 rounds;
outputs are checked against a reference once per configuration. Kernels run on the timed stream
(`PRIMUS_TURBO_EP_FORCE_CURRENT_STREAM=1`). The run-to-run spread is 0.2% median, 2.6% at most.
At EP4 the first minute of traffic after the GPUs have been idle runs about 3× slower and erratic, for
RCCL as for these kernels (DeepSeek-V3 routing at 160 CUs: 828 µs instead of 258 µs for the cached
dispatch, RCCL's alltoall 1108 µs instead of 413 µs): run a short warm-up first.

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
# kernel time alone and counters: the process must exit normally for rocprofv3 to write its files
P="EXIT_MODE=normal NCCL_CUMEM_ENABLE=0 CHECK=0 BACKENDS=flydsl_v2"
env $P ITERS=10 ROUNDS=1 CU_LIST=32,80,160,256 rocprofv3 --kernel-trace --output-format csv -d /tmp/ktrace -- \
  python3 -m torch.distributed.run --nproc_per_node=$N bench_ops.py
for C in FETCH_SIZE WRITE_SIZE; do                                                          # one group per run
  env $P ITERS=2 ROUNDS=1 CU_LIST=160 rocprofv3 --pmc $C --kernel-include-regex '(dispatch|combine)_v2' \
    --output-format csv -d /tmp/pmc_$C -- python3 -m torch.distributed.run --nproc_per_node=$N bench_ops.py
done
python3 rocprof_summary.py trace /tmp/ktrace; python3 rocprof_summary.py pmc /tmp/pmc_FETCH_SIZE
```

`roofline.py` needs no GPU: it prints the ceiling, roofline, phase and fixed-cost tables from the JSON.

## How DeepEP and nccl-tests define the throughput ceiling

| Source | Bandwidth it reports | Ceiling it compares with |
|---|---|---|
| nccl-tests (`doc/PERFORMANCE.md`), also rccl-tests and the MAIDAS data | `algbw = S / t`, S the per-rank buffer; `busbw = algbw × (n − 1) / n` for alltoall: the bytes a GPU sends to its peers per second | the per-GPU, one-direction link bandwidth B ("busbw … comparable with the hardware peak") |
| DeepEP V1 (`tests/test_intranode.py`, `docs/legacy.md`) | "bottleneck bandwidth" = `recv_x` bytes a rank receives, **its own rows included**, / time (CUDA events, L2 flushed once before the timed calls, mean of 50) | "~160 GB/s NVLink maximum" on H800: dispatch 153, combine 158 GB/s at EP8. The maintainers call it algorithm bandwidth: the link carries (n − 1) / n of it (PR #130), so about 134 / 138 GB/s, 84–86% of that maximum |
| DeepEP V2 (`tests/ep/test_ep.py`) | scale-up (SU) and scale-out (SO) bytes with the per-token metadata, own rows included unless `--ignore-local-traffic`, / kernel time from the PyTorch profiler (barrier before each call, L2 flushed); the receive copy is a separate kernel (`dispatch_copy_epilogue_impl`) with its own bandwidth, `2 × bytes / copy time` | – |
| DeepEP V2 model (`EPBuffer.get_theoretical_num_sms`) | link-bound time = link traffic / link speed (0.9 × the NVLink link speeds from `nvidia-smi nvlink -s`; the NIC rate from `ibstat`); own rows do not count as link traffic; tokens go to `n·(1 − C(E − E/n, k) / C(E, k))` ranks on average (balanced routing) | SMs needed so that no SM-side rate limits: link rate × SM bytes per link byte / per-SM HBM rate (defaults 180 GB/s read, 45 GB/s write per SM, Hopper), × 1.3, at least 64 when not overlapping compute |

The same ceiling on MI455X, in each convention (EP2, T = 8192, 160 CUs, link bytes 115.7 MB per rank and
direction, received rows 231.4 MB with the rank's own):

| | DeepEP V2 model: 0.9 × 1.8 = 1.62 TB/s | nccl-tests busbw (= our link GB/s) | DeepEP V1 algorithm bandwidth (own rows counted) |
|---|---|---:|---:|
| ceiling | 71 µs for 115.7 MB (equal to TransferBench's 1.61–1.65 TB/s) | 1.62 TB/s | 3.24 TB/s (twice the link at EP2: half the rows are local) |
| dispatch, counts known, 122.8 µs | 58% | 942 GB/s | 1.88 TB/s |
| combine, 145.8 µs | 49% | 793 GB/s | 1.59 TB/s |
| for reference: DeepEP V1 on H800, EP8 | | 134 / 138 GB/s physical | 153 / 158 GB/s |

- Use busbw (link bytes only) for "how far from the hardware"; DeepEP's V1 numbers overstate the link by
  n / (n − 1): 2× at EP2, 1.33× at EP4, 1.14× at EP8.
- DeepEP's own reference reaches 84–86% of its stated link maximum at EP8; the `flydsl_v2` kernels reach
  49–58% of the equivalent ceiling at EP2.
- The SM side of the DeepEP model with MI455X numbers: our TDM send moves about 22 GB/s per CU over the
  link, so 1.62 TB/s needs about 74 CUs, 96 with DeepEP's 1.3 margin; the register copy (9 GB/s per CU)
  would need all 256. This matches where the measured kernels stop gaining (96–160 CUs).
- The nccl-tests style baseline for the same bytes is already in `bench_ops.py`: RCCL `all_to_all_single`
  of the rows DeepEP sends (one per token and destination rank, "dedup"), 200 µs against 123 µs for the
  cached dispatch at EP2; at 4 GPUs RCCL's alltoall reaches 1.49 TB/s busbw at 1 GB (`06-rccl`).

## How far from the limits (EP2, measured)

Measured on this branch (`dev/boyang/flydsl_deepep_gfx1250_roofline`, the kernels of PR #583 `826936e5`) on
2026-10-29, GPUs 2,3, with the commands above (`ut_logs/roofline_ep2_20261029_094643/`); every number is
within 1% of the PR #583 run of 2026-10-27 (`8b19e035`). The run-to-run spread is under 3%.

| CUs | op | µs | link GB/s | of the link peak | of the best measured link copy (1513) | of the harness at these CUs |
|---:|---|---:|---:|---:|---:|---:|
| 32 | dispatch, counts known | 272.0 | 425 | 24% | 28% | 60% |
| 32 | combine | 348.1 | 332 | 18% | 22% | 47% |
| 80 | dispatch, counts known | 141.8 | 816 | 45% | 54% | 75% |
| 80 | combine | 167.8 | 690 | 38% | 46% | 63% |
| 160 | dispatch, counts known | 122.8 | 942 | 52% | 62% | 72% |
| 160 | dispatch with host wait | 180.2 | 642 | 36% | 42% | 49% |
| 160 | combine | 145.8 | 793 | 44% | 52% | 60% |
| 256 | dispatch, counts known | 118.2 | 979 | 54% | 65% | 72% |
| 256 | combine | 138.1 | 838 | 47% | 55% | 61% |

("the harness at these CUs": the TDM copy kernel at the same block count and 8192 rows per peer; the
summary's 74% uses the size fit of the same kernel at 160 blocks, 90.5 µs for these bytes. The copy-engine
reference measured 1511–1513 GB/s at 470 MB this time, 1539 on 2026-10-27.)

Fixed cost and streaming rate, fit over T = 256 … 32768 at 160 CUs:

| op | fixed µs | streaming GB/s | of the link peak | half the time is fixed below T ≈ |
|---|---:|---:|---:|---:|
| dispatch, counts known | 36.6 | 1323 | 74% | 3,400 |
| dispatch, `num_worst_tokens` | 55.7 | 1290 | 72% | 5,100 |
| dispatch with host wait | 89.3 | 1288 | 72% | 8,100 |
| combine | 60.0 | 1248 | 69% | 5,300 |
| copy kernel (harness) | 14.6 | 1524 | 85% | – |

At large T the kernels stream at 69–74% of the link peak; at the proxy shape the fixed part is 30–50% of
the time.

### Kernel time alone, and the bytes (rocprofv3)

`rocprofv3 --kernel-trace` over `bench_ops.py` (slowest rank, median of the timed calls; per CU count the
benchmark runs one checking pass, then 3 + 10 calls of each operation in order):

| CUs | dispatch kernel, counts known | with host wait | `num_worst_tokens` | combine kernel |
|---:|---:|---:|---:|---:|
| 32 | 255.8 | 265.4 | 267.3 | 336.9 |
| 80 | 127.4 | 132.5 | 131.6 | 157.4 |
| 160 | 108.9 | 116.7 | 114.7 | 136.0 |
| 256 | 104.9 | 111.0 | 110.3 | 127.7 |

The notify kernels take 9.9 µs (cached path) and 26.8 µs (with the count exchange) at every CU count.

Kernel time alone puts the dispatch kernel at 66–68% of 1.62 TB/s and the combine at 52–56% (160 / 256
CUs). DeepEP V1's 84–86% is timed with events around the whole call, like the tables above.

Counters at 160 CUs, per kernel and rank (`rocprofv3 --pmc`; `FETCH_SIZE` / `WRITE_SIZE` count the L2's
read and write traffic to memory or the fabric):

| kernel | `FETCH_SIZE` | model: cached reads | `WRITE_SIZE` | model: writes this GPU issues |
|---|---:|---:|---:|---:|
| dispatch | 122 MB | x 117 MB | 356 MB | to the peer 116 + own rows 116 + receive copy 116 = 347 MB, + metadata |
| combine | 241–256 MB | rows pushed 116 + own rows reduced 116 = 231 MB | 238 MB | push 116 + output 117 = 233 MB |

Reads from the uncached heap (the receive copy's and the reduce's) do not show in `FETCH_SIZE`; the rest
matches the byte table within 2–11%. `GL2C_EA_RDREQ_DRAM` / `GL2C_EA_WRREQ_DRAM` were collected too
(`pmc_summary.md`), but their units are not documented for gfx1250, so they are not used.

## How far from the limits (EP4, measured)

4 × MI455X, GPUs 0–3, 2026-10-29 (`ut_logs/roofline_ep4_20261029_102813/`), the same commands with
`--nproc_per_node=4`, after a warm-up (see Time). Ceilings, per GPU to all three peers at once
(`bench_ceiling.py`): TDM copy kernel 1.52–1.53 TB/s (32768 rows per peer, 64–256 blocks; 1.60 TB/s one
way), 1.44–1.46 TB/s at 8192 rows per peer (from 96 blocks; repeated after a warm-up, the first run fell
into the cold start); copy engines 1.51 TB/s (a stream per peer). A TDM copy block sends 33–37 GB/s at
EP4 against 22–25 at EP2: one load feeds three peers. The MAIDAS alltoall roofline at the same volume per
GPU (277.5 MB to the peers is the alltoall of a 353 MiB buffer): about 183 µs.

DeepSeek-V3 routing (top-8 of 128 experts in the top 4 of 8 groups), T = 8192; link bytes 277.5 MB:

| CUs | op | µs | link GB/s | of the link peak (154 µs) | of 1.62 TB/s (171 µs) | of the MAIDAS roofline (≈ 183 µs) |
|---:|---|---:|---:|---:|---:|---:|
| 32 | dispatch, counts known | 429.6 | 646 | 36% | 40% | 43% |
| 32 | combine | 668.7 | 415 | 23% | 26% | 27% |
| 80 | dispatch, counts known | 271.8 | 1021 | 57% | 63% | 68% |
| 80 | combine | 323.7 | 857 | 48% | 53% | 57% |
| 160 | dispatch, counts known | 258.1 | 1075 | 60% | 66% | 71% |
| 160 | dispatch, `num_worst_tokens` | 287.1 | 967 | 54% | 60% | 64% |
| 160 | dispatch with host wait | 319.9 | 868 | 48% | 54% | 57% |
| 160 | combine | 302.5 | 917 | 51% | 57% | 61% |
| 256 | dispatch, counts known | 254.6 | 1090 | 61% | 67% | 72% |
| 256 | combine | 285.4 | 972 | 54% | 60% | 64% |

Kernel time alone (rocprofv3), 160 / 256 CUs: dispatch kernel 242.4 / 239.1 µs (71% / 72% of 1.62 TB/s),
combine kernel 284.3 / 274.8 µs (60% / 62%); notify kernels 12.5 µs (cached path) and 29.4 µs (with the
count exchange). RCCL's alltoall of the same rows: 413 µs (dispatch), 413 µs (combine).

Uniform routing (top-8 of 256 experts, PR #583's EP4 case), T = 8192; link bytes 320.0 MB:

| CUs | op | µs | link GB/s | of the link peak (178 µs) | of 1.62 TB/s (198 µs) | of the MAIDAS roofline (≈ 210 µs) |
|---:|---|---:|---:|---:|---:|---:|
| 160 | dispatch, counts known | 295.4 | 1083 | 60% | 67% | 71% |
| 160 | dispatch with host wait | 368.1 | 869 | 48% | 54% | 57% |
| 160 | combine | 340.5 | 940 | 52% | 58% | 62% |
| 256 | combine | 320.1 | 1000 | 56% | 62% | 66% |

The host-wait dispatch (368 µs) and the combine (341 µs) match PR #583's EP4 table (0.36 / 0.34 ms).

Where the time goes at EP4 (DeepSeek-V3 routing, ceilings at 8192 rows per peer and the same block count;
a repeat of the CU sweep agrees within 1%):

| CUs | cached dispatch µs | send at the harness ceiling | own rows + receive copy at the local copy rate | rest | combine µs | send at the harness ceiling | rest |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 429.6 | 263.1 | 86.3 | 80.2 | 668.7 | 263.1 | 405.6 |
| 80 | 271.8 | 211.1 | 58.2 | 2.4 | 323.7 | 211.1 | 112.6 |
| 160 | 258.1 | 191.8 | 47.6 | 18.8 | 302.5 | 191.8 | 110.7 |
| 256 | 254.6 | 190.2 | 44.3 | 20.2 | 285.4 | 190.2 | 95.2 |

Fits over T = 256 … 32768 at 160 CUs: cached dispatch 41.7 µs + bytes / 1302 GB/s, with host wait
95.6 µs + 1264 GB/s, `num_worst_tokens` 62.3 µs + 1263 GB/s, combine 65.4 µs + 1198 GB/s. HBM traffic
is 1.04 GB per kernel and rank (45 µs at the peak, 29% of the link time). Counters at 160 CUs:
`WRITE_SIZE` 659–663 MB (dispatch) and 426–430 MB (combine) against 646 / 395 MB in the byte model;
`FETCH_SIZE` 288–294 MB (dispatch) and 673 MB (combine): at EP4 it counts reads of the uncached heap
that it did not count at EP2, so the read counters are not used.

- The dispatch's "rest" is 14–25 µs from 96 CUs up (6 µs at EP2); the combine is 95–118 µs above the
  send (41–64 µs at EP2).

## Where the time goes

EP2, T = 8192, every term from the ceilings measured with the same number of blocks:

| CUs | cached dispatch µs | send at the harness ceiling | own rows + receive copy at the local copy rate | rest | combine µs | send at the harness ceiling | rest |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 272.0 | 163.7 | 51.8 | 56.5 | 348.1 | 163.7 | 184.4 |
| 64 | 159.0 | 98.8 | 35.9 | 24.2 | 195.0 | 98.8 | 96.2 |
| 80 | 141.8 | 105.8 | 34.6 | 1.3 | 167.8 | 105.8 | 62.0 |
| 128 | 123.9 | 86.7 | 27.6 | 9.6 | 150.7 | 86.7 | 64.0 |
| 160 | 122.8 | 87.8 | 29.0 | 5.9 | 145.8 | 87.8 | 58.0 |
| 256 | 118.2 | 84.8 | 26.8 | 6.5 | 138.1 | 84.8 | 53.3 |

- **Dispatch**: each warp loads a row into LDS once, then stores it to every rank it goes to, this rank's
  own rows straight into `recv_x`, and waits for the stores before the next load (one row buffer per
  warp). After its send each block publishes its counts, polls, and copies the peers' rows from the heap
  into `recv_x` (`dispatch_v2_kernel`). The local writes, own rows and peer rows, are about half each of
  the 27–35 µs column. "Send, then local writes" matches the measurement within −5 to +10 µs from 80 CUs
  up; with fewer CUs the per-channel polling adds 24–57 µs. The send at the harness ceiling contains the
  copy kernel's own 15–20 µs of launch, ramp and tail.
- **Notify and host wait**: +57 µs over the cached dispatch (of it the notify kernel 27 µs against the
  cached path's 10 µs); `num_worst_tokens` (no host wait) +23 µs.
- **Combine**: below 96 channels every warp pushes first and reduces afterwards
  (`combine_v2_push_warps`), so push and reduce add up (32 CUs: 164 + 184 µs). From 96 channels 22 of 32
  warps push (LDS holds 22 row slots at hidden 7168) and 10 reduce from the start; the combine is then
  41–64 µs above the send while HBM runs at 17% of its peak: not a bandwidth limit (flag waits, the
  reduce tail, fewer pushers).
- **Few CUs**: the TDM send moves about 22 GB/s per CU, so 32 CUs give 707 GB/s, 39% of the link peak,
  whatever the protocol does.

## What could still be gained

EP2, 160 CUs; modelled from the terms above, not yet measured:

| change | removes | cached dispatch | combine |
|---|---|---:|---:|
| overlap or drop the receive copy (receive warps beside the senders, or `recv_x` in the heap) | about 15 µs | → about 108 µs | – |
| take the own-row stores off the send's path (two row buffers per warp, in pieces; the next load while the stores drain) | about 15 µs | → about 108 µs (94 with the row above) | – |
| overlap push and reduce below 96 channels | most of the 184 µs at 32 CUs | – | 32 CUs: toward the send alone, 164 µs |
| reach the achievable link rate on the send (1.32 → 1.6 TB/s) | about 15 µs | → about 108 µs (79 with both rows above) | → about 131 µs |
| the link peak | | 64 µs | 64 µs |

### What DeepEP does about the same parts

| part of the time | DeepEP V1 (`csrc/kernels/intranode.cu`) | DeepEP V2 (`EPBuffer`) | idea here |
|---|---|---|---|
| receive copy into `recv_x` | overlapped: even blocks send, odd blocks receive and copy from per-channel rings into `recv_x` as the tails advance (the design `flydsl` V1 ports) | a separate copy kernel after the transfer (`dispatch_copy_epilogue_impl`); `defer_epilogue=True` moves it, and the CPU count wait, to `event.current_stream_wait()` so compute can run in between; the benchmark's bandwidth leaves it out | overlap it inside the kernel (V1's way), or drop it by returning `recv_x` as a view of a double-buffered heap (neither DeepEP version does this) |
| notify and host wait | the dispatch needs the counts first; `num_worst_tokens` skips the host wait | the transfer writes into buffer slots before the counts are known; the CPU waits afterwards | the same as V2: give each source rank its own heap region so the send needs no prefix matrix, and let the receive copy order the rows |
| combine reduce | odd blocks reduce as the pushed rows arrive | a separate, deferrable reduce epilogue | reduce while pushing at every grid size; all warps pushing (LDS in pieces); flags per chunk |
| per-CU send rate | – (on H800 a handful of SMs fill NVLink: 160 GB/s against 45 GB/s of writes per SM; the benchmark uses 24 of 132) | TMA-based backends | on MI455X about 74 CUs fill the link (1.62 TB/s against 22 GB/s per CU), so the per-CU rate and copy engines matter far more than on H800 |
| copy engines | – | used for all-gather (`num_sms=0`); EP dispatch and combine need SMs | copy engines for the bulk rows at low CU counts (SDMA moves 1.38–1.54 TB/s here) |

The 84–86% of DeepEP V1 on H800 is its cached dispatch (the counts known) at the best chunk size, the
same case as the cached dispatch here (58%).

## Next measurements

1. ~~EP2 on this branch's head, kernel times and counters~~: done 2026-10-29 (above).
2. ~~EP4~~: measured 2026-10-29 (above), with the ceilings, the phase model and the write counters.
3. **Per-phase timing** inside the kernels (`s_memrealtime` per block: send start and end, counts
   published, receive copy start and end; combine push end and reduce end) to split the local writes
   between the send and the receive copy, and the combine's 40–60 µs.
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
- DeepEP: [V1 performance and method](https://github.com/deepseek-ai/DeepEP/blob/main/docs/legacy.md),
  V1 `tests/test_intranode.py` (before the V2 release, `b306af0^`), V2
  [`tests/ep/test_ep.py`](https://github.com/deepseek-ai/DeepEP/blob/main/tests/ep/test_ep.py) and
  `EPBuffer.get_theoretical_num_sms` in [`deep_ep/buffers/ep.py`](https://github.com/deepseek-ai/DeepEP/blob/main/deep_ep/buffers/ep.py),
  [PR #130](https://github.com/deepseek-ai/DeepEP/pull/130) (algorithm against physical bandwidth).
- nccl-tests [`doc/PERFORMANCE.md`](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)
  (algbw, busbw, the alltoall factor (n − 1) / n).
- PR #583 document `docs/flydsl_deepep_mi455.md` (Performance) for the measurement setup.
