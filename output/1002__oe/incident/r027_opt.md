# Round 27 -- attention backward, FlyDSL, gfx1250

Incumbent = `job_context/op/current` (s6). Beat = aiter prebuilt gfx1250 ASM bwd, must be beaten by 20%.
Inherited score 5.24624 ms = 1.047x beat at prod; need <= 4.576 ms (-12.8%).

## 1. What I read

`findings/facts.md` (current-bottleneck line first), `findings/dead_ends.md`, `findings/pool.md` (122 KB,
highest id g81), `findings/route.md`, and the round-24 deep profile `rounds/024/1-profiling`.

State of the pool coming in: **effectively empty.**
- `r25.i2.g79` -- died round 26 at zero card time (compile gate: `.vgpr_count` 1024, spill 616,
  `private_segment_fixed_size` 1852 B, 486 scratch instructions). LDS was *not* the killer (196608 B = 60.0%).
- `r26.i1.g80` -- died round 26 on its own Gate 1: geometry cannot prove single-SE-ness, and
  `flydsl/expr/rocdl/enum.py` `SyncScope` offers only Agent/Workgroup/Wavefront. No SE scope exists.
- `g73` / `g76` / `g78` -- closed.
- `r26.i2.g81` -- open, Gate 1 passed in round 26, carried to this round with the instruction to test it
  on **s6**, not on the fused arm. Refuted below at zero card time.
- `r24.i3.g74` -- deep-born, open, never executed. Blocker was "FlyDSL capability evidence is external and
  unverified". Unblocked below at zero card time.

Round 26 shipped nothing and round 25 was not accepted, so corpus sources (c) and (d) are **mandatory**.
Consulted files are listed in `explored.consulted` at the end of this file.

## 2. Instruments

### 2.1 GPU idle before starting -- PASS
`rocm-smi --showpids --showuse` clean, no scoring processes, `dmesg` clean (no `SMU: No response`).
The corpus warning is explicit: "If a gfx1250 number moves and the code did not, check `dmesg` before
believing the number."

### 2.2 rocprofv3 `--stats` survey -- BUILD FAILURE, recorded

    rocprofv3 --stats --kernel-trace --output-format csv --output-file survey -d $D -- \
      python3 benchmark.py --shapes prod --iters 11 --block 9 --lead 4 --arm-path cur=.../op/current --arms ""

Returned **rc=0**, ran the workload, printed `RESULT ... latency_ms=4.9894 ... sclk_start=2325 sclk_end=2239`,
and wrote `survey_bench.json`. **No `*.csv` was produced anywhere.** Searched `rounds/027/_scratch`,
`job_context/op`, and `/` to depth 4 with `-newermt "-30 minutes"`. Nothing.

This is the same silent-no-output symptom round 25 recorded for `rocprofv3 -o` -- except I used the form
round 25 recorded as the *working* one. So the per-kernel duration list and dispatch count are **not
available this round**. Not retried further: the round-24 deep profile already carries the per-kernel
budgets (`k_dkdv` 1073.6 cyc / 416 instr / 64 WMMA; `k_dqg` 1204.0 cyc / 417 instr / 96 WMMA) and the
corpus says of this part that "of 51 defined counters, 13 return data -- the entire memory hierarchy and
the entire WMMA family read zero", so the CSV would not have carried traffic numbers anyway.

### 2.3 TRAP recorded -- the job's power verdict rested on an experiment the corpus forbids

`pitfalls/measurement-traps.md`, "A single corrupted probe": *"any argument from input magnitude is
worthless ... Conclusions of the form 'throughput moves with the data, therefore this kernel is
power-limited' cannot be drawn this way."* Round 24's operand-zeroing experiment is exactly that shape,
and it is this job's primary evidence for the power verdict. Round 26 did measure power directly -- but
only under the **sustained** ruler, which `route.md` records as disagreeing **in sign** with the blocked
ruler that actually ships. So: **whether the ruler that ships sits at the cap had never been measured.**
Prediction before measuring: at the cap during the kernel, well below it during the L2 flush.

### 2.4 FREE INSTRUMENT -- power and clock under the BLOCKED (shipping) ruler. New first-class fact.

100 Hz raw `gpu_metrics` sampler (`raw/gmsample.py`): `u16@126` = socket W, `u16@584` = sclk MHz,
format_rev 1 / content_rev 9, 1176 B, 100% yield. Deliberately **not** hwmon `power1_input`, which round 26
proved is biased ~800 W low under load with power-correlated read failures. 8 s idle baseline, then
`benchmark.py --shapes prod --iters 51 --block 9 --lead 4` on `cur`, `CLOCK_MONOTONIC` brackets both sides.
2553 samples at exactly 100.0 Hz.

| window | n | duty | socket W (median) | sclk MHz (median) |
| --- | --- | --- | --- | --- |
| idle baseline | 801 | -- | 1132 | 2355 |
| whole bench window | 1752 | 100% | 2476 | 1478 |
| **in-kernel** (sclk <= 1600) | 951 | **54.3%** | **2500** | **1309** |
| between-iteration L2 flush (sclk >= 2200) | 718 | 41.0% | 1133 | 2355 |

**52.1% of all bench samples read >= 2450 W; p95 = 2501 W; max = 2508 W against a 2500 W socket cap.**

Verdict, now resting on a direct measurement of the shipping ruler rather than on a forbidden
data-magnitude argument: **the incumbent is power-limited at prod under the ruler that scores it.** The
clock is dragged 2355 -> 1309 MHz, a 44% derate, purely by the cap. `time = energy / P_limit`; the shader
clock is a dependent variable.

The consequence for this round's pricing is the whole point: **removing cycles pays at the measured
conversion factor 0.12 on this job; removing transactions pays at full value.** Both arms below were
chosen to be transaction-deleting for exactly this reason.

## 3. Zero-card-time adjudication of the two carried entries

### 3.1 `r26.i2.g81` -- REFUTED on s6. Closing it.

g81 is causal tail pairing: one workgroup takes K-tile `j` and K-tile `n-1-j`, so each pair does `n+1`
units regardless of `j`. The corpus calls it "worth copying" (aiter `fmha_v3_bwd_hd128_bf16` section 6 b6,
`gdx = (gdx + 1) / 2`), and round 26 measured a Gate-1 max/mean of **1.97** -- but **on the fused arm's
geometry**, which has far fewer, 4-wave workgroups. Round 26 itself wrote the instruction to re-test it on s6.

s6's `k_dkdv` at prod launches `grid = (nhkv=8, nblk=256, nb=4) = 8192` **one-wave** workgroups onto
256 CU x 4 SIMD = **1024 resident slots = exactly 8 dispatch waves**. Greedy list-scheduling census
(`raw/g81_census.py`, `raw/g81_census.txt`), dispatch order x fastest then y then z, so grid.y = the kv
block = the slow digit, which is already the longest-first ordering of `r1.i3.g03` / `r12.i2.g40`:

| shape | wgs | dispatch waves | max/mean per wg | current makespan | g81-paired | headroom |
| --- | --- | --- | --- | --- | --- | --- |
| prod | 8192 | 8.00 | 1.992 | 4112.0 | 4112.0 | **+0.000%** |
| proxy | 1024 | 1.00 | 1.984 | 512.0 | 516.0 | **-0.781%** |
| fast | 64 | 0.06 | 1.939 | 128.0 | 132.0 | **-3.125%** |

The per-workgroup skew *is* 1.99, exactly as round 26 measured -- but at prod it is absorbed completely:
dispatch efficiency is already **100.00%**, the makespan already equals the `sum/servers` lower bound, and
pairing cannot improve on a bound that is already met. This reproduces the census line already in
`impl.py`: *"prod launches 8192 -- eight dispatch waves, and the greedy hardware dispatcher balances the
256:1 causal work skew to a modelled 100%"*. At proxy and fast, pairing is actively **negative**, because
halving grid.y halves the number of workgroups and those shapes are already slot-starved (50.4% and 3.2%).

The corpus mechanism is real; it does not apply to a grid with eight dispatch waves. **g81 closed, zero
card time spent.** Lesson for `dead_ends.md`: a Gate-1 imbalance ratio computed on one arm's geometry does
not transfer to another arm's geometry -- the ratio was right and the conclusion was still wrong, because
the quantity that matters is the makespan against the lower bound, not the max/mean.

### 3.2 `r24.i3.g74` -- UNBLOCKED. Its only blocker was an unverified claim.

The pool entry's blocker, verbatim: *"The FlyDSL capability evidence is external and unverified."*
Verified in-container at `/opt/venv/lib/python3.12/site-packages/flydsl`:

- `compiler/kernel_function.py:315` -- `launch(..., cluster: Optional[DimType] = None, ...)`, docstring
  *"Cluster dimensions (x, y, z) for workgroup clustering. None means no clustering. **Enables MCAST and
  cluster barriers.**"* Normalised to `cluster_size` and passed to `gpu.LaunchFuncOp`.
- `expr/rocdl/cluster.py` -- `CLUSTER_BARRIER_ID = -3`, `is_wave_leader()`, `cluster_signal_once_per_wg()`,
  `cluster_wait()`, `cluster_barrier()`, `compute_cluster_position()`, and
  `compute_mcast_masks(local_x, local_y, cluster_m, cluster_n)` documenting
  `flat_wg_id = wg_x + wg_y*nwg_x = local_x + local_y*cluster_m`, returning (a_mask, b_mask) i32.
- `expr/rocdl/tdm_ops.py:216/253/515/549` -- `workgroup_mask: Union[int, "ir.Value"] = 0`,
  *"MCAST workgroup mask [15:0] for TDM GROUP1 descriptor ... 0 = no multicast (default)"*, plus the GL1
  `early_timeout` knob (bit 21, which the corpus calls "the most obviously untried lever on this part").
- `expr/rocdl/__init__.py:369` -- `cluster_load_async_to_lds(global_ptr, lds_ptr, size_bytes, offset=0, cpol=0, mask=None)`.
- `compiler/backends/rocm.py:76` -- a `fly-rocdl-cluster-attr` pass exists in the pipeline.

**g74 is build-ready.** One caveat carried into the build: the incumbent reaches TDM through
`make_tdm_atom` / `copy_atom_call`, which do **not** currently take a mask; the mask lives on the
lower-level `tdm_ops` path. The build has to get the mask onto the GROUP1 descriptor one way or the other,
and "the mask is silently zero" is the failure mode to gate against.

## 4. Why both arms are cluster multicast

The corpus measures workgroup-cluster multicast on **this exact part** at **+6.53% [+6.07, +6.99], n=5**
(`optimization/techniques/6-gfx1250-cdna5-mechanisms.md`, null-control floor 1.57%), and classifies it with
TDM as *transaction-deleting*, explicitly **not discounted by the power cap** -- unlike split barrier
(+4.42%, discounted) and `lock_simd` (~+1.9%, discounted, and worth nothing at one wave per SIMD, which is
what both our bodies run at). Section 2.4 above has now confirmed by direct measurement that the cap is
real on the shipping ruler. So a transaction-deleting mechanism is the only class of candidate that prices
at full value against a -12.8% gap, and this one is measured at half that gap on its own.

Hard rules taken from the same file and gated against in both builds: a cluster holds **at most 16
workgroups**; a multicast mask may name **at most 5 destinations** before the hardware **silently demotes
it to an ordinary load**; the grid must divide by the cluster dimension or the launch must refuse;
`D#.workgroup_mask` must be 0 outside a cluster; multicast loads **force an L1 miss**; and
`Cluster_ID == 0` means not in a cluster, with cluster barriers degrading to NOPs -- the silent-fallback
signature to check for.

Both TDM call sites in the incumbent were read and characterised:

- `k_dkdv._tdm_qdo(qt, gh, stage_off)` (kernels.py ~347) builds its global offset from `(bat, q0, qh)`
  and **not from kv0**. Peers along **grid.y** therefore fill byte-identical Q/dO tiles.
- `k_dqg._tdm_kv(kv0p, stage_off)` (kernels.py ~2046) builds its offset from `(bat, kv0p, hkv)` and **not
  from the q head**. With `DQ_NW = 1`, grid.x is the q head and `hkv = qh // G`, so peers along **grid.x**
  -- the four q heads of one GQA group -- issue identical K/V descriptors.

They are in **different kernels**, so the two arms are independent and mergeable, which satisfies the
"two ideas, each built and measured alone" rule without the arms contaminating each other.
