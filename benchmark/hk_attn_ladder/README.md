# HK attention ladders on gfx1250 (MI455X)

Three optimization ladders of the HipKittens-style (HK) attention kernels written for MI455X, in the format
of the HipKittens GEMM ladder (rocm.blogs.amd.com, "An Educational GEMM Ladder for Helios GPUs"): every
rung is the rung below it plus ONE feature, every rung builds to its own code object, and one harness
checks and times them all in one process on one GPU.

| Ladder | Problem (production shape) | Bottom | Top | Baseline |
|---|---|---|---|---|
| `llama/ladder_fwd.json` | Llama-3.1-8B attention forward: bf16, d 128, GQA 32 / 8, causal, BSHD (b4 s8192) | `f00` naive | `v11_best` (v1.1) | Primus-Turbo FlyDSL |
| `llama/ladder_bwd.json` | the same backward | `b00` naive 3-kernel FA2 | `v11_best` (v1.1) | Primus-Turbo FlyDSL |
| `mla/ladder_bwd.json` | DeepSeek-V3 MLA backward: bf16, qk 192 / v 128, MHA 128 heads, causal, SBHD (b2 s4096) | `m00` naive 3-kernel FA2 | `r7` (71bd0329) | Primus-Turbo FlyDSL |
| `mla/ladder_fwd.json` | the same forward | `x6_m1` (X6 M1, pure HIP) | `e2` (a08-1 E2, persistent) | FlyDSL (main; b12f5953 persistent) |

Every rung was built, checked and timed on e15-02 GPU1 on 2026-10-27: `results/20261027_e15-02_gpu1/` holds the
first campaign (the first-draft bottom rungs), `results/20261027b_e15-02_gpu1/` the one with the redesigned bottom
rungs (the current main lines). The `ref` / `ref_ms` values in the ladder files are a08-1 measurements from when a
rung was first written (GPU2 / GPU3 of another node, other sessions): orientation only.

## Where the rungs come from

* **Historical rungs** rebuild the exact code objects that were measured: the Llama v0 / v1 / v1.1 kernels
  (Primus-Turbo `dev/lhz/llama31_8b_hk` at cade7db6 plus the never-committed v1 sources from the a08-1
  backup) with every compile-time variant that was measured, and the MLA backward commits d3590be4 ..
  71bd0329 of `dev/lhz/dsv3-attn-hk-b0` as a patch chain, rounds 1 and 2 split into one-feature sub-rungs
  replayed from the a08-1 scratch variants. A `sha256` in a ladder file is the recorded hash of the code
  object that was measured; `hkl.build` checks it, so a matching rung is that binary, byte for byte
  (all 28 Llama and 27 MLA pins match with AMD clang 862306352beb, image `amdprimus/amdprimus:tas-20261019`).
* **New rungs** (`src/ladder/*.hip`, written for this ladder, no pin) fill the bottom the history does
  not have: nothing below Llama v0 / MLA d3590be4 survived. Each is the rung below it plus one feature, in the
  order the measurements support (the first draft added features in "natural" order and was non-monotonic; it
  is kept as side rungs):
  * Llama forward `f00` -> `f10z`: naive, TDM fill, double buffer, operand bursts, a 3-slot ring, software
    pipelining (S^T(c+1) under the softmax of c, co-execution scheduler), GQA packing, longest-first order,
    lock_simd, mask on the diagonal block only, lazy rescale; `f10z` runs at v0's speed.
  * Llama backward `b00` -> `b12l`: FlashAttention-2's three-kernel backward, TDM fill, double buffer, bursts,
    dQ fused in v0's decomposition (`b04f`: 128 kv rows per WG, 32-query steps, dS^T exchanged through LDS, each
    wave's K slice of dQ resident, 128-B-line atomics on a C-fragment dQ_acc), longest-first order, wave kinds
    (FULL steps first), LD rows by TDM, deferred dQ, the softmax pipelined under the WMMAs, v0's split barrier,
    lock_simd; v0 is 1.13x above `b12l`.
  * MLA backward `m00` -> `m07q`: the same three-kernel backward with the MLA head dims, TDM fill, the dS export
    of hk_v0, wider dQ waves, bursts, 32 kv rows per dK / dV wave, longest-first order, 64-query dQ waves;
    hk_v0 is still 1.48x above `m07q` (see Known issues).
  The MLA bottom rungs live in the Llama tree (`"tree": "../llama/hip"`, they use its hk_attn primitives) and
  launch through `mla/lab/bwd_mladder.py`. Every new rung passes `hkl.test` at every shape of its problem kind,
  production shape included.
* **MLA forward** (`mla/ladder_fwd.json`): the a08-1 HK forward, never merged into Primus-Turbo, rebuilt from the
  backup: X6 (pure HIP, `x6_m1` .. `x6_m2f`, from the MLA backward tree's include/) and the marker-shell + generated
  asm line S1a .. E2 (`mla/fwd_pkg/`: gfx1250_mla_fwd/ and asmkit/ at W-S1a, `mla/fwd_patches/` the later versions,
  built by `gen/build.py`, hkl.build "gen" kernels). All 22 code objects rebuild byte for byte to the shas a08-1
  measured (m1a / m2e / m2fx had no recorded sha: their notes match). The generated rungs are bitwise equal to the
  b12f5953 FlyDSL forward at every shape.
* **Side rungs** (`"side": true`) are measured dead ends that make good "what did not work" notes (v0 flags
  alone, TAIL, LSUM=1, the 8-wave ping-pong forward, ONEPHASE, the default-scheduling v1 backward, ...).

## Layout

```
hkl/            build.py    device-less build: patch chains, asm patches, sha256 pins, ISA (.s / .dis), meta.json
                runner.py   loads a rung (its own lab-module instance) or a baseline as fwd()/bwd() callables
                selfcheck.py CPU-only: every rung's code object vs its launcher's argument spec + a meta-tensor plan
                test.py     GPU correctness gate (fp32 reference, NaN-poisoned allocator, determinism, bitwise)
                ladder.py   GPU timing campaign (rotated rounds, L2 flush per call, null control, sclk)
                ref.py      fp32 reference (GQA BSHD / MLA SBHD), SNR, useful FLOPs
                one.py      one arm, a few calls (the workload of a profiler)
                table.py    a ladder file as a Markdown table
llama/hip/      the Llama HK package's hip/ (cade7db6) + v1 sources + src/ladder/ (new rungs), include/ladder/
llama/lab/      launch modules: fwd_v0, fwd_v1, fwd_pp, fwd_ladder, bwd_v0, bwd_v1, bwd_ladder (+ hiprt, codeobj)
mla/hip/        the HK MLA backward's hip/ at d3590be4; mla/patches/ the rest of its history; src/x6_mla_fwd_m*.hip
mla/fwd_pkg/    the HK MLA forward package (gfx1250_mla_fwd/, asmkit/) at W-S1a; mla/fwd_patches/ S1a2 .. E2
mla/lab/        bwd_hk (the production impl.py on a rung's build dir, SBHD folding), flydsl_bwd (baseline),
                fwd_hk (the MLA forward rungs), flydsl_fwd (the forward baselines: main, vendored b12f5953)
tools/          start_cpu_container.sh, start_gpu_container.sh, run_gpu.sh, att.sh, ktrace.sh, campaign.sh,
                vendor_flydsl_fwd.sh (b12f5953's FlyDSL MLA package -> build/vendor/)
```

## Use

```bash
tools/start_cpu_container.sh                                   # build container, no devices
python3 -m hkl.build llama/ladder_fwd.json --check             # every rung; exit 1 on a failed build / pin
docker exec -u $(id -u):$(id -g) -e HOME=/tmp -w $PWD lhz-hk-cpu python3 -m hkl.selfcheck llama/ladder_fwd.json

tools/start_gpu_container.sh 2                                 # GPU 2 only (/dev/kfd + its render node)
export HKL_GPU=2
tools/campaign.sh B                                            # new rungs, one per job, toy shapes
tools/campaign.sh C                                            # correctness, all shapes
tools/campaign.sh D                                            # timing tables -> logs/ladder_*.md
tools/campaign.sh E                                            # ATT of every main-line rung
```

`python3 -m hkl.test <ladder> --rung ... --shapes ...` and `python3 -m hkl.ladder <ladder> --rung ...` take
any subset (run them through `tools/run_gpu.sh <tag> <timeout> -- ...`).

## Measurement protocol

`hkl.ladder`: each arm verified once against the fp32 reference at the timing shape; pre-warmed for a few
seconds (sclk settles under the power cap); R rounds in a rotated order; a cell = W unmeasured calls + I
calls, each timed by its own event pair with a 512 MB fill enqueued before it; the top rung loaded twice
(the second, `#null`, bounds the noise of a step); a host thread samples the GPU's hwmon sclk. Steps are
per-round ratios against the rung's parent when it was timed, else the previous main-line rung (median and 95%
CI). On this card time follows the
clock while cycle counts are stable, so a step that changes power (e.g. the MLA dS export) can be worth
less in time than in cycles; the sclk column shows it.

Estimated GPU time: stage B ~30 min (34 short jobs), C ~45 min (the fp32 references dominate), D ~1 h,
E ~1 h. End-to-end Megatron runs (Llama-3.1-8B 1 GPU FlyDSL / HK p1 / HK v1.1; DSv3 proxy FlyDSL / HK
backward) use the Primus launchers and the Turbo branches, not this directory.

## GPU safety

From the a08-1 incidents (MES hangs, page faults, hive resets that needed power cycles): one job per GPU
(`run_gpu.sh` lock), a container that maps a single GPU, a hard timeout, refusal on a degraded card,
automatic stop with a `logs/GPU<n>.FAULT` interlock on any new amdgpu error line (only a person removes it);
new kernels run alone at toy shapes before anything larger; no multi-stream / CU-masked co-runs, no cluster
barriers; asm-patched kernels only from their verified patch (`hkl.build` refuses a mismatching `.s`).

## Known issues

* **dQ read error in the shipped MLA dQ kernel from r2d on (5c20de7e .. 71bd0329, `k_dqds_c` sha 5e3782d6),
  confirmed on MI455X (e15-02 GPU1, 2026-10-27)**: with causal and (Skv - Sq) % 64 == 32 the last steady
  unit's region-C read of row 2 nf takes the wrong half-block (found by a CPU replay of its idx.h). At b1
  Sq 4096 Skv 4128 h4 (`hkl.test --shapes rect32`) dq drops to 17.7 dB from r2d on while r2c and every
  rung before it are bitwise equal to hk_v0 (52.5 dB); dk / dv are unaffected. Production cannot reach it:
  the Primus-Turbo adapter's gate (`hk_mla_bwd_unsupported_reason`) applies the FlyDSL rules first, which
  require Sq and Skv to be multiples of 64, so Skv - Sq is too; only `limits.shape_unsupported_reason`
  alone (as the ladder's bwd_hk uses it) admits such shapes.
* The HK MLA forward (`mla/ladder_fwd.json`, top `e2`) only reaches parity with FlyDSL's persistent forward
  (b12f5953): 0.830 vs 0.821 ms on e15-02 GPU1 (0.5% more cycles), as on a08-1 (0.7396 vs 0.7378); it is 1.035x
  faster than main's FlyDSL forward (#567, 0.859 ms), which does not have the persistent schedule. It was never
  merged into Primus-Turbo. `x6_m2e` is not the version a08-1 measured (see the rung).
* `v11_best` of the Llama forward never ran on a GPU (it was blocked by a halted card on a08-1).
* `m07q` -> `hk_v0` is still a large step (1.48x): hk_v0's 48-kv-row waves on 2-wave workgroups (each Q / dO
  operand feeds 3 WMMAs at 998 VGPRs), its dS^T export in whole 128-B lines to a causal-packed layout and its
  384-query dQ workgroups on a K ring were not rebuilt as single steps; the Llama backward ladder's later
  steps (LD rows by TDM, pipelined softmax, barrier + readback, a deeper ring, wave kinds, 2-wave workgroups)
  measured flat or negative on MLA at `m07q`'s geometry and are side rungs. The pre-squash kernels between
  m02 and hk_v0 (k1, k1ds, q1 / q2, k3) were lost with the host they were written on.
* `b12l` -> Llama backward v0 is a 1.13x step (v0's register-double-buffered operand sets, 4-slot ring with TDM
  roles, packed LD records, sched_group_barrier interleave).

## Rungs

Generated with `python3 -m hkl.table <ladder> --build` (VGPRs / static LDS of the main kernel).

### Llama forward

| Rung | Parent | Source / flags | Pin | a08-1 ms | Feature | VGPR / LDS |
|---|---|---|---|---|---|---|
| f00 |  | `f00_naive.hip` | new |  | naive: 64 queries x 1 head per WG, register-mediated single-buffer K/V fill, 2 barriers per block, eager rescale | 238/35K |
| f01t | f00 | `f01t_tdm.hip` | new |  | TDM fill (one buffer, 2 barriers per block): no VGPRs, VALU or buffer-load waits for the fill | 218/35K |
| f02 | f01t | `f02_tdm.hip` | new |  | double-buffered TDM fill: block j+1's tensor_load_to_lds ops in flight during block j's compute, 1 barrier per block | 201/70K |
| f03b | f02 | `f03b_bursts.hip` | new |  | operand bursts: a phase's LDS operands loaded at once (V before the softmax), not one load before each WMMA | 298/70K |
| f04r | f03b | `f04r_ring3.hip` | new |  | 3-slot TDM ring, prefetch distance 2 (enabler of the pipelined rung) | 298/105K |
| f05p | f04r | `f05p_pipeline.hip` `-amdgpu-sched-strategy=coexec` | new |  | software pipelining: S^T(j+1) WMMAs under the softmax of block j (one branch-free region) | 284/105K |
| f06g | f05p | `f06g_gqa4.hip` `-amdgpu-sched-strategy=coexec` | new |  | GQA packing on the pipelined kernel: each K / V operand feeds the 4 q heads of its kv head | 672/105K |
| f07l | f06g | `f07l_longest_first.hip` `-amdgpu-sched-strategy=coexec` | new |  | longest-first workgroup order (causal load balance) | 672/105K |
| f08s | f07l | `f08s_lock_simd.hip` `-amdgpu-sched-strategy=coexec` | new |  | lock_simd: no WMMA arbitration pause at one wave per SIMD | 672/105K |
| f09m | f08s | `f09m_diag_mask.hip` `-amdgpu-sched-strategy=coexec` | new |  | causal mask on the diagonal block only (peeled loop, no branch in the hot loop) | 809/105K |
| f10z | f09m | `f10z_lazy_rescale.hip` `-amdgpu-sched-strategy=coexec` | new |  | lazy rescale behind one unlikely branch per chunk (v0's RESC form), max taken a phase ahead | 678/105K |
| v0 | f10z | `k_attn_fwd_v0.hip` `-amdgpu-sched-strategy=coexec` | yes | 1.123 | first HK forward (Primus-Turbo fd4a1bd0), the shipped form of f10z: 4-slot ring with split barriers, Q by TDM, O through LDS, a finer QK \|\| softmax interleave (same speed as f10z) | 692/320K |
| v0_resc | v0 | `k_attn_fwd_v0.hip` `-amdgpu-sched-strategy=coexec -DHK_FWD_OPT_RESC=1` | yes | 1.101 | lazy-rescale test as one max + one VCC branch | 696/320K |
| v0_resc_desc | v0_resc | `k_attn_fwd_v0.hip` `-amdgpu-sched-strategy=coexec -DHK_FWD_OPT_RESC=1 -DHK_FWD_OPT_DESC=1` | new |  | TDM descriptor math inside the QK WMMA run, burst LDS bases one stage ahead | 696/320K |
| p1 | v0_resc_desc | `k_attn_fwd_v0.hip` `-amdgpu-sched-strategy=coexec -DHK_FWD_OPT_ALL=1 -DHK_FWD_OPT_TAIL=0` | yes | 1.045 | K(c+2) and V(c+1) loads as one burst per chunk (Primus-Turbo cade7db6 pin) | 752/320K |
| v1_v0body | v0 | `k_attn_fwd_v1.hip` `-amdgpu-sched-strategy=coexec -DFWD1_MDS=0 -DFWD1_LSUM=0 -DFWD1_SM1L=0 -DFWD1_FSPLIT=0` | yes | 1.120 | v1 source with the v0 body (v0 grid) | 714/320K |
| v1_v0body_pairs | v1_v0body | `k_attn_fwd_v1.hip` `-amdgpu-sched-strategy=coexec -DFWD1_MDS=0 -DFWD1_LSUM=0 -DFWD1_SM1L=0 -DFWD1_FSPLIT=0` | yes | 1.099 | persistent grid: zig-zag causal pairs, XCD-major | 714/320K |
| v1_lsum0 | v1_v0body_pairs | `k_attn_fwd_v1.hip` `-amdgpu-sched-strategy=coexec -DFWD1_LSUM=0` | yes | 1.041 | lean SM1 / FSPLIT, F1 interleave, MDS2 (v1 as validated) | 768/320K |
| v11_best | v1_lsum0 | `k_attn_fwd_v1.hip` `-amdgpu-sched-strategy=coexec -DFWD1_LSUM=0 -DFWD1_OPT_RESC=1 -DFWD1_MDS=3 -DFWD1_OPT_UNROLL=1` | yes |  | v1.1: RESC + one burst per stage (MDS3) + 2-stage unroll (never run on a GPU) | 898/320K |
| f01 | f00 | `f01_double_buf.hip` | new |  | negative: register-path double buffer -- 64 more VGPRs hold block j+1's loads, occupancy 4 -> 3 waves/SIMD (ATT); the TDM path (f01t, f02) double-buffers without VGPRs *(side)* | 292/70K |
| f03 | f02 | `f03_gqa4.hip` | new |  | side chain (GQA before pipelining; one S^T -> softmax -> PV dependency chain per block): GQA packing on f02 -- 4x the work per WG at 1 wave/SIMD, nothing to hide the softmax under *(side)* | 657/70K |
| f04 | f03 | `f04_longest_first.hip` | new |  | side chain (GQA before pipelining; one S^T -> softmax -> PV dependency chain per block): longest-first workgroup order (causal load balance) *(side)* | 657/70K |
| f05 | f04 | `f05_ring4.hip` | new |  | side chain (GQA before pipelining; one S^T -> softmax -> PV dependency chain per block): 4-slot TDM ring with prefetch distance 2 (two blocks of compute cover each block's TDM latency) *(side)* | 657/140K |
| f06 | f05 | `f06_split_barrier.hip` | new |  | side chain (GQA before pipelining; one S^T -> softmax -> PV dependency chain per block): split barrier: signal after S^T(j), wait at the top of block j+1 (softmax + PV absorb wave skew) *(side)* | 659/140K |
| f07 | f06 | `f07_lock_simd.hip` | new |  | side chain (GQA before pipelining; one S^T -> softmax -> PV dependency chain per block): lock_simd: no WMMA arbitration pause at one wave per SIMD *(side)* | 659/140K |
| f08 | f07 | `f08_lazy_rescale.hip` | new |  | side chain (GQA before pipelining; one S^T -> softmax -> PV dependency chain per block): lazy rescale: O / l rescaled only when a query's max grew by > 8 nat (uniform branch) *(side)* | 788/140K |
| f09z | f08s | `f09z_lazy_rescale.hip` `-amdgpu-sched-strategy=coexec` | new |  | negative: lazy rescale with a per-head branch at every chunk end (the branches cut the interleaved schedule) *(side)* | 676/105K |
| x_desc | v0 | `k_attn_fwd_v0.hip` `-amdgpu-sched-strategy=coexec -DHK_FWD_OPT_DESC=1` | yes | 1.111 | DESC alone on v0 *(side)* | 684/320K |
| x_merge | v0 | `k_attn_fwd_v0.hip` `-amdgpu-sched-strategy=coexec -DHK_FWD_OPT_MERGE=1` | yes | 1.113 | MERGE alone on v0 *(side)* | 748/320K |
| x_p1_tail | p1 | `k_attn_fwd_v0.hip` `-amdgpu-sched-strategy=coexec -DHK_FWD_OPT_ALL=1` | yes | 1.065 | negative: row sums moved into the PV run (TAIL) *(side)* | 776/320K |
| x_v1_lsum1 | v1_lsum0 | `k_attn_fwd_v1.hip` `-amdgpu-sched-strategy=coexec` | yes | 1.052 | negative: row sums by a WMMA with an all-ones operand (LSUM=1) *(side)* | 770/320K |
| x_pp | v0 | `k_attn_fwd_pp.hip` | yes | 1.221 | negative: 8-wave ping-pong (HK's CDNA4 attention schedule) *(side)* | 424/320K |

### Llama backward

| Rung | Parent | Source / flags | Pin | a08-1 ms | Feature | VGPR / LDS |
|---|---|---|---|---|---|---|
| b00 |  | `b00_naive.hip` | new |  | naive FA2 backward: prep + dK/dV kernel + separate dQ kernel (S, dP recomputed), register-mediated single-buffer fills | 338/34K, 234/34K |
| b01t | b00 | `b01t_tdm.hip` | new |  | TDM fills (one buffer, 2 barriers per iteration): no VGPRs, VALU or buffer-load waits for the fill | 324/34K, 218/34K |
| b02 | b01t | `b02_tdm.hip` | new |  | double-buffered TDM fills: the next iteration's tiles in flight during this one's compute, 1 barrier per iteration | 320/68K, 220/68K |
| b03b | b02 | `b03b_bursts.hip` | new |  | operand bursts (next group's burst during this group's WMMAs, tr16 operands during the softmax), NL / delta requested a run ahead | 474/68K, 338/68K |
| b04f | b03b | `b04f_fused.hip` | new |  | dQ fused in v0's split: 128 kv / WG, 32-query steps, dS^T exchanged through LDS, each wave's K slice of dQ resident, 128-B-line fp32 atomics (5 GEMMs instead of 7) | 693/92K |
| b05g | b04f | `b05g_longest_first.hip` | new |  | longest-first workgroup order: kv heads fastest, kv blocks ascending (v0's grid order) | 693/92K |
| b06k | b05g | `b06k_wave_kinds.hip` | new |  | wave kinds per step (v0's FULL / DIAG / DEAD): the mask only on diagonal squares, fully masked squares skipped | 822/92K |
| b07l | b06k | `b07l_ld_tdm.hip` | new |  | NL / delta rows by TDM with the step's Q / dO tiles, read from LDS (v0's LD records): no global loads in the loop (flat alone; the enabler of b08d) | 822/92K |
| b08d | b07l | `b08d_deferred_dq.hip` | new |  | deferred dQ: dS operands requested right after the barrier, dQ(s) WMMAs + atomics in step s+1 after its S / dP run | 910/92K |
| b09i | b08d | `b09i_dq_softmax.hip` | new |  | dQ(s-1) WMMAs and step s's softmax in one scheduling region (no branch, no sched_barrier), atomics after the dS^T stores | 918/92K |
| b10p | b09i | `b10p_pipeline.hip` `-amdgpu-sched-strategy=coexec` | new |  | two pipelined regions per step (co-execution scheduler): S / dP(q tile 1) \|\| softmax(q tile 0), dQ(s-1) \|\| softmax(q tile 1) | 808/92K |
| b11s | b10p | `b11s_split_barrier.hip` `-amdgpu-sched-strategy=coexec` | new |  | v0's split step barrier: tr16 of dV / dK in flight past the signal, dV / dK WMMAs in the window, 3-slot ring with ops after the wait | 688/109K |
| b12l | b11s | `b12l_lock_simd.hip` `-amdgpu-sched-strategy=coexec` | new |  | lock_simd: no WMMA arbitration pause at one wave per SIMD | 688/109K |
| v0 | b12l | `k_attn_bwd_v0.hip` | yes | 3.491 | first HK backward (Primus-Turbo fd4a1bd0), the shipped form of b12l: next step's operand sets double-buffered in registers (x2-unrolled parity steps, readback a phase ahead), a 4-slot ring with TDM roles (wave 3 moves K / V), LD records per 4 queries, S / dP by kv tile with the softmax interleaved by sched_group_barrier | 754/260K |
| v0_regs | v0 | `k_attn_bwd_v0.hip` `-DHK_BWD_OPT_REGS=7` | yes | 3.433 | holds on the S / dP WMMA source registers (no WAR v_nops) | 770/260K |
| v0_regs_masked | v0_regs | `k_attn_bwd_v0.hip` `-DHK_BWD_OPT_REGS=7 -DHK_BWD_OPT_MASKED=1` | new |  | diagonal steps use the full step's operand layout | 770/260K |
| v0_regs_masked_atom4 | v0_regs_masked | `k_attn_bwd_v0.hip` `-DHK_BWD_OPT_REGS=7 -DHK_BWD_OPT_MASKED=1 -DHK_BWD_OPT_ATOM=4` | new |  | the 32 dQ atomics of a step as 4 sub-bursts of 8 | 776/260K |
| v0_rma_tdm3 | v0_regs_masked_atom4 | `k_attn_bwd_v0.hip` `-DHK_BWD_OPT_REGS=7 -DHK_BWD_OPT_MASKED=1 -DHK_BWD_OPT_ATOM=4 -DHK_BWD_OPT_TDM3=1` | new |  | wave 3 issues a real 16-byte dummy TDM op instead of a role branch | 768/260K |
| p1 | v0_rma_tdm3 | `k_attn_bwd_v0.hip` `-DHK_BWD_OPT_MASKED=1 -DHK_BWD_OPT_ATOM=4 -DHK_BWD_OPT_TDM3=1 -DHK_BWD_OPT_REGS=7 -DHK_BWD_OPT_ENTRY=1 -DHK_BWD_OPT_ADDR=1` | yes | 3.337 | loop-entry DS wait + early LDS base math (Primus-Turbo cade7db6 pin) | 766/260K |
| v1_sched0 | v0 | `k_attn_bwd_v1.hip` `-amdgpu-expert-scheduling-mode -DHK_ATTN_EXPERT_SCHED_BUILD -DBWD1_SCHED=0` | yes | 3.286 | v1 protocol under expert scheduling, memory ops still in bursts | 830/260K |
| v1 | v1_sched0 | `k_attn_bwd_v1.hip` `-amdgpu-expert-scheduling-mode -DHK_ATTN_EXPERT_SCHED_BUILD` | yes | 3.242 | every memory op interleaved under the WMMA runs | 830/260K |
| v1_atom2 | v1 | `k_attn_bwd_v1.hip` `-amdgpu-expert-scheduling-mode -DHK_ATTN_EXPERT_SCHED_BUILD -DBWD1_ATOM=2` | yes | 3.206 | one dQ atomic per WMMA gap (v1 as validated) | 830/260K |
| v11_regs | v1_atom2 | `k_attn_bwd_v1.hip` `-amdgpu-expert-scheduling-mode -DHK_ATTN_EXPERT_SCHED_BUILD -DBWD1_ATOM=2 -DBWD1_OPT_REGS=15` | yes | 3.211 | v1.1: register holds | 842/260K |
| v11_regs_tdm3 | v11_regs | `k_attn_bwd_v1.hip` `-amdgpu-expert-scheduling-mode -DHK_ATTN_EXPERT_SCHED_BUILD -DBWD1_ATOM=2 -DBWD1_OPT_REGS=15 -DBWD1_OPT_TDM3=1` | yes | 3.203 | v1.1: dummy TDM op instead of a role branch | 860/260K |
| v11_best | v11_regs_tdm3 | `k_attn_bwd_v1.hip` `-amdgpu-expert-scheduling-mode -DHK_ATTN_EXPERT_SCHED_BUILD -DBWD1_ATOM=2 -DBWD1_OPT_REGS=15 -DBWD1_OPT_TDM3=1 -DBWD1_OPT_MASKED=2` | yes | 3.194 | v1.1: diagonal steps on the full layout (MASKED=2) | 860/260K |
| b01 | b00 | `b01_double_buf.hip` | new |  | negative: register-path double buffer -- 64 more VGPRs for the in-flight loads cost a wave per SIMD (ATT); the TDM path (b01t, b02) double-buffers without VGPRs *(side)* | 433/68K, 298/68K |
| b05r | b04f | `b05r_ring3.hip` | new |  | negative: 3-slot ring + split step barrier on b04f -- the signal's DS drain right after the dS^T stores costs more than the skew it absorbs *(side)* | 693/109K |
| b05s | b04f | `b05s_split_barrier.hip` | new |  | negative: split step barrier with two Q / dO buffers -- the early signal waits for step s+1's TDM op *(side)* | 693/92K |
| b05n | b04f | `b05n_prefetch.hip` | new |  | negative: the same prefetch on b04f (before the work order and the wave kinds) *(side)* | 819/92K |
| b07n | b06k | `b07n_prefetch.hip` | new |  | negative: next step's operands a step ahead in registers (NL / delta, Q / dO readback) -- 990 VGPRs *(side)* | 990/92K |
| x_b10p_nocoexec | b10p | `b10p_pipeline.hip` | new |  | b10p in the default scheduler (no co-execution strategy) *(side)* | 902/92K |
| b03 | b02 | `b03_gqa4.hip` | new |  | side chain (first draft: GQA / fusion before the v0 decomposition): dQ kernel takes the 4 q heads of one kv head per WG (K / V tiles shared) *(side)* | 320/68K, 872/68K |
| b04 | b03 | `b04_fused_dq.hip` | new |  | side chain (first draft: GQA / fusion before the v0 decomposition): dQ fused into the dK/dV kernel: dS^T through LDS, dQ = dS K, fp32 atomics (5 GEMMs per pair instead of 7) *(side)* | 342/96K |
| b05 | b04 | `b05_kv32.hip` | new |  | side chain (first draft: GQA / fusion before the v0 decomposition): 32 kv rows per wave (128 per WG): every Q / dO LDS operand feeds 2 WMMAs, half the Q / dO traffic per kv row *(side)* | 598/124K |
| b06 | b05 | `b06_deferred_dq.hip` | new |  | side chain (first draft: GQA / fusion before the v0 decomposition): dQ one iteration late from a double-buffered dS^T: 1 barrier per iteration instead of 2 *(side)* | 584/144K |
| b07 | b06 | `b07_lock_simd.hip` | new |  | side chain (first draft: GQA / fusion before the v0 decomposition): lock_simd in the dK/dV kernel (one wave per SIMD) *(side)* | 584/144K |
| x_masked | v0 | `k_attn_bwd_v0.hip` `-DHK_BWD_OPT_MASKED=1` | yes | 3.448 | MASKED alone on v0 *(side)* | 754/260K |
| x_atom4 | v0 | `k_attn_bwd_v0.hip` `-DHK_BWD_OPT_ATOM=4` | yes | 3.464 | ATOM=4 alone on v0 *(side)* | 760/260K |
| x_tdm3 | v0 | `k_attn_bwd_v0.hip` `-DHK_BWD_OPT_TDM3=1` | yes | 3.473 | TDM3 alone on v0 *(side)* | 754/260K |
| x_addr | v0 | `k_attn_bwd_v0.hip` `-DHK_BWD_OPT_ADDR=1` | yes | 3.471 | ADDR alone on v0 *(side)* | 762/260K |
| x_entry | v0 | `k_attn_bwd_v0.hip` `-DHK_BWD_OPT_ENTRY=1` | yes | 3.493 | ENTRY alone on v0 *(side)* | 754/260K |
| x_onephase | v0 | `k_attn_bwd_v0.hip` `-DHK_BWD_OPT_ONEPHASE=1` | yes | 4.233 | negative: one memory phase per step (zero-window barrier) *(side)* | 850/260K |
| x_v1_noexp | v1 | `k_attn_bwd_v1.hip` | yes | 5.263 | negative: v1's interleaved source in the default scheduling mode *(side)* | 830/260K |
| x_v1_atom0 | v1 | `k_attn_bwd_v1.hip` `-amdgpu-expert-scheduling-mode -DHK_ATTN_EXPERT_SCHED_BUILD -DBWD1_ATOM=0` | yes | 3.400 | negative: all 32 atomics as one burst *(side)* | 830/260K |
| x_v0_noint | v0 | `k_attn_bwd_v0.hip` `-DBWD_INTERLEAVE=0` | new |  | v0 without the sched_group_barrier interleave of the softmax into the S / dP and dQ WMMA runs *(side)* | 762/260K |
| x_v0_nodead | v0 | `k_attn_bwd_v0.hip` `-DBWD_SKIP_DEAD=0` | new |  | v0 with DEAD waves running the masked diagonal path *(side)* | 734/260K |

### DeepSeek-V3 MLA backward

| Rung | Parent | Source / flags | Pin | a08-1 ms | Feature | VGPR / LDS |
|---|---|---|---|---|---|---|
| m00 |  | `m00_naive.hip` | new |  | naive FA2 backward with MLA dims (qk 192 / v 128, MHA): prep + dK/dV kernel + separate dQ kernel recomputing S and dP | 424/42K, 319/42K |
| m01 | m00 | `m01_tdm_dbuf.hip` | new |  | double-buffered TDM fills (the Llama ladder's b01 + b02): 192-wide tiles as 128- and 64-column ops | 387/84K, 277/84K |
| m02 | m01 | `m02_ds_export.hip` | new |  | dS^T exported by the dK/dV kernel to a scratch, dQ = dS K read back from it: no S/dP recompute, no atomics | 364/84K, 134/70K |
| m03q | m02 | `m03q_dq_wide.hip` | new |  | dQ kernel: 32 queries per wave (128 per WG), each K^T operand feeds 2 WMMAs, a chunk's operands as one burst | 364/84K, 309/86K |
| m04b | m03q | `m04b_bursts.hip` | new |  | dK/dV kernel: operand bursts per q tile, tr16 operands of dV / dK during the softmax, NL / delta requested a run ahead | 540/84K, 309/86K |
| m05w | m04b | `m05w_kv32.hip` | new |  | dK/dV kernel: 32 kv rows per wave (128 per WG), 32-query steps: every Q / dO operand feeds 2 WMMAs | 771/42K, 309/86K |
| m06g | m05w | `m06g_longest_first.hip` | new |  | longest-first workgroup order in both kernels (heads fastest; kv blocks ascending, q blocks descending) | 771/42K, 309/86K |
| m07q | m06g | `m07q_dq64.hip` | new |  | dQ kernel: 64 queries per wave (256 per WG): each K^T operand feeds 4 WMMAs, a K tile serves 256 queries | 771/42K, 517/118K |
| hk_v0 | m07q | `k_dkdv96ds.hip`; `k_dqds_c.hip` | yes | 2.529 | first HK backward (d3590be4): 48 kv rows per wave on 2-wave workgroups (each Q / dO operand feeds 3 WMMAs), a 3-stage Q / dO ring with per-q-tile readback, dS^T exported in whole 128-B lines (DPP regroup) to a causal-packed layout (4.33 GB), 384-query dQ workgroups on a K ring, lock_simd | 1012/63K, 887/270K |
| r1a | hk_v0 | `01a_dkdv_dsexport_pin.patch`; `k_dkdv96ds.hip`; `k_dqds_c.hip` | part |  | dK/dV: the dS^T export (6 buffer stores) is pinned by a sched_barrier right after the dO^T tr16 burst, ahead of the first dV WMMA: without expert scheduling a store issued inside the WMMA run waits for every earlier WMMA, under the tr16 latency it is free. | 1016/63K, 887/270K |
| r1b | r1a | `01b_dkdv_split_barrier.patch`; `k_dkdv96ds.hip`; `k_dqds_c.hip` | part |  | dK/dV: split ring barrier (hk1.h lds_arrive = release fence + s_barrier_signal, lds_wait = s_barrier_wait + acquire) with the 24 register-only dK kh1/kh2 WMMAs between arrive and wait, so the other wave's late arrival overlaps matrix work instead of an idle s_barrier. | 1012/63K, 887/270K |
| r1c | r1b | `01c_dkdv_mha_loop_top.patch`; `k_dkdv96ds.hip`; `k_dqds_c.hip` | part |  | dK/dV: the full loop specialised for G == 1 (MHA: the q-pair counter is ii, the q head 0), so the clamped prefetch tiles are min(ii + k, n - 1) instead of the k1_nx_* / k1_pf_* select chains: loop-top SALU 156 -> 57 per 2 iterations (the G != 1 loop stays as a second copy). | 1008/63K, 887/270K |
| r1d | r1c | `01d_dkdv_nlnd_voffset.patch`; `k_dkdv96ds.hip`; `k_dqds_c.hip` | part |  | dK/dV: the 8 NL/ND loads put their constant on the VGPR offset (3 loop-invariant VGPRs) and share one SGPR soffset, so no SALU rewrites an address register that an in-flight buffer load still reads (the s_wait_xcnt 0 WAR stalls go). | 1004/63K, 887/270K |
| r1e | r1d | `01e_dkdv_dsexport_staging.patch`; `k_dkdv96ds.hip`; `k_dqds_c.hip` | yes |  | dK/dV: the dS^T export first regroups all 3 blocks into distinct staging VGPRs (24 DPP ops), then issues the 6 stores, so no store-data register is rewritten while an earlier store still reads it (the ~82-cyc store-data s_wait_xcnt goes; with r1d the full loop has no s_wait_xcnt left). | 1004/63K, 887/270K |
| r1f | r1e | `k_dkdv96ds.hip` `-amdgpu-expert-scheduling-mode`; `k_dqds_c.hip` | yes |  | dK/dV compiled with -amdgpu-expert-scheduling-mode: the compiler places explicit s_wait_alu va_vdst / vm_vsrc counts instead of relying on the hardware interlock, so DS / VMEM issue no longer waits for every earlier WMMA. | 1004/63K, 887/270K |
| r1g | r1f | `01g_dkdv_masked_ring.patch`; `k_dkdv96ds.hip` `-amdgpu-expert-scheduling-mode`; `k_dqds_c.hip` | yes |  | dK/dV: the masked (diagonal) iterations ride the full loop's 3-stage Q/dO TDM ring with prefetch distance 2 (tile j in stage (j - n_mask) mod 3, k3m_idx.h) instead of an own-tile TDM + tensorcnt 0 + 2 barriers each: ~3 exposed HBM round trips (~1.2k cyc) per 96-row workgroup and 4 barriers per wave go. | 1015/63K, 887/270K |
| r1h | r1g | `01h_dq_split_barrier.patch`; `k_dkdv96ds.hip` `-amdgpu-expert-scheduling-mode`; `k_dqds_c.hip` | part |  | dQ: split barrier P_{u+1} in every steady unit but the last: arrive (tensorcnt 1, dscnt 0, release, s_barrier_signal) after WMMA 48 of run(X) (32 on the head tile), wait after run(X), so the remaining WMMAs absorb the ~90-cyc lag of the dS-issuing waves 0 / 3. | 1015/63K, 887/270K |
| r1i | r1h | `k_dkdv96ds.hip` `-amdgpu-expert-scheduling-mode`; `k_dqds_c.hip` `-amdgpu-expert-scheduling-mode` | part |  | dQ compiled with -amdgpu-expert-scheduling-mode: a DS read no longer waits for earlier WMMAs (the prerequisite of r1j's interleave). | 1015/63K, 887/270K |
| r1j | r1i | `01j_dq_burst_interleave.patch`; `k_dkdv96ds.hip` `-amdgpu-expert-scheduling-mode`; `k_dqds_c.hip` `-amdgpu-expert-scheduling-mode` | part |  | dQ: in main-tile steady units the 36-tr16 operand bursts are interleaved into the WMMA runs with sched_group_barrier ({8 WMMA} then {1 tr16, 1 WMMA} x 36 before the arrive, {4 WMMA} then the same under run(Y)), hiding the 130-180-cyc bursts under matrix work. | 1015/63K, 887/270K |
| r1k | r1j | `01k_dq_split_at_66.patch`; `k_dkdv96ds.hip` `-amdgpu-expert-scheduling-mode`; `k_dqds_c.hip` `-amdgpu-expert-scheduling-mode` | part |  | dQ: the split-barrier arrive moves from WMMA 48 to 66 (44 on the head tile): region A now carries the whole interleaved Y burst, and the later arrive lets its 36 tr16 retire before the s_wait_dscnt 0 that precedes the signal. | 1015/63K, 887/270K |
| r1l | r1k | `01l_dq_masked_branchfree.patch`; `k_dkdv96ds.hip` `-amdgpu-expert-scheduling-mode`; `k_dqds_c.hip` `-amdgpu-expert-scheduling-mode` | part |  | dQ: the masked units' zero-select predicate in branch-free form (x4_ds_zero: \| instead of \|\|), so the selects compile to in-place v_cndmask instead of 17 scalar branches + v_dual_mov operand copies that split the scheduling region (VGPR 887 -> 871). | 1015/63K, 871/270K |
| r1m | r1l | `01m_dq_masked_split.patch`; `k_dkdv96ds.hip` `-amdgpu-expert-scheduling-mode`; `k_dqds_c.hip` `-amdgpu-expert-scheduling-mode` | yes |  | dQ: the masked units (all but the last) take the split barrier too, with the select pattern carried in the first part of run(X). | 1015/63K, 871/270K |
| r1 | r1m | `k_dkdv96ds.hip` `-amdgpu-expert-scheduling-mode`; `k_dqds_c.hip` `-amdgpu-expert-scheduling-mode` | yes | 2.299 | round 1: dK/dV scheduling (dS export placement, split barrier, MHA loop top, NL/ND voffsets), masked tiles on the TDM ring, dQ split barrier / tr16 interleave / branch-free masks; expert scheduling for both | 1015/63K, 871/270K |
| r2a | r1 | `02a_dq_masked_pair_skip.patch`; `k_dkdv96ds.hip` `-amdgpu-expert-scheduling-mode`; `k_dqds_c.hip` `-amdgpu-expert-scheduling-mode` | part |  | dQ: in masked units a wave-uniform branch per q pair skips a fully masked pair's WMMAs instead of selecting its B operand to zero (bitwise identical: those WMMAs only add +-0), so the selects and the masked pairs' WMMA issue (and energy) go. | 1015/63K, 871/270K |
| r2b | r2a | `02b_dq_subtile_epilogue.patch`; `k_dkdv96ds.hip` `-amdgpu-expert-scheduling-mode`; `k_dqds_c.hip` `-amdgpu-expert-scheduling-mode` | part |  | dQ: the epilogue runs per 16-row sub-tile through a private 16 x 400 B per-wave LDS image after the rings (LDS 276480 -> 302080, still 1 WG/CU) instead of the whole-wave image in the dead ring: shorter store / load chains, epilogue 3.1 -> 2.4 us per WG. | 1015/63K, 871/295K |
| r2c | r2b | `02c_dq_real_desc_nocheck.patch`; `k_dkdv96ds.hip` `-amdgpu-expert-scheduling-mode`; `k_dqds_c.hip` `-amdgpu-expert-scheduling-mode` | part |  | dQ: the steady loop's (REAL) TDM descriptors drop the runtime nv >= 1 check (a real unit's rows always hold a stored pair), so the descriptor SALU inside the WMMA run needs no select / branch that would split burst Y and run X into different basic blocks. | 1015/63K, 871/295K |
| r2d | r2c | `02d_dq_steady_unroll3.patch`; `k_dkdv96ds.hip` `-amdgpu-expert-scheduling-mode`; `k_dqds_c.hip` `-amdgpu-expert-scheduling-mode` | part |  | dQ: main-tile steady loop unrolled by the ring depth (3), so every stage index is a compile-time constant: the burst addresses become precomputed base VGPRs + immediates and the loop carries no VALU, which lets expert mode count va_vdst(k) waits instead of va_vdst(0) drains. | 1015/63K, 878/295K |
| r2e | r2d | `02e_dq_keep_ds.patch`; `k_dkdv96ds.hip` `-amdgpu-expert-scheduling-mode`; `k_dqds_c.hip` `-amdgpu-expert-scheduling-mode` | part |  | dQ: keep_ds() (an empty asm volatile taking the 6 dS operands) keeps each region's dS operands live to its end, so the next burst cannot be allocated to a register the newest WMMA just read (that WAR forced va_vdst(0)). | 1015/63K, 925/295K |
| r2f | r2e | `02f_dq_snake_pairs.patch`; `k_dkdv96ds.hip` `-amdgpu-expert-scheduling-mode`; `k_dqds_c.hip` `-amdgpu-expert-scheduling-mode` | yes |  | dQ: q pairs dealt to the waves in snake order {w, 7 - w, 8 + w} (head {w, 7 - w}) instead of contiguous blocks, so the fully masked pairs of the diagonal units spread over all 4 waves (skippable share per wave 83% / 8% -> 42-50%) and the barrier-bound masked critical path drops by a third. | 1015/63K, 934/295K |
| r2 | r2f | `02z_5c20de7e_leftover.patch`; `k_dkdv96ds.hip` `-amdgpu-expert-scheduling-mode`; `k_dqds_c.hip` `-amdgpu-expert-scheduling-mode` | yes | 2.274 | dQ round 2: snake q pairs, masked-pair skip, steady loop unrolled by the ring depth, per-sub-tile epilogue image | 1015/63K, 934/295K |
| r3 | r2 | `k_dkdv96ds.hip` `-amdgpu-expert-scheduling-mode -amdgpu-enable-delay-alu=false`; `k_dqds_c.hip` `-amdgpu-expert-scheduling-mode` | yes | 2.255 | dK/dV compiler knobs: bottom-up pre-RA scheduling, delay-alu off, no machine sink | 1016/63K, 934/295K |
| r4 | r3 | `04_2f3d60c6.patch`; `k_dkdv96ds.hip` `-amdgpu-expert-scheduling-mode -amdgpu-enable-delay-alu=false`; `k_dqds_c.hip` `-amdgpu-expert-scheduling-mode` | yes | 2.231 | dK/dV: Q^T loads interleaved into the dV run + asm register retarget (va_vdst(0) drains 15 -> 9 per 2 iterations) | 1015/63K, 934/295K |
| r5 | r4 | `05_5b19697f.patch`; `k_dkdv96ds.hip` `-amdgpu-expert-scheduling-mode -amdgpu-enable-delay-alu=false`; `k_dqds_c.hip` `-amdgpu-expert-scheduling-mode`; `k_delta.hip` | yes | 2.218 | HIP k_delta (several b128 loads in flight per lane) | 1015/63K, 934/295K, 48/0K |
| r6 | r5 | `06_7ccfba3b.patch`; `k_dkdv96ds.hip` `-amdgpu-expert-scheduling-mode -amdgpu-enable-delay-alu=false`; `k_dqds_c.hip` `-amdgpu-expert-scheduling-mode`; `k_delta.hip` | yes |  | hardening: s_wait_xcnt before the tail block reuses its K/V descriptor SGPRs (perf-neutral) | 1015/63K, 934/295K, 48/0K |
| r7 | r6 | `07_71bd0329.patch`; `k_dkdv96ds.hip` `-amdgpu-expert-scheduling-mode -amdgpu-enable-delay-alu=false`; `k_dqds_c.hip` `-amdgpu-expert-scheduling-mode`; `k_delta.hip` | yes | 2.220 | redundant causal test dropped from the masked dK/dV loop | 1015/63K, 934/295K, 48/0K |
| m07k | m06g | `m07k_wave_kinds.hip` | new |  | flat on MLA (1.00x): q steps descending, FULL steps without the mask, wave kinds on the diagonal -- the mask is a smaller share of a 6-k-step S tile, at 966 VGPRs *(side)* | 966/42K, 309/86K |
| m07l | m06g | `m07l_ld_tdm.hip` | new |  | flat on MLA (side): NL / delta rows by TDM with the step's Q / dO tiles (the Llama ladder's b07l; 1.00x) *(side)* | 771/42K, 309/86K |
| m08p | m06g | `m08p_pipeline.hip` `-amdgpu-sched-strategy=coexec` | new |  | flat on MLA (side): q tile 1's S / dP WMMAs \|\| q tile 0's softmax, co-execution scheduler (0.99x; by kv tile: 0.98x) *(side)* | 868/42K, 309/86K |
| m08q | m07l | `m08q_dq64.hip` | new |  | m07l + the 64-query dQ waves of m07q *(side)* | 771/42K, 517/118K |
| m08h | m07q | `m08h_two_waves.hip` | new |  | negative on MLA (side): 2-wave dK / dV workgroups (64 kv rows; 0.97x: twice the Q / dO staging per kv row) *(side)* | 771/42K, 517/118K |
| m09r | m08q | `m09r_readback.hip` | new |  | flat on MLA (side): step barrier between the dV and dK runs, the next step's first operands + NL / delta read back under the dK WMMAs (1.00x) *(side)* | 852/42K, 517/118K |
| m10t | m09r | `m10t_ring3.hip` | new |  | flat on MLA (side): 3-slot Q / dO ring, TDM ops two steps ahead (1.00x) *(side)* | 857/63K, 517/118K |

### DeepSeek-V3 MLA forward

| Rung | Parent | Source / flags | Pin | a08-1 ms | Feature | VGPR / LDS |
|---|---|---|---|---|---|---|
| x6_m1 |  | `x6_mla_fwd_m1.hip` | yes | 1.151 | first HK MLA forward (X6 M1, pure HIP): 256-query WG (4 waves x 4 q tiles), 64-kv tiles on a 4-stage TDM ring, each tile serial: LDS burst -> 96 QK WMMA -> softmax -> 64 PV WMMA | 1003/268K |
| x6_m2f | x6_m1 | `x6_mla_fwd_m2.hip` | yes | 0.934 | M2: 32-kv chunks software-pipelined (QK(c+1) \|\| exp(c), PV(c) \|\| P / row sums / next mask), split barrier, branch-free TDM (NULL descriptors), K double-buffered in registers, a chunk's LDS reads (V^T(c), K(c+2)) as one burst at the head of phase 1 (X6's best) | 874/268K |
| s1a | x6_m2f | gen `(defaults)` | yes | 0.896 | marker shell (LLVM: prologue, TDM ring, barriers, epilogue) + generated asm tile regions (asmkit: MSB, expert waits, hazards; list scheduler), no cross-tile pipelining | 1024/268K |
| s1a_diag | s1a | gen `--diag` | yes | 0.872 | diagonal tiles specialized (TILE_D0..3 skip the q tiles past the diagonal) | 1024/268K |
| s1a2_diag | s1a_diag | `02_s1a2.patch`; gen `--csgpr --qgroup late --worder QK0,QK1,PV0,QK2,QK3,PV1,PV2,PV3 --trans-cost 3 --diag` | yes | 0.857 | softmax scale c from an SGPR pair, Q operands regrouped late, QK / PV WMMA order interleaved (QK0,QK1,PV0,QK2,QK3,PV1,PV2,PV3) | 1024/268K |
| s1b_A | s1a2_diag | `04_s1b.patch`; gen `--sched slot --csgpr --diag --pipe --ds-lat 290 --ds-first --ds-rate 6 --pipe-vafter pv --pipe-tree body --pipe-dsentry` | yes | 0.815 | cross-tile software pipelining (region boundary moved into the middle of the tile) with the slot scheduler (DS latency 290, DS first) | 1024/268K |
| e1 | s1b_A | `06_edev1.patch`; gen `--sched slot --csgpr --diag --pipe --ds-lat 290 --ds-first --ds-rate 6 --pipe-vafter pv --pipe-tree body --pipe-dsentry` | yes | 0.747 | epilogue: O / LSE staged through the wave's own Q rows and written as whole rows (straight line, rolling XCNT window) instead of 4 x 8 stores each followed by s_wait_xcnt 0 | 1024/268K |
| e2 | e1 | `07_edev2.patch`; gen `--sched slot --csgpr --diag --pipe --ds-lat 290 --ds-first --ds-rate 6 --pipe-vafter pv --pipe-tree body --pipe-dsentry` | yes | 0.740 | persistent: 256 WGs on a static schedule, the K / V ring continuous across blocks, the next block's Q and tiles 0..2 prefetched after its barrier (12-argument kernel) | 1024/303K |
| x6_m2e | x6_m1 | `x6_mla_fwd_m2.hip` `-DX6_BURST=0` | yes | 0.995 | not a measured version: the final M2 source built with X6_BURST=0 (same resources / kernarg notes as a08-1's m2e, no recorded sha), 1.41 ms on e15-02 against a08-1 m2e's 0.995; M2's chunk pipelining itself (m2d / m2e) was not preserved as source *(side)* | 824/268K |
| x6_m2fx | x6_m2f | `x6_mla_fwd_m2.hip` `-amdgpu-expert-scheduling-mode` | yes | 0.940 | negative: m2f under expert scheduling (s_wait_alu va_vdst(0) drains, 35-48 cyc each) *(side)* | 874/268K |
| ref | s1a | gen `(defaults)` ref | yes | 1.000 | REF: s1a's marker shell with the tile code in HIP (LLVM-compiled; the generator's correctness reference) *(side)* | 884/268K |
| ref_diag | s1a_diag | gen `--diag` ref | yes | 0.976 | REF with the diagonal specialization (HKGEN_DIAG=1) *(side)* | 826/268K |
| s1a_prog | s1a | gen `--sched program` | yes | 1.103 | negative: generated regions in program order (no list scheduling) *(side)* | 1024/268K |
| s1a2 | s1a | `02_s1a2.patch`; gen `--csgpr --qgroup late --worder QK0,QK1,PV0,QK2,QK3,PV1,PV2,PV3 --trans-cost 3` | yes | 0.884 | s1a2's tile code without the diagonal specialization *(side)* | 1024/268K |
| s1a3_n200 | s1a2_diag | `03_s1a3.patch`; gen `--csgpr --qgroup late --trans-cost 3 --diag --worder QK0,QK1,PV0,QK2,QK3,PV1,PV2,PV3 --ds-lat 200` | yes |  | flat: s1a2 scheduled with a 200-cycle DS latency model *(side)* | 1024/268K |
| s1a3_d290 | s1a2_diag | `03_s1a3.patch`; gen `--csgpr --qgroup late --trans-cost 3 --diag --defer3 --worder PV3p,QK0,QK1,PV0,QK2,QK3,PV1,PV2 --ds-lat 290` | yes |  | flat: LDS reads deferred into the previous tile (defer3; model 0.79-0.85, measured no gain) *(side)* | 1024/268K |
| s1b_B | s1a2_diag | `04_s1b.patch`; gen `--csgpr --trans-cost 3 --diag --pipe --ds-lat 290 --ds-first --ds-rate 8 --pipe-tree body --pipe-vafter pv --pipe-dsentry` | yes | 0.827 | s1b's pipelining with the old greedy list scheduler *(side)* | 1024/268K |
| fdev2_Ap | s1b_A | `05_fdev2.patch`; gen `--sched slot --csgpr --diag --pipe --ds-lat 290 --ds-first --ds-rate 6 --pipe-vafter pv --pipe-tree body --pipe-dsentry --pipe-q3after --arrive-lat 100` | yes | 0.821 | flat: barrier arrive moved from WMMA 132 to 99 (A'; barrier waits are not the bottleneck) *(side)* | 1024/268K |
| fdev2_A2 | s1b_A | `05_fdev2.patch`; gen `--sched slot --csgpr --diag --pipe --ds-lat 150 --ds-first --ds-rate 4 --pipe-vafter qk0 --pipe-tree body --pipe-dsentry` | yes | 0.812 | flat: rescheduled with the refitted r4 model (DS latency 150, rate 4; model -3..-4%, measured -1%) *(side)* | 1024/268K |
| e1d | e1 | `06b_e1d.patch`; gen `--sched slot --csgpr --diag --pipe --ds-lat 290 --ds-first --ds-rate 6 --pipe-vafter pv --pipe-tree body --pipe-dsentry` | yes |  | diagnostic: E1 with the old 16-row scattered write-back *(side)* | 1024/268K |
| ref_e2 | e2 | gen `--sched slot --csgpr --diag --pipe --ds-lat 290 --ds-first --ds-rate 6 --pipe-vafter pv --pipe-tree body --pipe-dsentry` ref | yes |  | E2's persistent shell with the tile code in HIP (REF) *(side)* | 834/303K |
| e2_np | e2 | gen `--sched slot --csgpr --diag --pipe --ds-lat 290 --ds-first --ds-rate 6 --pipe-vafter pv --pipe-tree body --pipe-dsentry` | yes |  | E2's code object launched non-persistent (npw = 0: grid (nqb, H, B), one block per WG) *(side)* | 1024/303K |
