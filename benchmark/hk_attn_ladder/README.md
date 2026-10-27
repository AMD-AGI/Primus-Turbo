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

Nothing in this directory has run on a GPU yet (prepared CPU-only, 2026-10-27). The `ref` / `ref_ms`
values in the ladder files are a08-1 measurements from when a rung was first written (GPU2 / GPU3 of
another node, other sessions): orientation only, every rung is re-measured by `tools/campaign.sh`.

## Where the rungs come from

* **Historical rungs** rebuild the exact code objects that were measured: the Llama v0 / v1 / v1.1 kernels
  (Primus-Turbo `dev/lhz/llama31_8b_hk` at cade7db6 plus the never-committed v1 sources from the a08-1
  backup) with every compile-time variant that was measured, and the MLA backward commits d3590be4 ..
  71bd0329 of `dev/lhz/dsv3-attn-hk-b0` as a patch chain, rounds 1 and 2 split into one-feature sub-rungs
  replayed from the a08-1 scratch variants. A `sha256` in a ladder file is the recorded hash of the code
  object that was measured; `hkl.build` checks it, so a matching rung is that binary, byte for byte
  (all 28 Llama and 27 MLA pins match with AMD clang 862306352beb, image `amdprimus/amdprimus:tas-20261019`).
* **New rungs** (`src/ladder/*.hip`, written for this ladder, no pin) fill the bottom the history does
  not have: nothing below Llama v0 / MLA d3590be4 survived. Llama forward `f00`..`f08` climb from a naive
  kernel to v0 minus software pipelining; Llama backward `b00`..`b07` from FlashAttention-2's three-kernel
  backward to v0 minus its step pipeline; MLA backward `m00`..`m02` from the same three-kernel backward with
  the MLA head dims, through TDM fills, to the dS-export idea of hk_v0 (dQ read back from a dS^T scratch
  instead of recomputing S and dP). The MLA bottom rungs live in the Llama tree (`"tree": "../llama/hip"`,
  they use its hk_attn primitives) and launch through `mla/lab/bwd_mladder.py`. They are reviewed on the CPU (an independent review of
  `f00`..`f08` found no correctness issue) but only a GPU run can validate them: stage B of the campaign
  runs each one alone at a toy shape first.
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
mla/hip/        the HK MLA backward's hip/ at d3590be4; mla/patches/ the rest of its history
mla/lab/        bwd_hk (the production impl.py on a rung's build dir, SBHD folding), flydsl_bwd (baseline)
tools/          start_cpu_container.sh, start_gpu_container.sh, run_gpu.sh, att.sh, campaign.sh
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
per-round ratios against the previous main-line rung (median and 95% CI). On this card time follows the
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
* The HK MLA forward reached parity with FlyDSL but was never integrated (a08-1 `fwd_summary.md`): a
  sidebar, not a ladder. The MLA forward is FlyDSL.
* `v11_best` of the Llama forward never ran on a GPU (it was blocked by a halted card on a08-1).
* `m02` -> `hk_v0` is still a large step (96-row kv blocks on a 3-stage ring, packed dS layout, 384-query dQ
  workgroups, lock_simd): the pre-squash kernels between them (k1, k1ds, q1 / q2, k3) were lost with the
  host they were written on.

## Rungs

Generated with `python3 -m hkl.table <ladder> --build` (VGPRs / static LDS of the main kernel).

### Llama forward

| Rung | Parent | Source / flags | Pin | a08-1 ms | Feature | VGPR / LDS |
|---|---|---|---|---|---|---|
| f00 |  | `f00_naive.hip` | new |  | naive: 64 queries x 1 head per WG, register-mediated single-buffer K/V fill, 2 barriers per block, eager rescale | 238/35K |
| f01 | f00 | `f01_double_buf.hip` | new |  | double-buffered LDS: block j+1's global loads in flight during block j's compute, 1 barrier per block | 292/70K |
| f02 | f01 | `f02_tdm.hip` | new |  | TDM fill: one 32-row tensor_load_to_lds per wave per block instead of the register path | 201/70K |
| f03 | f02 | `f03_gqa4.hip` | new |  | GQA packing: the 4 q heads of one kv head per WG share every K / V tile and LDS operand | 657/70K |
| f04 | f03 | `f04_longest_first.hip` | new |  | longest-first workgroup order (causal load balance) | 657/70K |
| f05 | f04 | `f05_ring4.hip` | new |  | 4-slot TDM ring with prefetch distance 2 (two blocks of compute cover each block's TDM latency) | 657/140K |
| f06 | f05 | `f06_split_barrier.hip` | new |  | split barrier: signal after S^T(j), wait at the top of block j+1 (softmax + PV absorb wave skew) | 659/140K |
| f07 | f06 | `f07_lock_simd.hip` | new |  | lock_simd: no WMMA arbitration pause at one wave per SIMD | 659/140K |
| f08 | f07 | `f08_lazy_rescale.hip` | new |  | lazy rescale: O / l rescaled only when a query's max grew by > 8 nat (uniform branch) | 788/140K |
| v0 | f08 | `k_attn_fwd_v0.hip` | yes | 1.123 | first HK forward (Primus-Turbo fd4a1bd0): f08 + software pipelining (QK(c+1) \|\| softmax(c) on 32-row chunks, double-buffered S^T), mask on the last stage only, Q through TDM, O through LDS, 320 KB LDS | 692/320K |
| v0_resc | v0 | `k_attn_fwd_v0.hip` `-DHK_FWD_OPT_RESC=1` | yes | 1.101 | lazy-rescale test as one max + one VCC branch | 696/320K |
| v0_resc_desc | v0_resc | `k_attn_fwd_v0.hip` `-DHK_FWD_OPT_RESC=1 -DHK_FWD_OPT_DESC=1` | new |  | TDM descriptor math inside the QK WMMA run, burst LDS bases one stage ahead | 696/320K |
| p1 | v0_resc_desc | `k_attn_fwd_v0.hip` `-DHK_FWD_OPT_ALL=1 -DHK_FWD_OPT_TAIL=0` | yes | 1.045 | K(c+2) and V(c+1) loads as one burst per chunk (Primus-Turbo cade7db6 pin) | 752/320K |
| v1_v0body | v0 | `k_attn_fwd_v1.hip` `-DFWD1_MDS=0 -DFWD1_LSUM=0 -DFWD1_SM1L=0 -DFWD1_FSPLIT=0` | yes | 1.120 | v1 source with the v0 body (v0 grid) | 714/320K |
| v1_v0body_pairs | v1_v0body | `k_attn_fwd_v1.hip` `-DFWD1_MDS=0 -DFWD1_LSUM=0 -DFWD1_SM1L=0 -DFWD1_FSPLIT=0` | yes | 1.099 | persistent grid: zig-zag causal pairs, XCD-major | 714/320K |
| v1_lsum0 | v1_v0body_pairs | `k_attn_fwd_v1.hip` `-DFWD1_LSUM=0` | yes | 1.041 | lean SM1 / FSPLIT, F1 interleave, MDS2 (v1 as validated) | 768/320K |
| v11_best | v1_lsum0 | `k_attn_fwd_v1.hip` `-DFWD1_LSUM=0 -DFWD1_OPT_RESC=1 -DFWD1_MDS=3 -DFWD1_OPT_UNROLL=1` | yes |  | v1.1: RESC + one burst per stage (MDS3) + 2-stage unroll (never run on a GPU) | 898/320K |
| x_desc | v0 | `k_attn_fwd_v0.hip` `-DHK_FWD_OPT_DESC=1` | yes | 1.111 | DESC alone on v0 *(side)* | 684/320K |
| x_merge | v0 | `k_attn_fwd_v0.hip` `-DHK_FWD_OPT_MERGE=1` | yes | 1.113 | MERGE alone on v0 *(side)* | 748/320K |
| x_p1_tail | p1 | `k_attn_fwd_v0.hip` `-DHK_FWD_OPT_ALL=1` | yes | 1.065 | negative: row sums moved into the PV run (TAIL) *(side)* | 776/320K |
| x_v1_lsum1 | v1_lsum0 | `k_attn_fwd_v1.hip` | yes | 1.052 | negative: row sums by a WMMA with an all-ones operand (LSUM=1) *(side)* | 770/320K |
| x_pp | v0 | `k_attn_fwd_pp.hip` | yes | 1.221 | negative: 8-wave ping-pong (HK's CDNA4 attention schedule) *(side)* | 424/320K |

### Llama backward

| Rung | Parent | Source / flags | Pin | a08-1 ms | Feature | VGPR / LDS |
|---|---|---|---|---|---|---|
| b00 |  | `b00_naive.hip` | new |  | naive FA2 backward: prep + dK/dV kernel + separate dQ kernel (S, dP recomputed), register-mediated single-buffer fills | 338/34K, 234/34K |
| b01 | b00 | `b01_double_buf.hip` | new |  | double-buffered fills: the next tiles' global loads in flight during this iteration, 1 barrier per iteration | 433/68K, 298/68K |
| b02 | b01 | `b02_tdm.hip` | new |  | TDM fills: one 32-row tensor_load_to_lds per wave per tile pair instead of the register path | 320/68K, 220/68K |
| b03 | b02 | `b03_gqa4.hip` | new |  | dQ kernel takes the 4 q heads of one kv head per WG (K / V tiles shared) | 320/68K, 872/68K |
| b04 | b03 | `b04_fused_dq.hip` | new |  | dQ fused into the dK/dV kernel: dS^T through LDS, dQ = dS K, fp32 atomics (5 GEMMs per pair instead of 7) | 342/96K |
| b05 | b04 | `b05_kv32.hip` | new |  | 32 kv rows per wave (128 per WG): every Q / dO LDS operand feeds 2 WMMAs, half the Q / dO traffic per kv row | 598/124K |
| b06 | b05 | `b06_deferred_dq.hip` | new |  | dQ one iteration late from a double-buffered dS^T: 1 barrier per iteration instead of 2 | 584/144K |
| b07 | b06 | `b07_lock_simd.hip` | new |  | lock_simd in the dK/dV kernel (one wave per SIMD) | 584/144K |
| v0 | b07 | `k_attn_bwd_v0.hip` | yes | 3.491 | first HK backward (Primus-Turbo fd4a1bd0): b07 + 32-query steps (q pairs, heads inner, unmasked steps first) on a 4-deep TDM ring of Q / dO / LD records with one split-barrier phase per step, dQ_acc in C-fragment order (one 128-B line per atomic), and the 9-phase software pipeline (softmax interleaved into the WMMA runs, dQ(s-1) under step s) | 754/260K |
| v0_regs | v0 | `k_attn_bwd_v0.hip` `-DHK_BWD_OPT_REGS=7` | yes | 3.433 | holds on the S / dP WMMA source registers (no WAR v_nops) | 770/260K |
| v0_regs_masked | v0_regs | `k_attn_bwd_v0.hip` `-DHK_BWD_OPT_REGS=7 -DHK_BWD_OPT_MASKED=1` | new |  | diagonal steps use the full step's operand layout | 770/260K |
| v0_regs_masked_atom4 | v0_regs_masked | `k_attn_bwd_v0.hip` `-DHK_BWD_OPT_REGS=7 -DHK_BWD_OPT_MASKED=1 -DHK_BWD_OPT_ATOM=4` | new |  | the 32 dQ atomics of a step as 4 sub-bursts of 8 | 776/260K |
| v0_rma_tdm3 | v0_regs_masked_atom4 | `k_attn_bwd_v0.hip` `-DHK_BWD_OPT_REGS=7 -DHK_BWD_OPT_MASKED=1 -DHK_BWD_OPT_ATOM=4 -DHK_BWD_OPT_TDM3=1` | new |  | wave 3 issues a real 16-byte dummy TDM op instead of a role branch | 768/260K |
| p1 | v0_rma_tdm3 | `k_attn_bwd_v0.hip` `-DHK_BWD_OPT_MASKED=1 -DHK_BWD_OPT_ATOM=4 -DHK_BWD_OPT_TDM3=1 -DHK_BWD_OPT_REGS=7 -DHK_BWD_OPT_ENTRY=1 -DHK_BWD_OPT_ADDR=1` | yes | 3.337 | loop-entry DS wait + early LDS base math (Primus-Turbo cade7db6 pin) | 766/260K |
| v1_sched0 | v0 | `k_attn_bwd_v1.hip` `-DHK_ATTN_EXPERT_SCHED_BUILD -DBWD1_SCHED=0` | yes | 3.286 | v1 protocol under expert scheduling, memory ops still in bursts | 830/260K |
| v1 | v1_sched0 | `k_attn_bwd_v1.hip` `-DHK_ATTN_EXPERT_SCHED_BUILD` | yes | 3.242 | every memory op interleaved under the WMMA runs | 830/260K |
| v1_atom2 | v1 | `k_attn_bwd_v1.hip` `-DHK_ATTN_EXPERT_SCHED_BUILD -DBWD1_ATOM=2` | yes | 3.206 | one dQ atomic per WMMA gap (v1 as validated) | 830/260K |
| v11_regs | v1_atom2 | `k_attn_bwd_v1.hip` `-DHK_ATTN_EXPERT_SCHED_BUILD -DBWD1_ATOM=2 -DBWD1_OPT_REGS=15` | yes | 3.211 | v1.1: register holds | 842/260K |
| v11_regs_tdm3 | v11_regs | `k_attn_bwd_v1.hip` `-DHK_ATTN_EXPERT_SCHED_BUILD -DBWD1_ATOM=2 -DBWD1_OPT_REGS=15 -DBWD1_OPT_TDM3=1` | yes | 3.203 | v1.1: dummy TDM op instead of a role branch | 860/260K |
| v11_best | v11_regs_tdm3 | `k_attn_bwd_v1.hip` `-DHK_ATTN_EXPERT_SCHED_BUILD -DBWD1_ATOM=2 -DBWD1_OPT_REGS=15 -DBWD1_OPT_TDM3=1 -DBWD1_OPT_MASKED=2` | yes | 3.194 | v1.1: diagonal steps on the full layout (MASKED=2) | 860/260K |
| x_masked | v0 | `k_attn_bwd_v0.hip` `-DHK_BWD_OPT_MASKED=1` | yes | 3.448 | MASKED alone on v0 *(side)* | 754/260K |
| x_atom4 | v0 | `k_attn_bwd_v0.hip` `-DHK_BWD_OPT_ATOM=4` | yes | 3.464 | ATOM=4 alone on v0 *(side)* | 760/260K |
| x_tdm3 | v0 | `k_attn_bwd_v0.hip` `-DHK_BWD_OPT_TDM3=1` | yes | 3.473 | TDM3 alone on v0 *(side)* | 754/260K |
| x_addr | v0 | `k_attn_bwd_v0.hip` `-DHK_BWD_OPT_ADDR=1` | yes | 3.471 | ADDR alone on v0 *(side)* | 762/260K |
| x_entry | v0 | `k_attn_bwd_v0.hip` `-DHK_BWD_OPT_ENTRY=1` | yes | 3.493 | ENTRY alone on v0 *(side)* | 754/260K |
| x_onephase | v0 | `k_attn_bwd_v0.hip` `-DHK_BWD_OPT_ONEPHASE=1` | yes | 4.233 | negative: one memory phase per step (zero-window barrier) *(side)* | 850/260K |
| x_v1_noexp | v1 | `k_attn_bwd_v1.hip` | yes | 5.263 | negative: v1's interleaved source in the default scheduling mode *(side)* | 830/260K |
| x_v1_atom0 | v1 | `k_attn_bwd_v1.hip` `-DHK_ATTN_EXPERT_SCHED_BUILD -DBWD1_ATOM=0` | yes | 3.400 | negative: all 32 atomics as one burst *(side)* | 830/260K |

### DeepSeek-V3 MLA backward

| Rung | Parent | Source / flags | Pin | a08-1 ms | Feature | VGPR / LDS |
|---|---|---|---|---|---|---|
| m00 |  | `m00_naive.hip` | new |  | naive FA2 backward with MLA dims (qk 192 / v 128, MHA): prep + dK/dV kernel + separate dQ kernel recomputing S and dP | 424/42K, 319/42K |
| m01 | m00 | `m01_tdm_dbuf.hip` | new |  | double-buffered TDM fills (the Llama ladder's b01 + b02): 192-wide tiles as 128- and 64-column ops | 387/84K, 277/84K |
| m02 | m01 | `m02_ds_export.hip` | new |  | dS^T exported by the dK/dV kernel to a scratch, dQ = dS K read back from it: no S/dP recompute, no atomics | 364/84K, 134/70K |
| hk_v0 | m02 | `k_dkdv96ds.hip`; `k_dqds_c.hip` | yes | 2.529 | first HK backward (d3590be4): m02 + 96-row kv blocks (two 48-row waves on a 3-stage Q/dO TDM ring, 1 wave/SIMD), causal-packed dS layout (4.33 GB), 384-query dQ workgroups on a K ring, lock_simd | 1012/63K, 887/270K |
| r1a | hk_v0 | `01a_dkdv_dsexport_pin.patch`; `k_dkdv96ds.hip`; `k_dqds_c.hip` | part |  | dK/dV: the dS^T export (6 buffer stores) is pinned by a sched_barrier right after the dO^T tr16 burst, ahead of the first dV WMMA: without expert scheduling a store issued inside the WMMA run waits for every earlier WMMA, under the tr16 latency it is free. | 1016/63K, 887/270K |
| r1b | r1a | `01b_dkdv_split_barrier.patch`; `k_dkdv96ds.hip`; `k_dqds_c.hip` | part |  | dK/dV: split ring barrier (hk1.h lds_arrive = release fence + s_barrier_signal, lds_wait = s_barrier_wait + acquire) with the 24 register-only dK kh1/kh2 WMMAs between arrive and wait, so the other wave's late arrival overlaps matrix work instead of an idle s_barrier. | 1012/63K, 887/270K |
| r1c | r1b | `01c_dkdv_mha_loop_top.patch`; `k_dkdv96ds.hip`; `k_dqds_c.hip` | part |  | dK/dV: the full loop specialised for G == 1 (MHA: the q-pair counter is ii, the q head 0), so the clamped prefetch tiles are min(ii + k, n - 1) instead of the k1_nx_* / k1_pf_* select chains: loop-top SALU 156 -> 57 per 2 iterations (the G != 1 loop stays as a second copy). | 1008/63K, 887/270K |
| r1d | r1c | `01d_dkdv_nlnd_voffset.patch`; `k_dkdv96ds.hip`; `k_dqds_c.hip` | part |  | dK/dV: the 8 NL/ND loads put their constant on the VGPR offset (3 loop-invariant VGPRs) and share one SGPR soffset, so no SALU rewrites an address register that an in-flight buffer load still reads (the s_wait_xcnt 0 WAR stalls go). | 1004/63K, 887/270K |
| r1e | r1d | `01e_dkdv_dsexport_staging.patch`; `k_dkdv96ds.hip`; `k_dqds_c.hip` | yes |  | dK/dV: the dS^T export first regroups all 3 blocks into distinct staging VGPRs (24 DPP ops), then issues the 6 stores, so no store-data register is rewritten while an earlier store still reads it (the ~82-cyc store-data s_wait_xcnt goes; with r1d the full loop has no s_wait_xcnt left). | 1004/63K, 887/270K |
| r1f | r1e | `k_dkdv96ds.hip`; `k_dqds_c.hip` | yes |  | dK/dV compiled with -amdgpu-expert-scheduling-mode: the compiler places explicit s_wait_alu va_vdst / vm_vsrc counts instead of relying on the hardware interlock, so DS / VMEM issue no longer waits for every earlier WMMA. | 1004/63K, 887/270K |
| r1g | r1f | `01g_dkdv_masked_ring.patch`; `k_dkdv96ds.hip`; `k_dqds_c.hip` | yes |  | dK/dV: the masked (diagonal) iterations ride the full loop's 3-stage Q/dO TDM ring with prefetch distance 2 (tile j in stage (j - n_mask) mod 3, k3m_idx.h) instead of an own-tile TDM + tensorcnt 0 + 2 barriers each: ~3 exposed HBM round trips (~1.2k cyc) per 96-row workgroup and 4 barriers per wave go. | 1015/63K, 887/270K |
| r1h | r1g | `01h_dq_split_barrier.patch`; `k_dkdv96ds.hip`; `k_dqds_c.hip` | part |  | dQ: split barrier P_{u+1} in every steady unit but the last: arrive (tensorcnt 1, dscnt 0, release, s_barrier_signal) after WMMA 48 of run(X) (32 on the head tile), wait after run(X), so the remaining WMMAs absorb the ~90-cyc lag of the dS-issuing waves 0 / 3. | 1015/63K, 887/270K |
| r1i | r1h | `k_dkdv96ds.hip`; `k_dqds_c.hip` | part |  | dQ compiled with -amdgpu-expert-scheduling-mode: a DS read no longer waits for earlier WMMAs (the prerequisite of r1j's interleave). | 1015/63K, 887/270K |
| r1j | r1i | `01j_dq_burst_interleave.patch`; `k_dkdv96ds.hip`; `k_dqds_c.hip` | part |  | dQ: in main-tile steady units the 36-tr16 operand bursts are interleaved into the WMMA runs with sched_group_barrier ({8 WMMA} then {1 tr16, 1 WMMA} x 36 before the arrive, {4 WMMA} then the same under run(Y)), hiding the 130-180-cyc bursts under matrix work. | 1015/63K, 887/270K |
| r1k | r1j | `01k_dq_split_at_66.patch`; `k_dkdv96ds.hip`; `k_dqds_c.hip` | part |  | dQ: the split-barrier arrive moves from WMMA 48 to 66 (44 on the head tile): region A now carries the whole interleaved Y burst, and the later arrive lets its 36 tr16 retire before the s_wait_dscnt 0 that precedes the signal. | 1015/63K, 887/270K |
| r1l | r1k | `01l_dq_masked_branchfree.patch`; `k_dkdv96ds.hip`; `k_dqds_c.hip` | part |  | dQ: the masked units' zero-select predicate in branch-free form (x4_ds_zero: \| instead of \|\|), so the selects compile to in-place v_cndmask instead of 17 scalar branches + v_dual_mov operand copies that split the scheduling region (VGPR 887 -> 871). | 1015/63K, 871/270K |
| r1m | r1l | `01m_dq_masked_split.patch`; `k_dkdv96ds.hip`; `k_dqds_c.hip` | yes |  | dQ: the masked units (all but the last) take the split barrier too, with the select pattern carried in the first part of run(X). | 1015/63K, 871/270K |
| r1 | r1m | `k_dkdv96ds.hip`; `k_dqds_c.hip` | yes | 2.299 | round 1: dK/dV scheduling (dS export placement, split barrier, MHA loop top, NL/ND voffsets), masked tiles on the TDM ring, dQ split barrier / tr16 interleave / branch-free masks; expert scheduling for both | 1015/63K, 871/270K |
| r2a | r1 | `02a_dq_masked_pair_skip.patch`; `k_dkdv96ds.hip`; `k_dqds_c.hip` | part |  | dQ: in masked units a wave-uniform branch per q pair skips a fully masked pair's WMMAs instead of selecting its B operand to zero (bitwise identical: those WMMAs only add +-0), so the selects and the masked pairs' WMMA issue (and energy) go. | 1015/63K, 871/270K |
| r2b | r2a | `02b_dq_subtile_epilogue.patch`; `k_dkdv96ds.hip`; `k_dqds_c.hip` | part |  | dQ: the epilogue runs per 16-row sub-tile through a private 16 x 400 B per-wave LDS image after the rings (LDS 276480 -> 302080, still 1 WG/CU) instead of the whole-wave image in the dead ring: shorter store / load chains, epilogue 3.1 -> 2.4 us per WG. | 1015/63K, 871/295K |
| r2c | r2b | `02c_dq_real_desc_nocheck.patch`; `k_dkdv96ds.hip`; `k_dqds_c.hip` | part |  | dQ: the steady loop's (REAL) TDM descriptors drop the runtime nv >= 1 check (a real unit's rows always hold a stored pair), so the descriptor SALU inside the WMMA run needs no select / branch that would split burst Y and run X into different basic blocks. | 1015/63K, 871/295K |
| r2d | r2c | `02d_dq_steady_unroll3.patch`; `k_dkdv96ds.hip`; `k_dqds_c.hip` | part |  | dQ: main-tile steady loop unrolled by the ring depth (3), so every stage index is a compile-time constant: the burst addresses become precomputed base VGPRs + immediates and the loop carries no VALU, which lets expert mode count va_vdst(k) waits instead of va_vdst(0) drains. | 1015/63K, 878/295K |
| r2e | r2d | `02e_dq_keep_ds.patch`; `k_dkdv96ds.hip`; `k_dqds_c.hip` | part |  | dQ: keep_ds() (an empty asm volatile taking the 6 dS operands) keeps each region's dS operands live to its end, so the next burst cannot be allocated to a register the newest WMMA just read (that WAR forced va_vdst(0)). | 1015/63K, 925/295K |
| r2f | r2e | `02f_dq_snake_pairs.patch`; `k_dkdv96ds.hip`; `k_dqds_c.hip` | yes |  | dQ: q pairs dealt to the waves in snake order {w, 7 - w, 8 + w} (head {w, 7 - w}) instead of contiguous blocks, so the fully masked pairs of the diagonal units spread over all 4 waves (skippable share per wave 83% / 8% -> 42-50%) and the barrier-bound masked critical path drops by a third. | 1015/63K, 934/295K |
| r2 | r2f | `02z_5c20de7e_leftover.patch`; `k_dkdv96ds.hip`; `k_dqds_c.hip` | yes | 2.274 | dQ round 2: snake q pairs, masked-pair skip, steady loop unrolled by the ring depth, per-sub-tile epilogue image | 1015/63K, 934/295K |
| r3 | r2 | `k_dkdv96ds.hip`; `k_dqds_c.hip` | yes | 2.255 | dK/dV compiler knobs: bottom-up pre-RA scheduling, delay-alu off, no machine sink | 1016/63K, 934/295K |
| r4 | r3 | `04_2f3d60c6.patch`; `k_dkdv96ds.hip`; `k_dqds_c.hip` | yes | 2.231 | dK/dV: Q^T loads interleaved into the dV run + asm register retarget (va_vdst(0) drains 15 -> 9 per 2 iterations) | 1015/63K, 934/295K |
| r5 | r4 | `05_5b19697f.patch`; `k_dkdv96ds.hip`; `k_dqds_c.hip`; `k_delta.hip` | yes | 2.218 | HIP k_delta (several b128 loads in flight per lane) | 1015/63K, 934/295K, 48/0K |
| r6 | r5 | `06_7ccfba3b.patch`; `k_dkdv96ds.hip`; `k_dqds_c.hip`; `k_delta.hip` | yes |  | hardening: s_wait_xcnt before the tail block reuses its K/V descriptor SGPRs (perf-neutral) | 1015/63K, 934/295K, 48/0K |
| r7 | r6 | `07_71bd0329.patch`; `k_dkdv96ds.hip`; `k_dqds_c.hip`; `k_delta.hip` | yes | 2.220 | redundant causal test dropped from the masked dK/dV loop | 1015/63K, 934/295K, 48/0K |
