# gfx1250_mla_bwd -- provenance

Starting point: the gfx1250 FlyDSL backward "s6" (k_delta + k_dkdv with a 3-stage TDM Q/dO
ring + k_dqg with a 3-stage TDM K/V ring, dQ chain on a side stream; head_dim 128, flydsl
0.3.4.1), as shipped in `origin/dev/lhz/flydsl-attn-b0:output/1002__e2e/arms_src/bwd_s6_0341/`
(design and measurements: `origin/dev/lhz/flydsl-attn-b0:output/0930__bwd/REPORT.md`).

First commit = that tree verbatim (md5 below), except `_env.py`, whose host-specific paths
were replaced by the `FLYDSL_PATH` / `AITER_SRC` environment variables.

    5e61678d53260a2fc17c68286298f4b9  kernels.py
    46e0e8127bd57c888944bbc6ee96249b  impl.py
    0024d09db2284ac48c29ab70e11a821d  __init__.py
    7df61bba26309ffc14b84a432fb451a7  _env.py (before the path edit)

## Port to DeepSeek-V3 MLA (D_QK = 192, D_V = 128)

1. Cleanup: only the code s6 runs at proxy/prod is kept (k_delta, k_dkdv, k_dqg; shipped
   switch values frozen; k_dq/k_dq_sp, the non-TDM k_dqg, split-K k_dkdv_sp/k_redsp/k_redsp_q
   and every experiment switch deleted; aiter helpers inlined). Compile-only at
   b4 s8192 hq32 hkv8 d128: instruction-identical to s6.
2. Head dims: q/k/dq/dk use D_QK, v/o/do/dv use D_V. With D_QK = 128 the same source still
   compiles to s6's exact instruction stream (the regression check for the port).
3. TDM: a pad interval must be a power-of-two number of dwords, so a 192-wide row is two TDM
   ops (128 + 64 columns, padded 36 / 68 dwords to one 400 B row stride); 3 ops per ring
   stage, and every `tensor_wait` immediate is derived from the op count (3, was 2). Rings:
   3 x 21504 B = 64512 B, inside LDS segment 0.
4. Registers: k_dqg at 64 queries per wave needs 1024 VGPR + 148 spilled at D_QK 192, so
   DQ_BQW = 32 there (709 VGPR). k_dkdv: 887 VGPR, 8 SGPRs spilled to VGPR lanes (s6: 0; the
   third TDM descriptor per stage and the separate q/do address families raise SGPR use
   69 -> 107).
5. `bounds_proof.py` (stdlib) replays the index math and both rings for every shape.
6. First launch (toy b1 s256 h2, gfx1250, serialized): dq/dk/dv fully written under NaN
   poisoning, SQNR vs an fp32 CPU reference 52.0 / 52.1 / 52.8 dB at 1/sqrt(192) and
   51.7 / 51.8 / 52.6 dB at the Megatron scale 0.13523, bitwise identical across calls.

Known gaps: the split-K paths are gone, so under-filled grids (the `fast` shape, 256
workgroups per kernel on 1024 SIMDs) run without them; inputs must be contiguous BSHD
(Megatron passes BSHD views of SBHD storage); the k_dqg epilogue still writes dQ with 2-byte
stores.

## Round 1 arm bwd_r1_b: k_dqg at 48 queries per wave (NQW 3) for D_QK 192

Base: arms/bwd_c1 (champion, hash 56cb4380). Lever: k_dqg's step cost (~2130 cycles per 32-row
kv step at prod for 64 WMMA) follows its LDS/TDM traffic (64 DS loads + 20 KB TDM per step), not
its WMMA issue; 48 queries per wave keep that traffic and raise the work per step to 96 WMMA
(S 36, dP 24, dQ 36), and cut the k_dqg tile-steps per (b, h) at prod from 8256 to 5547 (0.67x).

- kernels.py: `_dqg_tdm_impl` takes the query sub-tile count `nqw` (compile-time) and the
  runtime tile offset/count `q_off`, `ntile`: q0 = q_off + (ntile-1-grid.y)*16*nqw. `k_dqg`
  (NQW 2, unchanged body) and the new `k_dqg48` (NQW48 3) are both built from it. The S/dP,
  softmax/dS and dQ chunk schedule (`_sdp`, `_smx`, `_body_ck`, masked `_body`) was already
  written for any NQW (s6 ran NQW 4 at D 128): at NQW 3 the kt1 chunks dt 0..2 carry the kt0
  softmax of qh 0..2 and the dQ chunks qh 0..1 the kt1 softmax of qh 1..2. The kv-range math
  (nkvt_eff = ceil((q0+BQW+cshift)/32), nfull = floor((q0+cshift+1)/32)) is per tile from the
  48-row diagonal: 2 masked steps per tile at cshift 0 (32-row: 1).
- `dq_split(sq)` -> (q_split, n32, n48), q_split in {0, 32, 64} with (sq - q_split) % 48 == 0
  (prod/proxy/fast/toy: 64). impl._plan launches k_dqg48 over [q_split, sq) as plan name "dqg"
  (longest-first) and then k_dqg over the head [0, q_split) as "dqg_head" (a second launch, 512
  workgroups of <= 2 kv steps at prod). Each launcher is a single-kernel module, so the
  compile-only dumps/ISA table and the runtime code_sha check see both kernels. `_launch` keys
  its memo on the launcher too (one plan name maps to different launchers by shape).
- bounds_proof.py Q6: the split, both launches' tile-to-row maps (partition of [0, Sq), descending
  walk), per-row kv coverage / full-iteration / masked-iteration checks and the masked step count,
  on 11 shapes incl. q_split 0 (s192, rect_short_kv), 32 (s128, gqa) and 64 with n48 = 0 (edge64).
- k_delta and k_dkdv are untouched (code_sha equal to bwd_c1).

## Round 2 arm bwd_r2_a: k_dkdv64, two waves sharing one Q/dO TDM ring (64 kv rows per fetch)

Base: arms/bwd_r1_b (champion, hash ed198cd0). Lever: k_dkdv (2.78 ms, ~58% of the entry) pulls
Q/dO at the same ~15.6 TB/s ceiling as the other d192 kernels; an iteration costs 1.345 us at prod
vs <= 0.81 us uncontended (fast, one workgroup per CU), so >= 40% of it is fetch contention. More
kv rows per wave do not fit (NKV 3 ~ 1090-1160 VGPR). A second wave on another SIMD that reads the
same staged tile halves the Q/dO TDM bytes and LDS writes per FLOP and keeps each wave's body.

- kernels.py: `_dkdv_impl(..., nw)` (compile-time). nw = 1 is the legacy k_dkdv, instruction-
  identical to r1_b (code_sha 6339e909f586406a, checked by a one-off compile at prod:
  runs/arms/bwd_r2_a/legacy/). nw = DKDV_NW = 2 is the new `k_dkdv64` (block 64,
  `known_block_size` 64): grid (Hkv, Skv/64, B), kv block g ascending = longest first; wave w
  (`rocdl.wave_id`, SGPR) owns kv rows 64g + 32w + [0, 32), lane = tid % 32. Both waves walk the
  WORKGROUP's (q pair, q head) sequence: qp_start / nmaskp come from kv0g = 64g (block id), so the
  masked loop covers pairs 2g, 2g+1 at cshift 0 (wave 1's pair 2g is fully masked: P = 0, dS = 0)
  and the full loop needs no mask for either wave; inside the masked pairs each wave masks with
  its own kv0.
- One Q/dO ring (3 x 21504 B, segment 0) shared by both waves. TDM is cooperative
  (`make_tdm_atom(num_warps=2)`, the fwd's proven multi-wave TDM form): every wave issues the same
  3 ops per stage and moves rows [16w, 16w+16) of each 32-row tile, with its own tensorcnt (so the
  wait immediates stay TW_QDO = 3 / 0). Deviation from the brief ("wave 0 issues the 3 TDM ops"):
  with the split there is no wave-dependent branch anywhere, so both waves execute one instruction
  stream and take the same barriers by construction; bytes per stage are unchanged.
- Per-wave P/dS tiles at LDS_SEG + w*5120 (allocation 65536 + 2*5120 = 75776 B, 2 workgroups per
  CU), per-wave epilogue images at w*30720 inside the dead ring (61440 <= 64512).
- Barriers (`_lds_barrier` = `gpu.barrier memfence [workgroup]`: the fences carry
  `!mmra amdgpu-synchronize-as local`, so the release side waits dscnt only; a plain
  gpu.barrier() also emitted `s_wait_loadcnt 0` on the in-flight LSE/delta prefetch):
  masked iteration BARRIER (WAR) -> TDM stage 0 -> tensor_wait(0) -> BARRIER (RAW) -> reads;
  prologue BARRIER (WAR) -> TDM stages 0, 1 -> tensor_wait(3) -> BARRIER (RAW) -> readback;
  full iteration: TDM (ii+2)%3 at the top, tr16 of ii%3, dK/dV WMMAs of kv sub-tile 0,
  tensor_wait(3), BARRIER, readback of (ii+1)%3, WMMAs of kv sub-tile 1 (hide the readback);
  exit tensor_wait(0) -> BARRIER -> epilogue. Per wave 2*G*nmaskp + 2 + n_full + 1 barriers.
- impl.py: plan "dkdv" launches `launch_dkdv64` (nblk = Skv/64); `_check` requires Skv % 64 == 0.
  k_delta / k_dqg48 / k_dqg (head) untouched.
- bounds_proof.py: K1 per-wave TDM slices (16 rows, per-wave extent clamp), K4 workgroup-level
  causal split checked against EACH wave's 32 kv rows, K5 coverage over both waves, K6 the barrier
  protocol: a source scan of `_dkdv_impl` (the wave index only in the four address lines, every
  `if` a const_expr, trip counts from kv0g, six barrier sites) and a two-wave phase model of the
  ring (per-wave tensorcnt queues; a read needs every wave's slice retired before an earlier
  barrier and nothing re-targeting the stage since; no TDM into a stage read in the same phase;
  no TDM pending and no ring traffic in the epilogue phase); per-wave barrier counts checked.
  Mutation runs (runs/arms/bwd_r2_a/mut/summary.txt): dropping any barrier, reading back before
  the loop barrier or weakening the wait trips K2/K6.

## Round 2 arm bwd_r2_b: k_dqg96, two waves sharing one K/V TDM ring (96 queries per K/V fetch)

Base: arms/bwd_r1_b (champion, hash ed198cd0). Lever: k_dqg48 still streams ~14.8 TB/s of K/V TDM
at ~1.41 us per kv step (vs ~0.75-0.85 us uncontended), and one wave cannot hold more queries
(NQW 4 = 1024 VGPR + 148 spilled at D_QK 192). A second wave consuming the same staged K/V tile
halves the TDM bytes per query again while each wave keeps the k_dqg48 body.

- kernels.py: `_dqg_tdm_impl(..., nqw, nwave=1)`. nwave == 1 is byte-for-byte the old code path
  (k_dqg head code_sha 24acf8573e630a36 unchanged). The new `k_dqg96` = nqw 3, nwave DQ_NWAVE 2,
  block 64: lane = tid % 32, wave wv = readfirstlane(tid / 32); workgroup tile q0g = q_off +
  bid*96 (bid = ntile-1-grid.y, longest-first), wave wv owns [q0g + 48wv, q0g + 48wv + 48).
  The kv range (nkvt_eff, nfull) is computed from the WORKGROUP tile (to the last wave's
  diagonal), so both waves run the same trip counts; wave 0's extra steps run through the
  existing do_mask path (fully masked entries give p = exp2(NEG) = 0, ds = 0), not skipped.
  At cshift 0 every workgroup has 3 masked steps: wave 0 = 2 partial + 1 fully masked, wave 1
  = 1 unmasked (run masked) + 2 partial.
- TDM: one 3-stage K/V ring (64512 B) per workgroup. Deviation from the round sketch ("TDM issued
  by wave 0"): each wave runs the same `_tdm_kv` call with the atom's num_warps = 2, so wave w
  issues rows [16w, 16w+16) of every op (FlyDSL GFX1250 TDM lowering: per-wave offset from
  rocdl.wave_id = ttmp8[29:25], the pattern the fwd runs with num_warps 8). No divergent branch,
  equal SALU/TDM issue per wave, and each wave's tensor_wait(DQT_TW = 3) still retires exactly
  its own half of the older stage.
- Protocol per kv step i (cur = i%3, ncur = (i+1)%3, nxo = (i+2)%3), both waves:
  TDM(min(i+2, N-1)) -> nxo | S/dP WMMAs on the carried readback | tr16 of cur | softmax/dS
  (kt1, qh0) (ck body) or all softmax (masked body), covering the tr16 latency |
  tensor_wait(3) | `_wg_sync` = s_wait_dscnt 0 + fence(release, wg) + s_barrier_signal -1 +
  s_barrier_wait -1 + fence(acquire, wg) | readback of ncur | dQ WMMAs. Prologue: TDM tiles 0
  and min(1, N-1), tensor_wait(3), `_wg_sync`, readback of stage 0. Epilogue: tensor_wait(0),
  per-wave dQ stores; no final barrier. RAW: the readback of ncur follows the barrier both
  waves enter after retiring their half of ncur. WAR: the TDM into nxo at the top of i+1
  follows barrier i, which both waves enter after draining their tr16 of that stage (dscnt 0;
  its readback was consumed by the S/dP WMMAs of i). The readback moved from before to after the
  softmax chunk (single-wave order: tr16 | tensor_wait | readback | softmax).
- DQ_BARRIER_FENCE = True: compile-time ISA of the fence and no-fence builds has the same waits
  around every barrier (s_wait_tensorcnt 0x3, s_wait_dscnt 0x0; no tensorcnt 0, prologue
  loadcnt 0 present in both), so the fences cost nothing and stop IR passes moving LDS ops
  across the raw split barrier.
- dq_split(sq, bqw=96) -> (q_split, n32, n96): q_split 64 at toy/fast/proxy/prod (4032 = 42 x 96).
  impl._plan launches k_dqg96 as "dqg" (grid (Hq, n96, B), block 64) then k_dqg over the head
  [0, q_split) as "dqg_head". k_dqg48 stays in kernels.py, unlaunched.
- Outputs should be bitwise identical to bwd_r1_b (and c1): the same WMMAs in the same kv order
  per accumulator, the mask select is the identity on unmasked entries, fully masked steps add 0.
- bounds_proof.py: Q6 now replays the 96-row workgroup map (per-wave partition of [0, Sq)),
  per-row kv coverage, full/masked classification and the masked-step classes per wave; W1 the
  per-wave TDM blocks (global bounds, per-wave extent, union == single-wave LDS image); W2 a
  happens-before replay of the 2-wave ring with barrier epochs for N = 1..160 and every shape's
  trip counts, plus four mutated protocols that must be caught.
- ISA: tools/flydsl/isa_ring_barrier_check.py (signal/wait alternation, empty windows, dscnt 0
  and tensorcnt <= 3 at every signal, ring-op order readback -> TDM -> tr16 per barrier region).

## Round 3 arm bwd_r3_a: k_dkdv64 (bwd_r2_a) + k_dqg96 (bwd_r2_b)

Base: arms/bwd_r2_a (champion, hash 146b19c4). The two round-2 wins touch disjoint kernels: r2_a
changed only k_dkdv (-> k_dkdv64), r2_b only the main dQ kernel (k_dqg48 -> k_dqg96). This arm is
their union, nothing else.

- kernels.py: r2_a's tree plus r2_b's hunks (`DQ_NWAVE`/`DQ_BQW96`, `dq_split(sq, bqw=96)`,
  `_wg_sync` with DQ_BARRIER_FENCE, `_dqg_tdm_impl(..., nwave)`, `k_dqg96`/`launch_dqg96`). The
  only overlap is `_tdm_rows`, which both rounds gave a cooperative-TDM width argument; r2_a's
  name (`num_warps`) is kept and r2_b's call sites pass `nwave` positionally. The source from the
  `dkdv` banner to the `dqg` banner equals r2_a's byte for byte, the source from the `dqg` banner
  to the end equals r2_b's byte for byte.
- impl.py: r2_a's plus r2_b's dQ-chain block: plan "dqg" = `launch_dqg96` over [q_split, sq)
  (grid (Hq, n96, B), block 64, longest-first), "dqg_head" = the 32-query `launch_dqg` over
  [0, q_split); `_check` keeps r2_a's Skv % 64. All launches serial on the caller's stream.
- Expected binaries (compile-only, every variant): k_delta 69e06c22a4d3bb22, k_dkdv64
  edf417abcb4bd065 (= r2_a), k_dqg96 6f35100e3882e3f7 (= r2_b), k_dqg head 24acf8573e630a36.
  Outputs therefore bitwise identical to r2_a (dk/dv: same binary; dq: r2_b's dq is bitwise
  r1_b's, which is bitwise r2_a's).
- bounds_proof.py: r2_a's K1..K6 (k_dkdv64, incl. the two-wave barrier protocol scan) plus r2_b's
  Q6 / W1 / W2 (k_dqg96 workgroup map, per-wave TDM blocks, 2-wave ring happens-before replay with
  mutations N1..N4); per-check counts equal the parents' (K* = r2_a, Q*/W*/N* = r2_b).
- Risk named in the round plan: both 2-wave kernels now run back to back in one entry, so the
  L2/MALL state each kernel inherits changes (k_dqg96 starts after k_dkdv64 instead of k_dkdv);
  measured by the per-kernel arms of the same-process A/B.

## Round 4 arm bwd_r4_a: head-group launch of k_dkdv64 / k_dqg96 / k_dqg (fold traversal)

Base: arms/bwd_r3_a (champion, hash 698f5628). Finding (integration refresh, same process): r3_a
at the fold launch the e2e arm and the Turbo adapter use for Megatron's b2 h128 SBHD views,
[1, 4096, 256, d], took 4.387 ms against 3.928 ms for the b2 h128 launch over contiguous BSHD
(+11.7%; b4 h64 3.724 ms). The kernels' grids are (heads, tiles, B) with the head fastest; a
256-head row leaves half as many tiles of one head resident at a time (per XCD) as a 128-head row,
so a Q/dO (K/V) tile streamed by one workgroup is less often still in L2 for the next one.

- kernels.py: `HEAD_GROUP` (128) and `head_group(nh, target)` (shared constants block). k_dkdv64,
  k_dqg96 and k_dqg take one more i32 kernel argument `HG`; `_dkdv_impl` / `_dqg_tdm_impl` take
  it as `HG=None` (None = r3_a's code, kept for the unlaunched k_dkdv / k_dqg48). With HG the
  launch grid is (HG, tiles, B*nh/HG) -- the same workgroup count -- and the kernel decodes
  hb = block_idx.z*HG + x (k_dqg*: x after the XCD-major remap, now taken within the group of
  HG), bat = hb // nh, head = hb - bat*nh: one scalar division per workgroup. Every workgroup
  does exactly one (bat, head, tile) of r3_a, so outputs are bitwise identical for any HG.
- impl.py: `_plan` picks hg = head_group(Hkv) for k_dkdv64 and head_group(Hq) for the dQ chain:
  nh when nh <= HEAD_GROUP (toy/fast/proxy/prod b2h128: r3_a's grids), else the largest multiple
  of 8 dividing nh that is <= HEAD_GROUP (the fold [1, s, 256, d]: 128, i.e. grid (128, tiles, 2),
  the b2h128 grid). Launch-only knobs, one binary for every value: env FLY_BWD_HEAD_GROUP (read at
  import; 0 = r3_a's grids everywhere) and `flydsl_attn_bwd(..., head_group=N)` per call.
- bounds_proof.py: R1 (decode is a bijection onto [0, B) x [0, nh) for every launched kernel, every
  target in R_TARGETS and every shape; hg == nh reproduces r3_a's map; hg % 8 == 0 keeps a head
  on one XCD; source tie of the decode, grid and plan lines), R2 (prodfold at 128 walks (folded
  head, tile) exactly like r3_a's b2h128 launch, for all three kernels); new shapes prodfold,
  gqafold, gqa_hg. K/Q/W/N counts on r3_a's 11 shapes equal r3_a's except K6 +1 (the new
  const_expr branch in _dkdv_impl is one more scanned `if`).
- Compile-only (every variant incl. prodfold/toyfold): VGPR/SGPR 859/102, 868/68, 709/68 as r3_a,
  0 spill / 0 scratch / 0 lane spill; +36 / +24 / +54 instructions (prologue decode); the hot
  loops of k_dkdv64 are opcode-identical to r3_a's, k_dqg96 loops have 4 / 6 fewer s_set_vgpr_msb,
  k_dqg loops +1 s_set_vgpr_msb / +1 s_delay_alu; WMMA counts equal.
