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
