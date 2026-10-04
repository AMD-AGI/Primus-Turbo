# gfx1250_mla_bwd -- provenance

Starting point: the gfx1250 FlyDSL backward "s6" (k_delta + k_dkdv with a 3-stage TDM Q/dO
ring + k_dqg with a 3-stage TDM K/V ring, dQ chain on a side stream; head_dim 128, flydsl
0.3.4.1), as shipped in `origin/dev/lhz/flydsl-attn-b0:output/1002__e2e/arms_src/bwd_s6_0341/`.

First commit = that tree verbatim (md5 below), except `_env.py`, whose host-specific paths
were replaced by the `FLYDSL_PATH` / `AITER_SRC` environment variables.

    5e61678d53260a2fc17c68286298f4b9  kernels.py
    46e0e8127bd57c888944bbc6ee96249b  impl.py
    0024d09db2284ac48c29ab70e11a821d  __init__.py
    7df61bba26309ffc14b84a432fb451a7  _env.py (before the path edit)

Later commits port it to DeepSeek-V3 MLA attention (D_QK = 192, D_V = 128).
