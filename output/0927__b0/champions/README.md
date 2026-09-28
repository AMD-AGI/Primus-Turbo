# Champion snapshots at 2026-09-28 ~11:00 UTC (B0)

| dir | job | round | what |
|---|---|---|---|
| `fwd_r16_r13ns/` | gfx1250-flydsl-attn-fwd-b0-20260927 | 16 | r13 (small-grid m16x8/m32x2 kernels) with SPEC_STALE_MAX=False (h44); flydsl 0.3.4.1 |
| `bwd_r29_r19h_u2n/` | gfx1250-flydsl-attn-bwd-20260917-115934 | 29 | r19 + h33 clamps (h68) + k_dq kv-loop unroll x2 (u2n, h70); `_env.py` pins flydsl 0.3.2 (ISA identical under 0.3.4.1, e2e/recon/r29isa) |

Byte copies of each job's `job_context/op/current` (no `__pycache__`); md5 of every .py in `MD5SUMS`.
