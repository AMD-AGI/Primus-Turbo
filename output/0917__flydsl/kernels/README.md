# gfx1250 FlyDSL kernels we wrote

Bring-up sources, not yet integrated. **Placement is an open question, not an oversight:**
these are written against flydsl **0.3.2**, and `primus_turbo/flydsl/` is a 0.2.4 tree that
0.3.2 cannot import (`flydsl.expr.buffer_ops` was removed). Dropping them in beside it would
produce a tree that no single flydsl version can load. See `../STAGE1-FWD.md` section 4.

| file | status |
|---|---|
| `odo_gfx1250.py` | **works.** `delta = rowsum(dO*O)`, BSHD in, `[B,H,S]` fp32 out |

## odo_gfx1250.py

Ported from aiter's gfx942 `k_delta` (`fmha_bwd_gfx942/fmha_bwd_core.py`), which is almost
arch-neutral -- no MFMA, no `ds_read_tr16`, no `permlane32_swap`. Two changes:

1. **Wave size.** The row reduction is four `shuffle_xor` steps. gfx942 passes width 64;
   gfx1250 dispatches wave32 and must pass 32. The butterfly spans only 16 lanes either way,
   so a wrong width does not raise -- it reads lanes that are not there. Full `isfinite`
   coverage at both shapes is the evidence it is right.
2. **Layout.** gfx942's is varlen THD with `DEL [H, T]`; this is BSHD `[B, S, H, D]` with
   delta `[B, H, S]`, matching the LSE the gfx1250 forward emits.

Measured on `heliosr-1b114-c07-1` @1100 MHz, `AMD_SERIALIZE_KERNEL=3`, toy shape launched
first in the same process:

| shape | build + first launch | isfinite | SQNR |
|---|--:|---|--:|
| b=1 s=256 h=2 | 0.28 s | 512/512 | **159.24 dB** |
| b=4 s=8192 h=32 | cached | 1048576/1048576 | **156.49 dB** |

Passed on the first run, which matches how the 0915 ASM bring-up went (odo first, 147 dB,
first try). That is the reason to start here: the reference is three lines of torch and it
needs no main kernel, so it validates the whole toolchain against something that cannot be
subtly wrong.

**Cold JIT build for a kernel this size is 0.28 s** -- worth knowing before sizing any
unattended loop's per-turn timeout. aiter's attention forward, which is far larger, took
about 2.4 s.

Known gap: `n_rows % 32 == 0` is asserted rather than handled. Both shapes here divide; a
tail pass is needed before this is general.

```bash
docker cp output/0917__flydsl/kernels/odo_gfx1250.py fa-repro:/tmp/
docker exec -e AMD_SERIALIZE_KERNEL=3 -e TORCH_BLAS_PREFER_HIPBLASLT=0 fa-repro \
  bash -lc 'cd /tmp && python3 odo_gfx1250.py'
```
