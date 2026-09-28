# op/beat -- aiter prebuilt gfx1250 ASM forward

`op.target.beat`: aiter.ops.mha.fmha_fwd_with_sink_asm, sink=None, "beaten by 0%".

- Library: aiter, imported from the SOURCE CHECKOUT `/home/lihuzhan/code/aiter-src`
  (not pip-installed in the fa-repro image), HEAD `6963ae9d77acb207d8f03fc92dddd8a57faf8fa6`
  (2026-09-17), clean worktree. Never edited from this job; nothing is copied from it.
- Entry: `aiter/ops/mha.py:589` `fmha_fwd_with_sink_asm(q, k, v, softmax_scale, is_causal,
  return_lse, sink=None, out=None) -> (out, lse[B,Hq,Sq] fp32)`, `@compile_ops("module_fmha_fwd_with_sink_asm", ffi_type="ctypes")`.
- Host module: `aiter/jit/module_fmha_fwd_with_sink_asm.so` (built 2026-09-15 02:47 UTC,
  sha256 2002dcacd86d0a0f...), already built -- no JIT build happened in setup.
- Code objects (loaded, per aiter's own `LoadKernel` log line in the smoke run):
  `hsa/gfx1250/fmha_fwd_bf16/fmha_bf16_pertokenBf16_hd128_128x256_mask.co` (causal,
  sha256 d0e13b62d8bb665b...) and `..._hd128_128x256.co` (non-causal), both 2026-09-13.
- `_env.py` puts aiter-src and flydsl0341 on sys.path and ASSIGNS the two BLAS variables.
  `impl.py` asserts `aiter.ops.mha` resolved under aiter-src.
- Verified 2026-09-25 against op/eager: toy causal o 54.10 dB / lse 145.57 dB, toy
  non-causal 52.64 / 146.07, fast causal 53.56 / 147.17 (all finite).
- Only relied on at seqlen_q == seqlen_kv (the three op.shape entries). Its causal alignment
  for seqlen_q != seqlen_kv was not tested.
