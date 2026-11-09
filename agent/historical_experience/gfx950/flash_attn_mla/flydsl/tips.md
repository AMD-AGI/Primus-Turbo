# gfx950 / flash_attn_mla / flydsl — tips

- The dual-wave dense-causal pipeline (flash_attn_fwd.py) masks the causal diagonal only in its
  3-tile epilogue drain. That is correct only while BLOCK_M / BLOCK_N <= 3 (the D=128 build uses
  BLOCK_M=128 → 2 tiles). At BLOCK_M=256 the first diagonal tile lands in the main loop and must be
  masked there (`causal_mask_prologue_if_needed`, wave-uniform guard); symptom when missing: rows
  0-63 of every q-block except block 0 wrong, ~22 dB output SNR while LSE still looks fine.
- FlyDSL's on-disk cache (/root/.flydsl in the container) keys on kernel source + traits cache_tag,
  not on env vars read inside `_setup`; an env-var debug toggle silently reuses the old binary
  (tell-tale: compile+run ~0.2 s). Clear the cache before trusting such an experiment.
- K LDS for qk 192 = 3 strips × 8 lines × 520 elems; per buffer (K+V) 42.4 KB, double-buffered
  84.7 KB, so BLOCK_M=128 (2 CTAs/CU) does not fit 160 KB; BLOCK_M=256 / 8 waves / 1 CTA per CU does.
