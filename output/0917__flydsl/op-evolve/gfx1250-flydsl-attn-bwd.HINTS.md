# Hints for gfx1250-flydsl-attn-bwd

Read before round 1. Everything here was measured on this card on 2026-09-17 and is in
`Primus-Turbo/output/0917__flydsl/` with the probe that produced it.

## The two findings the PREVIOUS op-evolve job on this box discovered and then LOST

That job (`gfx1250-attn-llama31-8b-bwd-20260914-015415`) died on a bookkeeping rule because
its round 2 dropped these instead of writing them into `facts.md` / `dead_ends.md`. They are
seeded into `facts.md` already so nobody rediscovers them:

1. **Gate candidates on `vgpr_spill_count`, read from the compiled `.amdgcn`, BEFORE spending
   a timing run.** Four measured points: 326/325/307/266 spills against 9.983/9.302/9.347/
   8.691 ms -- monotone. Loop-body instruction count predicts nothing.
2. **Splitting HEAD_DIM in the dk/dv pass into two halves of 64** drops dk/dv/k/v from 768 to
   384 loop-invariant VGPRs.

**Append to `facts.md` and `dead_ends.md` EVERY round, including rounds where the gate
fails.** A round that produces a finding and does not write it down is recorded as a failed
round and the finding is gone.

## What the baseline is, and what it is not

`op/baseline/` is a CORRECT but deliberately unoptimised kernel: every output was validated
against torch at 141-159 dB before it was handed over. It is NOT a tuned kernel and the gap
to `op/beat/` is about **93x**, not a few percent. That is the opposite of the previous job
here, whose config axis had already been mined dry by its round 1.

Known-missing work, in the order the operator's own estimate ranks it. The first two have
known magnitudes; the rest do not, so **time each separately**:

| missing | expected |
|---|---|
| one wave (32 threads) per workgroup; the card wants 8 waves / 256 threads | up to ~8x |
| the query contraction is 16 wide, zero-padded to the WMMA's 32 | 2x |
| causally-masked tiles are computed, not skipped | already excluded from the "useful" figure |
| no software pipelining, no TDM async copy | unknown; the forward uses TDM heavily |
| Q/dO re-staged into LDS every iteration, no ring buffer | unknown |
| no bank-conflict swizzle on the LDS transpose | unknown |

## Arch facts that are NOT guesses

- `v_wmma_f32_16x16x32_bf16`, wave32. Operand fragment: lane `l`, element `e` ->
  16-axis `l%16`, 32-axis `(l//16)*8 + (e%8) + (e//8)*16`. Accumulator: lane `l`, element
  `si` -> `C[(l//16)*8 + si, l%16]`. Verified against torch at 150.34 dB.
- `ds_load_tr16_b128` -> lane `l`, element `e` gets `src[(l//16)*8 + e, l%16]`: one column of
  the source tile, eight rows. Decoded empirically.
- The accumulator is **not** operand-shaped, so `dkdv` needs an LDS transpose for `P^T` /
  `dS^T` (one b128 store + barrier + two transposing loads per 16x32 operand). `dq` does
  **not**: two consecutive kv-tile accumulators concatenate in lane. Do not "optimise away"
  dq's LDS traffic for P/dS -- there is none.
- **flydsl 0.3.2 is required.** On 0.2.4 the aiter forward does not build (the `ast_rewriter`
  rejects a list as a stateful-`if` state variable). 0.3.2 in turn removes
  `flydsl.expr.buffer_ops`, which Primus-Turbo's own FlyDSL tree imports, so **do not import
  `primus_turbo.pytorch` in the same process**. The baseline imports aiter and torch only.
- `is_rdna_arch("gfx1250")` returns False in flydsl, so a FlyDSL build classifies this card as
  CDNA/wave64 unless overridden. It does not raise; it drops the phantom upper lanes' work.
  aiter re-decides for itself. Any new kernel must too.

## Traps this box has actually sprung

- **Never call an autotuner on a training path.** Its `except: continue` cannot recover a HIP
  context that has taken a launch failure. This wedged the card on 2026-09-16.
- **A known out-of-bounds path has no floor to measure.** Timing `dkdv_heads="kv"` as a
  "performance floor" took a `GCVM_L2_PROTECTION_FAULT` at s=8192. Recoverable, but the run
  was lost.
- **Launch any newly compiled kernel at a toy shape first, in its own process**, with
  `AMD_SERIALIZE_KERNEL=3`. A bad TDM descriptor or a wave-size mismatch HANGS rather than
  raising: 0-byte log, no compile error, `SIGKILL` useless.
- **LDS reads must be plain intrinsics, never opaque inline asm.** Opaque blocks hide the RAW
  against the async global->LDS store from LLVM, which then mis-orders under `DEP_MODE=2`.
  The forward's comments record 55% silent NaN at 16384 causal from exactly this.
- **Statement-level `for i in range(N)` appending to a list** is rewritten into an `scf.for`
  with carried variables and fails. Use a list comprehension or `range_constexpr`.
- **This card is VR-throttled to 1100 MHz and drifts** (1100 -> 967 observed inside one timing
  window). Interleave arms ABAB and record a clock witness with every figure.
- **Only one GPU.** `runtime.gpu_pool` is empty; deep rounds cannot run their analyses in
  parallel and will be slow. That is why the schedule is weighted toward fast rounds.

## Correctness is the gate, and it is four tensors

`dq`, `dk`, `dv` separately, each >= 50 dB against the fp32 reference, with outputs
NaN-prefilled and full `isfinite` coverage checked BEFORE SQNR. On 2026-09-16 a run read
`out 53.67 / dq 52.24 / dk -inf / dv -inf` -- checking only one tensor would have passed it.
Of 74 runs scanned on 2026-09-17, 8 had `loss: nan`, and every one was the fastest in its
group. Corrupted computation is usually faster.
