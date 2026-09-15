# Weekly report — attention + GEMM on gfx1250 (2026-09-15)

Copy-paste paragraphs. Every number is measured on the box named; nothing is clock-converted
between machines.

---

## Short version (3 sentences)

Closed out the attention operator work on gfx1250: the aiter ASM forward plus ASM backward
path runs the production shape (b=4, s=8192, hq=32, hkv=8, d=128, bf16, causal) in
**11.726 ms against the 55.785 ms shipped baseline — 4.76x**, with SQNR held above 50 dB.
Separately, profiling the end-to-end training step found that **94% of it is hipBLASLt GEMM,
and 97% of that was running on a macro-tile-32x16x32 solution** picked because hipBLASLt has
no plain-bf16-GEMM tuning library for the NN transpose combination; routing around it gives
**1.894x end-to-end throughput (n=9), MFU 34% -> 65.4%**. The same defect turned out to be
the source of the ~3.9% trimodal run-to-run noise that had been invalidating the campaign's
A/B measurements — with it bypassed, between-run variance drops **9.3x to 0.42%**.

---

## Fuller version

**Attention operator (single layer, b=4 s=8192 hq=32 hkv=8 d=128 bf16 causal, A0-c07-1 at
1100 MHz, operator-level, n=5):**

| Stage | fwd | bwd | total | vs stock |
|---|--:|--:|--:|--:|
| Primus-Turbo stock `1cb2e183` | 10.651 | 45.134 | 55.785 ms | 1.00x |
| forced Triton fwd + fused bwd | 4.166 | 17.835 | 22.001 ms | 2.54x |
| aiter ASM fwd + fused bwd (now default) | 1.561 | 17.686 | 19.239 ms | 2.90x |
| aiter ASM fwd + ASM bwd (opt-in) | 1.572 | 10.160 | 11.726 ms | **4.76x** |

SQNR stays above the 50 dB gate throughout. The ASM backward is **opt-in, not default**: at
n=9 it shows no measurable end-to-end gain, and it carries 1 GiB of resident scratch plus
three prebuilt `.co` files to maintain.

**End-to-end (llama-3.1-8B, b=4 s=8192, 1 GPU, 8-layer config, seed pinned, n=9 per arm):**

| | tps | MFU | between-run sd |
|---|--:|--:|--:|
| baseline | 6,128 | ~34% | 3.91% (trimodal) |
| + GEMM layout workaround | **11,604** | **65.4%** | **0.42%** |

**Root cause, with hipBLASLt's own logs.** Same M/N/K, same 76 MB workspace; only the
transpose combination differs. The NN path resolves through a `GridBased` lookup table whose
nearest entry to (M=4096, N=32768, K=14336) is **N=1**, and MT32x16x32 is exactly the tile
one would choose for a GEMV — so a 65-million-workgroup launch takes 592-681 ms where the
correct solution takes 0.8 ms. The TN path has a trained `Prediction` model and picks
MT256x256x128. The physical evidence is in the library directory: of the bf16 libraries,
the TN combination ships `BB_BB_UA_Type_BB_HPA` plus `_CU96` and `_CU192` variants — the
plain-GEMM tuning library — and the NN combination ships none of the three (68 files vs 40).

This is a **packaging/coverage defect in the image, not a gfx1250 capability limit**: on the
same card at the same clock the NT path measures **1502 TF/s**, which is 1.67x faster than
the hand-written Triton GEMM we had been treating as the ceiling. It also corrects our
earlier "hipBLASLt is 13x off Triton" conclusion — that was measured on an NN square, which
happens to land squarely in the defective region.

**Two items to raise with the image maintainers:**
1. `Contraction_l_Ailk_Bljk` (NN) is missing `BB_BB_UA_Type_BB_HPA` and its CU variants.
   This affects every NN bf16 GEMM — in PyTorch training that is the dgrad of every Linear,
   i.e. the bulk of every step of every model.
2. `hipblaslt-bench` is unusable in this image: it expects `TensileLibrary_lazy_*.dat`, while
   the image ships the non-lazy per-problem-type layout with no master index. This blocks
   users from tuning or verifying solutions themselves.

**Measurement note worth carrying forward.** This box's end-to-end run-to-run noise floor was
~3.9% and trimodal, and every e2e conclusion drawn before we characterised it was n=1 —
including one that was off by 10 percentage points in the wrong direction. Seeds were also
not being set (torchtitan's `set_determinism` returns without seeding at world_size==1 when
`debug.seed is None`), so A/B arms differed in weight init as well as in the variable under
test. Both are fixed; A/B now runs seeded, order-alternated, n>=9.

**Caveats.** All end-to-end figures are on the 8-layer config; the 32-layer production config
sits at 88% memory and is untested with the workaround. A card wedge (`MES ring buffer is
full`) occurred once during the workaround's replication and could not be reproduced in six
subsequent runs — it is neither attributed to nor cleared of the workaround. The box is also
VR-limited to 1100 MHz (a sysfs sample under sustained load read 943 MHz), so none of these
absolute throughputs are representative of a healthy part.
