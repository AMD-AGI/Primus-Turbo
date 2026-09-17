# Candidate pool -- gfx1250-flydsl-attn-bwd

Ids are `r<round>.i<n>.g<global>`. `g` is monotonic across the whole job and never reused.
The birth round is where an idea was PROPOSED, not where it ran.

Highest `g` allocated so far: **g05**.

---

## r1.i1.g01 -- causal tile skipping in `k_dkdv` and `k_dq`  [EXECUTED round 1, arm A]

Both main kernels do the full `S^2` while bottom-right causal masks about half of it.
`k_dkdv`'s workgroup owns kv tile `[kv0, kv0+16)` and streams **every** q tile of every q
head in the GQA group; `k_dq`'s owns 16 queries and streams **every** 32-key block. In the
skipped tiles `p = exp2(NEG * LOG2E)` is exactly 0, so removing them is **bit-identical,
not an approximation** -- there is no precision risk against the 2.5 dB of headroom.

Evidence: Op Setup names it weakness #1, "roughly 2x is sitting there". `k_dkdv` is
**67-83% of runtime** (round 1 survey). AITER's card prices the residual after skipping:
*"the block-granular causal overhead is `1/ceil(S/256)`, so +50% matrix work at S=512 and
+6.25% at 4096"* (`aiter/attention/recipes/fmha_v3_bwd_hd128_bf16.md:38`) -- at this
kernel's **16-row** granularity the residual is smaller still. And the gfx1250 GEMM ladder
says this is the right FAMILY on a power-capped part: *"Under a cap, time is energy over a
fixed power budget, so removing work pays and merely overlapping it does not"*
(`hipkittens/gemm/recipes/bf16_gfx1250_ladder.md:596-601`).

No dead end recorded anywhere in the corpus for causal tile skipping.

## r1.i2.g02 -- full 32-deep contraction in `k_dkdv`'s dK/dV GEMM  [EXECUTED round 1, arm B]

`k_dkdv` stages 16 query rows and then issues `wmma_f32_16x16x32_bf16` for `dV += P^T dO`
and `dK += dS^T Q`, whose contraction is over **queries**. With only 16 staged, the
operand's upper half is zero-padded -- the source says so: *"a 16-wide contraction padded
to the WMMA's 32. Correct, not yet efficient."* Half of every dK/dV matrix instruction is
multiplying zeros, and those are 16 of the 24 WMMAs in the loop body.

Staging 32 query rows makes the contraction full. Per (16 kv x 32 q) the WMMA count goes
**48 -> 32**, so `k_dkdv` should go to ~0.667x. The idiom needed already exists in this
file: `k_dq` builds its K operand by shuffling `ds_load_tr16_b128` at row `lane_r` with row
`lane_r+16` into a v16. Arm B applies exactly that to the four dK/dV operands.

Corpus support: AITER's b4, *"pick the MFMA shape per product from the contracted
dimension"* so that *"no partial-K instruction is ever issued"*
(`fmha_v3_bwd_hd128_bf16.md:921`). This is the same defect, in the same place, in the same
op. Also an arithmetic-intensity change, which the ladder marks UP on a capped part.

## r1.i3.g03 -- AITER tile pairing, to load-balance a causally-skipped grid  [OPEN]

**Conditional on g01, and round 1's measurement is what prices it.** Once g01 skips, the
work per `k_dkdv` workgroup is `nqt - bid` q tiles: workgroup 0 does all of them and the
last does one. The TOTAL halves but the MAXIMUM does not move at all, so wherever the grid
fits the machine in one round the skip buys **nothing**. `k_dkdv`'s grid is **128
workgroups at the `fast` shape on a 256-CU part** -- one round, so `fast` is exactly that
case. `proxy` is 2048 (8 rounds) and `prod` is 16384 (64 rounds), where the average
dominates and the skip should land near 2x.

AITER's fix, which the corpus explicitly flags as worth copying and leaves unpriced:
*"`mha_bwd.cu` halves the grid for a causal mask -- `if (mt == 1 || mt == 2) gdx = (gdx +
1) / 2;`"*, one workgroup takes **K-tile `j` and K-tile `n-1-j`** so *"a pair sees `(n-j) +
(j+1) = n+1` regardless of `j`"*; *"a two-line host change and a duplicated loop body, and
it makes causal attention perfectly balanced across workgroups. **Worth copying.** It is
unpriced here"* (`fmha_v3_bwd_hd128_bf16.md:975-991`).

Caveat this job must weigh and AITER did not: pairing **halves the grid**, and at `fast`
that is 128 -> 64 workgroups on 256 CUs, which makes g05 worse. Pairing is right for
`prod`/`proxy` and probably wrong for `fast`; it may need to be conditional on grid size.

## r1.i4.g04 -- more than one wave per workgroup, and a tile larger than 16x16  [OPEN, large]

Every kernel here is **one 32-lane wave on a 16x16 tile**, inherited from bring-up probes.
The only gfx1250 performance recipe in the corpus is unambiguous that this is the wrong end
of the design space: the largest rung on the whole ladder is the **tile**, `03 128x128` at
**+81.10%**, then `02 async` at **+44.78%** and `07 tdm` at **+40.68%**
(`bf16_gfx1250_ladder.md:464-475`); *"the tile and the fill path are the win, and the new
mechanisms are the last 20%"* (`:862`). On waves it is equally specific and it cuts
**against** one wave: rung `11_one_wave` is a **-0.94% regression** and rung `12_two_waves`
is **+6.22%**, and the card reads that as *"the win is in the two-wave schedule"*
(`:480-487`). `lock_simd` is *"worth nothing at one [wave per SIMD]"* (`:830-836`).

This is the big structural item and it is **not a fast-round build** -- it is a rewrite of
both main kernels, their LDS staging and their fragment indexing together. Flagging it as
the thing a deep/large round should schedule, with the ladder's ranking as the plan.

⚠ Two gfx950 dead ends to respect when it IS built, both from
`flydsl/attention/dead-ends.md`: enlarging the KV block measured **3-4x slower** via
occupancy collapse (`:125`), and forcing occupancy with `waves_per_eu` measured **32%
slower** at 3 and **5x slower** at 4, with *"treat non-zero scratch as a refusal rather
than a cost"* (`:137`). On gfx1250 a spilling build is worse than slow -- it *"hangs after
the first launch"* (`bf16_gfx1250_ladder.md:701`).

## r1.i5.g05 -- the `fast` shape starves the machine: 128 workgroups on 256 CUs  [OPEN]

Measured this round, not previously written down anywhere. `k_dkdv`'s grid is
`(Skv/16, Hkv, B)`:

    fast   (64, 2, 1) =   128 workgroups    <-- 256-CU part; >half the machine idle
    proxy  (256, 8, 1) = 2048
    prod   (512, 8, 4) = 16384

This is why `fast` reads 3.0 TF/s against `prod`'s 57.2, and it is a **different bottleneck
from the other two shapes** -- no amount of work-removal fixes an empty machine. `fast` is
one third of the score, so it cannot be ignored, and it is the shape every round iterates
on. The fix is a split along an axis the grid does not currently use (split-K over the q
loop, which `flydsl/attention/techniques.md:373` measured *"worth +18% here"* at long
sequence, with the trap *"Split-K is also worthless when the inner range is already
short"*). Note it conflicts with `op.config.determinism`: a split-K dK/dV needs either a
deterministic tree reduction or a second pass, never atomics -- the bitwise-identity gate
over 200 runs will catch atomics immediately.

**Do not spend a round making `fast` faster by a change that costs `prod`.** g05 exists so
that a future round reads a `fast` result as an occupancy number rather than as evidence
about its candidate.
