# Report to AITER: three gfx1250 defects and one dispatch-policy question

**Hardware:** single gfx1250 (`AMD Radeon Graphics`, DID `0x75c1`), **VR-limited** — a 1100 MHz
DPM ceiling, and a sysfs sample inside a sustained timing window read 967 MHz.
**Every absolute figure below is at that clock and none is converted to any other clock.**
**Software:** `aiter` at the checkout in `/home/lihuzhan/code/aiter-src`, torch
`2.11.0+rocm7.14.0a20260625`, flydsl 0.2.4 (image) and 0.3.2 (installed side-by-side, see §3).
Shape unless stated: `b=4 s=8192 hq=32 hkv=8 d=128 bf16 causal` (Llama-3.1-8B, GQA ratio 4).

Four items. §1 is a correctness bug with a user-side workaround that costs 1 GiB. §2 is a
capability gap. §3 is a packaging conflict with no user-side fix. §4 is a question, not a defect.

---

## 1. The prebuilt gfx1250 ASM backward writes out of bounds under GQA

### Symptom

Two faces of one bug, depending on where the out-of-range write lands:

- `s=1024`: silent corruption. `dk`/`dv` collapse to about **−0.3 dB** SQNR while `dq` stays
  correct at 52.24 dB.
- `s=256`: a page-boundary crossing, so the process takes a fault instead. **dmesg records
  nothing and the card stays healthy** — this kills the process, not the GPU.

### Diagnosis

`dq` being correct is the locating clue: `dq` reads k/v per group, so the kernel's grouping
already agrees with the reference. Only the `dk`/`dv` **write** path is wrong.

The main kernel's grid is `(kv_tiles, nhead_q, batch)`. Under GQA, `ratio` workgroups own the
same `dk`/`dv` tile with no synchronisation, and the kernel indexes a **kv-sized** buffer by
**q head**. Three experiments pin it:

| | dq | dk | dv | |
|---|--:|--:|--:|---|
| ratio = 1 (no GQA) | 52.26 | 52.27 | 52.76 | all correct |
| ratio = 4, dk/dv indexed by kv head | 52.24 | **−0.94** | **−0.65** | corrupt |
| ratio = 4, dk/dv allocated per q head + host reduction | 52.24 | 50.55 | 51.09 | all correct |

Pre-zeroing `dk`/`dv` changed nothing, which rules out an accumulate-versus-assign mistake.

### What this is not

There is **no small-shape boundary problem** and an eligibility gate needs no seqlen lower
bound for *this* reason. We proposed one twice from single samples and were wrong both times;
the s=256 fault and the s=1024 corruption are the same bug landing in different memory.

### Our workaround, and what it costs

Allocate `dk`/`dv` **per q head** and reduce on the host. That is correct but costs
**1.000 GiB of resident scratch** plus a reduction on the critical path of every step.

### What we would like

`dk`/`dv` allocated per q head inside the kernel, or a workgroup-level reduction, so callers
do not each reinvent a 1 GiB workaround.

---

## 2. gfx1250 has no varlen ASM backward, and few backward variants at all

- The gfx1250 backward dispatch table carries only `mode=0`; the host passes `nullptr` for
  every `seqstart` pointer. **There is no varlen backward on this arch.**
- Asset count: gfx1250 ships **52** `.co` files against gfx950's 1466, and **6** backward
  variants against 124.
- No Python arch gate in aiter reaches the gfx1250 backward kernels at all:
  `can_impl_fmha_v3_bwd` starts from `get_gfx() == "gfx942"` and the only widening is for
  gfx950, so gfx1250 can never select them. We launch them by hand.

Concretely we would like a gfx1250 `psskddv` variant and a varlen backward. The variant gap is
what our own numbers keep running into; the varlen gap is the one with no workaround.

---

## 3. `flydsl==0.3.2` (which aiter pins) cannot coexist with a 0.2.4 consumer

`aiter-src/setup.py:17` pins `FLYDSL_VERSION = "flydsl==0.3.2"` and enforces it at build time.
This is correct for aiter — measured below — but it is unsatisfiable for any process that also
uses another flydsl consumer built against 0.2.4.

**Measured, both directions:**

- **aiter's gfx1250 FlyDSL forward genuinely needs 0.3.2.** On 0.2.4 it does not build. The
  blocker is not a missing symbol — it is the compiler front end: 0.2.4's `ast_rewriter`
  rejects a `list` as the state variable of a stateful dynamic `if`, which
  `fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py:1125` `_maybe_rescale` relies on. We first
  shimmed the four names 0.2.4 spells differently (`fx.ceildiv` → `fx.ceil_div`,
  `fx.to_llvm_ptr` → `buffer_ops.create_llvm_ptr`, integer `fx.max`/`fx.min` →
  `arith.max{si,ui}`); the build then got much further and still failed. On 0.3.2 it builds and
  runs with **zero shims**.
- **0.3.2 removes `flydsl.expr.buffer_ops` entirely** (nothing by that name in the wheel), and
  Primus-Turbo's FlyDSL tree imports it. So installing 0.3.2 to satisfy aiter breaks that tree.

There is no user-side fix: the two requirements are on the same package in the same process.
The only isolation available is `pip install --target` plus `sys.path` ordering, which works
but means two copies of a compiler runtime in one process.

**What would help, in rough order of preference:** a compatibility shim in aiter for the 0.2.x
front-end difference; or a documented minimum that is a *range* rather than `==`; or, if the
0.3.x dependency is genuinely hard, saying so prominently, since `==` in a build-time assert
reads like conservatism rather than a hard requirement.

---

## 4. Question: on gfx1250 the FlyDSL forward is the default, and it is slower than your own ASM forward

`fmha_kernels.py` routes gfx1250 bf16 `qk_hdim in (128, 192)` to the FlyDSL m32x8 kernel by
default. Measured against aiter's own prebuilt ASM forward, same process, same tensors, ABAB
interleaved, CUDA events, 256 MiB L2 flush between reps, n=20, correctness checked first
against an fp32 chunked reference with NaN-prefilled outputs:

| arm | median | best | sd | SQNR out | isfinite |
|---|--:|--:|--:|--:|---|
| `flydsl_flash_attn_batch_func` | **2.373 ms** | 2.328 ms | 2.68% | 51.63 dB | full coverage |
| `fmha_fwd_with_sink_asm` | **1.569 ms** | 1.555 ms | 1.09% | 53.62 dB | full coverage |

**The ASM forward is 1.51x faster on this shape.** Both are correct; this is not a correctness
report. Our measurement chain reproduces a separately recorded 1.572 ms for that ASM forward to
within 0.2%, so we believe the pair.

We are not claiming the default is wrong — the ASM forward may not cover the shapes FlyDSL
routing has to serve, and this is one shape on one throttled card. The question is whether the
gfx1250 default is expected to be the faster of the two here, because we were reading it as the
recommended path and it is not the fastest one you ship.

---

## Reproduction

`output/0917__flydsl/bin/stage1_fwd_ab.py` in our tree is self-contained: it imports aiter and
torch only, builds its own chunked fp32 reference, and writes `stage1_fwd.json`. §1's three-row
table comes from `tools/gfx1250/asm_bwd_launcher.py --dkdv-heads {kv,q}`.
