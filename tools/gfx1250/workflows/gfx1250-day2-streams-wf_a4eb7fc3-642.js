export const meta = {
  name: 'gfx1250-day2-streams',
  description: 'Three independent streams: aiter ASM forward probe on GPU0, HipKittens udna1 correctness gate (CPU), FlyDSL Gate A (CPU)',
  phases: [{ title: 'Streams', detail: 'one GPU probe plus two CPU-only investigations' }],
}

const COMMON = `
CONTEXT -- gfx1250 attention optimisation, 4-GPU host ctheliosp-1b112-a37-1, running now.

TODAY'S CHAMPION (just committed, 566e7798), production shape b=4 s=8192 hq=32 hkv=8 d=128
bf16 causal, median of 3 under an exclusive GPU window:
    fwd 2.579 ms | bwd 9.867 ms | total 12.412 ms | 620.5 TFLOP/s
    SQNR out/dq/dk/dv = 53.67 / 52.24 / 52.31 / 52.71 dB  (this exact quadruple is the
    signature of a numerically correct run on this shape -- any deviation is a red flag)

REFERENCE UPPER BOUND, pure tuned aiter Triton, same host, same harness:
    fwd 2.095 ms | bwd 9.174 ms | total 11.269 ms | 683.0 TFLOP/s
So we are 1.16x behind aiter overall, and MORE THAN HALF of that gap is now in the FORWARD
(2.579 vs 2.095 = 0.484 ms) rather than the backward (9.867 vs 9.174 = 0.693 ms but on a
2.4x larger base). That reordering is a measurement from this morning, not last night's
ranking, and it is why the ASM FORWARD is now ranked above the ASM backward.

ENVIRONMENT (already set up -- do NOT rebuild):
- Container "fa-repro" sees all 4 GPUs; fence every job with -e GPU=<N>:
    docker exec -e GPU=<N> -e PYTHONPATH=/home/lihuzhan/code/aiter-src \\
      -e TRITON_CACHE_DIR=/tmp/triton_cache_g<N> fa-repro bash -lc 'cd <REPO> && <cmd>'
- Container "fa-e2e" is fenced to GPU3 and is running a training job. DO NOT TOUCH IT.
- REPO = /home/lihuzhan/code/2026_0903__turbo/Primus-Turbo, branch gfx1250-attn-dispatch-and-tuning.
- Harness: tools/gfx1250/tune_attention.py -- prints ONE json line, gates on four-tensor SQNR,
  and asserts the requested config actually reached the kernel.
- aiter source clone (with the prebuilt gfx1250 .co files): /home/lihuzhan/code/aiter-src
- Campaign dir: $REPO/output/0914__campaign  (bin/, ledgers/, logs/)
- Yesterday's full analysis: $REPO/output/0913__opt_plan__claude/  (PROGRESS.md is the handoff;
  phase2/T7-ASM-FORWARD.md and phase2/T2-ASM-BACKWARD-SPEC.md are the offline archaeology)
- Today's reproduction + corrections: $REPO/output/0914__repro__c07/RESULTS.md

GPU DISCIPLINE -- THIS IS NOT OPTIONAL:
1. There is NO GPU lock. A measurement that lands on a card another stream is using does NOT
   raise -- it silently records a wrong number. That happened once already today: a combined
   A/B read 15.6 ms and SQNR 51.27 dB purely from contention; the real numbers were 12.4 and
   53.67. ALWAYS take measurements through:
       bash $REPO/output/0914__campaign/bin/exclusive.sh <gpu> <command...>
   which parks the sweep queue between candidates and releases on exit.
2. NEVER pkill -9 a process with work in flight on the GPU. Repeatedly doing that is the
   suspected contributor to a GPU wedge yesterday that needed a host reboot.
3. NEVER use \`pkill -f <pattern>\` where the pattern can match your own command line -- it
   kills your own shell. This also fired once today.
4. Health checks use \`timeout 15 dmesg | grep -E 'MES\\(|GPU Hang|wait for reset ack'\` ONLY.
   rocm-smi, ps and pgrep all walk device/process state and HANG on a wedged card.
5. NEVER run rocprofv3 PC sampling. Three attempts, three GPU faults, one needed a reboot.

REPORTING: your final message IS the return value. Report measured numbers, never estimates.
A negative result stated crisply is worth as much as a positive one -- say plainly when
something is refuted, and say what evidence refuted it.
`

const SCHEMA = {
  type: 'object',
  properties: {
    stream: { type: 'string' },
    verdict: { type: 'string', description: 'the decision this stream reached, one or two sentences' },
    numbers: { type: 'string', description: 'measured figures, or "none -- offline stream" ' },
    evidence: { type: 'array', items: { type: 'string' } },
    next_step: { type: 'string', description: 'the single most valuable follow-up, or "close this line"' },
    blockers: { type: 'string' },
  },
  required: ['stream', 'verdict', 'numbers', 'evidence', 'next_step', 'blockers'],
}

const TASKS = [
  {
    key: 't7-asm-forward',
    prompt: `YOUR GPU: 0 (exclusive -- nothing else is scheduled on it). YOUR STREAM: T7, the prebuilt
aiter gfx1250 ASM FORWARD.

GOAL: find out whether aiter's prebuilt gfx1250 ASM forward runs correctly on our production
shape and how fast it is. Accept if SQNR >= 40 dB on the output AND forward < 2.095 ms (the
tuned-aiter-Triton forward). Our own forward is 2.579 ms, so anything at or under ~2.1 ms
would close more than half our remaining gap to aiter.

READ FIRST: $REPO/output/0913__opt_plan__claude/phase2/T7-ASM-FORWARD.md. It is the offline
archaeology and it already resolved most unknowns. Summary of what it establishes:
- Python entry point EXISTS: aiter.ops.mha.fmha_fwd_with_sink_asm(q,k,v,scale,is_causal,
  return_lse,...) -- no hand-written launcher needed, unlike the backward.
- Layout is bshd, which is OUR layout. is_causal is a parameter. return_lse gives [B,Hq,Sq]
  fp32, which is exactly the flat LSE our fused backward wants.
- pertokenBf16 requires NO quantization scales (the 132 B parameter struct has no scale
  tensor, and its size matches the ELF kernarg exactly).
- LSE is the NATURAL logarithm (aiter's own test compares against torch.log(denom)+max_total
  and allclose's it), matching our convention -- no base conversion.
- Shape constraints are only: bf16, 4-D, stride(-1)==1, hq%hkv==0, head_dim in {64,128}.
  No seqlen divisibility limit. is_causal=True selects the ..._mask.co variant.
- CAVEAT ON THE RECORD: aiter's own tests only cover gqa=8. We are gqa=4 (hq=32, hkv=8).
  That combination has never been exercised, so check numerics before believing any timing.
- KNOWN BLOCKER: \`import aiter.ops.mha\` drags in aiter/ops/triton/gluon/pa_decode_gluon.py,
  which unconditionally does \`import jax\`, and jax is NOT in the image. Two ways round:
  install jax into the container, or bypass the package import and reach the ctypes entry
  point directly (it is declared ffi_type="ctypes" already). Try the cheap one first and say
  which you used.

DO:
1. Confirm the .co files are actually present under /home/lihuzhan/code/aiter-src/hsa/gfx1250/
   and identify which one is_causal=True + hd128 + bf16 selects.
2. Get the call to happen at all. Work in a SCRATCH script under $REPO/output/0914__campaign/,
   never by patching the installed aiter or the source clone -- other streams depend on both.
3. Numerics FIRST, timing second. Compare out against an fp32 reference (chunked per (b,h) to
   fit memory -- tools/gfx1250/tune_attention.py already contains a correct chunked fp32
   reference you can lift). Report SQNR in dB. Also validate the returned LSE against the
   reference's logsumexp, because the fused backward consumes it.
4. Only if numerics pass, time it the same way the harness does: CUDA events, median of 20,
   L2 flushed between reps with a 256 MiB buffer. Compare against 2.579 (ours) and 2.095
   (tuned aiter Triton) measured in the same session on the same card.
5. If it is fast AND correct, the follow-on is wiring it behind an arch gate in
   primus_turbo/pytorch/kernels/attention/ -- describe what that would take, but DO NOT ship
   it in this stream; report and let the main session sequence it.

ABANDON TRIGGER: if after 90 minutes you do not have a numerically validated forward, stop
and write up exactly where it failed. A crisp "the ABI is not what the document says, here is
the evidence" is a valuable result -- it closes a line that would otherwise be re-opened.`,
  },
  {
    key: 'hipkittens-gate',
    prompt: `NO GPU unless you ask for it at the very end -- GPU0 belongs to another stream right now and
GPU1/2/3 are all committed. YOUR STREAM: the HipKittens udna1 correctness gate. This is
CPU-dominant work (building and authoring tests); if you reach a point where you genuinely
need a GPU, say so in your report rather than taking one.

THE QUESTION, and it is binary: does the byte-identical HipKittens "register tier" actually
COMPUTE THE RIGHT THING under wave32 + WMMA on gfx1250?

Why it matters: 3rdparty/hipkittens/include/udna1/ is a complete 72-header gfx1250 port that
is already in the tree (I rsync'd the submodule in this morning; verify it is populated).
\`conversions.cuh::transpose\` has ELEVEN gfx950 backward call sites riding on it, and it is a
PURE REGISTER RELABEL with no cross-lane operation. That identity holds only because MFMA's
A-operand map and C-accumulator map are duals on CDNA wave64. WMMA f32_16x16x32_bf16 has A as
v16bf16 and C as v8f32 across 32 lanes -- a different duality. The identity is ASSERTED
EVERYWHERE AND EXECUTED NOWHERE. Answering it green or red both have value: green turns a
7-10 engineer-week estimate into a plan; red reprices the backward by +5-8 days and makes
ds_load_tr16_b128 plumbing a hard prerequisite (note \`grep -rn 'ds_load_tr' include/udna1/\`
currently returns zero).

DO, in this order:
1. Verify 3rdparty/hipkittens is populated (expect ~72 .cuh under include/udna1 and both
   tests/unit/cdna4 and tests/unit/udna1 present). Report what is actually there.
2. Build wiring. setup.py:348 appends -DKITTENS_CDNA4 UNCONDITIONALLY, and
   -DBUILD_HIPKITTENS_BACKEND is gated on gfx950 only (~setup.py:360) -- so a gfx1250 build
   today compiles HipKittens against wave64 headers. Work out the ~20-line change (plus the
   *_gfx1250.cu suffix, which filter_files_by_arch at setup.py:152 already dispatches on).
   You may EDIT setup.py. Do not run a full Primus-Turbo build -- it is long and would fight
   the other streams; compile the unit tests directly instead.
3. THE MAIN WORK: tests/unit/udna1/ is ~444 lines covering only warp/memory/tile.
   tests/unit/cdna4/ is ~2839 lines covering warp/{memory,shared,register} plus group. The
   ~2400-line delta IS the register and shared tiers -- exactly the byte-identical files whose
   wave32 correctness is in question, and exactly mma_ABt/wmma161632, which no test in the
   tree exercises. Copy cdna4/warp/{register,shared}/ into udna1/ and switch the Makefile
   define. This is mechanical BECAUSE the headers under test are byte-identical.
   Priority order, strictly: conversions.cuh::transpose -> reductions.cuh row/col max+sum
   (it uses __builtin_amdgcn_permlane32_swap under a comment describing "row 2 and 3" of a
   SIXTY-FOUR-lane accumulator, unchanged on wave32) -> mma_ABt/mma_AtB shape coverage ->
   maps.cuh::exp2.
4. Get as far as COMPILING the transpose test for gfx1250. Compilation alone is informative:
   a static assert or a codegen failure answers the question without a GPU.
5. Report precisely what would need to run on a GPU and for how long, so the main session can
   schedule a short exclusive window.

Local clang is at /opt/rocm-10.1.0a20260807 inside the containers; check what the host has.
Note gemm_tdm_arrive's own header warns it HANGS on runtimes that do not model
DS_ATOMIC_ASYNC_BARRIER_ARRIVE_B64 -- never run it first and never without a timeout.

ABANDON TRIGGER: if by 2.5 hours you cannot compile a single udna1 register-tier test, stop.
The primitive the whole port rests on does not work in this toolchain and no further test
porting changes that. Write that up -- it is the answer, not a failure.`,
  },
  {
    key: 'flydsl-gate-a',
    prompt: `NO GPU. YOUR STREAM: FlyDSL "Gate A" -- a strictly time-boxed investigation whose deliverable
is ONE FACT, not a kernel. Hard stop at 2 hours.

THE QUESTION: is FlyDSL blocked or not blocked as a route to a gfx1250 attention kernel?

Yesterday's plan ($REPO/output/0913__opt_plan__claude/phase2/PLAN-4GPU-TOMORROW.md) opens with
a six-point refusal to start FlyDSL at all. I have already REFUTED its first point this
morning and you should not re-derive it: it claims "the pinned 0.2.4 does not expose tdm_ops,
s_wait_tensorcnt, ds_load_tr16_b128 or permlanex16". Measured inside the image, flydsl 0.2.4
has ALL FOUR -- flydsl/expr/rocdl/tdm_ops.py exists, and the other three plus the string
"gfx1250" are all in flydsl/_mlir/dialects/_rocdl_ops_gen.py. pip index also offers 0.3.2.
So the TOOLCHAIN gate is open. That matches PROGRESS.md's own correction (its item 8) and
contradicts the plan document, which should be fixed.

The plan's OTHER three points still stand and are why this is ranked last:
- setup.py (around line 530) explicitly SKIPS installing flydsl (and triton) for any build
  whose offload archs include gfx1250. Verify and quote it.
- The FlyDSL attention kernels that exist are gfx950 wave64 + MFMA with HAND-SCHEDULED
  instruction tables. Look at branches origin/dev/kyle/flydsl-attn-bwd-nd (flash_attn_bwd.py,
  ~3289-line diff) and origin/feat/flydsl-attn-fwd-opt. A wave32/WMMA part invalidates every
  row of a hand-built schedule; that is where the 18-36 engineer-day estimate comes from.
- Even a PERFECT FlyDSL forward is worth only ~0.48 ms of our 12.412 ms (3.9%), and the ASM
  forward probe running right now targets the same 0.48 ms for about an hour of work.

DO:
1. Verify the setup.py gate and quote the exact lines.
2. Read primus_turbo/pytorch/kernels/attention/attention_flydsl_impl.py and
   attention_impl.py's _flydsl_common_ok. Today's branch deliberately TIGHTENED the arch test
   from \`get_device_compute_capability() >= (9,5)\` to \`is_gfx950()\`, because gfx1250 reports
   (12,5) which the old comparison let through -- straight into a gfx950 JIT. Confirm that
   reading and note what _gqa_group_ok does with our G=4.
3. NOTE AND EVALUATE: branch origin/dev/kyle/flydsl-attn-gqa4 is titled "Admit GQA groups
   below 8, so llama 7B and 8B reach the flydsl attention" -- that is precisely the gate our
   shape needs. Read its diff. Say whether it is only a gate relaxation or whether it also
   restructures the kernel.
4. The actual Gate A experiment, CPU only: take the existing gfx950 FlyDSL FORWARD and try to
   get it to BUILD for gfx1250 (flip the arch gate in a scratch copy; do not commit). Record
   the FIRST REAL FAILURE POINT with its exact error text. That failure point is the answer:
   if it is a missing primitive, the gate is open-ish; if it is "this schedule table assumes
   wave64", the gate is shut and the 18-36 day estimate is confirmed rather than assumed.
5. Do NOT attempt to port any FlyDSL BACKWARD. Do not build the whole of Primus-Turbo.

ABANDON TRIGGER: 2 hours elapsed, OR the first failure point lands in the "needs a rewritten
schedule table" class. Either way stop immediately and write the conclusion.

Deliverable: a blocked/not-blocked verdict with the specific evidence, plus a corrected
paragraph I can paste over the FlyDSL section of PLAN-4GPU-TOMORROW.md.`,
  },
]

phase('Streams')
const results = await parallel(TASKS.map(t => () =>
  agent(`${COMMON}\n\n========================================\n${t.prompt}`, {
    label: t.key,
    phase: 'Streams',
    schema: SCHEMA,
  })
))

return results.filter(Boolean)
