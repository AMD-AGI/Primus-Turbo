export const meta = {
  name: 'gfx1250-repro-verify',
  description: 'Independently verify the gfx1250 attention reproduction: flex anchor, num_warps non-transfer, clock sustain, ledger cross-check',
  phases: [{ title: 'Verify', detail: 'four GPU-fenced probes plus one offline ledger audit' }],
}

const COMMON = `
CONTEXT — read carefully before running anything.

We are reproducing, on this 4-GPU gfx1250 host, an attention-optimization result that was
measured yesterday on a DIFFERENT host (heliosr-1b114-c07-1) whose GPU was VR-throttled to
1100 MHz. This host runs at 2357-2400 MHz.

Environment (already set up, do NOT rebuild it):
- Running docker container named "fa-repro" (image amdprimus/amdprimus:gfx1250-20260910).
  Run everything through it:
    docker exec -e GPU=<N> -e PYTHONPATH=/home/lihuzhan/code/aiter-src fa-repro \\
      bash -lc "cd <REPO> && python3 tools/gfx1250/tune_attention.py <args>"
  The GPU env var is mandatory; the harness turns it into HIP_VISIBLE_DEVICES.
- Three checkouts of Primus-Turbo, all with compiled .so already copied in:
    HEAD     /home/lihuzhan/code/2026_0903__turbo/Primus-Turbo        (branch gfx1250-attn-dispatch-and-tuning, d2f75576)
    BAKEOFF  /home/lihuzhan/code/2026_0903__turbo/wt-bakeoff          (1cb2e183: fused kernel vendored, NOT yet routed by default)
    MAIN     /home/lihuzhan/code/2026_0903__turbo/wt-main             (c1325c7e: shipped state, two-kernel backward, --tune unsupported)
- Harness: tools/gfx1250/tune_attention.py. It prints ONE json line. It gates on four-tensor
  SQNR (out/dq/dk/dv >= 50 dB) and asserts the requested config actually reached the kernel.
- Production shape is --shape llama31-8b  (b=4 s=8192 hq=32 hkv=8 d=128 bf16 causal).
- Existing ledgers: /home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0914__repro__c07/*.jsonl
- Yesterday's docs/ledgers are in that repo under output/0913__opt_plan__claude/ (readable via
  git show gfx1250-attn-dispatch-and-tuning:<path> or directly, since that branch is checked out at HEAD).

MEASUREMENTS ALREADY IN HAND (median, ms / TFLOP-per-s, this host), all SQNR 53.7/52.2/52.3/52.7:
  MAIN    turbo shipped  (two-kernel bwd, untuned)   5.226 fwd / 25.025 bwd / 30.258 tot / 254.4
  BAKEOFF turbo shipped  (two-kernel bwd, untuned)   5.179 / 22.942 / 28.121 / 273.7
  BAKEOFF turbo cfgtuned "fwd:num_stages=2; bwd:num_warps=2"  2.718 / 22.977 / 25.695 / 299.5
  HEAD    fused champion "--impl fused --tune fwd:num_stages=2" 2.745 / 10.284 / 13.023 / 591.0
  HEAD    aiter shipped                               3.366 / 13.838 / 17.204 / 447.4
  HEAD    aiter tuned                                 2.095 /  9.174 / 11.269 / 683.0
  torch flex (our own script, output/0914__repro__c07/flex_anchor.py)  6.929 / 30.296 / 37.225 / 206.8

YESTERDAY'S NUMBERS on the throttled host (the targets being reproduced):
  turbo shipped 10.673 / 48.969 / 59.642 / 129.0
  turbo cfgtuned 4.152 / 32.252 / 36.405 / 211.4
  torch flex     7.506 / 23.831 / 31.337 / 245.6
  fused champion total 24.435 / 315.0
  aiter tuned    3.259 / 18.425 / 21.684 / 354.9
  aiter shipped  6.676 / 27.888 / 34.565 / 222.7

RULES:
- Use ONLY the GPU assigned to you below. Another agent owns each of the others; running on
  someone else's card corrupts their timing and yours. Never run a job on more than one GPU.
- Report measured numbers, never estimates. If something fails, report the failure and the
  error text; do not paper over it.
- Your final message IS the return value. Be specific and quantitative.
`

const SCHEMA = {
  type: 'object',
  properties: {
    question: { type: 'string' },
    verdict: { type: 'string', description: 'the direct answer, one or two sentences' },
    evidence: { type: 'array', items: { type: 'string' }, description: 'measured numbers / commands / file:line that support the verdict' },
    numbers: { type: 'string', description: 'a compact table of what you measured, or "none" if offline' },
    caveats: { type: 'string' },
  },
  required: ['question', 'verdict', 'evidence', 'numbers', 'caveats'],
}

const TASKS = [
  {
    key: 'flex',
    gpu: 0,
    prompt: `YOUR GPU: 0. YOUR QUESTION: is the torch-flex anchor number (37.225 ms) real, or is our flex_anchor.py not equivalent to what produced yesterday's 31.337 ms?

This is the ONE row that did not get faster on the faster host. Every other row improved ~1.4-2.1x. Either flex genuinely does not benefit from the clock here, or our script is measuring something different.

Yesterday's flex number came from a Primus-side bench that is NOT present on this host
(/home/lihuzhan/code/2026_0828__primus/Primus/benchmark/kernel/attention/bench_attention.py does not exist).
Check whether any equivalent bench exists anywhere under /home/lihuzhan/code (look in the Primus checkout,
and in the Primus-Turbo repo's own benchmark/ directory - there is a benchmark/ops/training/bench_attention_turbo.py
referenced in the harness comments).

Then attack the discrepancy directly. Things worth checking and MEASURING:
1. Read output/0914__repro__c07/flex_anchor.py. Compare its timing/FLOP accounting to
   tools/gfx1250/tune_attention.py (timed_ms, and the fwd/bwd timing closures around line 540-590).
   In particular check how the harness times the BACKWARD - whether it rebuilds the graph, uses
   retain_graph, zeroes grads, and whether our flex script's backward does strictly the same work.
   A backward that also re-runs part of the forward, or that accumulates into .grad instead of
   replacing, would inflate our number.
2. Is torch.compile actually compiling flex here, or silently falling back to eager?
   Check with TORCH_LOGS=graph_breaks or by inspecting whether a Triton flex kernel is generated.
3. Try the obvious variants and time them the same way: without torch.compile; with
   mode="max-autotune-no-cudagraphs"; with and without the block_mask (score_mod-free causal);
   with enable_gqa=True vs materialising k/v to 32 heads.
4. Sanity-bound it: flex at 37.2 ms is slower than our OWN two-kernel Triton backward (28.1 ms)
   on the same card and shape. Yesterday flex BEAT that path (31.3 vs 59.6). Which of those two
   worlds is right here?

Deliverable: the best defensible flex number on this host for b=4 s=8192 hq=32 hkv=8 d=128 bf16 causal,
with fwd and bwd split, plus a clear statement of whether 37.225 stands or is superseded, and why.
If flex genuinely regressed relative to the throttled host, say so and give the mechanism you can support.`,
  },
  {
    key: 'numwarps',
    gpu: 1,
    prompt: `YOUR GPU: 1. YOUR QUESTION: yesterday, on the throttled host, "bwd:num_warps=2" was a large win on the in-tree TWO-KERNEL backward (48.969 -> 32.252 ms, 1.52x). On this host it appears to do NOTHING (22.942 untuned -> 22.977 tuned). Verify that, and characterise it.

Work in BAKEOFF (/home/lihuzhan/code/2026_0903__turbo/wt-bakeoff), which has the two-kernel
backward as the default path AND supports --tune. (At HEAD the backward is routed to the fused
kernel above s=2048, so HEAD cannot measure this.)

Do:
1. Confirm the harness's assert_config_applied really proves num_warps reached the backward kernel.
   Read tools/gfx1250/tune_attention.py assert_config_applied and
   primus_turbo/triton/attention/attention_kernel.py's _parse_tune_spec / autotune list construction.
   A silently-ignored spec is the exact failure mode that produces a "flat sweep". Print the
   "configs" field from the harness json for each run and check it actually changed.
2. Sweep the backward num_warps on the production shape: 1, 2, 4, 8. Also sweep num_stages 1,2
   jointly with the best num_warps. Use --shape llama31-8b. Record fwd/bwd/total and SQNR for each.
3. If num_warps genuinely does not matter now, the interesting question is WHY the throttled card
   cared and this one does not. State the mechanism you can actually support from the data
   (e.g. latency-hiding knobs matter more when the core clock is low relative to memory, so the
   win was a throttle artifact) - but only claim what your numbers support.
4. Guardrail from yesterday's notes: waves_per_eu=4 and BLOCK_M1=16 were catastrophic on aiter;
   do not go hunting outside the knobs named here.

Deliverable: a table of num_warps x num_stages on the two-kernel backward at this clock, with the
"configs" field proving each config applied, and a verdict on whether yesterday's 1.52x transfers.`,
  },
  {
    key: 'clock',
    gpu: 2,
    prompt: `YOUR GPU: 2. YOUR QUESTION: what clock did this host ACTUALLY sustain during the measurements, and does per-row clock scaling explain the reproduction ratios?

Background: yesterday's host was VR-throttled with a DPM table of only 500/1100 MHz. This host's
DPM table reads 500 / ~2357-2400 MHz on all four cards, BUT dmesg also contains
"amdgpu ...: WARN: GPU is throttled, expect performance decrease. VR." lines from boot time on all
four cards. So the throttle warning is present here too, and we must not assume the ceiling is real.

Do:
1. Establish the clock story properly. Read /sys/class/drm/card*/device/pp_dpm_sclk on the host.
   Find the four cards that are actually the gfx1250 devices (there are many card* nodes; the GPUs
   are PCI 1002:75c1 at 0001:04:00.0, 0002:04:00.0, 0003:04:00.0, 0004:04:00.0). Check whether
   rocm-smi exists inside the fa-repro container and whether it reports achieved sclk.
   IMPORTANT SAFETY NOTE from yesterday's incident report: on a WEDGED card rocm-smi hangs forever
   and "timeout" does not bound it. This host's cards are healthy (many runs have completed), but
   still prefer /sys reads, and if you do call rocm-smi and it does not return within ~20s, abandon
   that approach rather than retrying.
2. MEASURE the sustained clock under load on YOUR card (GPU 2). Run a long attention job
   (e.g. the harness with --shape llama31-8b --iters 200 --impl fused --tune "fwd:num_stages=2")
   and sample the clock from /sys (or rocm-smi if safe) while it runs. Report the distribution,
   not one sample: is it holding ~2350 MHz, or sagging?
3. Compute the per-row reproduction ratio (yesterday_ms / today_ms) for every row in the table
   above and compare it to the clock ratio you measured. Rows:
   turbo shipped, turbo cfgtuned, fused champion, aiter tuned, aiter shipped.
   Which rows scale WITH the clock and which do not? A row that scales less than the clock ratio
   is telling us it is not core-clock bound - say which bound you think it hit (memory bandwidth,
   launch overhead, or something else) and what in the data supports that.
4. Note that the aiter-tuned row's forward (3.259 -> 2.095) scaled only 1.56x while its backward
   scaled 2.01x. Explain what you can.

Deliverable: the measured sustained clock on this host under load, the clock ratio vs yesterday,
and a per-row table of observed speedup vs clock-predicted speedup with an interpretation.`,
  },
  {
    key: 'crosscheck',
    gpu: 3,
    prompt: `YOUR GPU: 3. YOUR QUESTION: is our reproduction faithful to yesterday's measurement in every respect OTHER than the clock? Hunt for anything that makes today's numbers not comparable to yesterday's.

This is an adversarial audit. Your job is to FIND PROBLEMS, not to confirm the result. Default to
skepticism. Concretely:

1. Compare yesterday's recorded json rows to ours, field by field. Yesterday's ledgers are at
   output/0913__opt_plan__claude/phase2/ledgers/*.jsonl (wq.jsonl, wq2.jsonl, wq3.jsonl,
   forever.jsonl, det_fast.log, det500.log, shapes.log) in the Primus-Turbo checkout.
   Find the rows corresponding to: turbo shipped, turbo cfgtuned, aiter tuned (tag "fwd|aiter|st2"
   is known to be the 21.684 row), and the fused champion (24.435). Diff every field against ours:
   the "configs" applied, aiter_bwd_config / aiter_fwd_config, sqnr_db, batch/seqlen/hq/hkv/head_dim,
   peak_mem_gib, dtype, causal. Report ANY field that differs.
2. Environment drift. Yesterday's container was image "fa-tune:deps" which was built FROM
   amdprimus/amdprimus:gfx1250-20260910 (same image id as the base we are using, 6e656de79e6c) but
   then rebuilt Primus-Turbo inside it at TURBO_REF=2f00979. We instead copied the prebuilt
   _C.cpython-312-x86_64-linux-gnu.so and libprimus_turbo_kernels.so out of the BASE image. The md5
   of our _C.so (3959ea9edc4bf954fcf738296d276c1e) differs from the remote checkout's
   (e9dc4e78a7052e64bab2b27fad09301a). Determine whether that can affect the attention numbers at
   all. The attention path is Triton/Python - verify that claim rather than assuming it; check what
   _C actually exports and whether anything on the flash_attn_func path touches it.
   You can read the remote machine: ssh -o BatchMode=yes lihuzhan@heliosr-1b114-c07-1.mnb.dcgpu '<cmd>'
   (it works, non-interactively; it prints a long login banner you can ignore).
   Also compare torch/triton/python versions between our container and the remote's image if you
   can get at them.
3. We applied ONE source edit to the wt-main checkout that yesterday's run did not have: we copied
   primus_turbo/pytorch/core/low_precision.py from the branch onto c1325c7e, because c1325c7e's
   version raises TypeError on torch 2.11 (opaque type must subclass torch._opaque_base.OpaqueBase)
   and the process could not even import. Verify this file is genuinely off the attention path.
4. We used our own flex_anchor.py rather than the Primus bench. Note this as a known divergence;
   another agent is measuring it, so do NOT run flex yourself - just record what yesterday's
   31.337 figure was produced by, if the provenance is recoverable from the docs.
5. Re-run ONE row yourself on GPU 3 as an independent replication - the fused champion:
   docker exec -e GPU=3 fa-repro bash -lc "cd /home/lihuzhan/code/2026_0903__turbo/Primus-Turbo && python3 tools/gfx1250/tune_attention.py --shape llama31-8b --impl fused --tune 'fwd:num_stages=2'"
   Confirm it lands at ~13.0 ms with SQNR 53.67/52.24/52.31/52.71. Report the exact SQNR digits:
   if they match yesterday's to 2+ decimal places that is strong evidence the numerics are identical.

Deliverable: a list of every difference you found between yesterday's measurement conditions and
ours, each labelled as MATTERS / DOES-NOT-MATTER / UNKNOWN with your reasoning, plus your
independent replication number.`,
  },
  {
    key: 'ladder',
    gpu: null,
    prompt: `NO GPU FOR YOU - this task is offline analysis only. Do not run any GPU job; four other
agents own all four cards and a stray job would corrupt their timing.

YOUR QUESTION: what does the reproduced ladder actually say, and does it change any of the
conclusions or priorities written in yesterday's handoff documents?

Read, in the Primus-Turbo checkout (branch gfx1250-attn-dispatch-and-tuning is checked out):
  output/0913__opt_plan__claude/PROGRESS.md                     (the handoff, in Chinese)
  output/0913__opt_plan__claude/phase2/PLAN-4GPU-TOMORROW.md    (today's plan)
  output/0913__opt_plan__claude/phase1/RESULTS.md               (where the ladder came from)
  output/0913__opt_plan__claude/phase0/PLATFORM-ESCALATION.md   (the throttle escalation)
  docs/gfx1250-attention-tuning.md

Key things to work out from the reproduced numbers (listed in COMMON context above):

1. The handoff quantifies "VR throttle" as worth 1.65x. Our measured ratios are roughly:
   turbo shipped 59.642/28.121 = 2.12x, aiter shipped 34.565/17.204 = 2.01x,
   fused champion 24.435/13.023 = 1.88x, aiter tuned 21.684/11.269 = 1.92x,
   turbo cfgtuned 36.405/25.695 = 1.42x.
   Is the 1.65x estimate in PLATFORM-ESCALATION.md now superseded? By how much? Does the
   escalation document need to be rewritten, and what is the corrected value of the escalation?
2. PLAN-4GPU-TOMORROW.md quotes ceilings: "7.68 ms is the 7-GEMM MFMA floor, 9.2-9.9 ms is the
   real wall, 5.48 ms is the 5-GEMM floor" and "Below ~13 ms total is not Triton". Those were all
   computed at the throttled clock. Recompute every one of them for this host's clock and state
   which conclusions flip. Note especially: our fused champion is now at 13.023 ms TOTAL and its
   BACKWARD alone is 10.284 ms. Compare that to the quoted backward wall of 9.2-9.9 ms.
   If the reproduced backward is already at/near the wall the plan said was the target,
   say so plainly and say what that does to the day's ranked items.
3. The handoff insists the backward be reported on BOTH FLOP bases - nominal 5-GEMM (5.498 TFLOP)
   and issued 7-GEMM (7.697 TFLOP) - because the vendored kernel issues seven score-matrix passes.
   Recompute our fused champion's backward on both bases and state the achieved TFLOP/s each way,
   and what fraction of the machine roof that is. You will need the roof: gfx1250, 256 CUs,
   wave32, WMMA bf16. Derive it from the clock (another agent is measuring the sustained clock;
   use 2350 MHz as your working number and label it as such) and state your arithmetic.
4. Yesterday's ITEM ordering (1: aiter ASM v3 backward probe, 3: occupancy/num_warps,
   4: VALU diet, 5: udna1, 6: 5-GEMM prototype) was ranked using throttled-clock evidence.
   Given the reproduced numbers - in particular that aiter-tuned is STILL ahead of our champion
   (11.269 vs 13.023, 1.16x) exactly as it was yesterday (21.684 vs 24.435, 1.13x) - does the
   ranking change? Which items gain and which lose expected value?
5. The handoff lists "不要重做" (do not redo) items that were refuted on the throttled card.
   Flag any of them whose refutation depended on the throttle and therefore deserves a re-test
   at full clock. Be specific about which ones and why; do not just list them all.

Deliverable: a concise, decision-useful analysis. Where a document's number is now wrong, give the
corrected number. Where a conclusion flips, say so explicitly. Do not hedge everything - commit to
the reading the numbers support.`,
  },
]

phase('Verify')
const results = await parallel(TASKS.map(t => () =>
  agent(`${COMMON}\n\n========================================\n${t.prompt}`, {
    label: `verify:${t.key}`,
    phase: 'Verify',
    schema: SCHEMA,
  })
))

return results.filter(Boolean)
