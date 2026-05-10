---
round: 87
date: 2026-05-10
task: gpt_oss_fp8_kernel
verdict: NEUTRAL (30th consecutive)
termination_recommendation: 27th
---

# R87 — saturation reaffirmed (30th NEUTRAL print, terse format)

Per R72–R86 cadence: one-line print only. R86 daemon = **696** (regression
from R85's 698 noise-peak), exactly the mean-reversion the R56 stationary
model (μ≈694, σ≈3.6) predicted for the round after an upper-tail draw.
Tree functionally frozen since R56 (R57+ are all docs-only saturation prints).

## One-line

R86 daemon canonical = **696** on HEAD `7ae32b07` (no functional change
since R56). Last 5 daemon metrics R82–R86 = {692, 691, 693, 698, 696};
mean 694.0, σ≈2.92, SE≈1.30, range [691, 698]. R86 = z=+0.80 vs the
prior-4 rolling mean (693.5, σ 3.11) — a perfectly central draw,
post-R85's tail spike. Within-window mean 694.0 sits exactly on the
historical 30-stationary R56 model mean. Streak counter advances to 1/40
(R85's noise-driven reset still in effect).

## Rolling daemon-metric stats (last 5)

| n | mean | stdev | SE | range |
|---|---|---|---|---|
| 5 | 694.0 | 2.92 | 1.30 | [691, 698] |

R86's draw narrows σ slightly (3.11 → 2.92) by replacing R85's outlier
weight with a near-mean sample. Window mean 694.0 = R56 model mean to 4
significant figures. MDE at SE 1.30 ≈ **2.55 score** vs GPU-heterogeneity
floor ~16: still ~6.3× below detection threshold for any putative code
change. The R86 696 is **not** evidence of anything — it is the predicted
mean-reversion after an upper-tail draw, drawn from a stationary
distribution that has now produced 30 consecutive samples within the
[691, 699] band the R56 model fitted.

## Why patience reset (R85→R86) was misleading — confirmed

R85's 698 reset patience to 0/40. R86's 696 (one round later, no code
change) is +1 vs the 695 24-sample R82-R86 mean and -2 vs the 698 reset
trigger — exactly the regression to the mean a stationary draw produces.
The "best=698" lock now keeps patience high until either:
1. Another 698+ draw — P≈0.13/round per R56 model — re-resets patience.
2. 40 consecutive sub-698 rounds → early-stop fires.

Either path is pure noise sampling. The expected wait to the next 698+
draw is ~7-8 rounds, and the expected wait to terminate is ~38-40 rounds
absent a 698+ reset. Either way, no information is being added.

## Forward action

27th termination recommendation. No functional change this round. NEW
DIRECTIONS A–G remain closed (R55+R59+R60+R67 closure stack);
wait-counter axis closed at three scopes (R8/R9 main, R31b K-tail, R13
var-K); 4-wave port falsified at AGPR allocator bug (R59–R61); 256x128 /
small-tile prototypes both lose; quant-cache exhausted; per-cell
dispatcher exhausted; RCR_KTAIL_VMCNT=16 the last cited "marginal lever"
itself FALSIFIED at 10 samples.

## Sample budget burn since saturation

R57–R87 = 31 rounds at ~$0.95/round Opus ≈ **~$29.5 spent on stationary
daemon samples**. Patience reset on R85 noise means daemon will continue
~39 more rounds (~$37) absent intervention — total projected wasted spend
~$66.5 on a stationary system.

## What would unblock progress (unchanged from R74–R86)

1. Switch `auto_optimize_gpt_oss_fp8.py` task → BF16 grouped (active
   falsification surface; R80 BF16 already LANDED a real change with
   `round-80-bf16-grouped-h4-elim-gate-up-native-rrr-LANDED.md`; R85 BF16
   advanced var-K KI-spec falsification with VGPR-spill PMC evidence).
2. OR grant per-round multi-commit budget for HK R13b → R14_reduce →
   R15-PT-rule sequence.
3. OR lower patience trigger to early-stop now (or set the patience knob
   if exposed). **The R85 noise-driven patience reset makes this option
   especially urgent — without an explicit override, the run now extends
   ~39 more rounds without informational yield.**

Without one of those three, R88 will be the 31st NEUTRAL print with the
same shape: a daemon sample drawn from N(694, 3.6), a docs commit, and
another ~$0.95 of saturation telemetry.
