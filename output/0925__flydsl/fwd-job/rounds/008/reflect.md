# Round 8 reflect (fast)

**Not accepted.** Python's validation: geomean vs beat 0.6673 (fast 0.4894, proxy 0.7795, prod 0.7788). `op/current`
stays at round 4.

## What happened

- Two ideas, five card arms, each alone from op/current, screened at prod in 3 rotated sessions:
  - nodelay (`amdgpu-enable-delay-alu=False`): **+0.7%, 3/3.**
  - noexpert: -2.3%.
  - mmc (h12 L24): -1.2%.
  - p12s (h24 G=1 split): -3.5%.
  - p12n: -1.8%.
- g21 lost, so there was no merge. nodelay shipped to the working copy. Its output is bitwise identical to the
  champion.
- In four all-shape sessions with beat, prod was +0.62%, but fast was 0.961 and proxy 0.956 of the champion. The
  loss appears only when nodelay runs after beat in the same process (proxy 0.91). Without beat, fast is 1.002 and
  proxy 1.004 in 4/4 sessions.

## What I got wrong

- **I judged the round at prod only, and planned it that way.** The route and the pool entry said "judge at prod"
  because fast/proxy are bimodal. But the acceptance rule scores all three shapes as the benchmark measures them,
  with beat in the process. A schedule change can move the beat-preceded penalty unevenly, and this one did. The
  screen should have carried fast/proxy with beat from the start. I found the problem only after shipping.
- **Signs.** noexpert (-3 instructions) and mmc were predicted null; both lost 1-2%. p12s was predicted +1.5% and lost 3.5%,
  with the split *worse* than no-split. My census said +0.6% instructions for p12s, so the instruction count missed
  a 3.5% loss. The count is a direction test at best, and it failed as one for p12s and noexpert.
- nodelay was predicted at +1%; it measured +0.7%. That held in direction and roughly in size, at prod only.

## What did not work

- The h24 split signal/wait lever at G=1. Round 7's +3-point split gain was specific to the G=2 loop. I record the
  numbers but not the mechanism of the reversal. My guess (exposing partner-wave lag where the barrier absorbed it,
  or the proto ring's fences and layout) is unverified. r8.i2.g21 stays in the pool: I may move only the candidate.
  Its route outcome says lost, and the next reflect that owns it should retire it.

## For the next round

- nodelay is a free prod +0.7% whose blocker is the beat-preceded fast/proxy state, not prod.
- The fast/proxy-after-beat penalty (round 6, still unexplained) is now the thing that decides acceptance for
  small schedule changes. Understanding it is worth more than another 0.7% at prod.
