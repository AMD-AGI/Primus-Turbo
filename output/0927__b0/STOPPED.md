> **Stale (day 1).** This note records the 2026-09-27 15:43 UTC stop. The campaign resumed on 2026-09-28; current state: `HANDOFF-A0.md`.

# B0 campaign stopped 2026-09-27 ~15:44 UTC -- the machine was taken over

At ~15:43 UTC all four per-card containers fa-g0..fa-g3 were REMOVED by someone else (`docker ps -a` no longer
lists them; a new container `gfx1250dbg_maxtext_g31fp8_L6_at4` started right after). Per the user's rule we shut
everything down and did not recreate or start any container:
- `op-evolve stop` + TERM for both loops; watcher, auto-resume, patrol cron, ruler-audit agent stopped;
  orphaned agent poll loops (docker exec fa-g0 / dmesg -w) killed by PID. No KFD process of ours remains.
- `fa-repro-parked` (the original fa-repro, stopped and renamed at campaign start) is untouched. To restore it:
  `docker rename fa-repro-parked fa-repro` (only when the machine is ours again).

## Job state at stop
| job | round in flight | champion | notes |
|---|---|---|---|
| bwd `gfx1250-flydsl-attn-bwd-20260917-115934` | r25 deep, act at 04_report (arm g77, predicted <= +1.8%) | r20 (unchanged) | r24 rejected 0.9665; resume redoes r25 act (profiling+plan kept). Spec: runner container fa-g1 -> must be re-pointed (e.g. fa-repro) before any resume |
| fwd `gfx1250-flydsl-attn-fwd-b0-20260927` | r9 fast, opt just started | r6 (speculative softmax, prod 1524 TF/s, 84.8% ASM) | r6 gain pending the ruler audit (h35 bimodal timing); h37 re-land m16x8 gate (fast +29%) queued; spec container fa-g0 -> re-point before resume |

## Open items for the next session
1. Ruler audit (bimodal prod timing ~1.44 vs ~1.52 ms for identical code) was stopped mid-way; partial work in `ruler/`.
   Round 6 (+5.7%) and lab L12 (+6.4%) must be re-validated under a fixed ruler.
2. Lab L12 4-wave x 2 WG/CU: prod +6.4% but proxy -6.8% (`lab2-L12/VERIFY.md`).
3. op-evolve `evolve.gain_weights` (prod-weighted acceptance, as A0 r10) not applied on B0 (needs user approval
   to edit op-evolve source).
4. `~/.op_evolve_openai` absent on B0; reviewer switched to claude in both specs.
