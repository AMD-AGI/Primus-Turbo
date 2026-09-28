# B0 (ctheliosp-1b112-a37-1) 4-card attention campaign, started 2026-09-27

Local-only work (branch `b0/0927`, never pushed; A0 pushes `dev/lhz/flydsl-attn`).

| card | container | stream |
|---|---|---|
| GPU 0 | fa-g0 | fwd op-evolve job `gfx1250-flydsl-attn-fwd-b0-20260927` (clone of A0's fwd job at round 4) |
| GPU 1 | fa-g1 | bwd op-evolve job `gfx1250-flydsl-attn-bwd-20260917-115934` (resumed at round 24) |
| GPU 2 | fa-g2 | fwd sweep lab (L22-L24, L4/L5) |
| GPU 3 | fa-g3 | bwd FUSED5 lab (P2 -> P1 -> P3/P4) |

- Isolation: each fa-gN is a `docker commit` snapshot of fa-repro (`fa-tune:b0-snap`) with only `/dev/kfd` and
  `/dev/dri/renderD{128+8N}`; verified device_count 1, PCI domain N+1, load lands only on card N (`iso/`).
  fa-repro is stopped and renamed `fa-repro-parked` so stale commands fail.
- `patches/`: every hand edit to the op-evolve job dirs (they live in another repo, untracked).
- `probe/`: interference probe (neighbour-load effect on same-process ratios).
- `LAB-RULES.md`: rules given to every lab agent. `mon/watch.sh`: silent watcher.
- `fwd-hint.md`, `bwd-hint.md`: copies of the jobs' hint.md as delivered.
