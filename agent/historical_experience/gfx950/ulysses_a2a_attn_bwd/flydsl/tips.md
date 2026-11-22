# ulysses_a2a_attn_bwd (FlyDSL, gfx950) historical tips

Kernel: `primus_turbo/flydsl/mega_attn/a2a_attn_bwd_kernel.py` (fused Ulysses a2a(dO) + causal attention
backward + a2a(dq, dk, dv), symmetric-memory exchange over XGMI on one MI355X node; the attention is the
stock FlyDSL dK / dV KV-outer body run per work item). Campaign `mega_attn_a2a_bwd_flydsl_gfx950_20261122`:
bwd geomean over qwen3-30b-a3b n4 and llama3.1-70b n8 at 32K / 64K went 11493.2 -> 10536.6 us (-8.3%);
speedup vs the unfused a2a + attention + a2a 1.00 -> 1.06-1.14x. Correctness gate: dK / dV bitwise vs the
stock backward, dQ within tolerance. Times below are UTC.

### 2026-11-22 07:33 round-1 — BASELINE (analysis)
- Context: items (band, split, kv head) waited for every dO row above their band before starting; the dO
  pull ran top down; `num_early` top bands were claimed first to fill the start.
- Signal: fused = unfused; items ran 5% slower than in attn_only after the pull; band 0 needed all dO.
- Takeaway: an item-level wait on a causal suffix forces a trade between the start bubble and the tail;
  `num_early` sweeps were non-monotonic within 1.5%.
- Applicability: any fused causal backward whose inputs arrive progressively.

### 2026-11-22 07:58 round-2 — ACCEPTED
- Context: each item's suffix wait ended with an agent-scope acquire.
- Signal: removing the per-item acquire (a watermark of rows already acquired) -0.6%.
- Takeaway: on gfx950 an agent acquire (`buffer_inv sc1`) invalidates the XCD L2 under every resident item.
- Applicability: every persistent kernel that acquires per work item.

### 2026-11-22 08:50 round-4 — ACCEPTED
- Context: the claim loop held comm bases / descriptors and the dQ push / gather code across the stock body.
- Signal: 34 SGPR / 139 VGPR spills, 10 scratch reloads in the body's 1280-MFMA main loop; rocprofv3
  GRBM clocks equal, MFMA busy equal, SQ busy +13% (stalls, not clocks).
- Takeaway: count `scratch_load` in the hottest MFMA block of the final ISA (`FLYDSL_DUMP_IR=1`, cache
  off). Keep 64-bit bases in an LDS stash read at each use (readfirstlane), not in registers (-4.2%).
- Applicability: any kernel that embeds a register-heavy body inside a persistent claim loop.

### 2026-11-22 09:10 round-5 — ROLLED BACK
- Context: claiming the shortest (top) bands last to even out the end.
- Signal: worse at every setting (+0.3..+7%).
- Takeaway: the end is set by dQ rows waiting on the last-finishing bands, not by band order.
- Applicability: LPT orders for causal bands.

### 2026-11-22 09:33 round-6 — ACCEPTED
- Context: replace the item-level suffix wait with a per-q-block readiness hook inside the body (before each
  q block's first dO / delta fetch: an LDS "all pulled" word, else a relaxed poll of that pull unit's flag);
  pull the first wave's diagonal rows bottom up first, then top down; num_early = 0.
- Signal: first version slower (13 reloads in the main loop) — the timeout `fx.printf` inside the hook.
  Without printf: 1 reload; in-body waits ~0; -3.6% (both 64K shapes below the standalone stock bwd).
- Takeaway: never put `printf` (hostcall) in code inlined into a register-heavy loop, even on a cold
  timeout path; trap only. A fine-grained data-readiness hook beats any coarse claim-order trick.
  The pull order must match what the first wave of items reads first (`num_prefix=0`: +2..4%).
- Applicability: every fused producer -> consumer kernel where the consumer walks the data progressively.

### 2026-11-22 10:05 round-7 — ROLLED BACK
- Context: push dQ units between items (capped concurrency) instead of all at the end.
- Signal: single-call trace -1.3%, but same-session A/B neutral (qwen) / +2% (llama n8).
- Takeaway: end-phase CUs idle anyway while the last items finish, so end-phase pushes are free; pushes
  between items delay the next item wave. Per-rank GPU speed differs by up to 7%: single-call trace deltas
  below ~2% are noise — decide with alternating A/B benches.
- Applicability: any schedule that moves link-bound work earlier.

### 2026-11-22 10:21 round-8 — ROLLED BACK
- Context: fold the per-call prep launches (Q prescale, LSE transpose-prescale, dQ image zero) into the kernel.
- Signal: neutral; the Q prescale's 128 MB of local traffic on the pull blocks delayed the pull 399 -> 439 us.
- Takeaway: HBM-bound prep costs the same wherever it runs; on the producer blocks it delays the consumers.
- Applicability: prep folds into fused kernels.

### 2026-11-22 11:00 round-9 — ACCEPTED
- Context: fold only the LSE transpose-prescale (into the pull) and the dQ image zero (into its push).
- Signal: A/B 3x at 32K -0.17..-0.35%; geomean -0.08%.
- Takeaway: fold small launch-dominated prep and work that can run on idle end-phase CUs.
- Applicability: per-call buffer resets of persistent fused kernels.

### Remaining headroom (qwen 32K)
- Bwd - AttnOnly ~330 us: comm work on CUs (stage ~9K, pull ~13K, gather ~6K block-us), the link-bound
  push of the dQ rows that depend on the last item wave (~80 us tail), cross-rank coupling to the slowest GPU.
- The tail would need finer items at the end (another q_split), which breaks the dK / dV bitwise match.
