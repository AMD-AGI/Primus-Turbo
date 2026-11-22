# gfx950 / agkv_cp_attention_mega / flydsl tips

### 2026-11-22 09:57 round-1 — BASELINE
- Context: mega all-gather-KV CP attention (Primus-Turbo `attention_agkv.py`), 8 x MI355X XGMI mesh, cp 8 / 4 / 2
- Signal: the same code measured in two processes differs by up to 4 % on some cp8 shapes (l70 cp8 32K 9630 vs 10033 us), while repeats inside one process stay within 1 %
- Takeaway: compare variants interleaved inside one process (load each snapshot of `attention_agkv.py` as its own module, rotate the order per repeat) and accept on the paired delta; a single bench run per variant cannot resolve 1-3 % changes
- Applicability: any multi-GPU op whose run-to-run drift exceeds the change being measured; not needed for single-GPU kernels with < 0.5 % drift

### 2026-11-22 10:12 round-2 — ACCEPTED
- Context: FlyDSL push kernel over XGMI, units alternating between destination peers, blocks striding over units
- Signal: a grid of 48 at n = 8 (6.86 blocks per link) ran cp8 shapes 2-5 % slower than grids that are multiples of 7; n = 2 (one link) unchanged
- Takeaway: size a multi-peer push grid as a multiple of the number of links so each block stays on one link; give the forward (latency-critical) more blocks than the backward pushes that run under attention (28 vs 21 at n = 8)
- Applicability: any CU-driven peer-to-peer copy over the full XGMI mesh; recheck the counts when the unit order or per-unit size changes

### 2026-11-22 10:25 round-3 — ROLLED BACK
- Context: mega backward tail (dq copy / add after the last attention call, before the reduce kernel)
- Signal: moving a few-us copy to a side stream was +0.1 % (up to +0.9 % on 16K cp4 / cp8)
- Takeaway: the backward end is bound by the reduce waiting for peer flags; small local work in front of it is already hidden, and a stream hop (two event waits + launch on another queue) costs about as much as it saves
- Applicability: do not move sub-10-us work to side streams in front of a flag-wait kernel; check the wait duration in a trace first

### 2026-11-22 10:43 round-4 — ROLLED BACK
- Context: mega split backward, q0 (feeds the push of the lower owners' gradient slots) and q1-own (feeds only the local reduce)
- Signal: running them concurrently at equal priority sped hkv = 4 shapes 3-8 % but slowed hkv = 8 shapes 1-5.7 %; trace: q0 and its push 100-170 us later on high ranks
- Takeaway: in a push / reduce pipeline the end is set by the last push to land; never let independent tail work take compute units from a call that feeds a push. Use rocprofv3 kernel traces (host clock, all ranks) to see which push the reduces wait for
- Applicability: any overlap of local compute with a CU-driven exchange; re-try only with the push-feeding work protected (priority or ordering)

### 2026-11-22 10:50 round-5 — ROLLED BACK
- Context: round-4's overlap of q0 and q1-own, q0 and its push on a `priority=-1` stream launched first
- Signal: -0.19 % (quick); same split as round-4: hkv = 4 -4.3 %, hkv = 8 +1.4 .. +2.2 %
- Takeaway: on gfx950 stream priority orders dispatch between queues but does not take back compute units that another call's work-groups already hold; it cannot protect a push-feeding call from filler work with many waves
- Applicability: any attempt to share the GPU between a critical call and filler work; size the filler or CU-mask instead

### 2026-11-22 11:10 round-6 / round-7 — NOT ACCEPTED
- Context: mega forward, q1's remote non-causal call split into first halves (land first) and second halves, merged by LSE; round-7 gates the split to ranks with n - 1 - r > r + 1
- Signal: l8_cp8_16k -3.3 .. -3.6 % (rank 0 fwd end 619 -> 567 us, second wait 4 us), other representative shapes flat; geomean -0.66 / -0.69 % (quick); without the gate the extra call + wait + merge cost ~25 us on ranks that never waited
- Takeaway: the forward wait sits on the low ranks only (most second halves to receive, least remote q0 work before q1); splitting by arrival order recovers ~50 us of rank 0's 146 us wait at 16K cp8 and nothing at 32K+ or cp2. The remaining gap needs per-tile waits inside the attention kernel, not more host-side calls
- Applicability: multi-call schedules over a staggered exchange; each extra call + merge costs ~25 us at 16K, so split only where a trace shows the wait
