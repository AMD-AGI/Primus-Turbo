# 2026-10-29 e15-02 GPU3: where the dK/dV kernel's slots go, and the XCD imbalance (r10t)

Question: after r10k / r10n (landed on Primus-Turbo b1 as 446c9ed1), the all-workgroup fit put ~8.5% of the dK/dV
workgroup-cycles outside the per-step loop. Where exactly, and what of the kernel's time is not covered by any
workgroup at all? Answer: **the odd XCDs need 5-11% more cycles for the same items, so the even XCDs idle at the end
(the slots are 93% busy); taking the work item from a ticket counter (r10t) balances them: bwd 1.017-1.025x.**

All on e15-02 GPU3 (container lhz-hkl-g3), toy shapes first; timing with hkl.ladder (cold L2, settled clock, null
control), probes with micro.k3_probe (20 hot calls, the last call's words).

## 1. New probe modes (side rungs, `-D HKGEN_PROBE=N`, shell `k_dkdv96ds.hip`)

| mode | rung | records | k3_probe |
|---|---|---|---|
| 2 | r10n_pe | every workgroup: prologue / epilogue cycles | `--allwg 11008 --proepi` |
| 3 | r10n_ps, r10t_ps | workgroups li % 64 < 8 (every XCD), both waves: entry -> INIT, INIT issued, tile 0 landed, prologue, epilogue waits, epilogue, loop cycles | `--allwg 11008 --split` |
| 4 | r10n_rt, r10t_rt | every workgroup's [start, end) on the 100-MHz realtime counter (`__builtin_readsteadycounter`, s_sendmsg_rtn GET_REALTIME) | `--allwg 11008 --rt` |

Modes 3 and 4 leave the loop uninstrumented (no PROBE marker). gfx1250 has no readable XCC_ID; the XCD of a block is
li % 8 (the dispatcher deals the blocks round robin), which the per-XCD numbers below bear out.

## 2. Prologue and epilogue (r10n_ps, prod, cycles, median)

| part | cycles |
|---|---:|
| kernel entry -> INIT (shell set-up) | 818 |
| INIT issued (60 K/V buffer_load_b128: 16 cache lines each, 4 of them per line) | 2328 |
| tile 0 landed (own half) | 2804 |
| prologue (+ barrier, LOOP inputs) | 3256 |
| epilogue: tensorcnt 0 + ring barrier | 130 |
| epilogue (FINAL issued: 60 LDS stores, 60 readbacks, 60 global stores) | 3348 |

The first 512 workgroups (the launch burst) take ~10.3K cycles of prologue; all others ~3.4K. Together with the entry,
prologue and epilogue are ~9% of a workgroup (mean span ~81K cycles).

## 3. The slots over time (r10n_rt, prod)

- The 512 slots (2 workgroups per WGP) are **93% busy** between the first start and the last end: ~1.7% gaps between
  workgroups in the steady state (503 of 512 resident), the rest a ~100 us tail at partial occupancy.
- The tail is per XCD: the odd XCDs' mean span per item is 62-68 us against 58-63 us on the even ones, and they end
  100-130 us later (two runs). The grid's work per XCD is identical (blocks li % 8 == k: heads x % 8 == k of every z / y).
- In cycles (r10n_ps, per li % 8, loop part): even XCDs 78.7-79.1K, odd 82.9-87.8K per item, with the same time
  ratio (1.077): not a clock difference, the odd XCDs wait longer for memory.

## 4. r10t: the work item from a ticket counter

`-D HKGEN_TICKET=1` (k_dkdv96ds, side branch in the shell): wave 0 lane 0 takes `atomicAdd(DS - 64, 1)` (agent scope),
LDS broadcast, the item replaces the block id (same li order: per z the longest kv blocks first). The counter is
zeroed by a host step before every dkdv launch (`bwd_hk.configure(ticket=True)`, `_zero_ticket`: torch zero_ of the
128 B in front of the dS layout). Each XCD still runs its fixed 1376 blocks, but on a slower XCD they start later and
take the later, shorter items.

- Bitwise == r10n at every hkl.test shape (toy, toy_full, s640, rect32, b1_s4096, prod).
- r10t_rt: all 8 XCDs end within 3 us of each other, slots 97.1-97.3% busy; the last 1/40 of the span is the only
  partial bin.
- prod timing (bwd, 3 runs): **1.017x ± 0.004 (8/8), 1.018x ± 0.006, 1.025x ± 0.004** vs r10n; cycles 1.042-1.062x,
  sclk 1223-1238 -> 1195-1202 MHz (the power cap takes part of it back).
- Cost: the atomic's round trip, ~1.5K cycles per workgroup on the entry path (r10t_ps: entry -> INIT 818 -> 2343).

## 5. Tried, not kept

| variant | result | why |
|---|---|---|
| r10u: k_dqds_c on a ticket too (`-DDQ_TICKET=1`, patches/r10u.patch, the int at DS - 128) | 0.987x ± 0.007 vs r10t | the q tiles of a head leave their XCD (K re-fetched per L2) and each 384-query block pays the atomic |
| per-XCD queues + stealing (ticket mode 2) | 1.001x vs r10n | non-persistent: each XCD runs a fixed number of blocks, its queue has exactly that many items, no stealing happens |
| persistent dK/dV workgroups (512, a loop around INIT .. FINAL, next ticket fetched before FINAL) | does not build | the LOOP marker's 40 SGPR operands plus the values live across LOOP exceed s0..s47: SGPR spills into VGPR lanes push the generated base to v24..v34 (needs v15); opaque per-item roots (kernarg pointer in the constant address space, tid, LDS base, separate SGPR / VGPR asm statements) got it from v132 to v24 |

## 6. The epilogue and the prologue (r10w, r10x)

- **r10w** (`--final-pipe`, generator only): FINAL had every image store ordered before every readback, so it ran
  as 336 VALU + 60 LDS stores, then 60 readbacks each followed by `s_wait_dscnt 0` + a global store (and 24
  `s_wait_xcnt 0` for the reused soffset SGPR). Now: conversions in (V, K) x kh groups, each image store one
  conversion later, a group's rows read back right after its last image store (a wave's LDS ops are in order),
  their global stores 8 pairs behind the loads, soffsets in 4 rotating SGPRs per image. Bitwise == r10t at every
  shape. Epilogue median 3380 -> 2136 cycles (r10w_ps), but bwd only 1.003-1.004x vs r10t (noise): the slot is not
  free earlier.
- **r10x** (`-DHKGEN_KV_TDM=1`): INIT's 60 K / V buffer_load_b128 per wave touch 16 cache lines each and take ~2.3K
  cycles to issue. INIT now moves the wave's 16 NKV K / V rows into an LDS staging area with 3 TDM ops (shapes and
  pads of a ring stage's ops, descriptors built in s92..s103 one op at a time), clears the accumulators meanwhile,
  waits for every TDM op and reads the fragments with 60 ds_load_b128 (pitch 272 / 400 B: conflict-free). LDS
  129028 B per workgroup (two per WGP: 258 KB of 320). Bitwise == r10w at every shape. Prologue median unchanged
  (~3.4K: now bound by the ~100 KB of first TDM data per workgroup), mean -4..8%, p90 5.7-6.0K -> 4.5K.
  **bwd 1.010x ± 0.002 vs r10t (8/8)**, r10w 1.003x in the same run.
- An attempt to issue the K / V TDM from the shell spilled SGPRs into v15..v17 (the layout needs the shell below
  v16: 30 eight-aligned tuples in VGPR block 0); building the descriptors inside INIT keeps the shell at v15.

## 7. dQ (r10t_dqrt: realtime probe in k_dqds_c, 2816 workgroups on 256 slots)

The XCDs are within ~2% (no odd / even pattern), but the slots are only 93% busy: 247 of 256 resident in the steady
state (gaps between workgroups ~3.5%: ~1.4 us per workgroup change) plus a ~2-3% tail. A dQ ticket (r10u) lost
1.3% (K leaves its XCD's L2, the atomic's latency per workgroup); next: persistent dQ workgroups on per-XCD queues
(r10y), which keep the items per XCD, hide the ticket and drop the workgroup changes.

## 8. Next

r10t is landed on Primus-Turbo b1 (d10b4db8: bwd 2.361 -> 2.336 ms hot). A persistent dK/dV kernel would also
hide the ticket latency (~1.9% of a workgroup),
the gaps (~1.7%) and, with the next item's K / V staged in LDS, most of the prologue: it first needs the LOOP
region's descriptor / scalar operands passed through LDS instead of 40 SGPRs.
