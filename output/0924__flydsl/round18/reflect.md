# Round 18 (fast) -- reflect

**Not accepted.** prod 469.6 = 93.2% of its best ever 504.0. `op/current/` still holds round 17.

## What happened

Candidate `r18.i3.g56`: take `k_dkdv`'s 4 LSE/delta `buffer_load_b32` off the in-order LOADcnt
FIFO with `global_load_async_to_lds_b32`, which retires on ASYNCcnt instead. Chosen because this
round's new instrument (`attrib_ss.py`, steady-state stall attribution) put **470 of `.LBB0_8`'s
562 stall cycles on that one site**.

It took three builds. Build #1 emitted 6 async loads and **0** `s_wait_asynccnt` -- the backend
does not derive the wait from the memory dependence because the `inttoptr` LDS address is opaque
to it. Build #2 passed all four offline screens with `spill: 0` and was **silently wrong**:
dk/dv at **-56 / -73 dB**. Build #3 fixed it and passed `op/validation.py` (dk/dv 52.60 / 52.71 dB
at prod, determinism x200 bitwise).

Then it lost: **prod -6.84%, proxy -8.40%, fast -5.53%**, same sign on all three, against a 0.36%
same-code floor.

## What I got wrong

**1. I applied the wrong hardware contract, and the screen could not catch it.**
`global_load_async_to_lds_*` takes a **per-lane** LDS destination with **no implicit lane
stride**. I passed `_ldslot(qt, gh)` -- uniform across lanes -- so 32 lanes collided on the same
4 bytes and 31 values were dropped. That is the gfx90a/gfx940 `global_load_lds` contract (uniform
base via `m0`, hardware strides by lane) applied to an op that does not have it. Worse: my own
comment in the same function said "per-lane VGPR LDS destination confirmed by asyncprobe". I
verified the capability and then did not use it.

**2. I named the wrong suspect before measuring, and said so with confidence.**
Before build #2 went on the card I wrote that the `v4`/`v5` WAR drain at the top of `.LBB0_8`
"may cost more than the 470 cycles removed" and was "the first thing to look at" if it lost.
Build #3 has no such drain -- 2 `s_wait_loadcnt 0x0`, identical to the incumbent -- and lost
anyway. The suspect was cleared and the crime still happened.

**3. The bigger one: I treated a stall attribution as a price.**
Build #3 deletes the source instruction of the site that owned 83.6% of the body's stall.
`buffer_load_b32` 4 -> 0, `s_wait_loadcnt` 7 -> 3, body 678 -> 667 instructions, VGPR 724 -> 718,
`v_wmma` unchanged at 64. **Every static number I have moved the right way and the kernel got
slower by 6.84%.** This is the third time in this job (after `g47` and `g49`), but the first
where the compiler's scheduling was *not* constrained -- so "I broke the scheduler" does not
explain it. The attribution tool locates stalls correctly and prices them wrongly, and I had no
instrument capable of noticing the difference before spending the card time.

## What I do not know

Why it lost. The leading hypothesis -- the async path adds an HBM->LDS->VGPR round trip for 16
bytes per iteration that previously went HBM->VGPR and was read out of a register, moving the
wait off LOADcnt and onto DScnt, this kernel's tightest queue -- is **unproven**, and I gave no
static cycle attribution in `opt.md` because the tool files async loads in the LOADcnt queue and
would have produced a confident fiction. Settling it needs a fifth (ASYNCcnt) queue in
`attrib.py`. That is `route.md` row 5 and it is now the gate in front of `r18.i5.g58`, which
moves 32x more bytes down the same path.

## Incidental

A GPU page fault (`GCVM_L2_PROTECTION_FAULT`, TCP client, IH ring overflow) during the ISA dump,
under round 2's sixteen-round-old `screen.py`, after it had printed its complete JSON. The same
binary had already cleared x200 determinism and 3 shapes x 3 slots x 51 iters with zero faults.
Attribution unresolved; leaning harness teardown, not the kernel. Killed explicitly, card
returned to idle, no power cycle needed.
