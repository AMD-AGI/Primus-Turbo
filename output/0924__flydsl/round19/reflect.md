# Round 19 reflect -- not accepted

`op/current/` still holds round 17. Throughput 0.9939x its own best ever
(score 0.82478 against the incumbent's 0.83107). That is the correct outcome: the arm I
shipped is a null and I said so before Python did.

## What happened

Two arms, built separately from `op/current`, measured apart, both correct
(52.52-52.83 dB, determinism x200), both validated before any benchmark ran.

- `r19.i1.g59` -- WMMA A-operand reuse hint. prod **+0.08%**, proxy **-0.10%**, inside a
  same-code floor of 0.61% / 1.42%. Shipped as the round's best arm; carries no speed.
- `r19.i2.g60` -- remove `r7.i1.g21`'s prefetch tuple. prod **-17.27%**, proxy -15.13%. Dead.

Zero-card-time work that cost no arm slot: the fifth (ASYNCcnt) queue for the attribution
model, an instruction-class census of `k_dkdv`'s hot body, and a calibration of the static
model against the wall clock (~1809 real cycles/iteration vs a modelled 1685).

## What did not work, and what I misjudged

**The reuse hint (g59).** I misjudged this one by not writing down what I expected it to buy.
I verified it would be *legal* -- the incumbent's emission order alternates A every
instruction, so the hint could never have fired -- and then treated legality as if it implied
throughput. It does not. The hint saves an operand fetch; this body's cost is ~686
non-issuing cycles per iteration, and it has no operand-fetch problem. 56 of 128 WMMA carry
`matrix_a_reuse` in the shipped ISA and the time did not move. I should have priced the
mechanism against the bottleneck before spending an arm on it; that check was one paragraph
of arithmetic and I skipped it because the change was cheap to write.

**The register-window tax (g60).** This one I got wrong in a more useful way: I predicted the
direction and the magnitude from the gfx950 `v_accvgpr` analogue (1.72x) and the answer came
back 17.3% the *other* way. The arm was built to be falsifiable and it was falsified. The
cost of the 128-VGPR tuple is real in the instruction count and is not on the critical path;
what the tuple buys is latency cover, which is worth ~21% on today's body, not the +8.3% it
landed at in round 7.

**The thing I keep getting wrong at the round level:** I ranked both arms with static
reasoning, and the static reasoning has now been wrong four times in a row about which way
time moves (`g47`, `g49`, `g56`, and this round's own upgraded model, which priced `g56` at
20% faster after it had measured 6.84% slower). Round 18 said the model needed a fifth queue;
I built the fifth queue and the model got *worse*. The honest reading is that a model
assuming issue order equals completion order cannot price a latency-bound body at all, and
that is now recorded as a mechanism rather than as another tuning note.

## What the round is actually worth

Three negative results with measured arms behind them -- not compute, not HBM, not issue
slots -- which is what raised `bound: latency` from LOW to MEDIUM confidence. And one
untested hypothesis with a clean control: `k_dkdv` runs 1.25 LDS ops per WMMA against
`k_dq`'s 0.33, and `k_dq`'s body has no stall. `r16.i3.g50` is the arm that tests it. It is
still in `pool.md` and it is still unpriced, because gfx1250 reports bytes at no level.

## The one thing I would hand the next round

Do not open with `g50`. Open with its precondition: 32 redundant `buffer_load_b128` in
`.LBB0_8`, held live by a runtime-false select, semantics provably unchanged, measured at
prod. Free loads mean L1 has the headroom `g50` needs; loads that cost mean `g50` is dead
before a line of it is written. That probe takes no arm slot and it cannot lie, which is more
than can be said for anything else this kernel has been ranked with.
