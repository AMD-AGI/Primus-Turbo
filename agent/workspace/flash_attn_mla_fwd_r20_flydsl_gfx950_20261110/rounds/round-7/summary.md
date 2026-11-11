# Round 7 — ACCEPTED — 2026-11-10 20:20 (UTC+8)

## Hypothesis
With `K_HEAD` = 1 each QK reads its first k-step of K right at its start, so the LDS latency is
exposed (ATT round 6: `waitcnt` 48 cycles at the C3 P*V -> C5 QK boundary). K buffer 0, which the C5
QK reads, is already visible after C1's cluster sync and is not refilled in C3 (C3's DMAs fill K / V
buffer 1), so its head can be read inside the C3 P*V site, under the MFMAs and the rebase check.

## Single change
`_pv_spec_site(..., k_head_buf)` appends `_load_k_head(k_head_buf)` to slot `PV_K_HEAD_SLOT` = 56
(the empty tail of the site) and returns the packs; C3 passes `k_head_buf=0` and C4 skips its own
`_load_k_head(0)`. C7 -> C1 is unchanged (that head would have to ride the loop back edge).
`CODEGEN_REV` 1041. ISA: 2 `ds_read_b128` moved from the C5 block into the C3 block; VGPR 512 / 8
spill slots unchanged.

## Results (full set, 2 rotations, `artifacts/r7.csv`, base = round-6 snapshot, same session)

| shape | cand TFLOPS | cand ms | base (R6) TFLOPS | base ms | cand / base | O rel | LSE max | bitwise |
|---|---|---|---|---|---|---|---|---|
| dsv3 B2 S4K H128 | 1029.7 | 1.3351 | 1026.2 | 1.3396 | 1.0034 | 2.01e-03 | 4.75e-03 | yes |
| dsv2lite B8 S4K H16 | 1003.1 | 0.6853 | 1002.6 | 0.6856 | 1.0005 | 1.99e-03 | 3.17e-03 | yes |
| kimik2 B4 S4K H64 | 1021.9 | 1.3453 | 1021.6 | 1.3456 | 1.0003 | 1.97e-03 | 4.01e-03 | yes |
| kimik3 B2 S7168 H96 | 1159.3 | 2.7235 | 1154.6 | 2.7345 | 1.0041 | 1.83e-03 | 4.73e-03 | yes |
| B1 S4K | 1016.0 | 0.6766 | 1013.0 | 0.6785 | 1.0030 | 1.95e-03 | 4.60e-03 | yes |
| B1 S8K | 1173.7 | 2.3422 | 1169.0 | 2.3516 | 1.0040 | 1.80e-03 | 4.75e-03 | yes |
| B1 S16K | 1263.0 | 8.706 | 1256.1 | 8.7541 | 1.0055 | 1.75e-03 | 4.93e-03 | yes |
| B1 S32K | 1305.7 | 33.6846 | 1297.0 | 33.9091 | 1.0067 | 1.61e-03 | 4.82e-03 | yes |
| B1 S64K | 1321.9 | 133.0866 | 1318.8 | 133.3946 | 1.0024 | 1.58e-03 | 5.14e-03 | yes |
| B2 S4K | 1029.7 | 1.335 | 1025.0 | 1.3412 | 1.0046 | 2.01e-03 | 4.75e-03 | yes |
| B2 S16K | 1266.7 | 17.3618 | 1262.2 | 17.4228 | 1.0036 | 1.76e-03 | 4.82e-03 | yes |

Geomean cand 1138.09 vs base 1134.18 (+0.34 %), 11/11 PASS, bitwise identical, every shape >= +0.03 %.

## Decision
ACCEPTED (+0.34 %, no regression).

## Attribution
One exposed LDS round trip (~50 cycles of ~3350 per KV tile) per iteration removed; a pure stall, so
it also converts at the B1 S16K power cap (+0.55 %).

## Real-training transfer check
- Bucket: K1 (instruction schedule). Cache key: no cache. id(...) audit: n/a.
- 4-step trace hit rate: n/a. Workload distribution: n/a (dense causal attention).
- Benchmark gain: +0.34 %. Estimated real-training gain: +0.34 % (bitwise identical, stateless).
  Decision: ACCEPT-as-real.

## Next direction
The same for C1: read K buffer 1's head in the C7 P*V site and carry it over the loop back edge (and
from the prologue into the first iteration).
