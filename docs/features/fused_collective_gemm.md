# Fused collective GEMM

## Summary

The fused collective GEMM kernels (`primus_turbo/flydsl/fused_collective_gemm/`) run a dense GEMM
and an evenly split collective in one launch, instead of a GEMM, pack / unpack copies and an RCCL
collective one after another. They pass data through a symmetric-memory heap over XGMI, one tile at
a time:

- each tile raises a flag when its data is ready, and the GEMM tiles on the receiving side start as
  soon as their rows have landed;
- received rows go straight into the layout the next op reads, so there are no pack or unpack
  copies;
- weight-gradient tiles can run in the same launch, on the CUs the exchange leaves idle;
- there is no global barrier and no host sync, and results are bitwise reproducible.

| Kernel | Collective | What it computes | Status |
|---|---|---|---|
| GemmA2A (`gemm_a2a`) | all-to-all | GEMM whose output columns are split over the group; each owner pulls its column block straight into a `[rows, heads, head_dim]` (q / k / v) layout. Optional dW GEMM in the same launch. | merged, [#669](https://github.com/AMD-AGI/Primus-Turbo/pull/669) |
| A2AGemm (`a2a_gemm`) | all-to-all | Every rank pushes each destination's rows into its receive bank; the destination's GEMM consumes them as they land. Optional residual epilogue (one bf16 rounding), copy of the gathered rows and dW GEMM. | merged, [#671](https://github.com/AMD-AGI/Primus-Turbo/pull/671) |
| AgGemm (`ag_gemm`) | all-gather | All-gather of MLA latent rows `[q_latent \| c_kv \| k_pe]` fused into the q / k / v up-projections of the local heads, with `k_pe` broadcast to the heads. Optional copy of the gathered latents. | merged, [#673](https://github.com/AMD-AGI/Primus-Turbo/pull/673) |
| GemmRS (`gemm_rs`) | reduce-scatter | Latent-gradient GEMMs fused with the reduce-scatter of their partials, optional dW. | merged, [#674](https://github.com/AMD-AGI/Primus-Turbo/pull/674) |

GemmA2A and A2AGemm are each other's backward. In A2AGemm a tile walks K one source at a time in a
fixed order, so the result does not depend on arrival order.

## Support matrix

| | gfx942 (MI300X / MI325X) | gfx950 (MI350X / MI355X) | gfx1250 (MI455X) |
|---|:-:|:-:|:-:|
| GemmA2A, A2AGemm, AgGemm, GemmRS | — | ✅ | — |

| | Supported |
|---|---|
| Precision | BF16 |
| Group | 2, 4 or 8 ranks on one node |
| GemmA2A / A2AGemm | layouts `nt` / `nn`; rows in `sbhd` or `bshd` order; head dim 64 or 128 |
| AgGemm | tested with `dn` 64 and 128 |
| GemmRS | tested with `dn` 32 and 128 |

Frameworks: PyTorch. These are kernel-level entry points with no `turbo.ops` wrapper yet, and their
signatures can change.

## Usage

Each kernel is a Python launcher in its own module under `primus_turbo.flydsl.fused_collective_gemm`.
Every call takes the group's workspace and a flag region:

- **Workspace.** `workspace.get_collective_gemm_workspace(group, bank_bytes)` returns the group's
  `CollectiveGemmWorkspace`, sized with the kernel's `*_bank_bytes` helper. All four kernels share
  it: it is allocated on the first call and grows to the largest size asked for. Growing it during
  CUDA-graph capture raises, so run one eager step first.
- **Flag regions.** There are 6 regions (`flag_region` 0 to 5). Give each call site its own: flags
  only grow, so two call sites whose launches can be in flight together must not share one.
- **Ordering.** On every rank, issue all calls on one group in the same stream order. The two banks
  of the workspace alternate between calls, so they need no barrier.
- **Launch parameters** (`BM`, the number of push / pull blocks, where the dW tiles go) default to
  shape heuristics tuned on MI355X.

In the snippets, `n` is the group size and rows are in `sbhd` order (sequence-major, batch inside):
each rank holds `M = seq_local * batch` (GemmA2A / A2AGemm) or `T` (AgGemm / GemmRS) local rows, and
the gathered sequence has `n` times as many.

### GemmA2A

GEMM on the local rows, each rank's column block of the output sent to that rank as q / k / v heads
over the gathered sequence:

```python
from primus_turbo.flydsl.fused_collective_gemm.gemm_a2a_kernel import gemm_a2a, gemm_a2a_bank_bytes
from primus_turbo.flydsl.fused_collective_gemm.workspace import get_collective_gemm_workspace

# a [M, K]; b [N, K] for layout "nt" ([K, N] for "nn"), N = n * groups * (nq + nk + nv) * d.
# A rank's column block is `groups` groups of nq q, nk k and nv v heads.
# q / k / v: [n * M, groups * nq | nk | nv, d], the rank's heads over the gathered sequence.
workspace = get_collective_gemm_workspace(group, gemm_a2a_bank_bytes(M, N, 256))
gemm_a2a(a, b, (q, k, v), workspace, layout="nt", head_dim=d, groups=G, nq=nq, nk=nk, nv=nv,
         seq_local=seq_local, qkv_format="sbhd", flag_region=0)
```

For a single output, set `nk = nv = 0` and pass the tensor three times, `(o, o, o)`.
`wgrad=(a, b_w, dw)` also computes `dw = a^T @ b_w` in the launch.

### A2AGemm

Every rank's heads over the gathered sequence come back to the rows' owner, which runs the GEMM on
them, optionally adding a residual:

```python
from primus_turbo.flydsl.fused_collective_gemm.a2a_gemm_kernel import a2a_gemm, a2a_gemm_bank_bytes

# o: one [n * M, groups * nq * d] tensor, or (dq, dk, dv) each [n * M, groups * h, d].
# w [N, K] for layout "nt" ([K, N] for "nn"), K = n * groups * (nq + nk + nv) * d.
# y [M, N]: this rank's rows. residual [M, N] or None; o_gathered [M, K] (copy of the gathered A) or None.
workspace = get_collective_gemm_workspace(group, a2a_gemm_bank_bytes(M, K))
a2a_gemm(o, w, residual, y, o_gathered, workspace, seq_local=seq_local, qkv_format="sbhd",
         flag_region=2, layout="nt", head_dim=d, groups=G)
```

`wgrad=(b_w, dw)` also computes `dw = A^T @ b_w` from the gathered `A [M, K]`, with `b_w [M, N]`
and `dw [K, N]`.

### AgGemm

All-gather of the MLA latent rows, fused into the q / k / v up-projections of the rank's `hl`
heads:

```python
from primus_turbo.flydsl.fused_collective_gemm.ag_gemm_kernel import ag_gemm, ag_gemm_bank_bytes

# xq [T, Cq]: q latent; xkv [T, Ckv + dr]: [c_kv | k_pe], this rank's rows.
# wq [hl * (dn + dr), Cq]; wkv [hl * (dn + dv), Ckv]: k_nope rows of every local head, then v rows.
# q, k [n * T, hl, dn + dr]; v [n * T, hl, dv]: k[..., dn:] is k_pe, the same for every head.
# lq [n * T, Cq] / lc [n * T, Ckv]: copies of the gathered latents (for dW), or None.
workspace = get_collective_gemm_workspace(group, ag_gemm_bank_bytes(n * T, Cq, Ckv, dr))
ag_gemm(xq, xkv, wq, wkv, q, k, v, lq, lc, workspace, dn=dn, dr=dr, dv=dv, flag_region=4)
```

### GemmRS

The backward of AgGemm: the latent-gradient GEMMs of the rank's heads, with the partials
reduce-scattered back to the rows' owners:

```python
from primus_turbo.flydsl.fused_collective_gemm.gemm_rs_kernel import gemm_rs, gemm_rs_bank_bytes

# gq, gk [n * T, hl * (dn + dr)]; gv [n * T, hl * dv]: gradients of the rank's heads.
# wq_t [Cq, hl * (dn + dr)] = Wq^T; wv_t [Ckv, hl * dv] = Wv^T;
# wk_t [Ckv + dr, hl * (dn + dr)]: every head's Wk_nope^T, over an identity block that sums the
#   heads' k_pe gradients.
# d_q_latent [T, Cq] / d_kv_latent [T, Ckv + dr]: this rank's rows of the summed latent gradients.
workspace = get_collective_gemm_workspace(group, gemm_rs_bank_bytes(n * T, Cq, Ckv, dr))
gemm_rs(gq, gk, gv, wq_t, wk_t, wv_t, d_q_latent, d_kv_latent, workspace,
        dn=dn, dr=dr, dv=dv, flag_region=5)
```

`wgrad=(a, b, dw)` and `lead_wgrad=(a, b, dw)` add up to two weight gradients `dw = a^T @ b` to the
launch, such as the q and kv up-projection gradients from the `lq` / `lc` that AgGemm saved.

Complete calls, each checked against an fp32 reference: `tests/pytorch/ops/test_gemm_a2a.py`,
`test_a2a_gemm.py`, `test_ag_gemm.py` and `test_gemm_rs.py`.

## Performance

MI355X, 2 or 4 ranks on GPUs of one node, BF16, 32,768 rows over the group. Times in µs: the median
of 30 runs, then the maximum over ranks. Shapes are per rank: `M × N × K` is the GEMM a rank runs (rows × output columns × reduction);
`+ dW` means the weight-gradient GEMM of the same size runs in the same launch. "GEMM + RCCL" is the
same work unfused: the hipBLASLt GEMM(s), dW included, and the RCCL collective, each timed alone;
the total is their sum. Speed-up is (GEMM + RCCL) / Fused. The GemmA2A, A2AGemm and GemmRS tables
show selected shapes; the benchmarks below print the full sets.

### GemmA2A

The `N` columns are split over the ranks. Plain rows are the `nt` GEMM; `+ dW` rows are the `nn`
GEMM with the dW GEMM. RCCL is `all_to_all_single` of the GEMM output
([#669](https://github.com/AMD-AGI/Primus-Turbo/pull/669)).

| Ranks | M × N × K | GEMM + RCCL: total (GEMM / all-to-all) | Fused | Speed-up |
|---:|---|---:|---:|---:|
| 2 | 16384 × 10240 × 8192 | 4978.2 (1956.2 / 3022.0) | 3139.8 | 1.59× |
| 2 | 16384 × 8192 × 4096 + dW | 4008.5 (1583.3 / 2425.3) | 2564.4 | 1.56× |
| 2 | 16384 × 12288 × 5120 + dW | 6578.4 (2962.7 / 3615.6) | 3660.6 | 1.80× |
| 4 | 8192 × 14336 × 5120 | 2119.5 (908.7 / 1210.7) | 1226.2 | 1.73× |
| 4 | 8192 × 10240 × 8192 | 1856.2 (985.9 / 870.2) | 1156.8 | 1.60× |
| 4 | 8192 × 6144 × 3072 + dW | 1039.4 (505.8 / 533.6) | 620.6 | 1.67× |

### A2AGemm

The `K` columns arrive from all ranks. Plain rows are the `nt` GEMM with the residual epilogue and
the gathered copy; `+ dW` rows are the `nn` GEMM with the dW GEMM. RCCL is `all_to_all_single` of
the exchanged rows ([#671](https://github.com/AMD-AGI/Primus-Turbo/pull/671)).

| Ranks | M × N × K | GEMM + RCCL: total (GEMM / all-to-all) | Fused | Speed-up |
|---:|---|---:|---:|---:|
| 2 | 16384 × 8192 × 8192 | 4034.9 (1612.8 / 2422.1) | 2487.4 | 1.62× |
| 2 | 16384 × 7168 × 16384 | 7745.3 (2940.4 / 4804.9) | 4909.0 | 1.58× |
| 2 | 16384 × 4096 × 9216 + dW | 4428.7 (1715.6 / 2713.1) | 3133.4 | 1.41× |
| 4 | 8192 × 7168 × 16384 | 2788.0 (1405.9 / 1382.1) | 1889.1 | 1.48× |
| 4 | 8192 × 2048 × 5120 + dW | 727.7 (273.9 / 453.8) | 474.2 | 1.53× |
| 4 | 8192 × 4096 × 9216 + dW | 1686.7 (898.8 / 787.8) | 1187.8 | 1.42× |

### AgGemm

Each rank gathers all 32,768 latent rows and runs two GEMMs on them for its local heads: the q
up-projection reads the q latent (`K` = 1536), and the kv up-projection reads the kv latent
(`K` = 512) and writes both `k_nope` and `v`. The same launch also broadcasts `k_pe` to the heads
and copies out the gathered latents; run unfused, these layout copies add another 200 to 760 µs on
top of GEMM + RCCL ([#673](https://github.com/AMD-AGI/Primus-Turbo/pull/673)).

| Ranks | q GEMM | kv GEMM | GEMM + RCCL: total (GEMM / all-gather) | Fused | Speed-up |
|---:|---|---|---:|---:|---:|
| 2 | 32768 × 6144 × 1536 | 32768 × 8192 × 512 | 2123.5 (862.4 / 1261.1) | 1294.0 | 1.64× |
| 2 | 32768 × 12288 × 1536 | 32768 × 16384 × 512 | 2979.3 (1713.6 / 1265.7) | 1757.5 | 1.70× |
| 4 | 32768 × 3072 × 1536 | 32768 × 4096 × 512 | 1084.2 (413.5 / 670.7) | 723.7 | 1.50× |
| 4 | 32768 × 6144 × 1536 | 32768 × 8192 × 512 | 1557.9 (888.3 / 669.6) | 899.4 | 1.73× |

### GemmRS

Each rank computes the latent gradients of the whole sequence from its heads' gradients: the q GEMM
(`K` = the heads' q width) and the kv GEMM over `[dk_nope | dv]` (`K` = the heads' k_nope and v
width), plus the `k_pe` gradient summed over the heads. The partials are reduce-scattered to the
rows' owners. `+ dW` rows also compute the q and kv up-projection weight gradients in the same
launch. RCCL is `reduce_scatter_tensor` of the partials; run unfused, packing `[dk_nope | dv]` and
the `k_pe` sum add another 320 to 1280 µs on top of GEMM + RCCL
([#674](https://github.com/AMD-AGI/Primus-Turbo/pull/674)).

| Ranks | q GEMM | kv GEMM | GEMM + RCCL: total (GEMM / reduce-scatter) | Fused | Speed-up |
|---:|---|---|---:|---:|---:|
| 2 | 32768 × 1536 × 6144 | 32768 × 512 × 8192 | 2055.2 (723.4 / 1331.8) | 1408.2 | 1.46× |
| 2 | 32768 × 1536 × 6144 + dW | 32768 × 512 × 8192 + dW | 2762.8 (1432.4 / 1330.5) | 1756.2 | 1.57× |
| 4 | 32768 × 1536 × 3072 | 32768 × 512 × 4096 | 1074.8 (356.1 / 718.7) | 748.0 | 1.44× |
| 4 | 32768 × 1536 × 6144 | 32768 × 512 × 8192 | 1459.1 (743.6 / 715.4) | 920.7 | 1.58× |
| 4 | 32768 × 1536 × 3072 + dW | 32768 × 512 × 4096 + dW | 1486.2 (768.0 / 718.3) | 928.4 | 1.60× |
| 4 | 32768 × 1536 × 6144 + dW | 32768 × 512 × 8192 + dW | 2239.6 (1518.1 / 721.5) | 1488.8 | 1.50× |

Benchmarks (each prints, per shape, the hipBLASLt GEMM and RCCL times alone and the fused time; set
`--nproc_per_node` to the group size):

```bash
torchrun --nproc_per_node 4 benchmark/ops/training/bench_gemm_a2a.py --seq 32768
torchrun --nproc_per_node 4 benchmark/ops/training/bench_a2a_gemm.py --seq 32768
torchrun --nproc_per_node 4 benchmark/ops/training/bench_ag_gemm.py --seq 32768
torchrun --nproc_per_node 4 benchmark/ops/training/bench_gemm_rs.py --seq 32768
```

## Known limitations

- gfx950 only, BF16 only, and the group must sit inside one node: the kernels exchange through XGMI
  peer memory.
- The launchers do not check the inputs' dtype yet: other dtypes give wrong results without an
  error.
- AgGemm accepts `dn = 32`, but review of #673 reported that this width puts data in the wrong
  place, and that some `num_push` values leave rows unwritten. Use `dn` 64 or 128 and the default
  `num_push`.
- Size limits: each GEMM operand must have fewer than 2^31 elements, and one call's tiles must fit
  in one flag region (32768 flags). An A2AGemm residual needs `N % 256 == 0`; the in-kernel dW needs
  its dimensions to be multiples of 256.

## Further reading

- Design and full results: [#669](https://github.com/AMD-AGI/Primus-Turbo/pull/669) (GemmA2A),
  [#671](https://github.com/AMD-AGI/Primus-Turbo/pull/671) (A2AGemm),
  [#673](https://github.com/AMD-AGI/Primus-Turbo/pull/673) (AgGemm),
  [#674](https://github.com/AMD-AGI/Primus-Turbo/pull/674) (GemmRS)
- Tests: `tests/pytorch/ops/test_gemm_a2a.py`, `test_a2a_gemm.py`, `test_ag_gemm.py`, `test_gemm_rs.py`
- Benchmarks: `benchmark/ops/training/bench_gemm_a2a.py`, `bench_a2a_gemm.py`, `bench_ag_gemm.py`,
  `bench_gemm_rs.py`
