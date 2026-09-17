# beat provenance

`op.target.beat`: *"aiter prebuilt gfx1250 ASM backward via
`tools/gfx1250/asm_bwd_launcher.py`, `dkdv_heads=q` plus the host reduction — must be beaten
by 0%"*. That is what is behind `beat/impl.py`. It was available, it builds and it runs, so
the "stop and report if the target is not available" rule did not trigger.

## Library and version

The arm is two pieces from two repositories.

| piece | library | version |
|---|---|---|
| the kernels themselves | **aiter**, prebuilt gfx1250 ASM `.co` objects under `<aiter>/hsa/gfx1250/fmha_v3_bwd` | commit **`6963ae9d77acb207d8f03fc92dddd8a57faf8fa6`** (2026-09-17 10:22:53 +0300), local checkout `/home/lihuzhan/code/aiter-src` |
| the launch machinery (kernarg packing, `HipModule`, `asm_backward`) | **Primus-Turbo**, `primus_turbo/pytorch/kernels/attention/_asm_bwd_kernargs.py` | commit **`5690a771874a0fe8ab6277c245a411a1f746e9ff`** (2026-09-17 12:01:15 +0000), local checkout `/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo` |

The three `.co` stems used: `bwd_hd128_odo_bf16`,
`bwd_hd128_bf16_causal_br_a32_pssk`, `bwd_hd128_dq_convert_bf16`.

## How it is loaded, and why that way

`_asm_bwd_kernargs.py` is loaded **by file path** via `importlib.util.spec_from_file_location`,
exactly as `tools/gfx1250/asm_bwd_launcher.py` (the file `op.target.beat_launcher` names)
loads it. Importing it as `primus_turbo.pytorch.kernels...` instead pulls
`primus_turbo.pytorch.__init__`, which reaches
`primus_turbo/flydsl/attention/flash_attn_bwd.py:27` → `import flydsl.expr.buffer_ops`.
**flydsl 0.3.2 deletes that module outright**, and the baseline requires 0.3.2. So importing
primus_turbo as a package in this process would break the baseline, not just the beat arm.
Loading by path sidesteps the package `__init__` entirely.

## Configuration, and the one thing it costs

`dkdv_heads="q"` is what `op.target.beat` specifies. The ASM kernel then writes dK/dV as
`[B, Skv, Hq, D]` bf16 — one slice per *query* head — and the GQA reduction is the caller's
job. `beat/impl.py` does it on the host:

```python
dk = dk_q.view(b, skv, hkv, g, d).sum(dim=3).to(k.dtype)
```

**Report this honestly: that host reduction sums four bf16 slices, and it costs precision.**
Measured against `op/eager/`, the beat arm's dq lands around 52.5 dB but its **dk/dv land at
50.2–51.0 dB** — only just clear of the 50 dB gate, against the baseline's uniform ~52.5 dB.
This is a property of the specified configuration, not a defect introduced here.

## Timed region

Allocation is cached per shape (`dq_acc` fp32 `[B,Hq,Sq,D]`, `dk`/`dv` bf16 `[B,Skv,Hq,D]`),
built at first call and reused, so no allocator traffic sits inside the timing loop. But
`dq_acc.zero_()` **stays inside** the timed region: the ASM dQ path accumulates into it and
requires it zeroed on every call, so skipping it would be measuring a kernel that computes the
wrong answer. The host GQA reduction is likewise timed, because it is part of producing the
result the API promises.

## Measured, same session, same runner

`op.target.beat_measured_same_run: true`, so these are the numbers `validation.py` re-takes
every round; the figure below is the round-0 record, not a stored anchor.

| shape | beat median | TFLOP/s |
|---|--:|--:|
| fast  | 0.1054 ms | 51.0 |
| proxy | 0.5947 ms | 577.9 |
| prod  | 7.6134 ms | 722.2 |

`op.target.beat_reference_figure` quotes 10.160 ms at the production shape as a sanity check.
The same-session measurement is **7.613 ms** — faster than the quoted figure, not slower, so
nothing here is being flattered by a stale anchor.
