{{preamble}}

## Step 3: panel metrics -- UNAVAILABLE ON THIS MACHINE

**Do not run anything in this step. Do not install anything.** Write
`{{out_dir}}/3-kernel-metrics/provenance.yaml` with `status: skipped` and the reason below,
reply, and stop. This step is not part of the module's spine: the round continues past it,
and every later step must treat panel metrics as absent rather than pending.

`rocprof-compute` is **not installed here and will not be installed**, and that is a decision
already taken, not a gap to close:

- On gfx1250 its panels collect almost nothing. The LDS panel asks for 152 counters and 1 is
  available on this part; the L2, EA, UTCL1 and TXD panels get 0. There are no derived metrics
  on this architecture at all -- `FETCH_SIZE`, `WRITE_SIZE`, `MfmaFlops`, `VALUBusy`,
  `OccupancyPercent`, `LdsBankConflict`, `MemUnitStalled` have no gfx1250 definition -- so the
  Speed-of-Light panel has nothing to fill.
- It overrides the runtime counter table through `ROCPROFILER_METRICS_PATH`, so it would
  request counters the driver accepts and returns zero for. On a part where unknown counters
  fail silently, that amplifies the one hazard this module exists to avoid, and it does it
  where you cannot see it.
- Installing it would mean writing into `/opt/venv` in a **shared** container, which the job
  spec forbids (`owned: false`).

The evidence for the panels' emptiness is in
[`{{knowledge_dir}}/arch/gfx1250/profiling-surface.md`](../../../../knowledge/arch/gfx1250/profiling-surface.md).

`tools/sol_chart.py` and `tools/mem_chart.py` are also unusable here for a second, independent
reason: they import `matplotlib`. As of 2026-09-24 it IS available, but ONLY under
`PYTHONPATH=/home/lihuzhan/.local/pyviz`; it is deliberately NOT in the container's own site-packages and must not be
installed into it.

**What replaces this step.** Nothing does, completely. The questions it would have answered
are answered partially by step 2 (counters; PC sampling is forbidden) and step 4 (the
instruction trace, which since the 09-29 reflash gives per-instruction stall reasons for FlyDSL kernels on this card). Cache hit rates, byte traffic and matrix-unit utilisation are
simply **not measurable on this card** -- say that in the summary's `non_findings` rather than
leaving a gap that reads like an oversight.

## Reply with

```yaml
status: skipped
reason: rocprof-compute not installed and not installable here; gfx1250 panels collect ~nothing
kernels_done: []
charts: []
installed_anything: false
```
