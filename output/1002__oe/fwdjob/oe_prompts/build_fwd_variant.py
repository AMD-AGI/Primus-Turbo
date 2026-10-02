#!/usr/bin/env python3
"""Build the fwd variant of op-evolve's deep-round prompts and the patch that installs it.

    python3 build_fwd_variant.py        # stdlib only, CPU only, never touches the OE repo

Inputs (staged next to this file by the operator on 2026-10-02):
  head/<path>  the 7 files at OE HEAD (ee16d5b, branch lhz/gfx1250)
  bwd/<path>   the same files with output/0930__bwd/oejob/oe_uncommitted_0930.diff applied (the bwd variant,
               which is what the OE working tree holds today)
  campaign_corrections_fwd.md   the fwd CAMPAIGN CORRECTIONS block
Outputs:
  fwd/<path>   the fwd variant (bwd's job-agnostic ATT enablement kept, every bwd-specific line rewritten)
  ../oe_deep_fwd_1002.diff      git-style patch HEAD -> fwd variant (apply with `git apply` from the OE root)

Every replacement asserts that its anchor occurs exactly once, so a changed input fails loudly.
"""
import difflib
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = [l.strip() for l in (HERE / "files.txt").read_text().splitlines() if l.strip()]
BLOCK = (HERE / "campaign_corrections_fwd.md").read_text()
HEADING = "## ⚠ CAMPAIGN CORRECTIONS"
PT = "/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo"


def sub1(text, old, new, where):
    n = text.count(old)
    if n != 1:
        sys.exit(f"{where}: anchor occurs {n} times, expected 1:\n{old[:200]}")
    return text.replace(old, new)


def preamble(path):
    bwd = (HERE / "bwd" / path).read_text()
    i = bwd.index(HEADING)
    assert bwd.count(HEADING) == 1
    return bwd[:i] + BLOCK


def select(path):
    t = (HERE / "bwd" / path).read_text()
    return sub1(t,
        "This op's kernels differ by shape: `k_dqg` runs only at proxy and prod (fast dispatches `k_dq_sp`), so\n"
        "profile prod.\n",
        "This op's kernel differs by shape: proxy and prod dispatch `fmha_fwd_prefill_a16w16_m32x8` (rocprofv3 name\n"
        "`kn_fmha_fwd_prefill_a16w16_m32x8_bshd_0`), while `fast` does not fill the device and runs the small-grid\n"
        "`m32x2` kernel (`op/current/impl.py`), so a fast profile says nothing about prod. Profile prod.\n",
        path)


def counters(path):
    t = (HERE / "bwd" / path).read_text()
    t = sub1(t,
        "since the 09-29 reflash ATT gives per-instruction stall and idle cycles for this op's FlyDSL kernels\n"
        "(verified 2026-09-30). In `analysis.md`, state the counter limitation and point at step 4. Do not\n"
        "substitute a static instruction count for it -- this job has measured static metrics improving on\n"
        "arms that lost, again and again (h26, h78).\n",
        "since the 09-29 reflash ATT gives per-instruction stall and idle cycles for FlyDSL JIT kernels on\n"
        "this card (verified 2026-09-30 on the backward job's kernels; this op's kernels are captured for the\n"
        "first time in step 4, which must confirm the capture armed). In `analysis.md`, state the counter\n"
        "limitation and point at step 4. Do not substitute a static instruction count for it -- this op has\n"
        "measured static metrics improving on arms that lost (A0 round 5's QK/softmax software pipeline,\n"
        "-15.2% at prod; hint h28, and L30 in h13).\n",
        path)
    t = sub1(t,
        "benchmark with `--warmup-seconds 0` under `--pmc`: the unsynchronised warmup floods the profiler and\n"
        "looks like a hang.\n",
        "benchmark with `--warmup-seconds 0` under `--pmc`: the unsynchronised warmup floods the profiler and\n"
        "looks like a hang. The champion's census to compare against (A0, 2026-09-30):\n"
        f"`{PT}/output/0930__roofline/REPORT.md` §6 and §9 -- r13ns prod m32x8\n"
        "1.98e6 cycles/SIMD (53% of the matrix floor), the ASM bar 1.43-1.45e6 (72%) at a lower, power-capped clock.\n",
        path)
    return t


def metrics(path):
    t = (HERE / "bwd" / path).read_text()
    return sub1(t,
        "instruction trace, which since the 09-29 reflash gives per-instruction stall reasons for this op).",
        "instruction trace, which since the 09-29 reflash gives per-instruction stall reasons for FlyDSL kernels on this card).",
        path)


def thread_trace(path):
    t = (HERE / "bwd" / path).read_text()
    t = sub1(t,
        "This exact form was verified on this card on 2026-09-30 (after the 09-29 reflash) for the FlyDSL JIT\n"
        "kernels and for the ASM `.co`, with a clean `dmesg` every time.",
        "This exact form was verified on this card on 2026-09-30 (after the 09-29 reflash) for FlyDSL JIT\n"
        "kernels (the backward job's) and for the ASM `.co`, with a clean `dmesg` every time.",
        path)
    t = sub1(t,
        "`FLYDSL_GPU_ARCH=gfx1250` as for every run here). For this op `\"k_d[kq]\"` captures `k_dkdv` and `k_dqg`\n"
        "at prod (two dispatch directories, ~tens of MB); **quote the regex** -- unquoted, the shell globs\n"
        "`k_d[kq]`.",
        "`FLYDSL_GPU_ARCH=gfx1250` as for every run here). For this op `\"fmha_fwd_prefill\"` captures our prod\n"
        "kernel `kn_fmha_fwd_prefill_a16w16_m32x8_bshd_0` (one dispatch directory per call). For `op/beat` the\n"
        "ASM kernel is `aiter::fmha_bf16_pertokenBf16_hd128_128x256_mask` (`\"fmha_bf16\"`); it comes from a\n"
        "runtime-loaded `.co`, so pass that name to `tools/att_views.py --kernel-name`. **Quote the regex.**",
        path)
    t = sub1(t,
        "to source lines. Two reading rules learned on this op:\n",
        "to source lines. Two reading rules learned on this card (backward job, 2026-09-30):\n",
        path)
    t = sub1(t,
        "Comparable reads already on disk (same recipe, prod, CU 1):\n"
        "`/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0930__bwd/probe/att_s6/` (the champion s6),\n"
        "`.../probe/p2_r29/` (r29), `.../probe/p2_asm/` (the ASM bar's main kernel), with the operator's reading in\n"
        "`.../0930__bwd/notes/report_{dkdv,dqg,asm}.md`. If the target is byte-identical to s6, say so and\n"
        "spend the step on the question those reads leave open rather than re-deriving them.\n",
        "Comparable reads on disk: **none for this op's FlyDSL kernels on the new firmware** -- this capture is\n"
        "the first. The beat's ATT (`{{job_context_dir}}/profiling/beat/4-thread-trace/`, A0 2026-09-27, OLD\n"
        "firmware) has the ASM kernel's instruction mix; its stall and clock picture predates the reflash. The\n"
        "champion's cycle census and light-speed ablations (PMC, not ATT: skeleton 28%, softmax 19%, per-tile\n"
        f"barrier 5%) are in `{PT}/output/0930__roofline/REPORT.md` §6 and §9.\n"
        "What a useful read looks like (same recipe, the backward job):\n"
        f"`{PT}/output/0930__bwd/notes/report_{{dkdv,dqg,asm}}.md`.\n",
        path)
    t = sub1(t,
        "## MEASURED ON THIS MACHINE -- ATT is the primary diagnostic for this op\n"
        "\n"
        "**2026-09-24 (old firmware):** ATT captured nothing for FlyDSL JIT kernels (ELF dumps only) while it did\n"
        "capture a torch kernel. **That finding is superseded.** **2026-09-30, after the 09-29 reflash\n"
        "(VBIOS 700E):** the recipe above captures FlyDSL `k_dkdv` / `k_dqg` and the ASM `.co` with full\n"
        "per-instruction statistics, every run rc 0 with no GPU fault line in `dmesg`\n"
        "(`/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0930__bwd/probe/P1-RESULTS.md`). The operator's\n"
        "hand campaign that day took this op from 83.5% to 103.8% of the ASM bar, and every accepted lever was\n"
        "found or confirmed from an ATT read (a `s_wait_loadcnt` stall pointing at operand latency, a\n"
        "`s_wait_dscnt` behind a readback burst, VALU and `v_nop` counts between WMMAs). So: **run this step; do\n"
        "not skip it.** A `status: skipped` here needs a measured reason from this round.\n",
        "## MEASURED ON THIS MACHINE -- ATT is the primary diagnostic here\n"
        "\n"
        "**2026-09-24 (old firmware):** ATT captured nothing for FlyDSL JIT kernels (ELF dumps only) while it did\n"
        "capture a torch kernel; this job's rounds 10, 15 and 20 skipped this step on that finding. **That finding\n"
        "is superseded.** **2026-09-30, after the 09-29 reflash (VBIOS 700E):** the recipe above captures FlyDSL JIT\n"
        "kernels (the backward job's `k_dkdv` / `k_dqg`) and the ASM `.co` with full per-instruction statistics,\n"
        "every run rc 0 with no GPU fault line in `dmesg`\n"
        f"(`{PT}/output/0930__bwd/probe/P1-RESULTS.md`). The backward's\n"
        "hand campaign that day went from 83.5% to 103.8% of its ASM bar, and every accepted lever was found or\n"
        "confirmed from an ATT read (a `s_wait_loadcnt` stall pointing at operand latency, a `s_wait_dscnt` behind\n"
        "a readback burst, VALU and `v_nop` counts between WMMAs). This op's kernels load the same way\n"
        "(runtime-loaded FlyDSL JIT code objects) but have not been captured on this firmware yet. So: **run this\n"
        "step; do not skip it**, and check that it armed. A `status: skipped` here needs a measured reason from\n"
        "this round.\n",
        path)
    return t


BUILDERS = {
    "act/prompts/_preamble.md": preamble,
    "plan/prompts/_preamble.md": preamble,
    "profiling/prompts/_preamble.md": preamble,
    "profiling/prompts/01_select.md": select,
    "profiling/prompts/02_counters.md": counters,
    "profiling/prompts/03_metrics.md": metrics,
    "profiling/prompts/04_thread_trace.md": thread_trace,
}

PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")
BWD_ONLY = re.compile(r"k_dkdv|k_dqg|k_dq_sp|k_dq\b|\bs6\b|r29|h7[4-9]|h8[0-3]|\bh26\b|0930__bwd/probe/att_s6")

patch = []
for path in FILES:
    key = path.split("deep_loop/", 1)[1]
    text = BUILDERS[key](path)
    out = HERE / "fwd" / path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    head = (HERE / "head" / path).read_text()
    bwd = (HERE / "bwd" / path).read_text()
    # A placeholder nothing supplies makes core/prompts.py raise at render time: the fwd variant may use only
    # placeholders that the HEAD or bwd variant of the same file already uses (both have rendered in real rounds).
    extra = set(PLACEHOLDER.findall(text)) - set(PLACEHOLDER.findall(head)) - set(PLACEHOLDER.findall(bwd))
    if extra:
        sys.exit(f"{path}: new placeholders {sorted(extra)}")
    left = [l for l in text.splitlines() if BWD_ONLY.search(l) and "backward" not in l and "report_{dkdv" not in l]
    if left:
        sys.exit(f"{path}: bwd-specific text left:\n" + "\n".join(left))
    patch += [f"diff --git a/{path} b/{path}\n"]
    patch += difflib.unified_diff(head.splitlines(True), text.splitlines(True), f"a/{path}", f"b/{path}", n=3)
    print(f"built {path}: {len(head.splitlines())} -> {len(text.splitlines())} lines")

(HERE.parent / "oe_deep_fwd_1002.diff").write_text("".join(patch))
print("wrote", HERE.parent / "oe_deep_fwd_1002.diff")
