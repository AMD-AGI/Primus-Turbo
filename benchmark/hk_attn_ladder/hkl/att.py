###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Summarize rocprofv3 thread traces (tools/att.sh output) of one or more rungs, side by side.

    python3 -m hkl.att logs/att/llama_ladder_fwd_f00 logs/att/llama_ladder_fwd_f01 ... [--top 12]

For every ui_output_* dispatch directory (one traced dispatch, one CU): the traced waves, how many of them
share a SIMD at a time, the mean wave lifetime in cycles, WMMA issues per wave, the WMMA-pipe busy fraction
per SIMD (8 cycles per v_wmma_f32_16x16x32_bf16, summed over the SIMD's waves and divided by the SIMD's
traced span), and a time attribution: each instruction is charged the cycles from its issue to its wave's
next issue (the a08-1 att_hot.py method), summed per category over all waves. With --top, the instructions
that cost the most (with their stall cycles) follow. Pure stdlib; reads the per-wave JSON and code.json.
"""

import argparse
import collections
import glob
import json
import os
import sys

WMMA_CYC = 8  # v_wmma_f32_16x16x32_bf16 issue cadence at one wave per SIMD with lock_simd (p1_research)

CATS = (  # (prefix of the opcode, category), first match wins
    ("v_wmma", "wmma"), ("v_exp", "exp"), ("ds_load_tr", "ds_tr16"), ("ds_load", "ds_load"),
    ("ds_store", "ds_store"), ("s_wait_dscnt", "wait_ds"), ("s_wait_tensorcnt", "wait_tdm"),
    ("s_wait_loadcnt", "wait_vmem"), ("s_wait_storecnt", "wait_vmem"), ("s_wait_xcnt", "wait_xcnt"),
    ("s_wait_alu", "wait_alu"), ("s_wait_kmcnt", "wait_sload"), ("s_barrier_wait", "barrier_wait"),
    ("s_barrier_signal", "barrier_signal"), ("tensor_", "tdm_issue"), ("buffer_atomic", "atomic"),
    ("global_atomic", "atomic"), ("buffer_load", "vmem_load"), ("global_load", "vmem_load"),
    ("buffer_store", "vmem_store"), ("global_store", "vmem_store"), ("v_cvt", "cvt"), ("v_nop", "v_nop"),
    ("s_set_vgpr_msb", "set_vgpr_msb"), ("v_", "valu"), ("s_cbranch", "branch"), ("s_branch", "branch"),
    ("s_", "salu"),
)  # fmt: skip


def category(text):
    op = text.split()[0] if text else ""
    for p, c in CATS:
        if op.startswith(p):
            return c
    return "other"


def dispatch_dirs(path):
    """ui_output_* directories under an att.sh output dir (or the dir itself)."""
    if os.path.isfile(os.path.join(path, "code.json")):
        return [path]
    return sorted(glob.glob(os.path.join(path, "ui_output_*")))


def summarize(d, top=0):
    code = json.load(open(os.path.join(d, "code.json")))["code"]
    txt = [c[0] for c in code]
    fn = json.load(open(os.path.join(d, "filenames.json")))["wave_filenames"]
    per = collections.Counter()  # instruction index -> attributed cycles
    stall = collections.Counter()
    hits = collections.Counter()
    simd = collections.defaultdict(lambda: [0, None, None])  # (se, sm) -> [wmma, begin, end]
    waves, life, nwmma = 0, 0, 0
    spans = []
    for se, sms in fn.items():
        for sm, slots in sms.items():
            for slot, wvs in slots.items():
                for wv, (f, b, e) in wvs.items():
                    ins = json.load(open(os.path.join(d, f)))["wave"]["instructions"]
                    waves += 1
                    life += e - b
                    spans.append((b, e, (se, sm)))
                    s = simd[(se, sm)]
                    s[1] = b if s[1] is None else min(s[1], b)
                    s[2] = e if s[2] is None else max(s[2], e)
                    for k, r in enumerate(ins):
                        t, idx = r[0], r[4]
                        nxt = ins[k + 1][0] if k + 1 < len(ins) else e
                        per[idx] += nxt - t
                        stall[idx] += r[2]
                        hits[idx] += 1
                        if txt[idx].startswith("v_wmma"):
                            nwmma += 1
                            s[0] += 1
    if not waves:
        return None
    # waves resident on a SIMD at the same time (mean over each wave's lifetime of the others overlapping it)
    conc = []
    for b, e, key in spans:
        ov = sum(min(e, e2) - max(b, b2) for b2, e2, k2 in spans if k2 == key and min(e, e2) > max(b, b2))
        conc.append(ov / max(e - b, 1))
    util = [s[0] * WMMA_CYC / max(s[2] - s[1], 1) for s in simd.values() if s[1] is not None]
    cats = collections.Counter()
    for idx, cyc in per.items():
        cats[category(txt[idx])] += cyc
    total = sum(cats.values())
    out = {
        "dir": d, "kernel": txt[0].lstrip("; ").strip(), "waves": waves, "waves_per_simd": sum(conc) / len(conc),
        "wave_cycles": life / waves, "wmma_per_wave": nwmma / waves, "wmma_util": sum(util) / len(util),
        "cats": {c: v / total for c, v in cats.most_common()},
    }  # fmt: skip
    if top:
        out["top"] = [(txt[i], per[i] / total, hits[i], stall[i]) for i, _ in per.most_common(top)]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--top", type=int, default=0)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    rows = []
    for path in a.dirs:
        for d in dispatch_dirs(path):
            s = summarize(d, a.top)
            if s:
                s["label"] = os.path.basename(os.path.normpath(path))
                rows.append(s)
    keys = ["wmma", "valu", "exp", "ds_load", "ds_tr16", "wait_ds", "wait_tdm", "wait_vmem", "barrier_wait", "atomic",
            "vmem_load", "salu", "wait_alu", "set_vgpr_msb"]  # fmt: skip
    print(f"{'rung':<34} {'kernel':<18} {'w/SIMD':>6} {'wave cyc':>9} {'WMMA/w':>7} {'WMMA%':>6}  "
          + " ".join(f"{k[:9]:>9}" for k in keys))  # fmt: skip
    for s in rows:
        print(f"{s['label'][:34]:<34} {s['kernel'][:18]:<18} {s['waves_per_simd']:6.2f} {s['wave_cycles']:9.0f} "
              f"{s['wmma_per_wave']:7.0f} {100 * s['wmma_util']:5.1f}%  "
              + " ".join(f"{100 * s['cats'].get(k, 0):8.1f}%" for k in keys))  # fmt: skip
        for t, share, n, st in s.get("top", []):
            print(f"      {100 * share:5.1f}%  x{n:<6d} stall {st:<8d} {t[:90]}")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
