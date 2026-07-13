#!/usr/bin/env python3
"""Campaign bench for dsv4 sparse-MLA BACKWARD (dsa_bwd.py) utilization.

Runs the 6 shapes (flash/pro x cr{0,4,128}) @seq=4096, times the flydsl bwd
(median), checks SNR vs the triton oracle, and prints ONE final JSON line
{"ok": <all bwd SNR>=35dB>, "tflops": <mean bwd TF>} for flydsl_campaign.py.

Score = mean bwd TF across the 6 shapes (comprehensive, never-regress). All
intermediate lines go to stderr so the JSON is the last stdout line.
"""
import json
import sys

import bench_mla as bm

GROUPS = [(v, cr) for v in ("flash", "pro") for cr in (0, 4, 128)]
WARM, ITERS = 8, 20


def main():
    btfs = []
    ok = True
    for v, cr in GROUPS:
        ftf, fsnr, btf, bsnr = bm.one(v, cr, 4096, WARM, ITERS)
        btfs.append(btf)
        if bsnr < 35.0:
            ok = False
        print(f"  {v:5s} cr={cr:<3d} bwd={btf:7.1f}TF  SNR={bsnr:5.1f}dB "
              f"{'ok' if bsnr >= 35.0 else 'FAIL'}", file=sys.stderr, flush=True)
    score = sum(btfs) / len(btfs)
    print(f"  MEAN bwd = {score:.1f}TF  (ok={ok})", file=sys.stderr, flush=True)
    print(json.dumps({"ok": bool(ok), "tflops": round(score, 2)}))


if __name__ == "__main__":
    main()
