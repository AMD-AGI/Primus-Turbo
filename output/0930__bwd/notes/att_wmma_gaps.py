#!/usr/bin/env python3
"""WMMA-to-WMMA gap decomposition from ATT wave timelines (stdlib only).
usage: att_wmma_gaps.py <ui_output_dir> <loop_lo_vaddr> <loop_hi_vaddr> [wave-index-list e.g. 0-5]
Each gap between consecutive v_wmma issues inside [lo,hi) is classified by the instructions in it:
tensor-wait > barrier > dscnt > vmem-wait(loadcnt/xcnt, gap>20) > atomic > has-ds > has-vmem > valu/salu > back-to-back.
Floor = 8 cycles per gap (REPORT.md W=8). Used for the ASM vs fly r29 bwd comparison (2026-09-30)."""
import json, glob, sys, collections
d, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
code = json.load(open(d + '/code.json'))['code']
txt = {i: c[0] for i, c in enumerate(code)}; va = {i: c[5] for i, c in enumerate(code)}
files = sorted(glob.glob(d + '/se*_wv*.json'), key=lambda s: int(s.split('wv')[1].split('.')[0]))
if len(sys.argv) > 4:
    a, b = map(int, sys.argv[4].split('-')); files = files[a:b + 1]
agg = collections.Counter(); cnt = collections.Counter(); trips = 0
for f in files:
    ins = json.load(open(f))['wave']['instructions']; last = None; between = []
    for x in ins:
        v = va[x[4]]; t = txt[x[4]]
        if not (lo <= v < hi): last = None; between = []; continue
        if v == lo: trips += 1
        if 'wmma' in t:
            if last is not None:
                g = x[0] - last; ops = [s.split()[0] for s in between]
                has = lambda p: any(o.startswith(p) for o in ops)
                if has('s_wait_tensorcnt'): k = 'tensor-wait'
                elif has('s_barrier_wait'): k = 'barrier'
                elif has('s_wait_dscnt') and g > 20: k = 'dscnt'
                elif (has('s_wait_loadcnt') or has('s_wait_xcnt')) and g > 20: k = 'vmem-wait'
                elif any('atomic' in o for o in ops): k = 'atomic'
                elif has('ds_'): k = 'has-ds'
                elif has(('buffer', 'global', 'image')): k = 'has-vmem'
                elif ops: k = 'valu/salu'
                else: k = 'back-to-back'
                agg[k] += g; cnt[k] += 1
            last = x[0]; between = []
        else:
            between.append(t)
tot = sum(agg.values())
print('trips(loop-top hits)', trips, 'total gap cycles/trip %.0f' % (tot / trips))
for k in sorted(agg, key=lambda k: -agg[k]):
    print('  %-12s n/trip=%5.1f cyc/trip=%6.0f each=%5.1f excess-over-8=%6.0f' % (
        k, cnt[k] / trips, agg[k] / trips, agg[k] / cnt[k], (agg[k] - 8 * cnt[k]) / trips))
