#!/usr/bin/env python3
"""Summarise an ATT stats_ui_output csv: per instruction class Hitcount/Latency/Stall/Idle, and top stall sites.
usage: attsum.py stats.csv [topN]"""
import csv, sys, collections
rows = list(csv.DictReader(open(sys.argv[1]))); top = int(sys.argv[2]) if len(sys.argv) > 2 else 25
def cls(ins):
    op = ins.split()[0] if ins.split() else ''
    if 'wmma' in op: return 'wmma'
    if op.startswith('ds_'): return 'ds'
    if op.startswith(('global_', 'buffer_', 'tensor_', 'flat_')): return 'vmem'
    if op.startswith('s_wait') or op.startswith('s_barrier'): return 'wait/bar'
    if op.startswith('s_'): return 'salu'
    if op.startswith('v_'): return 'valu'
    return 'other'
tot = collections.defaultdict(lambda: [0, 0, 0, 0]); hot = []
for r in rows:
    h = int(r['Hitcount'])
    if not h: continue
    c = cls(r['Instruction']); v = [h, int(r['Latency']), int(r['Stall']), int(r['Idle'])]
    for i in range(4): tot[c][i] += v[i]
    hot.append((int(r['Stall']) + int(r['Idle']), int(r['Vaddr']), r['Instruction'][:70], v))
L = sum(t[1] for t in tot.values())
print(f"{'class':9s} {'hits':>9s} {'latency':>11s} {'lat%':>6s} {'stall':>11s} {'idle':>11s}")
for c, t in sorted(tot.items(), key=lambda kv: -kv[1][1]):
    print(f"{c:9s} {t[0]:9d} {t[1]:11d} {100*t[1]/L:5.1f}% {t[2]:11d} {t[3]:11d}")
print(f"total latency {L}; wmma hits {tot['wmma'][0]}; latency per wmma {L/max(1,tot['wmma'][0]):.1f}")
print("-- top stall+idle sites --")
for s, a, ins, v in sorted(hot, reverse=True)[:top]:
    print(f"{a:7d} {ins:70s} hits={v[0]} lat={v[1]} stall={v[2]} idle={v[3]}")
