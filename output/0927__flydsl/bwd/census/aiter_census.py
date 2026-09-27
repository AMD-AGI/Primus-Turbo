#!/usr/bin/env python3
"""Barrier-region census of aiter's gfx1250 bwd .co main loop (objdump text in enc/aiter.dis).
Regenerate the input: /opt/rocm/llvm/bin/llvm-objdump -d --mcpu=gfx1250 \
  /home/lihuzhan/code/aiter-src/hsa/gfx1250/fmha_v3_bwd/bwd_hd128_bf16_causal_br_a32_pssk.co > enc/aiter.dis"""
import re, sys
from collections import Counter
L = []
for line in open(sys.argv[1] if len(sys.argv) > 1 else 'enc/aiter.dis'):
    m = re.match(r"\s+([a-z_.][a-z0-9_.]*)\s*(.*?)\s*//\s*([0-9A-F]+):", line)
    if m: L.append((m.group(1), m.group(2), int(m.group(3), 16)))
def kind(mn, ops):
    if mn.startswith('v_wmma'): return 'wmma'
    if mn.startswith(('ds_load', 'ds_read')): return 'dsl'
    if mn.startswith(('ds_store', 'ds_write')): return 'dss'
    if mn == '.long': return 'tdm' if 'd031' in ops.lower() else 'long'
    if mn.startswith(('global_atomic', 'buffer_atomic')): return 'atom'
    if mn.startswith('v_'): return 'valu'
    if mn.startswith('s_'): return 'salu'
    return 'o'
# main loop = between `s_cbranch_scc1 4529` and the back-edge `s_branch 61007` (causal_br_a32_pssk)
start = next(i for i, x in enumerate(L) if x[0] == 's_cbranch_scc1' and x[1].startswith('4529'))
end = next(i for i, x in enumerate(L) if x[0] == 's_branch' and x[1].startswith('61007'))
B = L[start + 1:end + 1]
print('loop instr', len(B), 'bytes', L[end][2] - L[start][2], dict(Counter(kind(m, o) for m, o, _ in B)))
cuts = [i for i, (m, o, a) in enumerate(B) if m in ('s_barrier_signal', 's_barrier_wait')]
prev = 0
for c in cuts + [len(B)]:
    seg = B[prev:c]; cc = Counter(kind(m, o) for m, o, _ in seg)
    waits = [f"{m.replace('s_wait_', '')} {o}" for m, o, _ in seg if m.startswith('s_wait_') and 'xcnt' not in m and 'kmcnt' not in m]
    tag = B[c][0].replace('s_barrier_', '') if c < len(B) else 'END'
    print(f" [{prev:4d},{c:4d}) n={c-prev:4d} wmma={cc['wmma']:3d} dsl={cc['dsl']:3d} dss={cc['dss']:3d} tdm={cc['tdm']:2d} atom={cc['atom']:3d} valu={cc['valu']:3d} -> {tag:6s} waits={waits}")
    prev = c + 1
