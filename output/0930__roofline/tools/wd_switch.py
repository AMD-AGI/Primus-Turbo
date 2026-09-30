# Hot-loop W/D switch census: for the largest loop (backward branch), count WMMA, ds_* and WMMA->DS->WMMA switches.
import re, sys
L = open(sys.argv[1]).read().split('\n')
labels = {}; loops = []
for i, l in enumerate(L):
    m = re.match(r'^(\.?L\w+|\.LBB\d+_\d+):', l.strip())
    if m: labels[m.group(1)] = i
    m = re.search(r's_cbranch_\w+\s+(\S+)', l)
    if m and m.group(1) in labels and labels[m.group(1)] < i: loops.append((labels[m.group(1)], i))
def cls(l):
    t = l.strip().split()
    if not t or t[0].startswith(('.', ';', '//')) or t[0].endswith(':'): return None
    m = t[0]
    return 'W' if m.startswith('v_wmma') else 'D' if m.startswith('ds_') else 'o'
best = None
for a, b in loops:
    seq = ''.join(c for c in (cls(x) for x in L[a:b + 1]) if c)
    if best is None or seq.count('W') > best[0].count('W'): best = (seq, a, b)
seq = best[0]
wd = re.sub('o', '', seq)
runs = re.findall(r'W+|D+', wd)
sw = sum(1 for i in range(1, len(runs)) if runs[i][0] == 'D' and runs[i - 1][0] == 'W')
print(f"{sys.argv[1][-60:]}: loop lines {best[1]}-{best[2]} instr={len(seq)} WMMA={seq.count('W')} DS={seq.count('D')} W->D switches={sw} "
      f"WMMA/switch={seq.count('W')/max(sw,1):.1f}")
print('   ', wd[:200])
