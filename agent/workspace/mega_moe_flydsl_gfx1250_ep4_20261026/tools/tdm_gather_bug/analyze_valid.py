"""Bad tiles vs their valid-row count: is the corruption next to the padding boundary?

python analyze_valid.py DUMP.pt [...]
"""
import sys, torch

BM = 256
tot_tiles = tot_partial = 0
for path in sys.argv[1:]:
    d = torch.load(path, weights_only=False)
    out, ref, valid = d["out"], d["ref"], d["valid"]
    bad = ((out != ref).any(1) & valid).nonzero().flatten()
    nv = valid.view(-1, BM).sum(1)
    if tot_tiles == 0:
        n_used = int((nv > 0).sum())
        print(f"tiles with rows {n_used}, partial {int(((nv > 0) & (nv < BM)).sum())}")
    line = []
    for tile in (bad // BM).unique().tolist():
        rows = bad[(bad // BM) == tile]
        slots = sorted({(int(p) % BM) // 16 for p in rows})
        line.append(f"tile {tile}: valid {int(nv[tile])} (last valid slot {(int(nv[tile]) - 1) // 16}) bad slots {slots}")
        tot_tiles += 1
        tot_partial += int(nv[tile]) < BM
    print(f"{path.split('/')[-1]}: " + " | ".join(line))
print(f"bad tiles {tot_tiles}, partial {tot_partial}")
