"""Which output columns of a bad tile are wrong: per 16-row slot, per 16-column WMMA block of the N-block.

python analyze_cols.py DUMP.pt [...]   (cols 0-127 = wave_n 0, 128-255 = wave_n 1)
"""
import sys, torch

BM, TN = 256, 256
for path in sys.argv[1:]:
    d = torch.load(path, weights_only=False)
    out, ref, valid = d["out"], d["ref"], d["valid"]
    bad = ((out != ref).any(1) & valid).nonzero().flatten()
    print(f"== {path.split('/')[-1]}: bad rows {bad.numel()}")
    for tile in (bad // BM).unique().tolist():
        rows = bad[(bad // BM) == tile]
        diff = (out[rows] != ref[rows])
        nbs = diff.view(len(rows), -1, TN).any(2).any(0).nonzero().flatten().tolist()
        print(f"  tile {tile} N-blocks {nbs}")
        for nb in nbs:
            dn = diff[:, nb * TN:(nb + 1) * TN]
            for slot in sorted({(int(p) % BM) // 16 for p in rows}):
                m = torch.tensor([(int(p) % BM) // 16 == slot for p in rows])
                blk = dn[m].view(int(m.sum()), 16, 16).float().mean((0, 2))  # per 16-col block: fraction wrong
                print(f"    slot {slot:2d} rows {int(m.sum()):2d}: " + " ".join(f"{x:.1f}" for x in blk.tolist()))
