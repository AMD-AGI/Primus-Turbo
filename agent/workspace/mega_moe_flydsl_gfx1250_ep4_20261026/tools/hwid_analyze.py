"""hwid_analyze.py DUMP.pt SPIN_US: start-wave histogram of probe_hwid and which blockIdx residues start late."""
import sys, collections, torch

z = torch.load(sys.argv[1])
spin = float(sys.argv[2])
hw1, t0 = z["hw1"], z["t0"]
G = len(t0)
wave = (t0 / spin + 0.5).floor().long()
print(f"GRID {G}: start waves {dict(sorted(collections.Counter(wave.tolist()).items()))}")
for mod in (2, 4, 8, 16, 32):
    late = torch.bincount(torch.arange(G)[wave > 0] % mod, minlength=mod).tolist()
    tot = torch.bincount(torch.arange(G) % mod, minlength=mod).tolist()
    print(f"  late WGs per bid%{mod}: {late}  (of {tot[0]} each)")
key = ((hw1 >> 10) & 0xF) + 16 * ((hw1 >> 16) & 1)
print(f"  distinct (SA,WGP) slots {len(set(key.tolist()))}; WGs per slot in wave 0: "
      f"{dict(sorted(collections.Counter(key[wave == 0].tolist()).items()))}")
