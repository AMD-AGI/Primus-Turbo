import os, sys
os.environ.update(NKFIX_CHUNK_BYTES=str(64*1024), NKFIX_CHECK="2", NKFIX_CHECK_EVERY="3", NKFIX_STATS_FILE="/tmp/nk_cpu_stats.txt")
sys.path.insert(0, sys.argv[1])
import torch, nkfix_b0 as nk
torch.manual_seed(0)
ok = True
for (M, K, N) in [(300, 700, 500), (1000, 64, 1300), (40, 3000, 90)]:
    for copy_a in (False, True):
        a = torch.randn(M, K, dtype=torch.bfloat16) if not copy_a else torch.randn(K, M, dtype=torch.bfloat16).t()
        b = torch.randn(K, N, dtype=torch.bfloat16)
        ref = torch.mm(a.float(), b.float())
        out = nk._rewrite(a, b, copy_a)
        full = torch.mm(a.contiguous(), b.t().contiguous().t()).float()
        e = (out.float() - ref).abs().max().item(); e2 = (full - ref).abs().max().item()
        print(M, K, N, copy_a, "chunked", nk.stats["chunked"], "err", e, "ref-rewrite err", e2); ok &= e <= 2 * e2 + 1e-3
for i in range(7):
    t = torch.randn(10, 10)
    if i == 5: t[3, 3] = float("nan")
    nk._check(t, t, t, "dgrad")
nk.report(); print(open("/tmp/nk_cpu_stats.txt").read()); print("OK" if ok else "MISMATCH")
