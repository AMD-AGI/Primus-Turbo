"""Audit RCR bn128 for same race pattern as RRR bn128.
RCR main loop line 2326 writes Bs[tic][0] (same slot as b0 source) between two mmas
— EXACT same pattern as RRR. Test determinism on RCR bn128 path."""
import sys, itertools, subprocess, os

if "--worker" in sys.argv:
    idx = int(sys.argv[2])
    K, N, B, M_g = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
    n_runs = int(sys.argv[7])
    sys.path.insert(0, "/workspace/code/Primus-Turbo")
    import torch, primus_turbo
    from primus_turbo.pytorch.ops.grouped_gemm import grouped_gemm_compute_offs
    from primus_turbo.pytorch.ops.quantization import quantize_fp8
    from primus_turbo.pytorch.core.low_precision import ScalingGranularity, float8_e4m3
    DEV = "cuda"
    hk_grp = torch.ops.primus_turbo_cpp_extension.hk_grouped_rcr_fp8
    M_total = B * M_g
    torch.manual_seed(42 + idx)
    a = (torch.randn((M_total, K), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
    # RCR: b is [G, N, K] (col-major view)
    b = (torch.randn((B, N, K), dtype=torch.bfloat16, device=DEV) * 0.05).contiguous()
    af, asc = quantize_fp8(a, float8_e4m3, ScalingGranularity.TENSORWISE)
    bf, bsc = quantize_fp8(b, float8_e4m3, ScalingGranularity.TENSORWISE)
    g_offs = grouped_gemm_compute_offs(torch.full((B,), M_g, dtype=torch.int64, device=DEV))
    # Force bn=128 to hit bn128 kernel path
    outs = [hk_grp(af, bf, asc, bsc, g_offs, 4, M_g, 4, torch.bfloat16, 128).detach().clone() for _ in range(n_runs)]
    torch.cuda.synchronize()
    diffs = [100.0 * (outs[i] != outs[0]).sum().item() / outs[0].numel() for i in range(1, n_runs)]
    max_diff = max(diffs) if diffs else 0.0
    print(f"RESULT {max_diff:.6f}", flush=True)
    sys.exit(0)

# Driver — same shape grid as RRR test
K_set   = [256, 384, 512, 768, 1024, 1536, 2048]
N_set   = [128, 256, 512, 1024, 2048, 4096]
B_set   = [1, 4, 16]
M_g_set = [256, 1024, 4096]
n_runs  = 3

def estimate_mem_gb(K, N, B, M_g):
    M_total = B * M_g
    return (M_total*K + B*K*N + M_total*N*2) / 1e9

shapes = [s for s in itertools.product(K_set, N_set, B_set, M_g_set) if estimate_mem_gb(*s) < 3.0]
print(f"=== RCR bn128 race audit ===")
print(f"Same shape grid as RRR; force bn=128. Each shape 3-run bit-equal check.")
print(f"Total shapes: {len(shapes)}")
print()

passed, nondet, exc = 0, [], []
for idx, (K, N, B, M_g) in enumerate(shapes):
    cmd = ["python", "-u", __file__, "--worker", str(idx), str(K), str(N), str(B), str(M_g), str(n_runs)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env={**os.environ})
    except subprocess.TimeoutExpired:
        exc.append((K, N, B, M_g, "TIMEOUT"))
        continue
    last_line = (r.stdout.strip().splitlines() or [""])[-1]
    if r.returncode != 0 or not last_line.startswith("RESULT "):
        exc.append((K, N, B, M_g, f"rc={r.returncode}"))
        print(f"  [{idx:3}] K={K:5} N={N:5} B={B:3} M_g={M_g:5}: EXC rc={r.returncode}")
        continue
    max_diff = float(last_line.split()[1])
    if max_diff == 0.0:
        passed += 1
        print(f"  [{idx:3}] K={K:5} N={N:5} B={B:3} M_g={M_g:5}: DET")
    else:
        nondet.append((K, N, B, M_g, max_diff))
        print(f"  [{idx:3}] K={K:5} N={N:5} B={B:3} M_g={M_g:5}: NONDET {max_diff:.4f}%")

print()
print(f"=== Summary: {passed}/{len(shapes)} det, {len(nondet)} nondet, {len(exc)} exc ===")
if nondet:
    print("NON-DETERMINISTIC RCR bn128 shapes (RCR also has the race):")
    for K, N, B, M_g, d in nondet[:20]:
        print(f"  K={K} N={N} B={B} M_g={M_g}: {d:.4f}%")
