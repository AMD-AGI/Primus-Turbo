#!/usr/bin/env python3
"""R27 probe: DSV3-GateUP dB var-K with extended cell set + Triton allclose.

R24 found only (gm=2, xcds=0) as uniform-positive for DSV3-GateUP
dB var-K — but it failed Triton allclose (xcds=0 chiplet bypass
produces self-consistent but Triton-divergent accumulation order).

R27 retries with:
  * Extended cell set covering xcds ∈ {1, 2, 8, 16, 32} variants
    that R24's 11-cell set excluded.
  * Per-cell Triton-reference allclose verification (instead of
    just bit_eq vs another HK cell — the trap that bit
    R24's DSV3-GateUP attempt).

If any cell is BOTH uniform-positive AND allclose-safe, it's a
landable rule (4 shapes × 1 weight = ~+0.3 expected score, but
combined with the R26 already-probed 3-rule aggregate would
make a 4-rule aggregate worth re-trying).
"""

import os
import sys
import statistics

sys.path.insert(0, "/workspace/code/Primus-Turbo")

import torch  # noqa: E402

os.environ.setdefault("PRIMUS_TURBO_HIPKITTEN_PATH", "/workspace/code/HipKittens")
os.environ.setdefault("PRIMUS_TURBO_GROUPED_GEMM_BACKEND", "HIPKITTEN")

from primus_turbo.pytorch.kernels.hipkitten import loader as hipkitten  # noqa: E402

hk = hipkitten.load_bf16()
var_k_fn = hk.grouped_variable_k_crr
assert var_k_fn is not None

device = "cuda"


def warmup_bf16_runtime():
    """K-tail cold-start prophylaxis (R22)."""
    import primus_turbo.pytorch as turbo
    from primus_turbo.pytorch.core.backend import (
        BackendType, GlobalBackendManager, PrecisionType,
    )
    GlobalBackendManager.set_grouped_gemm_backend(
        BackendType.HIPKITTEN, PrecisionType.BF16_FP16_FP32
    )

    def warm(B, M_per, N, K):
        M_total = B * M_per
        a = torch.randn(M_total, K, dtype=torch.bfloat16, device=device,
                        requires_grad=True)
        b = torch.randn(B, N, K, dtype=torch.bfloat16, device=device,
                        requires_grad=True)
        gl = torch.full((B,), M_per, dtype=torch.int64, device=device)
        out = turbo.ops.grouped_gemm(a, b, gl, trans_b=True)
        grad = torch.randn_like(out)
        out.backward(grad)
        torch.cuda.synchronize()

    warm(16, 2048, 4096, 7168)
    warm(16, 2048, 7168, 2048)
    warm(16, 2048, 3072, 4096)
    warm(16, 2048, 4096, 1536)
    warm(4, 2048, 5760, 2880)
    warm(4, 2048, 2880, 2880)
    warm(32, 2048, 5760, 2880)
    warm(32, 2048, 2880, 2880)
    torch.cuda.synchronize()


def make_tensors(B, M_per):
    N_fwd = 4096
    K_fwd = 7168
    M_total = B * M_per
    torch.manual_seed(42)
    grad_out = torch.randn(M_total, N_fwd, dtype=torch.bfloat16, device=device)
    x = torch.randn(M_total, K_fwd, dtype=torch.bfloat16, device=device)
    grad_b = torch.empty(B, N_fwd, K_fwd, dtype=torch.bfloat16, device=device)
    offs = torch.tensor([i * M_per for i in range(B + 1)],
                        dtype=torch.int64, device=device)
    return grad_out, x, grad_b, offs, N_fwd, K_fwd


def time_one(grad_out, x, grad_b, offs, gm, xcd, iters=120):
    for _ in range(15):
        var_k_fn(grad_out, x, grad_b, offs, gm, xcd)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        var_k_fn(grad_out, x, grad_b, offs, gm, xcd)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def triton_reference(grad_out, x, B, M_per, N_fwd, K_fwd):
    """Compute dB reference via per-group dense PyTorch matmul (truth)."""
    out = torch.zeros(B, N_fwd, K_fwd, dtype=torch.float32, device=device)
    for g in range(B):
        s = g * M_per
        e = s + M_per
        # dB[g] = grad_out[s:e].T @ x[s:e]  (N_fwd, K_fwd)
        out[g] = grad_out[s:e].float().T @ x[s:e].float()
    return out


def hk_to_compare(grad_out, x, B, N_fwd, K_fwd, offs, gm, xcd):
    out = torch.zeros(B, N_fwd, K_fwd, dtype=grad_out.dtype, device=device)
    var_k_fn(grad_out, x, out, offs, gm, xcd)
    return out


def allclose_check(grad_out, x, B, M_per, N_fwd, K_fwd, offs, gm, xcd,
                    rtol=1e-2, atol=1e-1):
    """Check HK output vs PyTorch reference within bf16 allclose tolerance.

    bf16 allclose ranges: max_abs ~ 1e-1 typical for accumulations of
    ~M_per K-iters at full bf16 precision. We use the metric's
    downsized check_allclose tolerance (rtol=1e-2, atol=1e-1).
    """
    ref = triton_reference(grad_out, x, B, M_per, N_fwd, K_fwd)
    test = hk_to_compare(grad_out, x, B, N_fwd, K_fwd, offs, gm, xcd).float()
    max_abs = (test - ref).abs().max().item()
    max_rel = ((test - ref).abs() / (ref.abs() + 1e-6)).max().item()
    is_close = torch.allclose(test, ref, rtol=rtol, atol=atol)
    return is_close, max_abs, max_rel


def sweep_shape(B, M_per, prod_gm, prod_xcd, label):
    grad_out, x, grad_b, offs, N_fwd, K_fwd = make_tensors(B, M_per)
    M_total = B * M_per
    flops = 2 * N_fwd * K_fwd * M_total

    # R24 had 11 cells. Extend with more xcds variants and gm
    # combinations not previously tried.
    candidates = [
        (prod_gm, prod_xcd),
        # R24 set
        (1, 4), (2, 4), (4, 4), (8, 4), (16, 4),
        (1, 0), (2, 0), (4, 0), (8, 0), (16, 0),
        (4, 2), (4, 8),
        # NEW R27 cells (allclose-safe candidates)
        (1, 2), (2, 2), (8, 2), (16, 2),
        (1, 8), (2, 8), (8, 8), (16, 8),
        (1, 16), (2, 16), (4, 16),
        (1, 32), (2, 32),
    ]
    seen = set(); uniq = []
    for c in candidates:
        if c not in seen:
            uniq.append(c); seen.add(c)
    candidates = uniq

    print(f"\n=== {label} ===")
    print(f"  fwd=(B={B}, M_per={M_per}, N={N_fwd}, K={K_fwd}), "
          f"M_total={M_total}, prod=(gm={prod_gm}, xcds={prod_xcd})")

    results = []
    for (gm, xcd) in candidates:
        ms_trials = [time_one(grad_out, x, grad_b, offs, gm, xcd, iters=120)
                     for _ in range(5)]
        med_ms = statistics.median(ms_trials)
        tflops = flops / (med_ms * 1e9)
        results.append((gm, xcd, med_ms, tflops))

    prod_tf = next(r[3] for r in results
                   if r[0] == prod_gm and r[1] == prod_xcd)
    results.sort(key=lambda r: r[3], reverse=True)

    print(f"  {'(gm,xcd)':>10s}  {'tflops':>7s}  Δ vs prod")
    for r in results:
        d = (r[3] - prod_tf) / prod_tf * 100
        marker = "  *PROD*" if (r[0] == prod_gm and r[1] == prod_xcd) else ""
        print(f"  gm={r[0]:>2d} xcd={r[1]:>2d}   {r[3]:6.1f}   {d:+5.2f}%{marker}")

    return [(gm, xcd, tf) for (gm, xcd, _, tf) in results]


def main():
    print("=" * 92)
    print("R27 BF16 dB var-K probe — DSV3-GateUP extended cell set + Triton allclose")
    print("=" * 92)
    print("Warmup...")
    warmup_bf16_runtime()
    print("Warmup OK.")

    shapes = [
        ("B16-M2k", 16, 2048),
        ("B16-M4k", 16, 4096),
        ("B32-M2k", 32, 2048),
        ("B32-M4k", 32, 4096),
    ]
    by_shape = {}
    for sh_name, B, M_per in shapes:
        res = sweep_shape(B, M_per, 4, 8, f"DSV3-GateUP {sh_name}")
        by_shape[sh_name] = {(g, x): tf for (g, x, tf) in res}

    print("\n" + "-" * 92)
    print("AGG DSV3-GateUP dB var-K (extended)  vs prod=(4, 8)")
    print("-" * 92)
    sh_keys = list(by_shape.keys())
    cells = sorted(by_shape[sh_keys[0]].keys())
    prod_by_shape = {sh: by_shape[sh][(4, 8)] for sh in sh_keys}
    hdr = f"  {'cell':>10s}  " + "  ".join(f"{sh:>9s}" for sh in sh_keys) + "  " + \
          f"{'avg':>7s}  uniform"
    print(hdr)
    candidates_uniform = []
    for cell in cells:
        gm, xcd = cell
        deltas = [(by_shape[sh][cell] - prod_by_shape[sh]) / prod_by_shape[sh] * 100
                  for sh in sh_keys]
        avg = sum(deltas) / len(deltas)
        mn = min(deltas)
        u = "  +" if mn > 0 else "  -"
        marker = "  *PROD*" if cell == (4, 8) else ""
        ds = "  ".join(f"{d:+8.2f}%" for d in deltas)
        print(f"  gm={gm:>2d} xc={xcd:>2d}  {ds}  {avg:+6.2f}%  {u}{marker}")
        if mn > 0:
            candidates_uniform.append((cell, avg))

    candidates_uniform.sort(key=lambda c: c[1], reverse=True)
    print("\nUniform-positive candidates sorted by avg Δ:")
    for cell, avg in candidates_uniform:
        print(f"  gm={cell[0]:>2d} xcds={cell[1]:>2d}  avg={avg:+.2f}%")

    if not candidates_uniform:
        print("  >> NO uniform-positive cell at all.")
        return

    print("\n" + "-" * 92)
    print("Triton-reference allclose check on top uniform-positive cells")
    print("-" * 92)
    for cell, avg in candidates_uniform[:5]:
        gm, xcd = cell
        print(f"\nCell (gm={gm}, xcds={xcd})  avg={avg:+.2f}%:")
        for sh_name, B, M_per in shapes:
            grad_out, x, _, offs, N_fwd, K_fwd = make_tensors(B, M_per)
            ok, max_abs, max_rel = allclose_check(
                grad_out, x, B, M_per, N_fwd, K_fwd, offs, gm, xcd
            )
            tag = "PASS" if ok else "FAIL"
            print(f"  {sh_name:9s}  {tag}  max_abs={max_abs:.4e}  max_rel={max_rel:.4e}")


if __name__ == "__main__":
    main()
