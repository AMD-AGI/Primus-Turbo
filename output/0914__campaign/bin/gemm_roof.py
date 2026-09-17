"""What is the real dense-GEMM gap on this host, and which BLAS is torch actually using?

The library-path hypothesis is already refuted (tuned hipBLASLt on LD_LIBRARY_PATH moved
nothing), so the remaining questions are: (a) does torch even reach hipBLASLt, and (b) how
far is whatever it reaches from a Triton kernel on the same tensors in the same process.
Yesterday's escalation compared 27.4 against a Triton roof of 1002.7 at 1100 MHz; both
halves have to be re-measured here before that ratio means anything.
"""
import json, os, sys

import torch
import triton
import triton.language as tl


@triton.jit
def _mm(a_ptr, b_ptr, c_ptr, M, N, K,
        sam, sak, sbk, sbn, scm, scn,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr):
    pid = tl.program_id(0)
    nm, nn = tl.cdiv(M, BM), tl.cdiv(N, BN)
    ng = GM * nn
    gid = pid // ng
    fm = gid * GM
    gsz = min(nm - fm, GM)
    pm = fm + ((pid % ng) % gsz)
    pn = (pid % ng) // gsz
    om = (pm * BM + tl.arange(0, BM)) % M
    on = (pn * BN + tl.arange(0, BN)) % N
    ok = tl.arange(0, BK)
    ap = a_ptr + (om[:, None] * sam + ok[None, :] * sak)
    bp = b_ptr + (ok[:, None] * sbk + on[None, :] * sbn)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        a = tl.load(ap, mask=ok[None, :] < K - k * BK, other=0.0)
        b = tl.load(bp, mask=ok[:, None] < K - k * BK, other=0.0)
        acc = tl.dot(a, b, acc)
        ap += BK * sak
        bp += BK * sbk
    c = acc.to(tl.bfloat16)
    cm = pm * BM + tl.arange(0, BM)
    cn = pn * BN + tl.arange(0, BN)
    tl.store(c_ptr + scm * cm[:, None] + scn * cn[None, :], c,
             mask=(cm[:, None] < M) & (cn[None, :] < N))


def triton_mm(a, b, cfg):
    M, K = a.shape
    K2, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(M, cfg["BM"]) * triton.cdiv(N, cfg["BN"]),)
    _mm[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
              c.stride(0), c.stride(1), **cfg)
    return c


def timed_ms(fn, iters=10, warmup=3):
    flush = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda")
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts, s, e = [], torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(iters):
        flush.zero_()
        torch.cuda.synchronize()
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def main():
    M = N = K = 8192
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    flops = 2 * M * N * K
    out = {
        "prefer_hipblaslt": os.environ.get("TORCH_BLAS_PREFER_HIPBLASLT", "(unset)"),
        "torch": torch.__version__, "triton": triton.__version__,
        "arch": torch.cuda.get_device_properties(0).gcnArchName,
    }
    try:  # which backend torch says it will use -- "prefers" is not "reaches"
        out["preferred_blas"] = str(torch.backends.cuda.preferred_blas_library())
    except Exception as exc:
        out["preferred_blas"] = f"unavailable: {type(exc).__name__}"

    ms = timed_ms(lambda: torch.mm(a, b))
    out["torch_mm"] = {"ms": ms, "tflops": flops / (ms * 1e-3) / 1e12}

    ref = torch.mm(a, b)
    best = None
    for cfg in (
        dict(BM=128, BN=256, BK=64, GM=8, num_warps=8, num_stages=2),
        dict(BM=256, BN=128, BK=64, GM=8, num_warps=8, num_stages=2),
        dict(BM=128, BN=128, BK=64, GM=8, num_warps=4, num_stages=2),
        dict(BM=128, BN=256, BK=32, GM=8, num_warps=8, num_stages=3),
        dict(BM=256, BN=256, BK=32, GM=8, num_warps=8, num_stages=2),
    ):
        try:
            c = triton_mm(a, b, cfg)
            err = (c.float() - ref.float()).abs().max().item()
            t = timed_ms(lambda: triton_mm(a, b, cfg))
            row = {"cfg": {k: v for k, v in cfg.items()}, "ms": t,
                   "tflops": flops / (t * 1e-3) / 1e12, "max_abs_err": err}
            if best is None or t < best["ms"]:
                best = row
        except Exception as exc:
            continue
    out["triton_mm_best"] = best
    if best:
        out["gap_triton_over_torch"] = best["tflops"] / out["torch_mm"]["tflops"]
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
