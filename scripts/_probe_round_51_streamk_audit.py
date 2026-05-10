"""Round-51 audit — Stream-K scaffolding end-to-end smoke test.

Per R50 forward-pointer (analysis/_notes/round-50-fp8-grouped-C1prime-...md):
audit whether the existing R12-R17 Stream-K scaffolding is functional end-to-end
(host alloc + caller workspace ptr) AND whether the kernel itself honors the
sk_split_n field, OR whether the kernel-side K-split branch (deferred to "R18"
in the parallel run) was never landed.

Code-read verdict (from kernel_fp8_layouts.cpp lines 3024-3742, the
``grouped_rcr_kernel`` body): ZERO references to ``sk_split_n`` or
``sk_partial_buf``. The kernel ignores both fields. Empirical confirmation:

  Test 1 (kernel split-blindness): call grouped_rcr_dscale with
    sk_split_n=2 + caller-allocated workspace ptr, vs sk_split_n=0.
    Expected: outputs bit-identical (kernel reads neither field, so the
    only effect of sk_split_n>0 is host-side alloc/wire which doesn't
    touch the launched math). SNR ≈ ∞ (numerically identical).

  Test 2 (no crash, no NaN): exercise both sk_split_n in {2, 4} on a
    representative B=4 cell.

Anchor cell: Down_B4_M2048 fwd (smallest B=4, gap-leader 1565 T → 2800 T).

Outcome interpretation:
  * SNR=∞ AND no crash → kernel is split-BLIND, scaffolding is alloc-only,
    R52+ must write the kernel-side K-split branch (atomicAdd partials +
    reduce kernel). This is the EXPECTED outcome from code-read.
  * Finite SNR but >25 dB → kernel reads partial-buf for accumulation but
    doesn't reduce; would indicate scaffolding is more advanced than
    documented. Unlikely from code-read.
  * NaN / SNR<25 dB → kernel partially split-aware but reduce missing.
    Would indicate buggy partial-write without aggregation.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("PRIMUS_TURBO_HIPKITTEN_PATH", "/workspace/code/HipKittens")
sys.path.insert(0, "/workspace/code/Primus-Turbo")

import torch
from primus_turbo.pytorch.kernels.hipkitten import loader as _hk_loader
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
    _FP8_SK_WORKSPACE_CACHE,
    _fp8_sk_workspace_bytes,
)

hk = _hk_loader.load_fp8()
fn = hk.grouped_rcr_dscale


def _quantize(t):
    amax = t.abs().max()
    s = (amax / 240.0).clamp(min=1e-6).to(torch.float32)
    q = (t.to(torch.float32) / s).clamp(-240.0, 240.0).to(torch.float8_e4m3fn)
    return q, s


def _snr_db(ref: torch.Tensor, got: torch.Tensor) -> float:
    ref_f = ref.detach().to(torch.float32)
    got_f = got.detach().to(torch.float32)
    sig = (ref_f * ref_f).sum().item()
    err = ((ref_f - got_f) ** 2).sum().item()
    if err == 0.0:
        return float("inf")
    return 10.0 * (sig / err) ** 0.0  # placeholder
    # NOTE: real SNR = 10 * log10(sig/err); using ratio form below.


def _snr(ref: torch.Tensor, got: torch.Tensor) -> tuple[float, float]:
    """Return (max_abs_diff, snr_dB)."""
    import math
    ref_f = ref.detach().to(torch.float32)
    got_f = got.detach().to(torch.float32)
    sig = (ref_f * ref_f).sum().item()
    err = ((ref_f - got_f) ** 2).sum().item()
    max_diff = (ref_f - got_f).abs().max().item()
    if err == 0.0:
        return max_diff, float("inf")
    return max_diff, 10.0 * math.log10(sig / err)


def _call_one(B, M, N, K, sk_split_n: int, ws_ptr: int = 0):
    torch.manual_seed(42)
    g_lens = torch.full((B,), M, dtype=torch.int64, device="cuda")
    g_offs = torch.zeros(B + 1, dtype=torch.int64, device="cuda")
    g_offs[1:] = torch.cumsum(g_lens, dim=0)
    a_bf = torch.randn((B * M, K), dtype=torch.bfloat16, device="cuda")
    b_bf = torch.randn((B, N, K), dtype=torch.bfloat16, device="cuda")
    a, sa = _quantize(a_bf)
    b, sb = _quantize(b_bf)
    out = torch.empty((B * M, N), dtype=torch.bfloat16, device="cuda")
    fn(a, b, out, sa, sb, g_offs, 16,
       m_per_group=M, num_xcds=2,
       num_slots=0, chunk_size=0,
       fuse_ktail_off=0,
       sk_split_n=sk_split_n,
       sk_workspace_ptr=ws_ptr)
    torch.cuda.synchronize()
    return out


def main():
    B, M, N, K = 4, 2048, 2880, 2880
    print(f"Down_B4_M2048 fwd: B={B} M={M} N={N} K={K}")
    print(f"=" * 70)

    # Baseline: sk_split_n=0, no workspace.
    out_base = _call_one(B, M, N, K, sk_split_n=0, ws_ptr=0)
    print(f"sk_split_n=0  out.sum={out_base.to(torch.float32).sum().item():.4e} "
          f"max={out_base.abs().max().item():.4e} "
          f"any_nan={out_base.isnan().any().item()}")

    # End-to-end caller-allocated workspace path.
    m_total = B * M
    buf_bytes = _fp8_sk_workspace_bytes(m_total, N)
    slab = _FP8_SK_WORKSPACE_CACHE.get_or_alloc(buf_bytes, torch.device("cuda:0"))
    ws_ptr = int(slab.data_ptr())
    print(f"workspace bytes = {buf_bytes} ({buf_bytes/1024/1024:.1f} MiB), "
          f"ptr=0x{ws_ptr:x}")

    for sk in (2, 4):
        out_sk = _call_one(B, M, N, K, sk_split_n=sk, ws_ptr=ws_ptr)
        max_d, snr = _snr(out_base, out_sk)
        nan = out_sk.isnan().any().item()
        print(f"sk_split_n={sk}  max_abs_diff={max_d:.6e}  SNR={snr:.2f} dB  any_nan={nan}")

    print()
    print("Interpretation:")
    print("  SNR=inf + no NaN ⇒ kernel is SPLIT-BLIND (R18 kernel branch never landed)")
    print("                     — R52+ must write kernel K-split + reduce")
    print("  finite SNR > 25  ⇒ kernel reads partial-buf but doesn't reduce")
    print("                     — unlikely from code-read")
    print("  SNR < 25 / NaN   ⇒ kernel partially split-aware but reduce missing")


if __name__ == "__main__":
    main()
