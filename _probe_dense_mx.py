"""Dense mxfp8 GEMM sanity check: the grouped round touched two shared gemm_helper
symbols (a new ceildiv/floordiv helper pair and StoreCPerTensor's col_safe kwarg), so
confirm the dense path that shares them is unchanged."""
import torch

import primus_turbo.pytorch  # noqa: F401
from primus_turbo.flydsl.gemm.mxfp8_gemm_kernel import gemm_mxfp8_flydsl_kernel as fly_dense
from primus_turbo.pytorch.core.low_precision import float8_e4m3

DEV = "cuda"


def f8(*s):
    t = torch.empty(s, dtype=float8_e4m3, device=DEV)
    t.view(torch.uint8).random_(0, 64)
    return t


def scv(*s):
    return torch.randint(120, 132, s, dtype=torch.uint8, device=DEV)


def deq(x, s):
    xf = x.to(torch.float32)
    e = (s.to(torch.int32) - 127).to(torch.float32)
    return xf * torch.pow(2.0, e).repeat_interleave(32, dim=1)[:, : x.shape[1]]


def snr(ref, got):
    r, g = ref.float(), got.float()
    return 10 * torch.log10((r * r).mean() / (((r - g) ** 2).mean() + 1e-30)).item()


torch.manual_seed(0)
allok = True
for M, N, K in ((1024, 2944, 2944), (2048, 5760, 2944), (512, 2944, 5760)):
    a, b = f8(M, K), f8(N, K)
    a_s, b_s = scv(M, K // 32), scv(N, K // 32)
    ref = deq(a, a_s) @ deq(b, b_s).t()
    got = fly_dense(a, a_s, b, b_s, out_dtype=torch.bfloat16)
    r0 = fly_dense(a, a_s, b, b_s, out_dtype=torch.bfloat16)
    det = torch.equal(got.view(torch.int16), r0.view(torch.int16))
    s = snr(ref, got)
    ok = s >= 25.0 and det
    allok &= ok
    print(f"M={M:5d} N={N:5d} K={K:5d} | {s:6.2f}dB det={det}  " + ("OK" if ok else "***FAIL***"))
print("ALL_OK" if allok else "SOME_FAILED")
