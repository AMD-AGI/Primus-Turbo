"""定位 race 发生的位置：哪些 row、哪个 group、哪种值差异。"""

import torch

import primus_turbo  # noqa: F401
import primus_turbo.pytorch  # noqa: F401

DEVICE = torch.device("cuda")
DTYPE_FP8 = torch.float8_e4m3fn
DTYPE_OUT = torch.bfloat16
N_ITERS = 10000


def quantize_mx(x, axis):
    return torch.ops.primus_turbo_cpp_extension.quantize_mxfp8(
        x, DTYPE_FP8, axis, False, False, False
    )


def dq(q, s):
    f = s.view(torch.uint8).to(torch.int32) - 127
    f = (2.0 ** f.float()).repeat_interleave(32, dim=-1)
    return q.float() * f


def run(G, lens_list, n=8192, k=2048, seed=2):
    torch.manual_seed(seed)
    total_m = sum(lens_list)
    a_hp = torch.randn(total_m, k, device=DEVICE, dtype=torch.bfloat16) * 0.5
    b_hp = torch.randn(G, n, k, device=DEVICE, dtype=torch.bfloat16) * 0.5
    oa = quantize_mx(a_hp, axis=1)
    a_fp8, a_s = oa[0], oa[1]
    ob = quantize_mx(b_hp.reshape(G * n, k), axis=1)
    b_fp8 = ob[0].reshape(G, n, k)
    b_s = ob[1].reshape(G, n, -1)
    lens_t = torch.tensor(lens_list, dtype=torch.int64, device=DEVICE)
    offs_t = torch.ops.primus_turbo_cpp_extension.grouped_gemm_compute_offs(lens_t)

    a_f = dq(a_fp8, a_s)
    b_f = dq(b_fp8.reshape(-1, k), b_s.reshape(-1, k // 32)).reshape(b_fp8.shape)
    ref = torch.empty(total_m, n, dtype=DTYPE_OUT, device=DEVICE)
    cum = 0
    group_row_ranges = []
    for g in range(G):
        Mg = int(lens_t[g].item())
        ref[cum : cum + Mg].copy_((a_f[cum : cum + Mg] @ b_f[g].T).to(DTYPE_OUT))
        group_row_ranges.append((cum, cum + Mg, g))
        cum += Mg

    bad_iters = []
    for t in range(N_ITERS):
        out = torch.ops.primus_turbo_cpp_extension.turbo_grouped_gemm_fp8(
            a_fp8, b_fp8, a_s, b_s, lens_t, offs_t, False, True, DTYPE_OUT, "MX_BLOCKWISE"
        )
        torch.cuda.synchronize()
        d = (ref.float() - out.float()).abs()
        max_diff = d.max().item()
        if max_diff > 1.0:
            # find which row, which col, which group
            row_diff = d.max(dim=1).values
            bad_rows = torch.nonzero(row_diff > 1.0).flatten().tolist()
            row_min = bad_rows[0]
            row_max = bad_rows[-1]
            n_bad_rows = len(bad_rows)
            # which group?
            groups_hit = set()
            for r in bad_rows:
                for s, e, g in group_row_ranges:
                    if s <= r < e:
                        groups_hit.add(g)
                        break
            bad_iters.append((t, max_diff, n_bad_rows, row_min, row_max, sorted(groups_hit)))

    print(f"\nG={G} {lens_list} k={k} seed={seed}: {len(bad_iters)}/{N_ITERS} BAD")
    for t, mx, nr, rmn, rmx, gs in bad_iters:
        print(f"  iter={t:4d} max_diff={mx:.2f} bad_rows={nr} row_range=[{rmn},{rmx}] groups={gs}")


for cfg in [
    (4, [8192] * 4),  # historically most race-prone large-K
]:
    run(*cfg)
