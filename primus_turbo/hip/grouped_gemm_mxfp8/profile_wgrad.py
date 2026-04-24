###############################################################################
# Fine-grained timing for the HIP wgrad v1 path — decompose the 4.1 ms into
# transpose / quant / fwd-kernel.
###############################################################################
from __future__ import annotations
import torch
from primus_turbo.hip.grouped_gemm_mxfp8 import grouped_gemm_mxfp8_hip_fwd
from primus_turbo.triton.quantization.mxfp8_quant_kernels import quant_mxfp8_rowwise


def time_stmt(f, iters=50):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(5):
        f()
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        f()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def main():
    m, k, n, g = 65536, 2880, 5760, 32
    m_g = m // g
    device = "cuda"
    torch.manual_seed(0)
    go = torch.randn(m, n, device=device, dtype=torch.bfloat16)
    a  = torch.randn(m, k, device=device, dtype=torch.bfloat16)

    # Step 1: permute-contig
    def step1_go():  return go.view(g, m_g, n).permute(0, 2, 1).contiguous()
    def step1_a():   return a.view(g, m_g, k).permute(0, 2, 1).contiguous()

    go_T = step1_go()
    a_T  = step1_a()

    # Step 2: batched quant
    def step2_go():  return quant_mxfp8_rowwise(go_T.view(g * n, m_g))
    def step2_a():   return quant_mxfp8_rowwise(a_T.view(g * k, m_g))

    go_fp8, go_scale = step2_go()
    a_fp8,  a_scale  = step2_a()
    a_fp8_3d   = a_fp8.view(g, k, m_g)
    a_scale_3d = a_scale.view(g, k, m_g // 32)
    group_offs = torch.arange(0, g * n + 1, n, dtype=torch.int64, device=device)

    # Step 3: fwd kernel
    def step3():
        return grouped_gemm_mxfp8_hip_fwd(
            go_fp8, a_fp8_3d, go_scale, a_scale_3d, group_offs,
            out_dtype=torch.bfloat16,
        )

    t_p_go = time_stmt(step1_go)
    t_p_a  = time_stmt(step1_a)
    t_q_go = time_stmt(step2_go)
    t_q_a  = time_stmt(step2_a)
    t_fwd  = time_stmt(step3)

    total = t_p_go + t_p_a + t_q_go + t_q_a + t_fwd
    print(f"permute-contig grad_out [M,N]->[G,N,Mg]: {t_p_go:6.3f} ms")
    print(f"permute-contig a        [M,K]->[G,K,Mg]: {t_p_a:6.3f} ms")
    print(f"rowwise quant grad_out_T [G*N, Mg]     : {t_q_go:6.3f} ms")
    print(f"rowwise quant a_T        [G*K, Mg]     : {t_q_a:6.3f} ms")
    print(f"fwd kernel (G*N, K, Mg)                : {t_fwd:6.3f} ms")
    print(f"sum                                    : {total:6.3f} ms")


if __name__ == "__main__":
    main()
