#!/usr/bin/env python3
"""Benchmark: old Triton vs new Triton (e4m3/e4m3) vs mixed Triton (e4m3/e5m2) vs ASM_CO.

All variants use beta=1 (fused accumulation). Old Triton is the baseline.
ASM_CO is benchmarked first per shape to avoid Triton→ASM GPU state issues.

Scaling:
  - Old Triton (v_mfma_f32_16x16x32_fp8_fp8): natively fnuz, raw scales.
  - New Triton (tl.dot_scaled e4m3/e4m3): OCP, applies *0.25 internally.
  - Mixed Triton (tl.dot_scaled e4m3/e5m2): OCP, applies *0.25 internally. RHS is e5m2.
  - ASM_CO (same f8f6f4): expects pre-adjusted scales (2x each) to cancel implicit 0.25x.
"""

import ctypes
import struct
import time
import os

import torch

os.environ.setdefault("ROCR_VISIBLE_DEVICES", "0")
torch.cuda.set_device(0)

from primus_turbo.triton.gemm.gemm_kernel import _is_gfx950, _set_knobs_gfx950
from primus_turbo.triton.grouped_gemm.grouped_gemm_kernel import (
    NUM_XCDS, _get_num_cus, _grouped_variable_k_gemm_kernel,
)
from primus_turbo.triton.grouped_gemm.grouped_gemm_fp8_kernel import (
    _grouped_variable_k_dot_scaled_gemm_kernel,
    _grouped_variable_k_dot_scaled_mixed_gemm_kernel,
    _get_gg_fp8_tw_vk_config,
)
from primus_turbo.asm_co.grouped_gemm.loader import get_asm_co_wgrad_beta1_func
from primus_turbo.asm_co.hip_utils import asm_co_module_launch

SHAPES = [
    ("down_proj (2880x2880)", 131072, 32, 2880, 2880),
    ("gate_up  (2880x5760)", 131072, 32, 2880, 5760),
]

WARMUP = 10
REPS = 50
ASM_THREADS = 1024
ASM_LDS = 65536

if _is_gfx950():
    _set_knobs_gfx950()

num_sms = _get_num_cus()
asm_func = get_asm_co_wgrad_beta1_func()


def bench(fn, warmup=WARMUP, reps=REPS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000.0


def main():
    hdr = (f"{'Shape':<25} {'Triton-old':>12} {'new e4/e4':>12} {'mix e4/e5':>12} {'ASM_CO':>12}"
           f" {'e4e4/old':>9} {'mix/old':>9} {'asm/old':>9}"
           f" {'old ms':>9} {'e4e4 ms':>9} {'mix ms':>9} {'asm ms':>9}")
    sep = "-" * len(hdr)

    print("=" * len(hdr))
    print("Wgrad Kernel Benchmark  [beta=1, TFLOPS]")
    print(f"  Baseline: Triton-old (tl.dot)  |  GPU CUs: {num_sms}  |  Warmup: {WARMUP}  |  Reps: {REPS}")
    print("=" * len(hdr))
    print(hdr)
    print(sep)

    for name, M, G, OM, ON in SHAPES:
        lhs = torch.randn(M, OM, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        rhs_e4 = torch.randn(M, ON, device="cuda", dtype=torch.bfloat16)
        rhs_e4m3 = rhs_e4.to(torch.float8_e4m3fn)
        rhs_e5m2 = rhs_e4.to(torch.float8_e5m2)
        del rhs_e4

        ls = torch.tensor(1.0, device="cuda", dtype=torch.float32)
        rs = torch.tensor(1.0, device="cuda", dtype=torch.float32)
        ls_adj = (ls * 2.0).contiguous()
        rs_adj = (rs * 2.0).contiguous()

        seg = M // G
        go = torch.arange(0, M + 1, seg, device="cuda", dtype=torch.int64)[:G + 1]
        go[-1] = M

        bm, bn, bk, gm, ca, cb, ns, cs = _get_gg_fp8_tw_vk_config(
            OM, ON, max(M // max(G, 1), 256), lhs.dtype, rhs_e4m3.dtype, G, num_sms)

        tf = 2.0 * G * OM * ON * (M / G)

        out_asm = torch.zeros(G, OM, ON, device="cuda", dtype=torch.bfloat16)
        out_old = torch.zeros(G, OM, ON, device="cuda", dtype=torch.bfloat16)
        out_new = torch.zeros(G, OM, ON, device="cuda", dtype=torch.bfloat16)
        out_mix = torch.zeros(G, OM, ON, device="cuda", dtype=torch.bfloat16)

        def run_asm(_out=out_asm, _lhs=lhs, _rhs=rhs_e4m3, _ls=ls_adj, _rs=rs_adj,
                    _go=go, _G=G, _OM=OM, _ON=ON):
            buf = ctypes.create_string_buffer(96)
            struct.pack_into("<QQQQQQ", buf, 0,
                             _lhs.data_ptr(), _rhs.data_ptr(), _out.data_ptr(),
                             _ls.data_ptr(), _rs.data_ptr(), _go.data_ptr())
            struct.pack_into("<iiiiiiii", buf, 48,
                             _G, _OM, _ON, _lhs.stride(0), _rhs.stride(0),
                             _out.stride(0), _out.stride(1), _out.stride(2))
            asm_co_module_launch(asm_func, buf, 96, num_sms, ASM_THREADS, ASM_LDS,
                                 _lhs.device, "wgrad_beta1")

        def run_old(_out=out_old, _lhs=lhs, _rhs=rhs_e4m3, _ls=ls, _rs=rs,
                    _go=go, _G=G, _OM=OM, _ON=ON, _bm=bm, _bn=bn, _bk=bk,
                    _gm=gm, _ca=ca, _cb=cb, _ns=ns, _cs=cs):
            _grouped_variable_k_gemm_kernel[(num_sms,)](
                _lhs, _rhs, _out, _ls, _rs, _go, _G, _OM, _ON,
                _lhs.stride(0), _rhs.stride(0),
                _out.stride(0), _out.stride(1), _out.stride(2),
                stride_lhs_n=1, stride_rhs_n=1,
                BLOCK_SIZE_M=_bm, BLOCK_SIZE_N=_bn, BLOCK_SIZE_K=_bk,
                GROUP_SIZE_M=_gm, NUM_SMS=num_sms, NUM_XCDS=NUM_XCDS,
                CHUNK_SIZE=_cs, IS_FP8=True,
                CACHE_MODIFIER_A=_ca, CACHE_MODIFIER_B=_cb,
                BETA_IS_ONE=True,
                num_warps=8, num_stages=_ns,
                waves_per_eu=0, matrix_instr_nonkdim=16, kpack=1)

        def run_new(_out=out_new, _lhs=lhs, _rhs=rhs_e4m3, _ls=ls, _rs=rs,
                    _go=go, _G=G, _OM=OM, _ON=ON, _bm=bm, _bn=bn, _bk=bk,
                    _gm=gm, _ca=ca, _cb=cb, _ns=ns, _cs=cs):
            _grouped_variable_k_dot_scaled_gemm_kernel[(num_sms,)](
                _lhs, _rhs, _out, _ls, _rs, _go, _G, _OM, _ON,
                _lhs.stride(0), _rhs.stride(0),
                _out.stride(0), _out.stride(1), _out.stride(2),
                BLOCK_SIZE_M=_bm, BLOCK_SIZE_N=_bn, BLOCK_SIZE_K=_bk,
                GROUP_SIZE_M=_gm, NUM_SMS=num_sms, NUM_XCDS=NUM_XCDS,
                CHUNK_SIZE=_cs,
                CACHE_MODIFIER_A=_ca, CACHE_MODIFIER_B=_cb,
                num_warps=8, num_stages=_ns,
                waves_per_eu=0, matrix_instr_nonkdim=16, kpack=1)

        def run_mix(_out=out_mix, _lhs=lhs, _rhs=rhs_e5m2, _ls=ls, _rs=rs,
                    _go=go, _G=G, _OM=OM, _ON=ON, _bm=bm, _bn=bn, _bk=bk,
                    _gm=gm, _ca=ca, _cb=cb, _ns=ns, _cs=cs):
            _grouped_variable_k_dot_scaled_mixed_gemm_kernel[(num_sms,)](
                _lhs, _rhs, _out, _ls, _rs, _go, _G, _OM, _ON,
                _lhs.stride(0), _rhs.stride(0),
                _out.stride(0), _out.stride(1), _out.stride(2),
                BLOCK_SIZE_M=_bm, BLOCK_SIZE_N=_bn, BLOCK_SIZE_K=_bk,
                GROUP_SIZE_M=_gm, NUM_SMS=num_sms, NUM_XCDS=NUM_XCDS,
                CHUNK_SIZE=_cs,
                CACHE_MODIFIER_A=_ca, CACHE_MODIFIER_B=_cb,
                num_warps=8, num_stages=_ns,
                waves_per_eu=0, matrix_instr_nonkdim=16, kpack=1)

        ms_asm = bench(run_asm)
        ms_old = bench(run_old)
        ms_new = bench(run_new)
        ms_mix = bench(run_mix)

        tf_old = tf / (ms_old * 1e9)
        tf_new = tf / (ms_new * 1e9)
        tf_mix = tf / (ms_mix * 1e9)
        tf_asm = tf / (ms_asm * 1e9)

        print(f"{name:<25} {tf_old:>12.2f} {tf_new:>12.2f} {tf_mix:>12.2f} {tf_asm:>12.2f}"
              f" {ms_old/ms_new:>8.2f}x {ms_old/ms_mix:>8.2f}x {ms_old/ms_asm:>8.2f}x"
              f" {ms_old:>9.3f} {ms_new:>9.3f} {ms_mix:>9.3f} {ms_asm:>9.3f}")

        del lhs, rhs_e4m3, rhs_e5m2, out_asm, out_old, out_new, out_mix, ls, rs, ls_adj, rs_adj
        torch.cuda.empty_cache()

    print("=" * len(hdr))


if __name__ == "__main__":
    main()
