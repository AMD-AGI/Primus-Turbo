###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Backward STEP3 (fc1 dgrad + combine) fp8 -- smoke correctness + latency + breakdown.

Replicates the backward up to STEP3 on real mega-pool data: forward L1 -> (l1, dispatch_weights),
STEP1 (dispatch(dy)+fc2 dgrad), STEP2 swiglu_backward -> grad_l1 + grad_gate. Then STEP3
(``grouped_gemm_combine_mxfp8_flydsl_kernel`` with ``grad_gate``): fp8 fc1-dgrad + combine PUSH +
unweighted reduce + gate scatter -> dx [T, H] + grad_topk_weights [T, K].

Same kernel / overlap pattern as forward L2 (fc2+combine), but K=2I and ``with_gate=True``.

Run inside the dev container (8 GPUs):
  PYTHONPATH=<repo> python benchmark/ops/bench_step3_fp8.py --num-processes 8 --num-tokens 8192
  PYTHONPATH=<repo> python benchmark/ops/bench_step3_fp8.py --num-processes 8 --breakdown
"""

import argparse
import datetime
import math
import os
import signal
import subprocess
import sys

import numpy as np
import torch
import torch.distributed as dist

import primus_turbo.pytorch  # noqa: F401
from primus_turbo.flydsl.mega.fp8 import (
    dispatch_grouped_gemm_mxfp8,
    dispatch_prologue,
    get_symm_buffer_for_mega_moe,
    grouped_gemm_combine_mxfp8_flydsl_kernel,
    quantize_grouped_weight_mxfp8_flydsl as quantize_grouped_weight_mxfp8,
)
from primus_turbo.flydsl.mega import swiglu_backward_flydsl_kernel
from primus_turbo.flydsl.mega.fp8.quant import colwise_grouped_meta, quantize_rowwise_mxfp8_flydsl
from primus_turbo.pytorch.kernels.mega_moe.mega_moe_backward_fp8_impl import (
    _DW_FP8_FORMAT,
    _dispatch_l2_dgrad_mxfp8_flydsl_kernel,
    _w1t_combine_fp8_cached,
)

_H_GROUP_LENS = 9
_H_GROUP_OFFS = 10


def _run_bench_subprocess(cmd, env, *, timeout_s=180):
    proc = subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        out, _ = proc.communicate(timeout=timeout_s)
        return proc.returncode, out.decode(errors="replace")
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        try:
            out, _ = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            out = b""
        return -9, out.decode(errors="replace") + "\n[TIMEOUT killed process group]"


def _routing(T, K, E, *, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    scores = torch.rand(T, E, generator=g, device=device).abs() + 1
    w, idx = torch.topk(scores.softmax(-1), K, dim=-1)
    return idx.to(torch.int64), w.to(torch.float32)


def _global_weights(E, I, H, device):
    g = torch.Generator(device=device).manual_seed(1234)
    W1 = torch.randn((E, 2 * I, H), generator=g, device=device, dtype=torch.bfloat16) * (2.0 / math.sqrt(H))
    W2 = torch.randn((E, H, I), generator=g, device=device, dtype=torch.bfloat16) * (2.0 / math.sqrt(I))
    return W1, W2


def _bench(fn, *, warmup, iters, group):
    # BACK-TO-BACK timing (host custom-op dispatch overlaps GPU); single-call event-bracket timing
    # would count host dispatch/autotune-lookup as GPU-idle and inflate fast fp8 kernels.
    torch.cuda.synchronize(); group.barrier()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(); group.barrier()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return float(s.elapsed_time(e) / iters)


@torch.no_grad()
def profile(group, args):
    rank, world = group.rank(), group.size()
    H, I, E, K, T, BM, BN = args.hidden, args.inter, args.num_experts, args.num_topk, args.num_tokens, args.bm, args.bn
    epr = E // world

    torch.manual_seed(7 + rank)
    x = torch.randn((T, H), device="cuda", dtype=torch.bfloat16)
    topk_idx, topk_w = _routing(T, K, E, device="cuda", seed=100 + rank)
    W1g, W2g = _global_weights(E, I, H, "cuda")
    W1 = W1g[rank * epr : (rank + 1) * epr].contiguous()
    W2 = W2g[rank * epr : (rank + 1) * epr].contiguous()
    del W1g, W2g

    symm = get_symm_buffer_for_mega_moe(
        group, num_experts=E, num_max_tokens_per_rank=T, num_topk=K, hidden=H,
        intermediate_hidden=I, block_m=BM, block_n=BN, use_mxfp8=True,
    )
    sym_layout = symm.make_sym_layout()
    handle = tuple(dispatch_prologue(
        topk_idx, topk_w, sym_layout=sym_layout, num_tokens=T, num_topk=K, num_experts=E,
        world_size=world, rank=symm.rank, experts_per_rank=epr, block_m=BM,
        num_max_pool_tokens=symm.num_max_pool_tokens,
    ))

    w1q, w1s = quantize_grouped_weight_mxfp8(W1)
    torch.cuda.synchronize(); group.barrier()
    l1 = dispatch_grouped_gemm_mxfp8(x, w1q, w1s, handle, sym_layout, symm, BM=BM, BN=BN)
    dispatch_weights = symm.weight_recv_buf.clone()

    dy = torch.randn((T, H), device="cuda", dtype=torch.bfloat16)
    grad_swiglu, _ = _dispatch_l2_dgrad_mxfp8_flydsl_kernel(dy, W2, group, handle, BM, BN)
    grad_l1, grad_gate = swiglu_backward_flydsl_kernel(
        grad_swiglu, l1, symm.meta_scalars[1:2], scale=dispatch_weights, return_gate=True,
    )

    group_lens = handle[_H_GROUP_LENS]
    group_offs = handle[_H_GROUP_OFFS]
    meta = colwise_grouped_meta(group_lens, group_offs)
    gl1_q_row, gl1_a_sp = quantize_rowwise_mxfp8_flydsl(grad_l1, preshuffle=True)
    w1tf = _w1t_combine_fp8_cached(W1)
    rowwise = (gl1_q_row, gl1_a_sp)
    tki = topk_idx.contiguous().view(-1)
    combine_cu = getattr(args, "combine_cu", 28)
    combine_slots = symm.combine_slots

    def _step3():
        return grouped_gemm_combine_mxfp8_flydsl_kernel(
            None, w1tf, list(handle), group, topk_indices=tki, grad_gate=grad_gate,
            x_fp8_rowwise=rowwise, BM=BM, BN=BN, num_combine_cu=combine_cu,
        )

    dx, grad_topk_weights = _step3()
    dx_ok = tuple(dx.shape) == (T, H) and bool(torch.isfinite(dx.float()).all())
    dtw_ok = (
        grad_topk_weights is not None
        and tuple(grad_topk_weights.shape) == (combine_slots,)
        and bool(torch.isfinite(grad_topk_weights.float()).all())
    )
    dx_norm = float(dx.float().norm())

    t_step3 = _bench(_step3, warmup=args.warmup, iters=args.iters, group=group)
    m_pad = int(handle[_H_GROUP_OFFS][-1].item())
    flops = 2.0 * m_pad * (2 * I) * H  # fc1 dgrad GEMM: [P,2I] @ [2I,H]
    symm.destroy()
    return {
        "dx_ok": float(dx_ok), "dtw_ok": float(dtw_ok), "dx_norm": dx_norm,
        "t_step3": t_step3, "flops": flops, "m_pad": m_pad,
        "mode": getattr(args, "mode", "full"),
    }


def _amax(group, v):
    t = torch.tensor([v], device="cuda"); dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group); return float(t)


def _amin(group, v):
    t = torch.tensor([v], device="cuda"); dist.all_reduce(t, op=dist.ReduceOp.MIN, group=group); return float(t)


def worker(local_rank, world, args):
    ip = os.getenv("MASTER_ADDR", "127.0.0.1")
    port = int(os.getenv("MASTER_PORT", "8492"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl", init_method=f"tcp://{ip}:{port}", world_size=world, rank=local_rank,
        timeout=datetime.timedelta(seconds=int(os.getenv("MEGA_BENCH_TIMEOUT_S", "600"))),
    )
    torch.set_default_device("cuda")
    group = dist.new_group(list(range(world)))
    rank = dist.get_rank()
    try:
        r = profile(group, args)
        dx_ok, dtw_ok = _amin(group, r["dx_ok"]), _amin(group, r["dtw_ok"])
        dx_norm = _amax(group, r["dx_norm"])
        t_step3 = _amax(group, r["t_step3"])
        if rank == 0:
            tf = r["flops"] / (t_step3 * 1e-3) / 1e12
            ok = dx_ok >= 1.0 and dtw_ok >= 1.0
            mode = r.get("mode", "full")
            label = {
                "full": "full (dgrad GEMM+PUSH+reduce+gate)",
                "gemm": "GEMM_ONLY (fc1 dgrad GEMM + mxfp8 epilogue)",
                "push": "PUSH_ONLY (combine XGMI, no GEMM wait)",
                "no_reduce": "NO_REDUCE (GEMM||PUSH overlap)",
            }.get(mode, mode)
            if mode == "full" and not getattr(args, "_breakdown_child", False):
                print(f"\n{'='*72}")
                print(f"[backward STEP3  fc1 dgrad+combine  fp8]  EP{world} T={args.num_tokens} "
                      f"cc={args.combine_cu} H={args.hidden} I={args.inter} E={args.num_experts} K={args.num_topk}")
                print(f"{'='*72}")
            print(f"  {label:42s} : {t_step3:8.3f} ms | {tf:8.1f} TFLOPS  (M_pool={int(r['m_pad'])})")
            if mode == "full" and not getattr(args, "_breakdown_child", False):
                print(f"  [smoke] dx [T,H] finite={bool(dx_ok>=1.0)} (norm={dx_norm:.3e}) | "
                      f"grad_topk [T,K] finite={bool(dtw_ok>=1.0)}  {'PASS' if ok else 'FAIL'} "
                      f"(rigorous dx SNR -> e2e backward gradcheck)")
        torch.cuda.synchronize(); group.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="backward STEP3 (fc1 dgrad + combine) fp8 smoke + breakdown")
    ap.add_argument("--num-processes", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=7168)
    ap.add_argument("--inter", type=int, default=2048)
    ap.add_argument("--num-experts", type=int, default=256)
    ap.add_argument("--num-topk", type=int, default=8)
    ap.add_argument("--num-tokens", type=int, default=8192)
    ap.add_argument("--bm", type=int, default=256)
    ap.add_argument("--bn", type=int, default=256)
    ap.add_argument("--combine-cu", type=int, default=28, help="prod combine default (unified w/ fwd L2)")
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--mode", choices=["full", "gemm", "push", "no_reduce"], default="full")
    ap.add_argument("--breakdown", action="store_true",
                    help="run full/gemm/push/no_reduce in separate processes + overlap summary")
    args = ap.parse_args()

    if args.breakdown:
        if args.num_processes == 1:
            print("[STEP3 dgrad+combine breakdown] need EP>1"); sys.exit(1)
        base_env = os.environ.copy()
        base_env["PYTHONPATH"] = base_env.get("PYTHONPATH", os.getcwd())
        specs = [
            ("full", {}),
            ("gemm", {"PT_COMBINE_GEMM_ONLY": "1"}),
            ("push", {"PT_COMBINE_PUSH_ONLY": "1"}),
            ("no_reduce", {"PT_COMBINE_NO_REDUCE": "1"}),
        ]
        cmd_base = [
            sys.executable, __file__,
            "--num-processes", str(args.num_processes),
            "--num-tokens", str(args.num_tokens),
            "--combine-cu", str(args.combine_cu),
            "--warmup", str(args.warmup),
            "--iters", str(args.iters),
        ]
        print(f"[STEP3 fc1 dgrad+combine breakdown] EP{args.num_processes} T={args.num_tokens} cc={args.combine_cu}")
        times = {}
        for mode, extra in specs:
            env = base_env.copy()
            env.update(extra)
            env["MASTER_PORT"] = str(8700 + hash(mode) % 400)
            cmd = cmd_base + ["--mode", mode]
            rc, out = _run_bench_subprocess(cmd, env, timeout_s=300)
            if rc != 0:
                print(out)
                sys.exit(rc)
            for line in out.splitlines():
                if "ms" in line and ":" in line and "TFLOPS" in line:
                    try:
                        times[mode] = float(line.rsplit(":", 1)[1].split("ms")[0].strip())
                    except ValueError:
                        pass
                    print(line)
        if len(times) >= 4:
            g, p, nr, f = times["gemm"], times["push"], times["no_reduce"], times["full"]
            serial = g + p
            reduce_est = max(f - nr, 0.0)
            overlap_eff = serial / max(nr, 1e-6)
            roofline = max(g, p) / max(nr, 1e-6)
            print(f"\n--- overlap analysis (bwd STEP3, K=2I) ---")
            print(f"  GEMM leg (isolated)          : {g:.3f} ms")
            print(f"  PUSH leg (isolated)          : {p:.3f} ms")
            print(f"  serial sum GEMM+PUSH         : {serial:.3f} ms")
            print(f"  GEMM||PUSH (NO_REDUCE)       : {nr:.3f} ms")
            print(f"  reduce+gate est (full-nr)    : {reduce_est:.3f} ms")
            print(f"  full STEP3 combine           : {f:.3f} ms")
            print(f"  overlap saved                : {serial - nr:.3f} ms  ({100*(serial-nr)/serial:.0f}% of serial)")
            print(f"  overlap vs serial ratio      : {overlap_eff:.2f}x  (ideal {serial/max(g,p):.2f}x if perfect)")
            print(f"  overlap vs max(GEMM,PUSH)    : {roofline:.2f}x  (1.0 = perfect hide shorter leg)")
        sys.exit(0)

    if args.num_processes == 1:
        worker(0, 1, args)
    else:
        if args.mode == "gemm":
            os.environ["PT_COMBINE_GEMM_ONLY"] = "1"
        elif args.mode == "push":
            os.environ["PT_COMBINE_PUSH_ONLY"] = "1"
        elif args.mode == "no_reduce":
            os.environ["PT_COMBINE_NO_REDUCE"] = "1"
        torch.multiprocessing.spawn(worker, args=(args.num_processes, args), nprocs=args.num_processes)
