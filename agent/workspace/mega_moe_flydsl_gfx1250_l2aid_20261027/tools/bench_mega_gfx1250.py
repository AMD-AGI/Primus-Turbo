"""bench_mega_gfx1250.py: benchmark/ops/training/bench_mega_moe.py's per-stage view of Mega MoE -- gemm_only,
comm_only, fused, the comm the fused kernel hides and the overlap roofline -- on the gfx1250 kernels the
production step runs (fused_mega_moe_gfx1250_impl). The original bench drives the gfx950 kernels, whose
direct-to-LDS buffer loads do not lower on gfx1250.

HIP_VISIBLE_DEVICES=0,1,2,3 ONLY_SHAPE=<label> torchrun --nproc-per-node 4 bench_mega_gfx1250.py [--csv OUT]
One shape per process (a second symmetric-buffer key in one EP4 process can hang its rendezvous).

Per stage (the six GEMMs of one fwd+bwd step, in step order):
  fused     the production launch (dispatch_gemm / gemm_combine / variable-K; dW1 = x re-dispatch + variable-K,
            serial in production)
  gemm_only the gfx1250 FlyDSL grouped GEMM on the already-dispatched pool (the unfused base path's GEMM)
  comm_only dispatch: dispatch_only_gfx1250 at the fused kernel's 64 dispatch workgroups (all-CU figure in
            brackets); combine: bench_mega_moe.combine_only (the shared ep_intranode combine push)
  hidden    gemm_only + comm_only - fused; hidden/comm = the share of the comm the fusion hides
  roofline  max(gemm_only, comm_only) / fused (100% = perfect overlap)
Latencies are the slowest rank (bench_mega_moe.bench per rank, max over ranks).
"""
import argparse, csv, math, os, statistics, sys

import torch
import torch.distributed as dist
import torch.nn.functional as F

rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl", rank=rank, world_size=world, device_id=torch.device("cuda", rank))
TREE = os.environ.get("TURBO_SRC", "/home/xiaompen/Turbo")
sys.path.insert(0, TREE)
import primus_turbo.pytorch  # noqa: E402,F401  (first, so the bench's own sys.path entry cannot shadow TREE)
from primus_turbo.flydsl.grouped_gemm.grouped_gemm_bf16_kernel_gfx1250 import (  # noqa: E402
    grouped_gemm_bf16_nn_flydsl_kernel,
    grouped_gemm_bf16_nt_flydsl_kernel,
    grouped_gemm_bf16_variable_k_flydsl_kernel as variable_k,
)
from primus_turbo.flydsl.mega.bf16.dispatch_prologue_kernel import run_dispatch_prologue  # noqa: E402
from primus_turbo.flydsl.mega.bf16.symm_buffer import get_symm_buffer_for_mega_moe  # noqa: E402
from primus_turbo.flydsl.mega.bf16_gfx1250.dispatch_gemm_kernel import (  # noqa: E402
    dispatch_gemm_nt_gfx1250,
    dispatch_only_gfx1250,
)
from primus_turbo.flydsl.mega.bf16_gfx1250.gemm_combine_kernel import gemm_combine_nt_gfx1250  # noqa: E402
from primus_turbo.flydsl.utils.swiglu_kernel import (  # noqa: E402
    swiglu_backward_flydsl_kernel,
    swiglu_flydsl_kernel,
)
from primus_turbo.pytorch.ops.moe.fused_mega_moe import fused_mega_moe  # noqa: E402

sys.path.insert(1, "/home/xiaompen/Turbo/benchmark/ops/training")
import bench_mega_moe as B  # noqa: E402

SHAPES = [  # label, tokens/rank, hidden, inter, experts/rank, topk, routing
    # bench_mega_moe's DeepSeek-V3 case at EP4 (256 experts, its balanced generate_routing)
    ("bench_dsv3", 8192, 7168, 2048, 256 // world, 8, "balanced"),
    # the campaign shapes (quick_test_bench.py routing: randn logits, + skew on rank 0's experts)
    ("dsv3_t8192_random", 8192, 7168, 2048, 32, 8, 0.0),
    ("dsv3_t8192_skew1", 8192, 7168, 2048, 32, 8, 1.0),
    ("dsv3_t4096_random", 4096, 7168, 2048, 32, 8, 0.0),
]
SHAPES = [s for s in SHAPES if os.environ.get("ONLY_SHAPE", s[0]) == s[0]]
ap = argparse.ArgumentParser()
ap.add_argument("--csv", default=None)
ap.add_argument("--iters", type=int, default=20)
ap.add_argument("--no-dense", action="store_true", help="skip the dense single-weight roofline")
args = ap.parse_args()
dev = torch.device("cuda", rank)
g = dist.group.WORLD
DISPATCH_CU = 64  # dispatch_gemm_nt_gfx1250's num_dispatch_cu


def p0(*a):
    if rank == 0:
        print(*a, flush=True)


def max_over_ranks(v):
    t = torch.tensor([float(v)], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def mean_over_ranks(v):
    t = torch.tensor([float(v)], device=dev)
    dist.all_reduce(t)
    return t.item() / world


def timed(fn):
    torch.cuda.synchronize(); dist.barrier()
    ms = B.bench(fn, warmup=5, iters=args.iters)
    torch.cuda.synchronize(); dist.barrier()
    return max_over_ranks(ms)


def worst_gate(out, ref):
    cos, rel, ok = B.gate3(out, ref)
    t = torch.tensor([cos, -rel, float(ok)], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return t[0].item(), -t[1].item(), bool(t[2].item())


def dense_ms(M, N, K):
    """Dense roofline: the gfx1250 NT kernel on one group of the same M x N x K."""
    if args.no_dense:
        return float("nan")
    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16) / 8
    b = torch.randn(1, N, K, device=dev, dtype=torch.bfloat16) / 8
    offs = torch.tensor([0, M], device=dev, dtype=torch.int64)
    ms = timed(lambda: grouped_gemm_bf16_nt_flydsl_kernel(a, b, offs))
    del a, b
    return ms


rows_out = []
for label, T, H, I, EPR, K, routing in SHAPES:
    E = EPR * world
    gen = torch.Generator(device=dev).manual_seed(1234 + rank)
    randn = lambda *s, scale=1.0: (torch.randn(*s, device=dev, generator=gen) * scale).to(torch.bfloat16)
    w1 = randn(EPR, 2 * I, H, scale=2 / H**0.5)
    w2 = randn(EPR, H, I, scale=2 / I**0.5)
    x = randn(T, H)
    dy = randn(T, H)
    if routing == "balanced":
        torch.manual_seed(rank)
        idx, tw = B.generate_routing(T, K, E, device=dev)
    else:
        logits = torch.randn(T, E, device=dev, generator=gen)
        logits[:, :EPR] += routing
        tw, idx = logits.softmax(-1).topk(K, dim=-1)
    tw = tw.float()

    # ---- the production step's tensors, stage by stage
    handle, dispatch_weights = run_dispatch_prologue(x, w1, g, idx, tw)
    symm = get_symm_buffer_for_mega_moe()
    real_tiles = int(handle.num_tile_blocks[0].item())
    M_eff = real_tiles * 256
    rows = handle.pool_row_to_route.numel()
    counts = torch.bincount(handle.tile_to_expert[:real_tiles].long(), minlength=EPR)[:EPR] * 256
    offs = torch.zeros(EPR + 1, dtype=torch.int64, device=dev)
    offs[1:] = counts.cumsum(0)
    recv_idx = handle.pool_row_to_recv_token[:M_eff].long()
    real = recv_idx < int(symm.num_max_recv_tokens)
    prefix, tok_counts = handle.num_tokens_per_expert_prefix, handle.num_tokens_per_expert

    def gather_pool():
        return F.pad(symm.dispatch_token_buffer, (0, 0, 0, 1))[recv_idx].contiguous()

    def act_fwd(l1):
        return swiglu_flydsl_kernel(l1, num_tile_blocks=handle.num_tile_blocks, scale=dispatch_weights,
                                    tile_to_expert=handle.tile_to_expert)

    l1_out = dispatch_gemm_nt_gfx1250(x, w1, handle)
    torch.cuda.synchronize(); dist.barrier()
    pool_x = gather_pool()
    act = act_fwd(l1_out)
    grad_act = dispatch_gemm_nt_gfx1250(dy, w2, handle, b_nn=True)
    torch.cuda.synchronize(); dist.barrier()
    pool_dy = gather_pool()
    grad_l1, grad_gate, act_w = swiglu_backward_flydsl_kernel(
        grad_act, l1_out, scale=dispatch_weights, return_gate=True, return_act_w=True,
        num_tile_blocks=handle.num_tile_blocks, tile_to_expert=handle.tile_to_expert)
    torch.cuda.synchronize(); dist.barrier()

    disp_bytes = int(handle.expert_send_count[handle.expert_send_dst_rank != rank].sum().item()) * H * 2
    comb_bytes = int(handle.combine_recv_fold_count[handle.combine_recv_dst_rank != rank].sum().item()) * H * 2
    l2_rows = torch.zeros(rows, H, device=dev, dtype=torch.bfloat16)
    topk_flat = idx.to(torch.int64).contiguous().view(-1)
    reduce_ready = torch.zeros(int(symm.num_max_routes), dtype=torch.int64, device=dev)

    def combine_ref(gemm_out, grad_gate_=None):
        """Decoupled reference: GEMM rows -> combine push -> top-k reduce (bench_mega_moe's _ref_fwd)."""
        l2_rows[:M_eff] = gemm_out
        torch.cuda.synchronize(); dist.barrier()
        B.combine_only(g, handle=handle, l2_tokens=l2_rows, grad_gate=grad_gate_)
        torch.cuda.synchronize(); dist.barrier()
        out = torch.empty(int(symm.num_max_tokens_per_rank), H, device=dev, dtype=torch.bfloat16)
        B.topk_reduce_only(out, symm.combine_token_buffer, reduce_ready, topk_flat, symm.num_tokens_per_rank,
                           int(symm.num_max_routes), topk=K, num_experts=E, num_experts_per_rank=EPR, rank=rank)
        torch.cuda.synchronize(); dist.barrier()
        return out[:T]

    stages = []

    def record(name, kind, flops, dense_dims, gemm, comm, fused, check, comm_bytes=0, comm_all=None):
        stages.append(dict(stage=name, kind=kind, flops=flops, dense=dense_ms(*dense_dims), gemm=gemm, comm=comm,
                           fused=fused, check=check, comm_bytes=comm_bytes, comm_all=comm_all))

    # 1. L1 fwd: dispatch x + pool @ w1.T  (NT, N=2I, K=H)
    fl = 2.0 * mean_over_ranks(M_eff) * 2 * I * H
    t_gemm = timed(lambda: grouped_gemm_bf16_nt_flydsl_kernel(pool_x, w1, offs))
    t_comm = timed(lambda: dispatch_only_gfx1250(x, handle, num_dispatch_cu=DISPATCH_CU))
    t_comm_all = timed(lambda: dispatch_only_gfx1250(x, handle))
    t_fused = timed(lambda: dispatch_gemm_nt_gfx1250(x, w1, handle))
    ref = grouped_gemm_bf16_nt_flydsl_kernel(pool_x, w1, offs)
    got = dispatch_gemm_nt_gfx1250(x, w1, handle)[:M_eff]
    record("L1 fwd dispatch_gemm (NT)", "dispatch", fl, (M_eff, 2 * I, H), t_gemm, t_comm, t_fused,
           worst_gate(got[real], ref[real]), disp_bytes, t_comm_all)
    del ref, got

    # 2. L2 fwd: act @ w2.T + combine + reduce  (NT, N=H, K=I)
    fl = 2.0 * mean_over_ranks(M_eff) * H * I
    t_gemm = timed(lambda: grouped_gemm_bf16_nt_flydsl_kernel(act[:M_eff], w2, offs))
    t_comm = timed(lambda: B.combine_only(g, handle=handle, l2_tokens=l2_rows))
    t_fused = timed(lambda: gemm_combine_nt_gfx1250(act, w2, handle, idx))
    ref = combine_ref(grouped_gemm_bf16_nt_flydsl_kernel(act[:M_eff].contiguous(), w2, offs)).clone()
    got = gemm_combine_nt_gfx1250(act, w2, handle, idx)
    record("L2 fwd gemm_combine (NT)", "combine", fl, (M_eff, H, I), t_gemm, t_comm, t_fused,
           worst_gate(got, ref), comb_bytes)
    del ref, got

    # 3. L2 dgrad: dispatch dy + pool @ w2  (NN, N=I, K=H)
    fl = 2.0 * mean_over_ranks(M_eff) * I * H
    t_gemm = timed(lambda: grouped_gemm_bf16_nn_flydsl_kernel(pool_dy, w2, offs))
    t_comm = timed(lambda: dispatch_only_gfx1250(dy, handle, num_dispatch_cu=DISPATCH_CU))
    t_comm_all = timed(lambda: dispatch_only_gfx1250(dy, handle))
    t_fused = timed(lambda: dispatch_gemm_nt_gfx1250(dy, w2, handle, b_nn=True))
    ref = grouped_gemm_bf16_nn_flydsl_kernel(pool_dy, w2, offs)
    got = dispatch_gemm_nt_gfx1250(dy, w2, handle, b_nn=True)[:M_eff]
    record("L2 dgrad dispatch_gemm (NN)", "dispatch", fl, (M_eff, I, H), t_gemm, t_comm, t_fused,
           worst_gate(got[real], ref[real]), disp_bytes, t_comm_all)
    del ref, got

    # 4. dW2[e] = dy_pool[e].T @ act_w[e]  (variable-K, no comm: dy is still in the token buffer)
    fl = 2.0 * mean_over_ranks(M_eff) * H * I
    tb = symm.dispatch_token_buffer
    dw2 = lambda: variable_k(tb, act_w, prefix, masked_k=tok_counts, a_index=handle.pool_row_to_recv_token)
    t_gemm = timed(dw2)
    got = dw2()
    ref = torch.zeros_like(got)
    pd = pool_dy * real[:, None]
    for e in range(EPR):
        s, t = int(offs[e]), int(offs[e + 1])
        if t > s:
            ref[e] = (pd[s:t].T.float() @ act_w[s:t].float()).to(ref.dtype)
    record("dW2 variable-K", "wgrad", fl, (H, I, M_eff), t_gemm, 0.0, t_gemm, worst_gate(got, ref))
    del ref, got, pd

    # 5. L1 dgrad: grad_l1 @ w1 + combine (with the gate grad) + reduce  (NN, N=H, K=2I)
    fl = 2.0 * mean_over_ranks(M_eff) * H * 2 * I
    t_gemm = timed(lambda: grouped_gemm_bf16_nn_flydsl_kernel(grad_l1[:M_eff], w1, offs))
    t_comm = timed(lambda: B.combine_only(g, handle=handle, l2_tokens=l2_rows, grad_gate=grad_gate))
    t_fused = timed(lambda: gemm_combine_nt_gfx1250(grad_l1, w1, handle, idx, grad_gate=grad_gate, b_nn=True))
    ref = combine_ref(grouped_gemm_bf16_nn_flydsl_kernel(grad_l1[:M_eff].contiguous(), w1, offs)).clone()
    got = gemm_combine_nt_gfx1250(grad_l1, w1, handle, idx, grad_gate=grad_gate, b_nn=True)[0]
    record("L1 dgrad gemm_combine (NN)", "combine", fl, (M_eff, H, 2 * I), t_gemm, t_comm, t_fused,
           worst_gate(got, ref), comb_bytes)
    del ref, got

    # 6. dW1[e] = grad_l1[e].T @ x_pool[e]: x re-dispatch (all CUs) then variable-K, serial as in production
    fl = 2.0 * mean_over_ranks(M_eff) * 2 * I * H
    dispatch_only_gfx1250(x, handle)
    dw1 = lambda: variable_k(grad_l1, tb, prefix, masked_k=tok_counts, b_index=handle.pool_row_to_recv_token)
    t_gemm = timed(dw1)
    t_comm = timed(lambda: dispatch_only_gfx1250(x, handle))
    t_fused = timed(lambda: (dispatch_only_gfx1250(x, handle), dw1()))
    got = dw1()
    ref = torch.zeros_like(got)
    gl = grad_l1[:M_eff] * real[:, None]
    for e in range(EPR):
        s, t = int(offs[e]), int(offs[e + 1])
        if t > s:
            ref[e] = (gl[s:t].T.float() @ pool_x[s:t].float()).to(ref.dtype)
    record("dW1 x re-dispatch + variable-K", "wgrad+dispatch", fl, (2 * I, H, M_eff), t_gemm, t_comm, t_fused,
           worst_gate(got, ref), disp_bytes)
    del ref, got, gl

    # ---- the whole step through the production op, for the non-GEMM share
    xg = x.clone().requires_grad_()
    twg = tw.clone().requires_grad_()
    w1g, w2g = w1.clone().requires_grad_(), w2.clone().requires_grad_()

    def step():
        for t in (xg, twg, w1g, w2g):
            t.grad = None
        fused_mega_moe(g, xg, idx, twg, w1g, w2g).backward(dy)

    t_step = timed(step)
    step_flops = 2.0 * T * K * H * 3 * I * 3

    # ---- report (rank 0)
    if rank == 0:
        print(f"\n{'=' * 100}\n[{label}] EP{world} T={T} H={H} I={I} experts/rank={EPR} topk={K} routing={routing} "
              f"pool tiles={real_tiles} (rank 0)  max over ranks\n{'=' * 100}")
    tot = dict(gemm=0.0, comm=0.0, fused=0.0)
    for s in stages:
        gemm_tf = s["flops"] / (s["gemm"] * 1e-3) / 1e12
        fused_tf = s["flops"] / (s["fused"] * 1e-3) / 1e12
        dense_tf = s["flops"] / (s["dense"] * 1e-3) / 1e12 if not math.isnan(s["dense"]) else float("nan")
        serial = s["gemm"] + s["comm"]
        hidden = serial - s["fused"]
        bw = s["comm_bytes"] / (s["comm"] * 1e-3) / 1e9 if s["comm"] > 0 else float("nan")
        cos, rel, ok = s["check"]
        tot["gemm"] += s["gemm"]; tot["comm"] += s["comm"]; tot["fused"] += s["fused"]
        row = {
            "shape": label, "stage": s["stage"],
            "dense ms": s["dense"], "dense TF": dense_tf,
            "gemm_only ms": s["gemm"], "gemm_only TF": gemm_tf,
            "grouped/dense %": 100 * s["dense"] / s["gemm"] if not math.isnan(s["dense"]) else float("nan"),
            "comm_only ms": s["comm"], "comm GB/s": bw,
            "comm all-CU ms": s["comm_all"] if s["comm_all"] is not None else float("nan"),
            "fused ms": s["fused"], "fused TF": fused_tf,
            "serial ms": serial, "hidden ms": hidden,
            "hidden/comm %": 100 * hidden / s["comm"] if s["comm"] > 0 else float("nan"),
            "roofline %": 100 * max(s["gemm"], s["comm"]) / s["fused"],
            "fused/gemm_only %": 100 * s["fused"] / s["gemm"],
            "check": f"{cos:.5f}/{'PASS' if ok else 'FAIL'}",
        }
        rows_out.append(row)
        if rank == 0:
            print(f"  -- {s['stage']}  [{row['check']}]")
            print(f"     dense_gemm : {s['dense']:8.3f} ms | {dense_tf:7.1f} TF (single-weight roofline)")
            print(f"     gemm_only  : {s['gemm']:8.3f} ms | {gemm_tf:7.1f} TF | grouped/dense = "
                  f"{row['grouped/dense %']:.1f}%")
            if s["comm"] > 0:
                extra = f" | all-CU {s['comm_all']:.3f} ms" if s["comm_all"] is not None else ""
                print(f"     comm_only  : {s['comm']:8.3f} ms | {bw:7.1f} GB/s (XGMI){extra}")
            print(f"     fused      : {s['fused']:8.3f} ms | {fused_tf:7.1f} TF | serial {serial:.3f} ms, hidden "
                  f"{hidden:.3f} ms ({row['hidden/comm %']:.0f}% of comm) | roofline {row['roofline %']:.1f}% | "
                  f"fused/gemm_only {row['fused/gemm_only %']:.1f}%")
    hidden = tot["gemm"] + tot["comm"] - tot["fused"]
    summary = {
        "shape": label, "stage": "SUM of the 6 GEMM stages", "gemm_only ms": tot["gemm"],
        "comm_only ms": tot["comm"], "fused ms": tot["fused"], "serial ms": tot["gemm"] + tot["comm"],
        "hidden ms": hidden, "hidden/comm %": 100 * hidden / tot["comm"],
        "roofline %": 100 * max(tot["gemm"], tot["comm"]) / tot["fused"],
        "fused/gemm_only %": 100 * tot["fused"] / tot["gemm"], "step ms": t_step,
        "step TF": step_flops / (t_step * 1e-3) / 1e12, "non-GEMM ms": t_step - tot["fused"],
    }
    rows_out.append(summary)
    p0(f"  == SUM 6 GEMM stages: gemm_only {tot['gemm']:.3f} ms + comm_only {tot['comm']:.3f} ms = serial "
       f"{tot['gemm'] + tot['comm']:.3f} ms -> fused {tot['fused']:.3f} ms, hidden {hidden:.3f} ms "
       f"({summary['hidden/comm %']:.0f}% of comm), fused/gemm_only {summary['fused/gemm_only %']:.1f}%")
    p0(f"  == fused_mega_moe fwd+bwd step {t_step:.3f} ms ({summary['step TF']:.1f} TF); outside the 6 GEMM "
       f"stages (prologue, swiglu fwd/bwd, small ops, gaps): {summary['non-GEMM ms']:.3f} ms")
    del xg, twg, w1g, w2g, pool_x, pool_dy, act, grad_act, grad_l1, l1_out, l2_rows
    torch.cuda.empty_cache()

if rank == 0 and args.csv:
    keys = []
    for r in rows_out:
        keys += [k for k in r if k not in keys]
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows_out)
dist.barrier()
dist.destroy_process_group()
