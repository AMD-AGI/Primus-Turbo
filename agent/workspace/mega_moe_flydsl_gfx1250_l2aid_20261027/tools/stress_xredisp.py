"""stress_xredisp.py T ITERS: the copy-engine x re-dispatch (round 4) against the dispatch_only path.

Every iteration draws a new x, dy and routing, runs the gfx1250 stages fwd + bwd with the side-stream
copies, then checks bitwise: (a) every rank's x slice of this rank's second buffer equals that rank's x,
(b) the dW1 equals variable_k over a dispatch_only of the same x. A stale cache line of the second
buffer from the previous iteration's dW1 would show in (b)."""
import os, sys, torch, torch.distributed as dist

rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl", rank=rank, world_size=world, device_id=torch.device("cuda", rank))
g = dist.group.WORLD
import primus_turbo.pytorch  # noqa
import primus_turbo.pytorch.kernels.fused_mega_moe.fused_mega_moe_gfx1250_impl as G
from primus_turbo.flydsl.grouped_gemm.grouped_gemm_bf16_kernel_gfx1250 import (
    grouped_gemm_bf16_variable_k_flydsl_kernel as VK,
)
from primus_turbo.flydsl.mega.bf16.symm_buffer import get_symm_buffer_for_mega_moe
from primus_turbo.flydsl.mega.bf16_gfx1250.dispatch_gemm_kernel import dispatch_only_gfx1250

T, ITERS = int(sys.argv[1]), int(sys.argv[2])
H, I, EPR, K = 7168, 2048, 32, 8
E = EPR * world
gen = torch.Generator(device="cuda").manual_seed(777 + rank)
randn = lambda *s, scale=1.0: (torch.randn(*s, device="cuda", generator=gen) * scale).to(torch.bfloat16)
w1 = randn(EPR, 2 * I, H, scale=2 / H**0.5)
w2 = randn(EPR, H, I, scale=2 / I**0.5)

bad_buf = bad_dw1 = 0
for it in range(ITERS):
    t = T
    x = randn(t, H)
    dy = randn(t, H)
    logits = torch.randn(t, E, device="cuda", generator=gen)
    if it % 4 == 1:
        logits[:, : E // 4] += 2.0  # skew toward rank 0's experts
    tw, idx = logits.softmax(-1).topk(K, dim=-1)
    l1, dw, h = G.stage1_forward(x, w1, g, idx, tw)
    G.stage2_forward(l1, w2, h, idx, dw)
    gl1, gg, _, _, _ = G.stage2_backward(dy, l1, dw, w2, h)
    _, _, dW1 = G.stage1_backward(gl1, x, w1, h, idx, gg)

    symm = get_symm_buffer_for_mega_moe()
    buf = G.get_x_redispatch(symm).buffers[rank]
    xs = [torch.empty(T, H, dtype=torch.bfloat16, device="cuda") for _ in range(world)]
    ts = [torch.zeros(1, dtype=torch.int64, device="cuda") for _ in range(world)]
    dist.all_gather(ts, torch.tensor([t], device="cuda"))
    xpad = torch.zeros(T, H, dtype=torch.bfloat16, device="cuda")
    xpad[:t] = x
    dist.all_gather(xs, xpad)
    tm = symm.num_max_tokens_per_rank
    ok_buf = all(
        torch.equal(buf[s * tm : s * tm + int(ts[s])], xs[s][: int(ts[s])]) for s in range(world)
    )
    ref_recv = dispatch_only_gfx1250(x, h)
    dW1_ref = VK(
        gl1, ref_recv, h.num_tokens_per_expert_prefix, masked_k=h.num_tokens_per_expert,
        b_index=h.pool_row_to_recv_token,
    ).to(w1.dtype)
    ok_dw1 = torch.equal(dW1, dW1_ref)
    bad_buf += not ok_buf
    bad_dw1 += not ok_dw1
    if not (ok_buf and ok_dw1):
        print(f"rank {rank} it {it} t {t}: buf_ok={ok_buf} dw1_ok={ok_dw1} "
              f"max|d|={(dW1.float() - dW1_ref.float()).abs().max().item():.3e}", flush=True)

torch.cuda.synchronize()
res = torch.tensor([bad_buf, bad_dw1], device="cuda")
dist.all_reduce(res)
if rank == 0:
    print(f"T={T} iters={ITERS}: buffer mismatches={int(res[0])} dW1 mismatches={int(res[1])} (summed over ranks)")
dist.destroy_process_group()
