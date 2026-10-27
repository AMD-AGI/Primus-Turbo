"""EP1 staged-bias GPT-OSS case of test_mega_bf16_gfx1250_fused_moe.py with every SNR printed (no assert).

python3 probe_staged.py [--shape 1024 2880 2880 32 4] [--no_bias]
"""
import argparse, os, socket, sys, torch, torch.distributed as dist

sys.path.insert(0, os.environ.get("TURBO_SRC", "/home/xiaompen/Turbo"))
import primus_turbo.pytorch  # noqa
from primus_turbo.flydsl.utils.glu_activation import GLUActivation
from primus_turbo.pytorch.ops.moe.fused_mega_moe import fused_mega_moe_stage1, fused_mega_moe_stage2
from tests.pytorch.ops.test_mega_bf16_gfx1250_fused_moe import _reference, _snr

ap = argparse.ArgumentParser()
ap.add_argument("--shape", type=int, nargs=5, default=[1024, 2880, 2880, 32, 4])
ap.add_argument("--no_bias", action="store_true")
ap.add_argument("--reps", type=int, default=2)
args = ap.parse_args()
num_tokens, hidden, inter, num_experts, num_topk = args.shape
with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(s.getsockname()[1]))
dist.init_process_group("nccl", rank=0, world_size=1, device_id=torch.device("cuda", 0))
g = dist.group.WORLD
activation = GLUActivation.swigluoai()
torch.manual_seed(num_tokens)
w1 = (torch.randn(num_experts, 2 * inter, hidden, device="cuda") * 5 / hidden**0.5).to(torch.bfloat16)
w2 = (torch.randn(num_experts, hidden, inter, device="cuda") * 2 / inter**0.5).to(torch.bfloat16)
b1 = torch.randn(num_experts, 2 * inter, device="cuda").to(torch.bfloat16)
b2 = torch.randn(num_experts, hidden, device="cuda").to(torch.bfloat16)
if args.no_bias:
    b1 = b2 = None
for rep in range(args.reps):
    x = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
    topk_w, topk_idx = torch.randn(num_tokens, num_experts, device="cuda").softmax(-1).topk(num_topk, dim=-1)
    dy = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
    src = [x, topk_w, w1, w2] + ([] if args.no_bias else [b1, b2])
    leaves = [t.clone().requires_grad_() for t in src]
    refs = [t.detach().float().requires_grad_() for t in src]
    lb = leaves[4:] if not args.no_bias else [None, None]
    rb = refs[4:] if not args.no_bias else [None, None]
    l1_out, dispatch_weights, handle = fused_mega_moe_stage1(leaves[0], topk_idx, leaves[1], leaves[2], g)
    y = fused_mega_moe_stage2(l1_out, dispatch_weights, handle, topk_idx, leaves[1], leaves[3], g, activation, *lb)
    y.backward(dy)
    yr = _reference(refs[0], topk_idx, refs[1], refs[2], refs[3], activation, *rb)
    yr.backward(dy.float())
    names = ("dx", "d topk_w", "dW1", "dW2", "d fc1_bias", "d fc2_bias")
    snrs = [f"y {_snr(y, yr.detach()):.1f}"] + [f"{n} {_snr(a.grad, b.grad):.1f}" for n, a, b in zip(names, leaves, refs)]
    print(f"rep {rep}: " + ", ".join(snrs), flush=True)
dist.destroy_process_group()
