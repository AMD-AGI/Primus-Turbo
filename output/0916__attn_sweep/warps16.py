import torch, triton, triton.language as tl
@triton.jit
def k(X, BLOCK: tl.constexpr):
    o = tl.arange(0, BLOCK); tl.store(X+o, tl.load(X+o)*2.0)
x = torch.ones(1024, device="cuda")
for w in (8,16,32):
    try:
        h = k[(1,)](x, BLOCK=1024, num_warps=w)
        print("num_warps", w, "-> compiled, metadata num_warps =", h.metadata.num_warps)
    except Exception as e:
        print("num_warps", w, "-> RAISES", type(e).__name__, str(e)[:120])
