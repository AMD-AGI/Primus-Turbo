"""Time the a16 un-permute pass alone, against its 2x|dQ| roofline."""
import os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import primus_turbo.flydsl.attention.flash_attn_bwd as M

B, Hq, S, D = (int(x) for x in sys.argv[1:5])
if os.environ.get("BLK"):
    M._A16_BLOCK = int(os.environ["BLK"])
if os.environ.get("UC"):
    M._A16_UC = int(os.environ["UC"])
img = torch.randn(B * S * Hq * D, dtype=torch.bfloat16, device="cuda")
dq = torch.empty(S, B, Hq, D, dtype=torch.bfloat16, device="cuda")
st = torch.cuda.current_stream()
for _ in range(5):
    M._unpermute_dq_a16(img, dq, B, S, Hq, D, 0.6931, st)
torch.cuda.synchronize()
b = 1e9
for _ in range(60):
    t0 = time.perf_counter()
    M._unpermute_dq_a16(img, dq, B, S, Hq, D, 0.6931, st)
    torch.cuda.synchronize()
    b = min(b, (time.perf_counter() - t0) * 1e3)
n = B * S * Hq * D * 2
print("  blk=%d unpermute %.4f ms  %.2f TB/s (1R:1W %d MB)" % (M._A16_BLOCK, b, 2 * n / b * 1e-9, n >> 20))
