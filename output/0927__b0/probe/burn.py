# Worst-case neighbour load for the interference probe: bf16 GEMM until killed or DURATION s.
import os, time, torch
dur = float(os.environ.get("DURATION", "3600"))
a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
t = time.time()
while time.time() - t < dur:
    for _ in range(20):
        a @ a
    torch.cuda.synchronize()
