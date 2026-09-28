import os, sys, json
sys.path.insert(0, "/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/e2e/attn_backends")
sys.path.insert(0, "/home/lihuzhan/.local/flydsl0341")
os.environ["E2E_ATTN"] = "asm"
import torch
from e2e_attn import arms
b, s, hq, hkv, d = 4, 8192, 32, 8, 128
q = torch.randn(b, s, hq, d, device="cuda", dtype=torch.bfloat16); k = torch.randn(b, s, hkv, d, device="cuda", dtype=torch.bfloat16); v = torch.randn_like(k)
f = arms.get_fwd("flyd"); g = arms.get_fwd("fly")
o1, l1 = f(q, k, v, d ** -0.5); o2, l2 = g(q, k, v, d ** -0.5); torch.cuda.synchronize()
print("equal", torch.equal(o1, o2), torch.equal(l1, l2), os.listdir(os.environ["PROF_DUMP_DIR"]))
