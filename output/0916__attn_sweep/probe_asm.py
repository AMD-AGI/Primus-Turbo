import sys, torch
sys.path.insert(0,'/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo')
import os
os.environ.setdefault("PRIMUS_TURBO_ASM_FWD_TRACE","1")
from primus_turbo.pytorch.kernels.attention.attention_asm_fwd_impl import asm_forward_eligible
q=torch.randn(4,4096,32,128,device="cuda",dtype=torch.bfloat16)
k=torch.randn(4,4096,8,128,device="cuda",dtype=torch.bfloat16)
print("eligible:", asm_forward_eligible(q,k,k,dropout_p=0.0,bias=None,alibi_slopes=None,sink=None,window_size=(-1,-1)))
try:
    import aiter; print("aiter:", aiter.__file__)
except Exception as e:
    print("aiter import FAILED:", type(e).__name__, str(e)[:120])
