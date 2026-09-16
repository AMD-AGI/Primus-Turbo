import os, sys, torch
sys.path.insert(0,'/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo')
from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager, PrecisionType
from primus_turbo.pytorch.ops import flash_attn_func
from primus_turbo.pytorch.ops.attention import flash_attn_interface as F
import primus_turbo.pytorch.kernels.attention.attention_fused_bwd_impl as FB

hits = {"fused":0, "twokernel":0, "asmfwd":0}
_of = F.dense_fused_backward
def spy(*a, **k):
    hits["fused"] += 1; return _of(*a, **k)
F.dense_fused_backward = spy
_ot = F.triton_dense_backward
def spy2(*a, **k):
    hits["twokernel"] += 1; return _ot(*a, **k)
F.triton_dense_backward = spy2
_oa = F.asm_dense_forward
def spy3(*a,**k):
    hits["asmfwd"] += 1; return _oa(*a,**k)
F.asm_dense_forward = spy3

if os.environ.get("FORCE_TWOKERNEL"):
    F.fused_backward_eligible = lambda *a, **k: False

GlobalBackendManager.set_attn_backend(BackendType.TRITON, PrecisionType.BF16_FP16_FP32)
b,s,hq,hkv,d = 4,4096,32,8,128
q=torch.randn(b,s,hq,d,device="cuda",dtype=torch.bfloat16,requires_grad=True)
k=torch.randn(b,s,hkv,d,device="cuda",dtype=torch.bfloat16,requires_grad=True)
v=torch.randn(b,s,hkv,d,device="cuda",dtype=torch.bfloat16,requires_grad=True)
do=torch.randn(b,s,hq,d,device="cuda",dtype=torch.bfloat16)
o=flash_attn_func(q,k,v,causal=True); o.backward(do)
print("FORCE_TWOKERNEL=",os.environ.get("FORCE_TWOKERNEL",""), hits)
