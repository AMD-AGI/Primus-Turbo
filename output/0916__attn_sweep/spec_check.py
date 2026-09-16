import os, sys
sys.path.insert(0,'/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo')
os.environ["PRIMUS_TURBO_ATTN_TRITON_TUNE"]="fwd:num_warps=2,num_stages=3,waves_per_eu=1,PRE_LOAD_V=True;bwd:num_warps=16,num_stages=1,waves_per_eu=2"
from primus_turbo.triton.attention import attention_kernel as ak
for half, fn in (("fwd", ak.get_autotune_fwd_configs), ("bwd", ak.get_autotune_bwd_configs)):
    cfgs, keys = fn()
    print(half, [ (dict(c.kwargs), c.num_warps, c.num_stages) for c in cfgs ])
print("sweep grid size:", len(ak._build_configs("sweep", {"num_stages":1,"num_warps":4}, {"num_stages":1,"num_warps":4})))
print("sweep grid kwargs sample:", ak._build_configs("sweep", {"PRE_LOAD_V":False,"num_stages":2,"num_warps":2}, {"PRE_LOAD_V":False,"num_stages":2,"num_warps":2})[0].kwargs)
try:
    ak._parse_tune_spec("bwd:nwarps=2","bwd")
except Exception as e:
    print("typo ->", type(e).__name__, e)
