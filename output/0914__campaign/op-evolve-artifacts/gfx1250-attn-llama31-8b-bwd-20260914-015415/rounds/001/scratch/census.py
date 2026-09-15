import os,sys,json,glob
sys.argv=[sys.argv[0]]
OP=os.environ["OPDIR"]
cache=os.environ["TRITON_CACHE_DIR"]
sys.path.insert(0,os.path.join(OP,".."))
JC="/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-attn-llama31-8b-bwd-20260914-015415/job_context"
sys.path.insert(0,os.path.join(JC,"op"))
sys.path.insert(0,os.path.join(JC,"op","ut"))
import torch, benchmark
from shapes import ALL_SHAPES
shape=[s for s in ALL_SHAPES if s["name"]=="b4_s8192_hq32_hkv8_d128"][0]
mod=benchmark.load_impl(OP)
q,k,v,do=benchmark.make_inputs(shape)
out=mod.attention(q,k,v,causal=True)
torch.autograd.grad(out,(q,k,v),do,retain_graph=True)
torch.cuda.synchronize()
for f in sorted(glob.glob(cache+"/*/*.json")):
    d=json.load(open(f))
    if "name" not in d: continue
    print(f"{d['name']:<28} regs={d.get('num_regs')} spills={d.get('num_spills')} lds={d.get('shared')} warps={d.get('num_warps')} warp_size={d.get('warp_size')} stages={d.get('num_stages')} waves_per_eu={d.get('waves_per_eu')}")
