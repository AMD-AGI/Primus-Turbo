import os, sys
os.environ["TORCH_BLAS_PREFER_HIPBLASLT"]="0"
os.environ["HIP_VISIBLE_DEVICES"]=os.environ.get("GPU","1")
spec = sys.argv[1] if len(sys.argv)>1 else ""
if spec: os.environ["PRIMUS_TURBO_ATTN_TRITON_TUNE"]=spec
sys.path.insert(0,"/home/lihuzhan/code/2026_0903__turbo/wt-bakeoff")
import primus_turbo.triton.attention.attention_kernel as ak
print("module file:", ak.__file__)
def dump(name,k):
    cfgs=[(c.num_warps,c.num_stages,dict(c.kwargs)) for c in k.configs]
    print(f"  {name}: n={len(cfgs)} {cfgs}")
print("SPEC=",repr(spec))
print("decorated kernel objects (the ones that actually launch):")
dump("attn_fwd", ak.attn_fwd)
dump("_bwd_kernel_dkdv", ak._bwd_kernel_dkdv)
dump("_bwd_kernel_dq", ak._bwd_kernel_dq)
print("identity: dkdv.configs is module list:", ak._bwd_kernel_dkdv.configs is ak.autotune_bwd_configs,
      "| dq is same list:", ak._bwd_kernel_dq.configs is ak.autotune_bwd_configs)
print("harness assert_config_applied sees:")
sys.path.insert(0,"/home/lihuzhan/code/2026_0903__turbo/wt-bakeoff/tools/gfx1250")
