import torch, triton, triton.language as tl
@triton.jit
def k(X, n, BLOCK: tl.constexpr):
    o = tl.arange(0, BLOCK)
    tl.store(X+o, tl.load(X+o, mask=o<n)*2.0, mask=o<n)
x = torch.ones(128, device="cuda")
for wpe in (0,1,2,4):
    h = k[(1,)](x, 128, BLOCK=128, num_warps=2, waves_per_eu=wpe)
    md = h.metadata
    print("wpe", wpe, "-> metadata.waves_per_eu =", getattr(md,"waves_per_eu","ABSENT"), " hash:", h.hash[:16])
    asm = h.asm["amdgcn"]
    line = [l for l in asm.splitlines() if "waves_per_eu" in l or "amdhsa_accum" in l]
    print("   asm marks:", line[:2])
try:
    k[(1,)](x, 128, BLOCK=128, bogus_key=3)
    print("BOGUS ACCEPTED SILENTLY")
except Exception as e:
    print("BOGUS raises:", type(e).__name__, str(e)[:200])
