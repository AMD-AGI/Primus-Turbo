"""Split attention into forward and backward, per profiled step, for each trace.

The JIRA tables give one "AITER fmha" number per machine and do not split fwd from bwd, which
is the split that decides where the gap actually is. Kernel names carry it explicitly on both
stacks -- aiter's fmha_fwd/fmha_bwd, and inductor's flex_attention vs
flex_attention_backward -- so it can be read off rather than inferred.
"""
import gzip, json, sys, re
from collections import defaultdict

def load(f):
    ev = json.load(gzip.open(f))["traceEvents"]
    return [e for e in ev if e.get("ph")=="X" and e.get("cat") in ("kernel","Kernel")]

def bucket(n):
    l = n.lower()
    is_attn = ("fmha" in l or "flex_attention" in l or "attn" in l)
    if not is_attn:
        return None
    # "backward" appears in the inductor kernel name; aiter spells it fmha_bwd.
    if "bwd" in l or "backward" in l:
        return "bwd"
    return "fwd"

for f in sys.argv[1:]:
    k = load(f)
    tot = sum(e["dur"] for e in k)/1000.0
    agg = defaultdict(lambda: [0.0,0])
    gemm = 0.0
    for e in k:
        b = bucket(e["name"])
        if b:
            a = agg[(b, e["name"])]; a[0] += e["dur"]/1000.0; a[1] += 1
        nl = e["name"].lower()
        if "cijk" in nl or "gemm" in nl:
            gemm += e["dur"]/1000.0
    fwd = sum(v[0] for (b,_),v in agg.items() if b=="fwd")
    bwd = sum(v[0] for (b,_),v in agg.items() if b=="bwd")
    print(f"\n=== {f.split('/')[-1][:60]} ===")
    print(f"  GPU kernel 合计 {tot:8.1f} ms     GEMM {gemm:7.1f} ms ({gemm/tot*100:.1f}%)")
    print(f"  attention 前向  {fwd:8.1f} ms")
    print(f"  attention 反向  {bwd:8.1f} ms")
    print(f"  attention 合计  {fwd+bwd:8.1f} ms ({(fwd+bwd)/tot*100:.1f}%)   反向/前向 = {bwd/fwd:.2f}x")
    for (b,n),(ms,c) in sorted(agg.items(), key=lambda x:-x[1][0]):
        print(f"      [{b}] {ms:7.1f} ms n={c:<4} {n[:72]}")
