import os, re, shutil, sys
JC = "/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-attn-llama31-8b-bwd-20260914-015415/job_context"
SRC = os.path.join(JC, "op", "current")
KERN = "vendor/primus_turbo/triton/attention/fused_mha_bwd_kernel.py"
LOOP = "    for blk_idx in range(num_steps):\n"
DST, NEW = sys.argv[1], sys.argv[2] + "\n"
if os.path.exists(DST): shutil.rmtree(DST)
shutil.copytree(SRC, DST)
p = os.path.join(DST, KERN); s = open(p).read()
idxs = [m.start() for m in re.finditer(re.escape(LOOP), s)]
assert len(idxs) == 2, len(idxs)          # [0] = dk/dv loop, [1] = dq loop
i = idxs[0]
s2 = s[:i] + NEW + s[i + len(LOOP):]
assert s2 != s
open(p, "w").write(s2); import ast; ast.parse(s2); print("ok", DST)
