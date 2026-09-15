"""r4.i2.g17 -- split the dk/dv pass into two sequential m-loops.

Peak loop-invariant residency goes 768 -> 512 VGPRs: dv (256) is accumulated and
STORED before dk (256) is ever allocated, so the two accumulators are never live
at the same time. Tile, BLOCK_N1, the fusion and the launch config are untouched.
"""
import os, re, shutil, sys

JC = "/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-attn-llama31-8b-bwd-20260914-015415/job_context"
SRC = os.path.join(JC, "op", "current")
KERN = "vendor/primus_turbo/triton/attention/fused_mha_bwd_kernel.py"
DST = sys.argv[1]

if os.path.exists(DST): shutil.rmtree(DST)
shutil.copytree(SRC, DST)
p = os.path.join(DST, KERN)
s = open(p).read()
orig = s

def sub1(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:90])
    s = s.replace(old, new)

# ---- 1. _bwd_dkdv_inner gains PHASE (default 2 = both, so the noncausal
#         caller keeps working unchanged).
sub1("    SLIDING_WINDOW: tl.constexpr,\n):\n    # if HEAD_DIM is padded\n",
     "    SLIDING_WINDOW: tl.constexpr,\n    PHASE: tl.constexpr = 2,  # 0 = dv only, 1 = dk only, 2 = both\n):\n    # if HEAD_DIM is padded\n")

# ---- 2. guard the dV accumulation with PHASE != 1
a0 = "        # Compute dV.\n"
a1 = "                dv = tl.dot(pT.to(do.type.element_ty), do, acc=dv)\n"
i0 = s.index(a0); i1 = s.index(a1, i0) + len(a1)
blk = s[i0:i1]
s = s[:i0] + "        if PHASE != 1:\n" + "".join(
    ("    " + ln if ln.strip() else ln) for ln in blk.splitlines(True)) + s[i1:]

# ---- 3. guard the dK half (Di load, dP, dS, dk accumulate) with PHASE != 0
b0 = "        # D (= delta) is pre-divided by ds_scale.\n"
b1 = "        # Increment pointers.\n"
i0 = s.index(b0); i1 = s.index(b1, i0)
blk = s[i0:i1]
s = s[:i0] + "        if PHASE != 0:\n" + "".join(
    ("    " + ln if ln.strip() else ln) for ln in blk.splitlines(True)) + s[i1:]

# ---- 4. the causal caller: one accumulator, two phases, store in between.
L = s.splitlines(True)
def find(pred, start=0):
    for i in range(start, len(L)):
        if pred(L[i]): return i
    raise AssertionError

z0 = find(lambda l: l == "        dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)\n")
z1 = find(lambda l: l == "        dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)\n", z0)
assert z1 - z0 == 6, z1 - z0
H = find(lambda l: l.startswith("        for hqid in range(hkid * GROUP_SIZE"), z1)
E = find(lambda l: l == "        # end of GQA/MQA of dkdv\n", H)
assert L[E + 13] == "            tl.store(DK + adj_dk + offs_dk_pe, dk_pe, mask=mask_kv)\n", L[E + 13]

body = "".join(("    " + ln if ln.strip() else ln) for ln in L[H:E])
body = body.replace(
    "                dk, dk_pe, dv = _bwd_dkdv_inner(\n"
    "                    dk,  # output tensor\n"
    "                    dk_pe,  # optional output tensor\n"
    "                    dv,  # output tensor\n",
    "                acc_dk, dk_pe, acc_dv = _bwd_dkdv_inner(\n"
    "                    acc,  # output tensor (dv when phase 0, dk when phase 1)\n"
    "                    dk_pe,  # optional output tensor\n"
    "                    acc,  # same buffer; PHASE selects which one is written\n")
assert body.count("acc_dk, dk_pe, acc_dv") == 2, body.count("acc_dk, dk_pe, acc_dv")
body = body.replace(
    "                    SLIDING_WINDOW=SLIDING_WINDOW,\n                )\n",
    "                    SLIDING_WINDOW=SLIDING_WINDOW,\n"
    "                    PHASE=phase,\n                )\n"
    "                if phase == 0:\n"
    "                    acc = acc_dv\n"
    "                else:\n"
    "                    acc = acc_dk\n")
assert body.count("PHASE=phase") == 2, body.count("PHASE=phase")

prologue = (
    "        # r4.i2.g17: two sequential m-loops. `phase` is a plain Python loop,\n"
    "        # so the body is emitted twice and `acc` is dead after each\n"
    "        # store -- dv and dk are never live at the same time.\n"
    "        for phase in tl.static_range(2):\n"
    "            acc = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)\n"
    "            if HAS_PE:\n"
    "                dk_pe = tl.zeros([BLOCK_N1, PE_HEAD_DIM], dtype=tl.float32)\n"
    "            else:\n"
    "                dk_pe = acc\n")
epilogue = (
    "            # end of GQA/MQA of dkdv for this phase\n"
    "            if phase == 0:\n"
    "                adj_dv = bid * stride_dvb + hkid * stride_dvh + k_start * stride_dvn\n"
    "                offs_dv = offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvd\n"
    "                tl.store(DV + adj_dv + offs_dv, acc, mask=mask_kv)\n"
    "            else:\n"
    "                adj_dk = bid * stride_dkb + hkid * stride_dkh + k_start * stride_dkn\n"
    "                offs_dk = offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkd\n"
    "                acc *= sm_scale\n"
    "                tl.store(DK + adj_dk + offs_dk, acc, mask=mask_kv)\n"
    "                if HAS_PE:\n"
    "                    offs_dk_pe = offs_n[:, None] * stride_dkn + offs_d_pe[None, :] * stride_dkd\n"
    "                    dk_pe *= sm_scale\n"
    "                    tl.store(DK + adj_dk + offs_dk_pe, dk_pe, mask=mask_kv)\n")

L = L[:z0] + L[z0 + 7:H] + [prologue, body, epilogue] + L[E + 14:]
s = "".join(L)
assert s != orig
open(p, "w").write(s)
import ast; ast.parse(s)
print("patched", p)
