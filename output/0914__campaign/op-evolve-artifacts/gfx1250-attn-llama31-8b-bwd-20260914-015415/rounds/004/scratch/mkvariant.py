"""Build a variant copy of op/current with one source edit applied.

usage: mkvariant.py <name> <dest_dir>
Every edit is asserted to have actually fired -- round 3 lost a sweep point to a
patch that silently did nothing.
"""
import os, re, shutil, sys

JC = "/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-attn-llama31-8b-bwd-20260914-015415/job_context"
SRC = os.path.join(JC, "op", "current")
KERN = "vendor/primus_turbo/triton/attention/fused_mha_bwd_kernel.py"

DKDV_LOOP = "    for blk_idx in range(num_steps):\n"   # occurs twice: [0]=dkdv, [1]=dq

ASSUMES = """
    # r4 arm: tell the compiler the strides and ids are non-negative, so the AMD
    # backend can keep the buffer-op offset in 32 bits instead of widening it.
    tl.assume(stride_qm >= 0)
    tl.assume(stride_qd >= 0)
    tl.assume(stride_kn >= 0)
    tl.assume(stride_kd >= 0)
    tl.assume(stride_vn >= 0)
    tl.assume(stride_vd >= 0)
    tl.assume(stride_dom >= 0)
    tl.assume(stride_dod >= 0)
    tl.assume(stride_dqm >= 0)
    tl.assume(stride_dqd >= 0)
    tl.assume(stride_dkn >= 0)
    tl.assume(stride_dkd >= 0)
    tl.assume(stride_dvn >= 0)
    tl.assume(stride_dvd >= 0)
    tl.assume(stride_deltam >= 0)
    tl.assume(stride_qb >= 0)
    tl.assume(stride_qh >= 0)
    tl.assume(stride_kb >= 0)
    tl.assume(stride_kh >= 0)
    tl.assume(stride_vb >= 0)
    tl.assume(stride_vh >= 0)
    tl.assume(stride_dob >= 0)
    tl.assume(stride_doh >= 0)
    tl.assume(stride_dqb >= 0)
    tl.assume(stride_dqh >= 0)
    tl.assume(stride_dkb >= 0)
    tl.assume(stride_dkh >= 0)
    tl.assume(stride_dvb >= 0)
    tl.assume(stride_dvh >= 0)
    tl.assume(stride_deltab >= 0)
    tl.assume(stride_deltah >= 0)
    tl.assume(tl.program_id(0) >= 0)
    tl.assume(tl.program_id(1) >= 0)
    tl.assume(tl.program_id(2) >= 0)
"""


def patch_loop(src, which, newline):
    """which: 0 = dk/dv inner loop, 1 = dq inner loop."""
    idxs = [m.start() for m in re.finditer(re.escape(DKDV_LOOP), src)]
    assert len(idxs) == 2, f"expected 2 inner loops, found {len(idxs)}"
    i = idxs[which]
    return src[:i] + newline + src[i + len(DKDV_LOOP):]


def apply(name, src):
    if name.startswith("base"):
        return src
    if name.startswith("dkdv_unroll"):
        n = int(name.replace("dkdv_unroll", ""))
        return patch_loop(src, 0,
            f"    for blk_idx in tl.range(num_steps, loop_unroll_factor={n}):\n")
    if name.startswith("dq_unroll"):
        n = int(name.replace("dq_unroll", ""))
        return patch_loop(src, 1,
            f"    for blk_idx in tl.range(num_steps, loop_unroll_factor={n}):\n")
    if name == "dkdv_nolicm":
        return patch_loop(src, 0,
            "    for blk_idx in tl.range(num_steps, disable_licm=True):\n")
    if name == "both_nolicm":
        s = patch_loop(src, 0, "    for blk_idx in tl.range(num_steps, disable_licm=True):\n")
        return patch_loop(s, 1, "    for blk_idx in tl.range(num_steps, disable_licm=True):\n")
    if name == "assume":
        anchor = "    # program ids\n    hkid = tl.program_id(0)\n"
        assert src.count(anchor) >= 1, src.count(anchor)
        return src.replace(anchor, anchor + ASSUMES)
    if name == "assume_unroll2":
        s = apply("assume", src)
        return apply("dkdv_unroll2", s)
    if name == "assume_nolicm":
        s = apply("assume", src)
        return apply("dkdv_nolicm", s)
    if name == "unroll2_nolicm":
        s = patch_loop(src, 0,
            "    for blk_idx in tl.range(num_steps, loop_unroll_factor=2, disable_licm=True):\n")
        return s
    raise SystemExit("unknown variant " + name)


name, dest = sys.argv[1], sys.argv[2]
if os.path.exists(dest):
    shutil.rmtree(dest)
shutil.copytree(SRC, dest, ignore=shutil.ignore_patterns("__pycache__"))
p = os.path.join(dest, KERN)
orig = open(p).read()
new = apply(name, orig)
if not name.startswith("base"):
    assert new != orig, f"variant {name} changed nothing"
open(p, "w").write(new)
print(f"[mkvariant] {name} -> {dest}  ({'edited' if new != orig else 'verbatim'})")
