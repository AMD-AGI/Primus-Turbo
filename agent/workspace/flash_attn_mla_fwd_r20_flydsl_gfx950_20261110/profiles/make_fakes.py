"""Build wrong-result upper-bound variants of src/branch_mla under profiles/fakes/<name>/branch_mla.

noexp: exp2 -> one FMA; nokv: no K / V DMA inside the main loop (the `load_[kv]_tile(j_idx ...)`
calls; the drain's use max_m*); both: the two.
"""

import os
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src", "branch_mla")
EXP_OLD = "        e = rocdl.exp2(T.f32, as_mlir_value(x))\n"
EXP_NEW = "        e = (fx.Float32(x) * fx.Float32(0.0625) + fx.Float32(0.5)).ir_value()\n"
VARIANTS = {"noexp": (True, False, 2001), "nokv": (False, True, 2002), "both": (True, True, 2003)}
# ldshalf: odd K slots and odd P*V V steps reuse the preceding packs (half the ds_read bytes, same MFMAs).
K_OLD = "            k_lo[ks] = _load_k_pack_aligned(\n"
K_NEW = (
    "            if const_expr(ks % 2 == 1 and k_lo[ks - 1] is not None):\n"
    "                k_lo[ks] = k_lo[ks - 1]\n"
    "                k_hi[ks] = k_hi[ks - 1]\n"
    "                continue\n"
    "            k_lo[ks] = _load_k_pack_aligned(\n"
)
V_OLD = (
    "                a = _ds_read_tr16_b64(traits, self.v_lds_read_vec4_type, lane_ptrs[dc % 2], imm, buf_id)\n"
)
V_NEW = (
    "                if const_expr(step % 2 == 1 and packs[step - 1][dc] is not None):\n"
    "                    packs[step][dc] = packs[step - 1][dc]\n"
    "                    continue\n"
    + V_OLD
)


def make_ldshalf():
    dst = os.path.join(HERE, "fakes", "ldshalf", "branch_mla")
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(SRC, dst, ignore=shutil.ignore_patterns("__pycache__"))
    path = os.path.join(dst, "attn_helper_mla.py")
    helper = open(path).read()
    helper, n = re.subn(r"^CODEGEN_REV = \d+$", "CODEGEN_REV = 2004", helper, flags=re.M)
    assert n == 1 and helper.count(K_OLD) == 1 and helper.count(V_OLD) == 1
    helper = helper.replace(K_OLD, K_NEW).replace(V_OLD, V_NEW)
    open(path, "w").write(helper)
    print("ldshalf ->", dst)


make_ldshalf()
for name, (noexp, nokv, rev) in VARIANTS.items():
    dst = os.path.join(HERE, "fakes", name, "branch_mla")
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(SRC, dst, ignore=shutil.ignore_patterns("__pycache__"))
    helper = open(os.path.join(dst, "attn_helper_mla.py")).read()
    helper, n = re.subn(r"^CODEGEN_REV = \d+$", f"CODEGEN_REV = {rev}", helper, flags=re.M)
    assert n == 1
    if noexp:
        assert helper.count(EXP_OLD) == 1
        helper = helper.replace(EXP_OLD, EXP_NEW)
    open(os.path.join(dst, "attn_helper_mla.py"), "w").write(helper)
    if nokv:
        path = os.path.join(dst, "flash_attn_fwd_mla.py")
        lines = open(path).read().split("\n")
        hits = 0
        for i in range(len(lines)):
            if re.match(r"^\s+ctx\.load_[kv]_tile\(j_idx", lines[i]):
                lines[i] = re.sub(r"ctx\.load_[kv]_tile\(.*\)", "pass", lines[i])
                hits += 1
        assert hits == 10, hits
        open(path, "w").write("\n".join(lines))
    print(name, "->", dst)
