"""Wrong-result upper bounds on the per-job costs left after round 16, under profiles/fakes/<name>/branch_mla.

noq: the Q loads return a constant (no Q round trip per job; the scale folds away too).
diag1 / diag2: every wave skips the diagonal's dead 32-col halves as if it were wave 1 / 2, so the
waves do equal diagonal work (8 / 12 tile-quarters each vs 4-16 now); a balanced interleave would
give 10.
"""

import os
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src", "branch_mla")

Q_OLD = (
    "            q_i32_pack = fly.copy_atom_call_ssa(\n"
    "                [self.q_load_i32x4_type],\n"
    "                self.load_atom_128,\n"
    "                fx.slice(self.q_div, (None, fx.Int32(elem_index))),\n"
    "            )\n"
)
Q_NEW = "            q_i32_pack = Vec.from_elements([fx.Int32(0x3E003E00)] * 4, fx.Int32).ir_value()\n"
DIAG_OLD = "        q_lo = fx.Int32(self.q_start_pos_i32 + self.delta_i32)\n"


def make(name, rev, edits):
    dst = os.path.join(HERE, "fakes", name, "branch_mla")
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(SRC, dst, ignore=shutil.ignore_patterns("__pycache__"))
    path = os.path.join(dst, "attn_helper_mla.py")
    helper = open(path).read()
    helper, n = re.subn(r"^CODEGEN_REV = \d+$", f"CODEGEN_REV = {rev}", helper, flags=re.M)
    assert n == 1
    for old, new in edits:
        assert helper.count(old) == 1, old
        helper = helper.replace(old, new)
    open(path, "w").write(helper)
    print(name, "->", dst)


make("noq", 2010, [(Q_OLD, Q_NEW)])
for w, rev in ((1, 2011), (2, 2012)):
    diag_new = f"        q_lo = fx.Int32(fx.Int32(self.q_start) + fx.Int32({64 * w}) + self.delta_i32)\n"
    make(f"diag{w}", rev, [(DIAG_OLD, diag_new)])
