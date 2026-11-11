"""Wrong-result bounds on the next job's Q fetch in drain C10 after round 18, under
profiles/fakes/<name>/branch_mla. The epilogue read back stays; its wait no longer covers the
K prefetch either way.
nostage: the 16 Q stage DMAs are dropped (the read back gets stale LDS); the 8 gathers stay.
noqc10:  the 8 middle-block gathers are dropped too; all 24 packs come from the stale stage.
"""

import os
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src", "branch_mla")

DMA_OLD = "            fx.copy(self.dma_atom, src, dst, soffset=fx.Int32(0))\n"
DMA_NEW = "            pass\n"
KEEP_OLD = "keep=lambda ks16: ks16 >= QK16_PAIRED_STEPS)"
KEEP_NEW = "keep=lambda ks16: False)"
READ_OLD = "        for s in range_constexpr(QK16_PAIRED_STEPS):\n            chunk ="
READ_NEW = "        for s in range_constexpr(_qk16_k_steps(traits)):\n            chunk ="
WAIT_OLD = "                _waitcnt_vm_n(sum(p is not None for p in q_next))\n"
WAIT_NEW = "                _waitcnt_vm_n(20)\n"


def _patch(path, edits):
    text = open(path).read()
    for old, new in edits:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    open(path, "w").write(text)


def make(name, rev, helper_edits):
    dst = os.path.join(HERE, "fakes", name, "branch_mla")
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(SRC, dst, ignore=shutil.ignore_patterns("__pycache__"))
    path = os.path.join(dst, "attn_helper_mla.py")
    helper = open(path).read()
    helper, n = re.subn(r"^CODEGEN_REV = \d+$", f"CODEGEN_REV = {rev}", helper, flags=re.M)
    assert n == 1
    open(path, "w").write(helper)
    _patch(path, helper_edits)
    _patch(os.path.join(dst, "flash_attn_fwd_mla.py"), [(WAIT_OLD, WAIT_NEW)])
    print(name, "->", dst)


make("nostage", 2030, [(DMA_OLD, DMA_NEW)])
make("noqc10", 2031, [(DMA_OLD, DMA_NEW), (KEEP_OLD, KEEP_NEW), (READ_OLD, READ_NEW)])
