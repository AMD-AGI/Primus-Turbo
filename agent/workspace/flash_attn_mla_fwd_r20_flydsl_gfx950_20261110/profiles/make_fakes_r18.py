"""Wrong-result bounds on the Q fetch's memory latency after round 17, under profiles/fakes/<name>/branch_mla.

The loads, their count, their destinations and their position are unchanged and the data is real Q
(so the MFMA operands still switch); only where it comes from moves.
warmq:  every job reads batch 0, head 0, q block 0 -- one 96 KB block, L2-hot on every XCD.
warmqh: every job reads batch 0, its own head, q block 0 -- 12 MB over 128 heads, MALL-resident.
"""

import os
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src", "branch_mla")

BATCH_OLD = (
    "            traits.HEAD_DIM,\n"
    "            batch_idx,\n"
    "            batch_idx * self.seq_len_v,\n"
    "        )\n"
    "        q_head = (h_idx % traits.NUM_HEADS_KV)"
)
BATCH_NEW = (
    "            traits.HEAD_DIM,\n"
    "            batch_idx * 0,\n"
    "            batch_idx * 0,\n"
    "        )\n"
    "        q_head = (h_idx % traits.NUM_HEADS_KV)"
)
OFF_OLD = "        offset = q_block_idx * traits.BLOCK_M * self.stride_q_n_v + q_head * traits.HEAD_DIM\n"
OFF_ALL = "        offset = (q_block_idx * traits.BLOCK_M * self.stride_q_n_v + q_head * traits.HEAD_DIM) * 0\n"
OFF_HEAD = "        offset = q_block_idx * 0 + q_head * traits.HEAD_DIM\n"


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


COAL_OLD = (
    "                elem_index = q_elem_offset + (\n"
    "                    (wave_q_offset + lane % 16 + 16 * qt) * self.stride_q_n_v + (_qk16_step_d(ks16) + lane_d)\n"
    "                )\n"
)
COAL_NEW = (
    "                elem_index = q_elem_offset + (\n"
    "                    (wave_q_offset + 16 * qt) * self.stride_q_n_v + ks16 * 512 + lane * 8\n"
    "                )\n"
)

make("warmq", 2020, [(BATCH_OLD, BATCH_NEW), (OFF_OLD, OFF_ALL)])
make("warmqh", 2021, [(BATCH_OLD, BATCH_NEW), (OFF_OLD, OFF_HEAD)])
# coalq: each load reads 1 KB contiguous (8 lines) of the right job's Q instead of 16 B from each
# of 16 rows per quarter-wave (64 lines); same loads, destinations and bytes, wrong positions.
make("coalq", 2022, [(COAL_OLD, COAL_NEW)])
