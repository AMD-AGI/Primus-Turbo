"""Host cost of building one persistent schedule: python wip/sched_time.py"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ab_bench as ab  # noqa: E402

pkg = ab.load_pkg("src/branch_mla", "mla_cand")
ops = sys.modules["mla_cand.ops"]
for B, S, H in [(1, 65536, 128), (4, 4096, 64), (1, 4096, 128)]:
    mod = ops._fwd_module(H, H, ab.DQK, ab.DV, True, True, 256, 1, False, 1, 64, 256)
    t = time.perf_counter()
    flat, ctas, stride = mod.schedule(B, S)
    dt = time.perf_counter() - t
    print(
        f"B={B} S={S} H={H}: {ctas} CTAs x {stride - 1} jobs max, {len(flat) * 4 / 1024:.1f} KiB, {dt * 1e3:.1f} ms"
    )
