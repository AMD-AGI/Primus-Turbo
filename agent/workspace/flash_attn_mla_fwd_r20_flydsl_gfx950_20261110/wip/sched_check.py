"""Host-side checks of the persistent schedule: order twin, coverage, modelled balance."""

import collections
import contextlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.argv = sys.argv[:1]
with contextlib.redirect_stdout(io.StringIO()):
    import persist_sim as ps
import ab_bench as ab  # noqa: E402

pkg = ab.load_pkg(os.path.join(os.path.dirname(HERE), "src", "branch_mla"), "mla_p")
from mla_p import ops  # noqa: E402
from mla_p.attn_helper_mla import host_job_order  # noqa: E402

for name, B, S, H in ps.SHAPES:
    nq = (S + 255) // 256
    o = host_job_order(H, nq, True)
    assert o == [ps.item_of(lin, H, nq) for lin in range(H * nq)], name
    assert sorted(o) == [(h, q) for h in range(H) for q in range(nq)], name
    mod = ops._fwd_module(H, H, 192, 128, True, False, 256, 1, False, 1, 64, 256)
    flat, ctas, stride = mod.schedule(B, S)
    rows = [flat[c * stride : (c + 1) * stride] for c in range(ctas)]
    jobs = collections.Counter(j for r in rows for j in r[1 : 1 + r[0]])
    want = {h | q << 9 | b << 24 for b in range(B) for h in range(H) for q in range(nq)}
    assert set(jobs) == want and all(v == 1 for v in jobs.values()), name
    cost = [sum(8 + ps.tiles((j >> 9) & 32767) for j in r[1 : 1 + r[0]]) for r in rows]
    print(
        f"{name:9s} ctas {ctas} stride {stride} modelled load max/mean {max(cost) / (sum(cost) / len(cost)):.3f}"
    )
