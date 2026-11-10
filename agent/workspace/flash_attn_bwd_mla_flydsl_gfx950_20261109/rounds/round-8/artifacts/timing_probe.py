"""Same-process bwd timing of the chosen configs under the bench's perf_counter sampler.
Usage: python timing_probe.py B S repeats cfg[,cfg...] where cfg names a key of ALL; configs build in
the given order. A cfg prefixed with '+' runs once at B1 S1024 before any timing and is not timed."""

import statistics
import sys

sys.path.insert(0, "/perf_apps/xiaoming/Primus-Turbo/agent/workspace/flash_attn_bwd_mla_flydsl_gfx950_20261109")
import quick_test_bench as qb  # noqa: E402
import torch  # noqa: E402

B, S, REPEATS = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
ALL = {"r7": {"g2d": 2, "g3_sb": 1}, "r8": {"g2d": 1, "g3_sb": 0}, "r8s1": {"g2d": 1, "g3_sb": 1}}
pre = [n[1:] for n in sys.argv[4].split(",") if n.startswith("+")]
names = [n for n in sys.argv[4].split(",") if not n.startswith("+")]


def make(b, s):
    g = torch.Generator(device="cuda").manual_seed(0)
    q, k = (torch.randn((s, b, 128, 192), device="cuda", dtype=torch.bfloat16, generator=g) for _ in range(2))
    v = torch.randn((s, b, 128, 128), device="cuda", dtype=torch.bfloat16, generator=g)
    do = torch.randn((s, b, 128, 128), device="cuda", dtype=torch.bfloat16, generator=g)
    o, lse = qb.mla_fwd_sbhd(q, k, v, causal=True, return_lse=True)
    return do, q, k, v, o, lse


if pre:
    small = make(1, 1024)
    for n in pre:
        qb.mla_bwd_sbhd(*small, **ALL[n])
    torch.cuda.synchronize()
    del small
data = make(B, S)
fns = {n: (lambda over=ALL[n]: qb.mla_bwd_sbhd(*data, **over)) for n in names}
samplers = {n: qb.timed(fns[n], 5, 3.0) for n in names}
res = {n: [] for n in names}
for r in range(REPEATS):
    for n in names[r % len(names) :] + names[: r % len(names)]:
        res[n].append(samplers[n]())
print(f"B{B} S{S} pre {pre} timed {names}: " + "  ".join(
    f"{n} {statistics.median(res[n]):8.3f} ms {['%.2f' % x for x in res[n]]}" for n in names), flush=True)
