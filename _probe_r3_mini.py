"""Round-3 mini-bench: the 6 scored balanced cells, with optional trace-time patches.

usage: _probe_r3_mini.py [noprio] [nopair]
Mirrors _bench_mx8tw_bal.py's shapes (G=32, per-expert 4096, H=2944, preshuffle=False) but
skips the guard distributions, so one variant costs ~3 min instead of a full bench. Patches are
applied BEFORE any kernel is traced; run each variant in its own process with a cleared cache.
Absolute numbers run a little above the harness (fewer cells competing for L2), so only ever
read the A/B ratio off this -- confirm a keeper with bench.sh.
"""

import statistics
import sys

import torch

import primus_turbo.flydsl.grouped_gemm.mxfp8_grouped_kernel as mk

FLAGS = set(sys.argv[1:])
if "noprio" in FLAGS:
    mk.rocdl.s_setprio = lambda v: None
if "nopair" in FLAGS:
    mk._gnt_pair_n = lambda N, BLOCK_N=256: False
print(f"[variant] {sorted(FLAGS) or ['base']}")

DEV = "cuda"
G = 32
PER_EXPERT = 4096
MTOT = G * PER_EXPERT
H = 2944
F8 = torch.float8_e4m3fn
PROJ = {"gate_up": 5760, "down": 2944}


def q8(*shape):
    return (torch.randn(*shape, device=DEV) * 0.5).to(F8)


def e8m0(*shape):
    return torch.full(shape, 127, dtype=torch.uint8, device=DEV)


def build(op, proj, offs):
    N = PROJ[proj]
    if op == "wgrad":
        l, lsc = q8(H, MTOT), e8m0(H, MTOT // 32)
        r, rsc = q8(N, MTOT), e8m0(N, MTOT // 32)
        return (
            lambda ps=False: mk.grouped_gemm_mxfp8_variable_k_flydsl_kernel(
                l, lsc, r, rsc, offs, H, N, G, torch.bfloat16, -1, 4, ps
            ),
            2.0 * MTOT * H * N,
        )
    K, NN = (H, N) if op == "fwd" else (N, H)
    a, asc = q8(MTOT, K), e8m0(MTOT, K // 32)
    b, bsc = q8(G, NN, K), e8m0(G, NN, K // 32)
    return (
        lambda ps=False: mk.grouped_gemm_mxfp8_flydsl_kernel(
            a, asc, b, bsc, offs, NN, K, None, torch.bfloat16, -1, ps
        ),
        2.0 * MTOT * K * NN,
    )


def main():
    torch.manual_seed(0)
    o = torch.zeros(G + 1, dtype=torch.int64, device=DEV)
    o[1:] = torch.full((G,), PER_EXPERT, dtype=torch.int64, device=DEV).cumsum(0)
    tot, flops = 0.0, 0.0
    for op in ("fwd", "dgrad", "wgrad"):
        for proj in ("gate_up", "down"):
            fn, flop = build(op, proj, o)
            fn(True)
            torch.cuda.synchronize()
            out = []
            for _ in range(3):
                for _ in range(5):
                    fn()
                torch.cuda.synchronize()
                a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                a.record()
                for _ in range(30):
                    fn()
                b.record()
                torch.cuda.synchronize()
                out.append(a.elapsed_time(b) / 30)
            ms = statistics.median(out)
            tot += ms
            flops += flop
            print(f"  {op:6s} {proj:8s} {ms:.4f} ms  {flop / (ms * 1e9):7.1f} TF")
    print(f"MINI_SUM_MS={tot:.4f}  MINI_TF={flops / (tot * 1e9):.1f}")


main()
