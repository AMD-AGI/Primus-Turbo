"""Round-3 A/B probe: time the two wgrad cells of the mx8tw board.

usage: _probe_r3_ab.py [nopair]
`nopair` monkeypatches `_gnt_pair_n` to False BEFORE any compile, which turns off
the pair-major RHS feed + col_safe epilogue while leaving everything else alone.
Run each variant in its own process with a cleared FlyDSL cache.
"""

import statistics
import sys

import torch

import primus_turbo.flydsl.grouped_gemm.mxfp8_grouped_kernel as mk

if len(sys.argv) > 1 and sys.argv[1] == "nopair":
    mk._gnt_pair_n = lambda N, BLOCK_N=256: False
    print("[variant] pair-major RHS feed OFF")
else:
    print("[variant] pair-major RHS feed ON")

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


def time_cell(proj, offs, iters=40, reps=3):
    N = PROJ[proj]
    l, lsc = q8(H, MTOT), e8m0(H, MTOT // 32)
    r, rsc = q8(N, MTOT), e8m0(N, MTOT // 32)

    def run(ps=False):
        return mk.grouped_gemm_mxfp8_variable_k_flydsl_kernel(
            l, lsc, r, rsc, offs, H, N, G, torch.bfloat16, -1, 4, ps
        )

    run(True)
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        for _ in range(5):
            run()
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        for _ in range(iters):
            run()
        b.record()
        torch.cuda.synchronize()
        out.append(a.elapsed_time(b) / iters)
    ms = statistics.median(out)
    print(f"wgrad/{proj:8s} ms={ms:.4f} TF={2.0 * MTOT * H * N / (ms * 1e9):.1f} raw={[round(x, 4) for x in out]}")
    return ms


def main():
    torch.manual_seed(0)
    o = torch.zeros(G + 1, dtype=torch.int64, device=DEV)
    o[1:] = torch.full((G,), PER_EXPERT, dtype=torch.int64, device=DEV).cumsum(0)
    tot = sum(time_cell(p, o) for p in ("gate_up", "down"))
    print(f"wgrad_sum_ms={tot:.4f}")


main()
