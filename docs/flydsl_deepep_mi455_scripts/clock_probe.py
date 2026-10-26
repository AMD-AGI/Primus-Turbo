# Measures the frequency of prims.read_clock() on the current GPU.
import time

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch

from primus_turbo.flydsl.utils.buffer_ops import buffer_store, create_buffer_resource
from primus_turbo.flydsl.utils.launch_cache import run_launcher
from primus_turbo.flydsl.utils.prims import cast, read_clock


@flyc.kernel(known_block_size=[64, 1, 1])
def clock_kernel(OUT: fx.Tensor):
    res = create_buffer_resource(OUT, max_size=True)
    c = read_clock()
    if fx.thread_idx.x == fx.Int32(0):
        buffer_store(cast(c, fx.T.i32), res, fx.Int32(0))
        buffer_store(cast(c >> fx.Int64(32), fx.T.i32), res, fx.Int32(1))


@flyc.jit
def launch_clock(out, stream: fx.Stream):
    clock_kernel(out).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


def sample():
    out = torch.zeros(2, dtype=torch.int32, device="cuda")
    run_launcher(launch_clock, out, torch.cuda.current_stream())
    torch.cuda.synchronize()
    lo, hi = out.tolist()
    return (hi << 32) | (lo & 0xFFFFFFFF)


sample()
c0, t0 = sample(), time.perf_counter()
time.sleep(2.0)
c1, t1 = sample(), time.perf_counter()
print(f"read_clock: {(c1 - c0) / (t1 - t0) / 1e6:.2f} MHz ({c0} -> {c1})")
