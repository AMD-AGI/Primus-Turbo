# Checks that a deliberate s_trap 2 aborts the queue cleanly on this GPU.
import flydsl.compiler as flyc
import flydsl.expr as fx
import torch

from primus_turbo.flydsl.deep_ep.arch import get_arch_traits, spin_timeout_hit
from primus_turbo.flydsl.utils.launch_cache import run_launcher


@flyc.kernel(known_block_size=[64, 1, 1])
def trap_kernel(OUT: fx.Tensor):
    if fx.thread_idx.x == fx.Int32(0):
        fx.printf("trap_probe: about to trap\n")
    spin_timeout_hit(get_arch_traits())


@flyc.jit
def launch_trap(out, stream: fx.Stream):
    trap_kernel(out).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


out = torch.zeros(2, dtype=torch.int32, device="cuda")
torch.cuda.synchronize()
print("launching", flush=True)
try:
    run_launcher(launch_trap, out, torch.cuda.current_stream())
    torch.cuda.synchronize()
    print("no error raised", flush=True)
except Exception as e:
    print("raised:", type(e).__name__, str(e)[:200], flush=True)
