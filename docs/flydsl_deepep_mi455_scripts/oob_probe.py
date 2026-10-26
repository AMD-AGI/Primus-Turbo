# P1: buffer-descriptor out-of-bounds semantics on the current GPU.
import flydsl.compiler as flyc
import flydsl.expr as fx
import torch

from primus_turbo.flydsl.deep_ep.arch import get_arch_traits
from primus_turbo.flydsl.deep_ep.common import tensor_addr
from primus_turbo.flydsl.utils.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource,
    create_buffer_resource_from_addr,
)
from primus_turbo.flydsl.utils.launch_cache import run_launcher

CPOL = get_arch_traits().cpol_sys_load
# (records bytes, element offset, cache policy) of each 16-byte load
LOADS = [(0, 0, 0), (0, 0, CPOL), (16, 4, 0), (8, 0, 0), (8, 0, CPOL), (64, 0, 0), (20, 4, 0)]
# (byte offset of the resource into GUARD, records bytes, element offset) of each 16-byte store
STORES = [(0, 16, 0), (0, 16, 4), (32, 8, 0), (48, 0, 0)]


@flyc.kernel(known_block_size=[64, 1, 1])
def oob_kernel(SRC: fx.Tensor, DST: fx.Tensor, GUARD: fx.Tensor):
    dst = create_buffer_resource(DST, max_size=True)
    if fx.thread_idx.x == fx.Int32(0):
        for j, (records, offset, cpol) in enumerate(LOADS):
            res = create_buffer_resource_from_addr(tensor_addr(SRC), num_records_bytes=records)
            value = buffer_load(res, fx.Int32(offset), vec_width=4, dtype=fx.T.i32, cache_modifier=cpol)
            buffer_store(value, dst, fx.Int32(4 * j))
        for j, (base, records, offset) in enumerate(STORES):
            res = create_buffer_resource_from_addr(tensor_addr(GUARD) + fx.Int64(base), num_records_bytes=records)
            value = buffer_load(dst, fx.Int32(4 * len(LOADS) + 4 * j), vec_width=4, dtype=fx.T.i32)
            buffer_store(value, res, fx.Int32(offset))


@flyc.jit
def launch_oob(src, dst, guard, stream: fx.Stream):
    oob_kernel(src, dst, guard).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


src = torch.full((64,), 7, dtype=torch.int32, device="cuda")
dst = torch.full((4 * (len(LOADS) + len(STORES)),), -2, dtype=torch.int32, device="cuda")
for j in range(len(STORES)):
    dst[4 * (len(LOADS) + j) : 4 * (len(LOADS) + j + 1)] = 10 + j
guard = torch.full((16,), -1, dtype=torch.int32, device="cuda")
run_launcher(launch_oob, src, dst, guard, torch.cuda.current_stream())
torch.cuda.synchronize()
for j, spec in enumerate(LOADS):
    print(f"load  records={spec[0]:2d}B elem={spec[1]} cpol={spec[2]:#x}: {dst[4 * j : 4 * j + 4].tolist()}")
for j, spec in enumerate(STORES):
    b = spec[0] // 4 + spec[2]
    print(f"store base={spec[0]:2d}B records={spec[1]:2d}B elem={spec[2]} (value {10 + j}): guard[{b}:{b + 4}] = {guard[b : b + 4].tolist()}")
print("guard", guard.tolist())
