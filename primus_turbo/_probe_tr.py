"""Empirically probe ds_read_tr16_b64 lane/element transpose mapping on gfx950.

Fill LDS[i] = i (i16). Each of the 64 lanes issues one ds_read_tr16_b64 with
element base = lane*4; the 4 returned i16 values are written to OUT[lane*4 + j].
Reading OUT back reveals, for every (lane, slot) -> which LDS element it got,
i.e. the exact hardware transpose permutation.
"""
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import buffer_ops, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.compiler.kernel_function import CompilationContext
from flydsl._mlir import ir

N_LDS = 1024


def build():
    arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=arch, global_sym_name="probe_tr_smem")
    lds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_off + N_LDS * 2

    @flyc.kernel(known_block_size=[64, 1, 1])
    def probe(OUT: fx.Tensor):
        v4i16 = Vec.make_type(4, fx.Int16)
        base_ptr = allocator.get_base()
        lds = SmemPtr(base_ptr, lds_off, fx.Int16.ir_type, shape=(N_LDS,)).get()
        tid = fx.Index(gpu.thread_idx.x)
        for c in range_constexpr(N_LDS // 64):
            i = tid + fx.Index(c * 64)
            Vec.from_elements([fx.Int16(i)], fx.Int16).store(lds, [i])
        gpu.barrier()
        elem = tid * fx.Index(4)
        byte_off = elem * fx.Index(2) + fx.Index(lds_off)
        ptr = buffer_ops.create_llvm_ptr(fx.Int64(byte_off), address_space=3)
        val = rocdl.ds_read_tr16_b64(v4i16, ptr).result
        out_rsrc = buffer_ops.create_buffer_resource(OUT, max_size=True)
        vals = [fx.Int32(fx.Int16(Vec(val)[j])) for j in range_constexpr(4)]
        buffer_ops.buffer_store(
            Vec.from_elements(vals, fx.Int32), out_rsrc, tid * fx.Index(16), offset_is_bytes=True
        )

    @flyc.jit
    def launch(OUT: fx.Tensor, stream: fx.Stream):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        probe(OUT).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    return launch


def main():
    launch = build()
    out = torch.full((64 * 4,), -1, device="cuda", dtype=torch.int32)
    launch(out, torch.cuda.current_stream())
    torch.cuda.synchronize()
    o = out.cpu().tolist()
    print("lane: [slot0 slot1 slot2 slot3]  (LDS element index received)")
    for l in range(64):
        print(f"{l:2d}: {o[l*4:l*4+4]}")


if __name__ == "__main__":
    main()
