"""Probe ds_read_tr8_b64 semantics empirically.

Strategy:
1. LDS filled with byte[i] = (i & 0xFF). For i < 64, byte = i.
2. Each of 64 lanes calls ds_read_tr8_b64(ptr = LDS + lane_id * 8).
   So lane i conceptually reads 8 bytes starting at LDS[i*8..i*8+7].
   Before HW transposition, lane i's "raw" 8 bytes = [i*8, i*8+1, ..., i*8+7].
3. Each lane writes its received 8 bytes to gmem at slot lane_id * 8.
4. CPU reads gmem; the byte each lane received reveals the transpose semantics.

Run: python -m primus_turbo.flydsl.gemm._tr8_probe
"""

import os
import sys

import torch

# Module-load FlyDSL discovery (same as gemm_fp8_kernel.py).
try:
    from kernels.fp8_gemm_utils import wait_barrier  # noqa: F401
except ImportError:
    import flydsl as _flydsl_probe
    _flydsl_root = os.path.abspath(
        os.path.join(os.path.dirname(_flydsl_probe.__file__), "..", "..")
    )
    if _flydsl_root not in sys.path:
        sys.path.insert(0, _flydsl_root)
    from kernels.fp8_gemm_utils import wait_barrier  # noqa: F401

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import buffer_ops as _buffer_ops, range_constexpr, rocdl
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T, Vector as Vec
from flydsl.expr.utils.arith import ArithValue

_LDS_PTR_TYPE = None


def _inttoptr_lds(byte_addr):
    global _LDS_PTR_TYPE
    if _LDS_PTR_TYPE is None:
        _LDS_PTR_TYPE = ir.Type.parse("!llvm.ptr<3>")
    return _llvm.inttoptr(_LDS_PTR_TYPE, _raw(fx.Int64(byte_addr)))


def _lds_ptr_from_i32(addr_i32, byte_offset=0):
    ptr = _inttoptr_lds(ArithValue(addr_i32).extui(T.i64))
    if byte_offset != 0:
        ptr = _buffer_ops.get_element_ptr(ptr, static_byte_offset=byte_offset)
    return ptr


def compile_probe():
    LDS_SIZE = 64 * 8  # 512 bytes (covers offsets 0..511)

    @fx.struct
    class SharedStorage:
        L: fx.Array[fx.Uint8, LDS_SIZE, 16]

    @flyc.kernel(known_block_size=[64, 1, 1])
    def probe_kernel(out: fx.Tensor):
        # Fill LDS via lane scalar stores. Each lane writes 8 bytes at lane*8.
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        lane = fx.thread_idx.x

        # Cast LDS to uint8 view; lane writes byte[lane*8 + i] = lane*8 + i for i=0..7
        lds_base_i32 = fx.Int32(fx.ptrtoint(lds.L.ptr))
        # Write 8 bytes per lane.
        for i in range_constexpr(8):
            byte_val = (lane * 8 + i) & 0xFF
            # Compute LDS addr for this byte
            addr_i32 = lds_base_i32 + fx.Int32(lane * 8 + i)
            ptr = _lds_ptr_from_i32(addr_i32.ir_value() if hasattr(addr_i32, "ir_value") else _raw(addr_i32))
            # Store one byte.
            _llvm.store(
                _raw(fx.Int8(byte_val)),
                ptr,
                volatile_=True,
            )
        rocdl.s_barrier()

        # Now each lane calls ds_read_tr8_b64 with ptr = LDS + lane*8.
        addr_i32 = lds_base_i32 + fx.Int32(lane * 8)
        ptr = _lds_ptr_from_i32(addr_i32.ir_value() if hasattr(addr_i32, "ir_value") else _raw(addr_i32))
        TR_TYPE = Vec.make_type(2, fx.Int32)
        v = rocdl.ds_read_tr8_b64(TR_TYPE, ptr).result

        # Dump each lane's v2i32 (= 8 bytes) to out[lane*2..lane*2+1] as i32.
        # out is a 1D i32 tensor of length 128 (= 64 lanes * 2 i32 each).
        rsrc = _buffer_ops.create_buffer_resource(out, max_size=True)
        v_vec = Vec(v)
        for i in range_constexpr(2):
            elem = v_vec[i]  # i32
            # Element index = lane * 2 + i  (buffer_store treats offset in elements)
            elem_idx = lane * 2 + i
            _buffer_ops.buffer_store(rsrc, _raw(fx.Int32(elem_idx)), elem, vec_width=1, dtype=T.i32)

    @flyc.jit
    def launch_probe(out: fx.Tensor, stream: fx.Stream):
        probe_kernel(out).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    return launch_probe


def main():
    # 64 lanes * 2 i32 per lane = 128 i32 = 512 bytes
    out = torch.zeros(128, dtype=torch.int32, device="cuda")
    launch = compile_probe()
    args = (out, torch.cuda.current_stream())
    compiled = flyc.compile(launch, *args)
    compiled(*args)
    torch.cuda.synchronize()

    # Reinterpret as bytes; each lane wrote 8 bytes at byte_offset = lane*8.
    arr = out.view(torch.uint8).cpu().numpy()
    print("Lane → 8 received bytes (in order):")
    for lane in range(64):
        bytes_recv = arr[lane * 8 : lane * 8 + 8].tolist()
        print(f"  lane {lane:2d}: {bytes_recv}")


if __name__ == "__main__":
    main()
