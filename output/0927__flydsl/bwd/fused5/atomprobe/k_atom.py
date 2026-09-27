"""FUSED5 card probe P2: fp32 SCOPE_DEV atomic throughput with k_dkdv_f5's exact address stream.

No WMMA, no loads. Same grid (hkv, kv tile, batch), same causal q-pair walk (longest-first,
q-pair outer / q-head inner), same 2 x 64-atomic clauses per iteration, same per-lane
index formula as op/kernels.py:_dq_fused -> same L2/HBM contention pattern.
  KVG  = 1, 2, 4  : one workgroup stands for 32*KVG kv rows (grid.y = Skv/(32*KVG)),
                    i.e. BLOCK_KV 32 / 64 / 128 atomic traffic = 68.99 / 34.6 / 17.4 GB at prod.
  ATOM_LDS = 70656 (default) pins occupancy to k_dkdv_f5's 1 wave/SIMD; 0 = free.
  MODE = "atom"   : global_atomic_add_f32 scope:SCOPE_DEV       (the thing we need)
         "store"  : buffer_store_b32 at the same addresses       (write-bandwidth control)
         "none"   : loop + index math only                       (issue floor control)
Bounds: identical index expression, proven in ../bounds_proof.py (max idx = size-1).
"""
import os
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr, rocdl

D = 128
MODE = os.environ.get("ATOM_MODE", "atom")
# ATOM_LDS=70656 pins occupancy to k_dkdv_f5's (4 WG/CU = 1 wave/SIMD); 0 = unconstrained.
LDS_B = int(os.environ.get("ATOM_LDS", "70656"))


def _pin_lds(v):
    """Allocate LDS_B bytes and touch them once so the allocation survives."""
    if LDS_B == 0:
        return
    from flydsl._mlir.dialects import llvm as llvm_dialect
    smem = fx.SharedAllocator().allocate(LDS_B)
    a = fx.Int32(fx.ptrtoint(smem.peek().ptr))
    pt = fx.PointerType.get(fx.Float32.ir_type, address_space=fx.AddressSpace.Shared, alignment=4)
    llvm_dialect.store(fx.as_ir_value(v), fx.as_ir_value(fx.to_llvm_ptr(fx.inttoptr(pt, a))))


# Mode dispatch lives in plain helpers: the kernel-body AST rewriter turns a Python `if`
# inside the kernel into a scoped region, so names bound under it do not escape.
def _mk_target(DQA, nbytes):
    if MODE == "atom":
        g = fx.Tensor(fx.make_view(fx.get_iter(DQA), fx.make_layout((1 << 30, 1), (1, 1))))
        return g, fx.make_copy_atom(
            fx.UniversalAtomicAdd(fx.Float32, rocdl.SyncScope.Agent), fx.Float32)
    b = fx.rocdl.make_buffer_tensor(DQA, num_records_bytes=fx.Int64(nbytes))
    g = fx.Tensor(fx.make_view(fx.get_iter(b), fx.make_layout((1 << 30, 1), (1, 1))))
    return g, fx.make_copy_atom(fx.rocdl.BufferCopy(32), fx.Float32)


def _emit(g, atom, idx, val):
    if MODE == "none":
        return val
    f = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
    fx.memref_store_vec(fx.Vector.from_elements([val], dtype=fx.Float32), f)
    fx.copy_atom_call(atom, f, fx.slice(g, (idx, None)))
    return val


def _keepalive(DQA, v):
    if MODE != "none":
        return
    b2 = fx.rocdl.make_buffer_tensor(DQA, num_records_bytes=fx.Int64(4))
    t2 = fx.Tensor(fx.make_view(fx.get_iter(b2), fx.make_layout((1, 1), (1, 1))))
    f = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
    fx.memref_store_vec(fx.Vector.from_elements([v], dtype=fx.Float32), f)
    fx.copy_atom_call(fx.make_copy_atom(fx.rocdl.BufferCopy(32), fx.Float32), f,
                      fx.slice(t2, (fx.Int32(0), None)))


@flyc.kernel(known_block_size=[32, 1, 1])
def k_atom(DQA: fx.Tensor, Sq: fx.Int32, Skv: fx.Int32, Hq: fx.Int32, G: fx.Int32,
           nqt2: fx.Int32, cshift: fx.Int32, KVG: fx.Int32, B_: fx.Int32):
    lane = fx.Int32(fx.thread_idx.x)
    hkv = fx.Int32(fx.block_idx.x)
    bid = fx.Int32(fx.block_idx.y)
    bat = fx.Int32(fx.block_idx.z)
    row = lane % fx.Int32(16)
    half = lane // fx.Int32(16)
    kv0 = bid * fx.Int32(32) * KVG
    _c = kv0 - cshift
    qp_start = ((_c < fx.Int32(0)).select(fx.Int32(0), _c)) // fx.Int32(32)
    n = G * (nqt2 - qp_start)
    rs_dq = Hq * fx.Int32(D)
    g, atom = _mk_target(DQA, B_ * Sq * rs_dq * fx.Int32(4))

    @flyc.jit
    def loop(state, n):
        final = state
        for it, carried in range(fx.Index(fx.Int32(0)), fx.Index(n), 1, init=state):
            ii = fx.Int32(it)
            qi = ii // G
            gh = ii - qi * G
            qh = hkv * G + gh
            acc = fx.Vector(carried[0])
            val = acc[0] + fx.Float32(1.0)
            for hh in range_constexpr(2):
                q0 = (qp_start + qi) * fx.Int32(32) + fx.Int32(hh * 16)
                base = (bat * Sq + q0 + half * fx.Int32(8)) * rs_dq + qh * fx.Int32(D) + row
                for dtile in range_constexpr(8):
                    for si in range_constexpr(8):
                        idx = base + fx.Int32(si) * rs_dq + fx.Int32(dtile * 16)
                        val = _emit(g, atom, idx, val)
            final = yield [fx.as_ir_value(fx.Vector.from_elements([val], dtype=fx.Float32))]
        return final

    out = loop([fx.as_ir_value(fx.Vector.filled(1, 0.0, fx.Float32))], n)
    out = list(out) if isinstance(out, (list, tuple)) else [out]
    _keepalive(DQA, fx.Vector(out[0])[0])
    _pin_lds(fx.Vector(out[0])[0])


@flyc.jit
def launch_atom(DQA, Sq: fx.Int32, Skv: fx.Int32, Hq: fx.Int32, G: fx.Int32,
                nqt2: fx.Int32, cshift: fx.Int32, KVG: fx.Int32, B_: fx.Int32,
                nhkv: fx.Int32, nblk: fx.Int32, nb: fx.Int32, stream: fx.Stream):
    k_atom(DQA, Sq, Skv, Hq, G, nqt2, cshift, KVG, B_).launch(
        grid=(nhkv, nblk, nb), block=(32, 1, 1), stream=stream)
