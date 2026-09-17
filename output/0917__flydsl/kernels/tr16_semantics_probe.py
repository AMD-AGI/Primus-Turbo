"""Reverse-engineer ds_load_tr16_b128's transpose semantics on gfx1250.

This is the last unknown before dkdv can be written. aiter's VManager documents the
per-lane ADDRESS it presents:

    kv = (l // 16) * 8 + l % 8      d = ((l // 8) % 2) * 8

but not what the crossbar then returns. So: fill a 16x16 bf16 LDS tile with the index
pattern `value = row * 16 + col` (0..255, all exactly representable in bf16), do the
transposing load with exactly that addressing, and decode which (row, col) each
(lane, element) came back with.
"""
import os, sys, json
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")
# flydsl 0.3.2 is required (0.2.4 cannot build aiter's gfx1250 kernels) and is installed
# side by side rather than over the image's 0.2.4, which other things in this container
# still need. Durable path first: /tmp does not survive a container restart, and the
# unattended op-evolve job runs for up to 48h with a supervisor that restarts things.
for _p in ("/home/lihuzhan/.local/flydsl032", "/tmp/flydsl032"):
    if os.path.isdir(_p):
        sys.path.insert(0, _p)
        break
sys.path.insert(0, "/home/lihuzhan/code/aiter-src")
import torch, flydsl
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl, range_constexpr
from flydsl._mlir.dialects import llvm as llvm_dialect
from aiter.ops.flydsl.kernels.kernels_common import create_llvm_ptr
print(f"flydsl {flydsl.__version__}   arch {torch.cuda.get_device_properties(0).gcnArchName}")

ROWS, COLS = 16, 16
ROW_BYTES = COLS * 2          # unpadded: the swizzle is a bank-conflict fix, not correctness


def _bv(t, nb, dt, vec=1):
    nt = (1 << 31) // (vec * (dt.width // 8))
    b = fx.rocdl.make_buffer_tensor(t, num_records_bytes=fx.Int64(nb))
    return fx.Tensor(fx.make_view(fx.get_iter(b), fx.make_layout((nt, vec), (vec, 1))))
def _atom(dt, vec): return fx.make_copy_atom(fx.rocdl.BufferCopy(vec * dt.width), dt)
def _ldv(buf, tile, dt, vec):
    f = fx.make_rmem_tensor(fx.make_layout(vec, 1), dt)
    fx.copy_atom_call(_atom(dt, vec), fx.slice(buf, (tile, None)), f); return f.load()
def _stv(val, buf, tile, dt, vec):
    f = fx.make_rmem_tensor(fx.make_layout(vec, 1), dt)
    fx.memref_store_vec(val, f)
    fx.copy_atom_call(_atom(dt, vec), f, fx.slice(buf, (tile, None)))


@flyc.kernel(known_block_size=[32, 1, 1])
def k_tr16(SRC: fx.Tensor, OUT: fx.Tensor):
    lane = fx.Int32(fx.thread_idx.x)
    g_src = _bv(SRC, ROWS * COLS * 2, fx.BFloat16, 8)
    g_out = _bv(OUT, 32 * 8 * 2, fx.BFloat16, 8)

    smem = fx.SharedAllocator().allocate(ROWS * ROW_BYTES)
    lds0 = fx.Int32(fx.ptrtoint(smem.peek().ptr))

    # Fill LDS linearly: lane l writes the 8 elements at flat index l*8.
    v8_ty = fx.Vector.make_type(8, fx.BFloat16)
    vals = _ldv(g_src, lane, fx.BFloat16, 8)
    p_wr = create_llvm_ptr(lds0 + lane * fx.Int32(16), address_space=3)
    llvm_dialect.store(fx.as_ir_value(vals), p_wr)
    fx.barrier()

    # The documented transposing-load addressing.
    lane_kv = (lane // fx.Int32(16)) * fx.Int32(8) + lane % fx.Int32(8)
    lane_d = ((lane // fx.Int32(8)) % fx.Int32(2)) * fx.Int32(8)
    p_rd = create_llvm_ptr(lds0 + lane_kv * fx.Int32(ROW_BYTES) + lane_d * fx.Int32(2),
                           address_space=3)
    got = fx.Vector(rocdl.ds_load_tr16_b128(v8_ty, p_rd))
    _stv(got, g_out, lane, fx.BFloat16, 8)


@flyc.jit
def launch(SRC: fx.Tensor, OUT: fx.Tensor, stream: fx.Stream):
    k_tr16(SRC, OUT).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)


src = torch.arange(ROWS * COLS, device="cuda", dtype=torch.float32).to(torch.bfloat16)
out = torch.full((32 * 8,), float("nan"), device="cuda", dtype=torch.bfloat16)
launch(src, out, torch.cuda.current_stream()); torch.cuda.synchronize()

o = out.float().cpu().reshape(32, 8)
if not torch.isfinite(o).all():
    print("non-finite in result:", int((~torch.isfinite(o)).sum())); sys.exit(2)
idx = o.to(torch.int32)
print("\n(lane, elem) -> (row, col) of the source tile, first 8 lanes:")
for l in range(8):
    cells = " ".join(f"{int(v)//16:2d},{int(v)%16:<2d}" for v in idx[l])
    print(f"  lane {l:2d}: {cells}")
print("  ...")
for l in (16, 17, 24, 31):
    cells = " ".join(f"{int(v)//16:2d},{int(v)%16:<2d}" for v in idx[l])
    print(f"  lane {l:2d}: {cells}")

# Does it match the WMMA A-operand layout?  operand: lane l, elem e -> [l%16, (l//16)*8 + (e%8) + (e//8)*16]
# That needs 32 cols; with a 16-col tile only e=0..7 are meaningful, cols (l//16)*8 + e.
print("\nHypothesis A -- result is the TRANSPOSE, i.e. (lane,e) == src[col=?, row=?]:")
rows = idx // 16; cols = idx % 16
same_row_per_lane = int((rows == rows[:, :1]).all(dim=1).sum())
same_col_per_lane = int((cols == cols[:, :1]).all(dim=1).sum())
print(f"  lanes whose 8 values share ONE source row:    {same_row_per_lane}/32")
print(f"  lanes whose 8 values share ONE source column: {same_col_per_lane}/32")
json.dump({"map": idx.tolist(), "lanes_one_row": same_row_per_lane,
           "lanes_one_col": same_col_per_lane}, open("/tmp/tr16.json", "w"), indent=2)
