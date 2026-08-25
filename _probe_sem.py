#!/usr/bin/env python3
"""Can one work-group SEE another's stores while both are still running?

That is the single semantic a progressive dQ fold needs and the one this tree has never
used. Every work-group here is both producer and consumer: it stores its slice, adds 1 to a
shared counter with a device-scope atomic, then POLLS the counter and records the first turn
it saw all NWG increments.

The poll is a BOUNDED loop, not a spin -- a real spin hangs the GPU when the semantic does
not hold, and that needs a host-side `rocm-smi --gpureset` to clear.

Reading OUT: 0 means that work-group never saw the full count (semantics do NOT hold, or the
groups did not co-reside); >0 is the turn on which it did.

usage: _probe_sem.py [aux]   aux = CPol bits (0 default, 1 sc0, 3 sc0|nt, 17 sc0|sc1)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import buffer_ops, const_expr, gpu, rocdl
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue
from flydsl.expr.utils.arith import _to_raw as _raw
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm

AUX = int(sys.argv[1]) if len(sys.argv) > 1 else 17
NWG, BLOCK, POLLS = 64, 256, 2048


@flyc.kernel(known_block_size=[BLOCK, 1, 1])
def sem_kernel(DATA: fx.Tensor, FLAG: fx.Tensor, OUT: fx.Tensor):
    bid = fx.Index(gpu.block_idx.x)
    tid = fx.Index(gpu.thread_idx.x)
    d = buffer_ops.create_buffer_resource(DATA, max_size=True)
    f = buffer_ops.create_buffer_resource(FLAG, max_size=True)
    o = buffer_ops.create_buffer_resource(OUT, max_size=True)
    buffer_ops.buffer_store(
        _raw(fx.Float32(1.0)),
        d,
        bid * fx.Index(BLOCK) + tid,
        cache_modifier=AUX,
    )
    # every lane adds 1/BLOCK so the counter reaches exactly NWG with no lane predication
    rocdl.raw_ptr_buffer_atomic_fadd(
        _raw(fx.Float32(1.0 / BLOCK)),
        f,
        ArithValue(_raw(fx.Index(0))).index_cast(fx.Int32.ir_type),
        0,
        AUX,
    )
    def _opaque(v):
        """Identity LICM cannot see through, so the poll's load is re-issued every turn.

        Without it the address is loop-invariant and the load has no side effects, so the
        compiler hoists it and the poll reads ONE snapshot -- which is what "0 / 64 saw it"
        means, not a visibility failure. Same trick the kernel's own _opaque_idx uses.
        """
        r = llvm.inline_asm(
            ir.IntegerType.get_signless(32), [_raw(fx.Int32(v))], "", "=v,0",
            has_side_effects=True,
        )
        return fx.Index(r)

    hit = fx.Float32(0.0)
    for _t, _st in range(fx.Index(1), fx.Index(POLLS), fx.Index(1), init=[_raw(hit)]):
        _cur = fx.Float32(_st[0] if isinstance(_st, list) else _st)
        _v = buffer_ops.buffer_load(
            f, _opaque(fx.Index(0)), vec_width=1, dtype=fx.Float32, cache_modifier=AUX
        )
        _full = ArithValue(fx.Float32(_v) > fx.Float32(NWG - 0.5))
        _new = _cur + _full.select(fx.Float32(1.0), fx.Float32(0.0))
        hit = yield [_raw(_new)]
    hit = fx.Float32(hit[0] if isinstance(hit, list) else hit)
    buffer_ops.buffer_store(_raw(hit), o, bid * fx.Index(BLOCK) + tid, cache_modifier=0)


@flyc.jit
def launch_sem(DATA: fx.Tensor, FLAG: fx.Tensor, OUT: fx.Tensor, stream: fx.Stream):
    sem_kernel(
        DATA, FLAG, OUT,
        value_attrs={"rocdl.flat_work_group_size": f"{int(BLOCK)},{int(BLOCK)}"},
    ).launch(grid=(fx.Index(NWG), 1, 1), block=(BLOCK, 1, 1), stream=stream)


def main():
    dev = "cuda"
    data = torch.zeros(NWG * BLOCK, device=dev, dtype=torch.float32)
    flag = torch.zeros(4, device=dev, dtype=torch.float32)
    out = torch.zeros(NWG * BLOCK, device=dev, dtype=torch.float32)
    launch_sem(data, flag, out, torch.cuda.current_stream())
    torch.cuda.synchronize()
    turns = out.view(NWG, BLOCK)[:, 0].cpu()
    saw = int((turns > 0).sum())
    print("aux=%d  counter=%.3f (want %d)  data_sum=%.0f (want %d)"
          % (AUX, float(flag[0]), NWG, float(data.sum()), NWG * BLOCK))
    print("work-groups that SAW the full count: %d / %d  (out of %d polls each)"
          % (saw, NWG, POLLS - 1))
    if saw:
        seen = turns[turns > 0]
        print("  polls-seen-full: min %d  median %d  max %d  -> earliest WG saw it at poll ~%d"
              % (int(seen.min()), int(seen.median()), int(seen.max()), POLLS - 1 - int(seen.max())))
    print("VERDICT:", "concurrent visibility WORKS" if saw == NWG
          else ("PARTIAL (%d/%d)" % (saw, NWG) if saw else "NO visibility within %d polls" % POLLS))


if __name__ == "__main__":
    main()
