#!/usr/bin/env python3
"""Standalone probe: which vdata type makes `rocdl.raw_ptr_buffer_atomic_fadd` emit
`buffer_atomic_pk_add_bf16` on this flydsl build, and at what rate.

Nothing here imports flash_attn_bwd. One arm = one minimal kernel that adds a constant into
an image REPS times, so the arm passes only if
  (1) the ISA dump contains the instruction name the arm is asking for, and
  (2) the image reads back exactly REPS * 1.0 (bf16 holds every integer to 256, so a correct
      packed accumulation is EXACT and a wrong lane/opsel shows up as 2x, half, or garbage).

arms
  v2bf16   vector<2xbf16> vdata  -> want buffer_atomic_pk_add_bf16
  i32bc    the same pair bitcast to i32 (the untried lead from the round-1 write-up)
  f32      scalar f32 on an f32 image -> buffer_atomic_add_f32 (control: the op itself works)
  v1bf16   vector<1xbf16>, 2 B -> is there a narrow form?
  v4bf16   vector<4xbf16>, 8 B -> is there a wider packed form?
  v2f32    vector<2xf32>, 8 B -> is there a wider float form at all?

usage: _probe_pkatom.py [arm ...]
"""
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPS = 32
N = 1 << 22  # elements in the image; 8 MB at bf16, so the atomics stay in cache
BLOCK = 256
ARMS = ["f32", "v2bf16", "v2bf16s", "f32s", "i32bc", "v1bf16", "v4bf16", "v2f32"]
WANT = {
    "v2bf16": "buffer_atomic_pk_add_bf16",
    "v2bf16s": "buffer_atomic_pk_add_bf16",
    "i32bc": "buffer_atomic_pk_add_bf16",
    "f32": "buffer_atomic_add_f32",
    "f32s": "buffer_atomic_add_f32",
    "v1bf16": "buffer_atomic_pk_add_bf16",
    "v4bf16": "buffer_atomic_pk_add_bf16",
    "v2f32": "buffer_atomic_add_f32",
}
# vdata elements per lane-atomic, which is what the rate has to be normalised by.
LANES = {"v2bf16": 2, "v2bf16s": 2, "i32bc": 2, "f32": 1, "f32s": 1, "v1bf16": 1, "v4bf16": 4, "v2f32": 2}


def _run_arm(arm):
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    import torch
    from flydsl.expr import buffer_ops, const_expr, gpu, range_constexpr, rocdl
    from flydsl.expr.typing import Vector as Vec
    from flydsl.expr.utils.arith import ArithValue
    from flydsl.expr.utils.arith import _to_raw as _raw

    per_lane = LANES[arm]
    is_f32 = arm in ("f32", "f32s", "v2f32")
    spread = arm.endswith("s")
    esize = 4 if is_f32 else 2
    nthread = N // per_lane
    # `spread`: each of the REPS atomics of one lane lands in a different N/REPS-sized region,
    # so the 32 updates of one lane never touch the same cache line. That separates the L2
    # atomic-unit throughput from same-address RMW serialisation.
    stride = (N // REPS) if spread else 0

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def pkatom_kernel(OUT: fx.Tensor):
        tid = fx.Index(gpu.block_idx.x) * fx.Index(BLOCK) + fx.Index(gpu.thread_idx.x)
        rsrc = buffer_ops.create_buffer_resource(OUT, max_size=True)
        # Byte offset of this lane's slot; consecutive lanes take consecutive slots, so a
        # wave's atomics cover one contiguous run -- the coalesced case, not the scattered one.
        base = tid * fx.Index(per_lane)
        ones32 = [fx.Float32(1.0) for _ in range_constexpr(per_lane)]
        if const_expr(arm == "v2f32"):
            vdata = Vec.from_elements(ones32, fx.Float32)
        elif const_expr(is_f32):
            vdata = ones32[0]
        elif const_expr(arm == "i32bc"):
            vdata = Vec.from_elements(ones32, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0]
        else:
            vdata = Vec.from_elements(ones32, fx.Float32).to(fx.BFloat16)
        for r in range_constexpr(REPS):
            elem = (base + fx.Index(r * stride)) % fx.Index(N) if const_expr(spread) else base
            off = ArithValue(_raw(elem * fx.Index(esize))).index_cast(fx.Int32.ir_type)
            rocdl.raw_ptr_buffer_atomic_fadd(_raw(vdata), rsrc, off, 0, 0)

    @flyc.jit
    def launch(OUT: fx.Tensor, stream: fx.Stream):
        pkatom_kernel(OUT).launch(grid=(fx.Index(nthread // BLOCK), 1, 1), block=(BLOCK, 1, 1), stream=stream)

    tdt = torch.float32 if is_f32 else torch.bfloat16
    out = torch.zeros(N, device="cuda", dtype=tdt)
    st = torch.cuda.current_stream()
    fn = flyc.compile(launch, out, st)  # NB: compiling also launches once
    out.zero_()
    fn(out, st)
    torch.cuda.synchronize()
    err = float((out.float() - REPS).abs().max())
    print(
        f"# {arm} out[:4]={out[:4].float().tolist()} min={float(out.float().min())} "
        f"max={float(out.float().max())} mean={float(out.float().mean()):.4f}",
        file=sys.stderr,
    )

    ev = [torch.cuda.Event(True) for _ in range(2)]
    best = float("inf")
    for _ in range(20):
        ev[0].record()
        fn(out, st)
        ev[1].record()
        torch.cuda.synchronize()
        best = min(best, ev[0].elapsed_time(ev[1]))
    # One lane-atomic per vdata, REPS per lane, N/per_lane lanes.
    gatom = (N / per_lane) * REPS / best / 1e6
    gupd = N * REPS / best / 1e6
    return err, best, gatom, gupd


def _child(arm):
    d = f"/tmp/pkatom_{arm}"
    shutil.rmtree(d, ignore_errors=True)
    os.environ["FLYDSL_DUMP_IR"] = "1"
    os.environ["FLYDSL_DUMP_DIR"] = d
    res = {"arm": arm, "want": WANT[arm]}
    try:
        err, ms, gatom, gupd = _run_arm(arm)
        res.update(
            maxerr=round(err, 6), ms=round(ms, 4), Gatom_s=round(gatom, 1), Gupd_s=round(gupd, 1)
        )
    except Exception as e:
        txt = f"{type(e).__name__}: {e}"
        res["fail"] = txt.replace("\n", " | ")[:600]
    hits, isa = 0, ""
    for root, _, files in os.walk(d):
        for f in files:
            if f.endswith(".s") or f.endswith("_isa.s"):
                p = os.path.join(root, f)
                body = open(p, errors="ignore").read()
                if WANT[arm] in body:
                    isa, hits = p, max(hits, body.count(WANT[arm]))
    res["isa_hits"] = hits
    res["isa"] = isa
    print(json.dumps(res))


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        return _child(sys.argv[2])
    arms = sys.argv[1:] or ARMS
    for arm in arms:
        # Own process per arm: a failed instruction selection can wedge the MLIR context, and
        # the ISA dump directory has to be a clean read.
        r = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--child", arm],
            capture_output=True,
            text=True,
            timeout=900,
        )
        for ln in r.stderr.splitlines():
            if ln.startswith("#"):
                print(ln, flush=True)
        line = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
        if line:
            print(line[-1], flush=True)
        else:
            tail = (r.stdout[-600:] + r.stderr[-1400:]).replace("\n", " | ")
            print(json.dumps({"arm": arm, "crash": tail[-1600:]}), flush=True)


if __name__ == "__main__":
    main()
