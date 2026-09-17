"""gfx1250 FlyDSL `odo` kernel: delta = rowsum(dO * O), fp32.

The first kernel we write ourselves in FlyDSL for gfx1250. Chosen first deliberately: the
reference is three lines of torch, so it is checkable WITHOUT launching a main attention
kernel, and it is the cheapest way to find out whether we can build a gfx1250 FlyDSL kernel
at all and what a cold JIT build costs.

Ported from aiter's gfx942 `k_delta` (fmha_bwd_gfx942/fmha_bwd_core.py). That kernel is
almost arch-neutral -- no MFMA, no ds_read_tr16, no permlane32_swap -- so the port is two
changes:

  1. WAVE SIZE. The cross-lane reduction is `shuffle_xor(acc, 1 << s, WIDTH)`. gfx942 is
     wave64; gfx1250 dispatches wave32. The butterfly itself only spans 16 lanes either way,
     but the width argument must match the dispatch or the shuffle reads lanes that do not
     exist. This is the one place the wave32 trap could bite, and it would not raise.
  2. LAYOUT. gfx942's is varlen THD with DEL [H, T]; ours is BSHD [B, S, H, D] with delta
     [B, H, S], matching the LSE the gfx1250 forward emits.

Correctness is gated with NaN-prefill + full isfinite coverage BEFORE SQNR, because a
wave-size mismatch drops lanes' work rather than raising.
"""
import os, sys, time, json
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

import torch
import flydsl
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr

print(f"flydsl {flydsl.__version__} @ {flydsl.__file__}")
ARCH = torch.cuda.get_device_properties(0).gcnArchName
print(f"arch {ARCH}")
assert "gfx1250" in ARCH, f"this kernel targets gfx1250, got {ARCH}"

DV = 128
WAVE = 32                    # gfx1250 dispatches wave32. gfx942's version says 64 here.
DELTA_THREADS = 256
LANES_PER_ROW = DV // 8      # 16 threads cover one 128-wide row at 8 elements each
ROWS_PER_PASS = DELTA_THREADS // LANES_PER_ROW
ROWS_DELTA = 32
PASSES_PER_WG = ROWS_DELTA // ROWS_PER_PASS


def _buffer_view(tensor, nbytes, fx_dt, vec=1):
    ntile = (1 << 31) // (vec * (fx_dt.width // 8))
    buf = fx.rocdl.make_buffer_tensor(tensor, num_records_bytes=fx.Int64(nbytes))
    return fx.Tensor(fx.make_view(fx.get_iter(buf), fx.make_layout((ntile, vec), (vec, 1))))


def _atom(fx_dt, vec):
    return fx.make_copy_atom(fx.rocdl.BufferCopy(vec * fx_dt.width), fx_dt)


def _ldv(buf, tile, fx_dt, vec):
    frag = fx.make_rmem_tensor(fx.make_layout(vec, 1), fx_dt)
    fx.copy_atom_call(_atom(fx_dt, vec), fx.slice(buf, (tile, None)), frag)
    return frag.load().ir_value()


def _st1(val, buf, idx, fx_dt):
    frag = fx.make_rmem_tensor(fx.make_layout(1, 1), fx_dt)
    fx.memref_store_vec(fx.Vector.from_elements([val], dtype=fx_dt), frag)
    fx.copy_atom_call(_atom(fx_dt, 1), frag, fx.slice(buf, (idx, None)))


@flyc.kernel(known_block_size=[DELTA_THREADS, 1, 1])
def k_delta_bshd(DO: fx.Tensor, O: fx.Tensor, DEL: fx.Tensor,
                 S: fx.Int32, H: fx.Int32, n_rows: fx.Int32):
    """delta[b, h, s] = sum_d dO[b, s, h, d] * O[b, s, h, d], fp32.

    A row is one (b, s, h); its DV elements are contiguous in BSHD, so the flat row index
    runs in DO/O memory order and only the STORE index has to be de-interleaved.
    """
    tid = fx.Int32(fx.thread_idx.x)
    bid = fx.Int32(fx.block_idx.x)
    g_do = _buffer_view(DO, n_rows * (DV * 2), fx.BFloat16, 8)
    g_o = _buffer_view(O, n_rows * (DV * 2), fx.BFloat16, 8)
    g_delta = _buffer_view(DEL, n_rows * 4, fx.Float32)

    tile = bid * (ROWS_DELTA * DV // 8) + tid
    do_vecs = [_ldv(g_do, tile + u * (ROWS_PER_PASS * DV // 8), fx.BFloat16, 8)
               for u in range_constexpr(PASSES_PER_WG)]
    o_vecs = [_ldv(g_o, tile + u * (ROWS_PER_PASS * DV // 8), fx.BFloat16, 8)
              for u in range_constexpr(PASSES_PER_WG)]

    lane_in_row = tid % fx.Int32(LANES_PER_ROW)
    row_in_group = tid // fx.Int32(LANES_PER_ROW)
    for u in range_constexpr(PASSES_PER_WG):
        do8 = fx.Vector(do_vecs[u])
        o8 = fx.Vector(o_vecs[u])
        e0 = fx.Float32(0.0)
        e1 = fx.Float32(0.0)
        for c in range_constexpr(4):
            e0 = e0 + fx.Float32(do8[2 * c]) * fx.Float32(o8[2 * c])
            e1 = e1 + fx.Float32(do8[2 * c + 1]) * fx.Float32(o8[2 * c + 1])
        acc = e0 + e1
        # 4 XOR shuffles reduce the 16 lanes of a row. WAVE is the dispatch width.
        for sft in range_constexpr(4):
            acc = acc + fx.gpu.shuffle_xor(acc, 1 << sft, WAVE)
        idx = bid * fx.Int32(ROWS_DELTA) + u * fx.Int32(ROWS_PER_PASS) + row_in_group
        ok = (lane_in_row == fx.Int32(0)) & (idx < n_rows)
        # BSHD row idx -> (b, s, h); delta is [B, H, S]
        b = idx // (S * H)
        rem = idx - b * (S * H)
        s_ = rem // H
        h = rem - s_ * H
        dst = (b * H + h) * S + s_
        # Lanes that are not lane 0, and rows past the end, are steered one past DEL so the
        # buffer descriptor drops the store -- the same bounds trick the template uses.
        _st1(acc, g_delta, ok.select(dst, n_rows), fx.Float32)


@flyc.jit
def launch_delta(DO: fx.Tensor, O: fx.Tensor, DEL: fx.Tensor,
                 S: fx.Int32, H: fx.Int32, n_rows: fx.Int32,
                 nblk: fx.Int32, stream: fx.Stream):
    k_delta_bshd(DO, O, DEL, S, H, n_rows).launch(
        grid=(nblk, 1, 1), block=(DELTA_THREADS, 1, 1), stream=stream)


def run(B, S, H, label):
    torch.manual_seed(0)
    do = torch.randn(B, S, H, DV, device="cuda", dtype=torch.bfloat16)
    o = torch.randn(B, S, H, DV, device="cuda", dtype=torch.bfloat16)
    n_rows = B * S * H
    assert n_rows % ROWS_DELTA == 0, "pad or handle the tail; the toy/production shapes divide"
    delta = torch.full((B, H, S), float("nan"), device="cuda", dtype=torch.float32)
    nblk = n_rows // ROWS_DELTA
    t0 = time.time()
    launch_delta(do, o, delta, S, H, n_rows, nblk, torch.cuda.current_stream())
    torch.cuda.synchronize()
    build_s = time.time() - t0

    fin = int(torch.isfinite(delta).sum())
    ref = (do.float() * o.float()).sum(-1).permute(0, 2, 1).contiguous()   # [B,S,H] -> [B,H,S]
    err = (ref - delta).pow(2).mean().clamp_min(1e-30)
    db = float(10 * torch.log10(ref.pow(2).mean() / err))
    print(f"  {label:16s} build+launch {build_s:6.2f}s   isfinite {fin}/{delta.numel()}"
          f"   SQNR {db:7.2f} dB" + ("" if fin == delta.numel() else "   <-- INCOMPLETE"))
    return {"label": label, "build_s": round(build_s, 3), "isfinite": f"{fin}/{delta.numel()}",
            "full_coverage": fin == delta.numel(), "sqnr_db": round(db, 2)}


print("\n--- toy shape first, in this process, before anything large ---")
res = [run(1, 256, 2, "toy b1 s256 h2")]
print("\n--- production shape ---")
res.append(run(4, 8192, 32, "prod b4 s8192 h32"))

# second call: cold build vs cached
t0 = time.time(); r = run(4, 8192, 32, "prod (cached)"); print()
json.dump(res + [r], open("/tmp/odo1250.json", "w"), indent=2)
ok = all(x["full_coverage"] and x["sqnr_db"] >= 100 for x in res)
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 2)
