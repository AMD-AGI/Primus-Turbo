#!/usr/bin/env python3
"""Hand launcher for aiter's prebuilt gfx1250 ASM kernels.

The BACKWARD machinery now lives in
primus_turbo/pytorch/kernels/attention/_asm_bwd_kernargs.py, because it went behind the
dispatcher and product code must not import from tools/. It is re-exported here so this
script's bring-up and ABI self-test keep working unchanged.

The FORWARD launcher below is still experiment-only: the shipping forward path reaches
aiter through aiter.ops.mha, and this direct .co launch exists to take the @compile_ops JIT
and hipcc off the critical path if that ever matters.
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

# Loaded by file path rather than as primus_turbo.pytorch.kernels...: importing it through
# the package pulls primus_turbo.pytorch.__init__, which imports torch. The ABI self-test
# below is meant to run with no torch and no GPU -- that is the whole reason it exists, and
# it is what caught a hand-transcribed field table that had silently dropped its last seven
# fields including mask_x/mask_y.
import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location(
    "_asm_bwd_kernargs",
    _REPO / "primus_turbo" / "pytorch" / "kernels" / "attention" / "_asm_bwd_kernargs.py",
)
_m = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_m)

ASM_DIR, BDX, BF16, CO = _m.ASM_DIR, _m.BDX, _m.BF16, _m.CO
DQDKDV_FIELDS, DQDKDV_SIZE, FP32 = _m.DQDKDV_FIELDS, _m.DQDKDV_SIZE, _m.FP32
HipModule, MASK_X, MASK_Y = _m.HipModule, _m.MASK_X, _m.MASK_Y
ODO_FIELDS, ODO_SIZE = _m.ODO_FIELDS, _m.ODO_SIZE
POST_FIELDS, POST_SIZE, SYMBOLS = _m.POST_FIELDS, _m.POST_SIZE, _m.SYMBOLS
TS_DQ, TS_KV, TS_ODO, TS_QO = _m.TS_DQ, _m.TS_KV, _m.TS_ODO, _m.TS_QO
asm_backward, pack_compact, pack_padded = _m.asm_backward, _m.pack_compact, _m.pack_padded

# ---- forward -------------------------------------------------------------
# The prebuilt gfx1250 ASM forward. Launching it here rather than through
# aiter.ops.mha.fmha_fwd_with_sink_asm takes three things off the critical
# path: the missing jax dependency that aiter.ops.mha's import chain pulls in,
# the @compile_ops JIT build, and hipcc. The ABI is fully documented in
# csrc/py_itfs_cu/asm_fmha_fwd_with_sink.cu, whose static_assert of 132 bytes
# matches the ELF kernarg_segment_size exactly.
FWD_DIR = Path("/home/lihuzhan/code/aiter-src/hsa/gfx1250/fmha_fwd_bf16")
FWD_CO = {True: "fmha_bf16_pertokenBf16_hd128_128x256_mask.co",
          False: "fmha_bf16_pertokenBf16_hd128_128x256.co"}
FWD_SYM = {True: "_ZN5aiter41fmha_bf16_pertokenBf16_hd128_128x256_maskE",
           False: "_ZN5aiter36fmha_bf16_pertokenBf16_hd128_128x256E"}

SUB_Q = 128   # Q-tile size (ts_qo), asm_fmha_fwd_with_sink.cu:265
FWD_OPT = 7   # bit0 reverse_kv | bit1 double_q | bit2 remap_xy; the host hardcodes 7

# struct KernelArgs is "#pragma pack(push, 1)" -- genuinely packed, unlike the
# 16-byte-padded backward struct. Offsets are from the .cu comments.
FWD_FIELDS = [
    ("d_addr", 0x00, 8), ("q_addr", 0x08, 8), ("k_addr", 0x10, 8), ("v_addr", 0x18, 8),
    ("lse_addr", 0x20, 8), ("scalar", 0x28, 4), ("q_seq_len", 0x2C, 4),
    ("q_seqs", 0x30, 4), ("q_ts", 0x34, 4), ("q_hs", 0x38, 4), ("q_bas", 0x3C, 4),
    ("gqa", 0x40, 4), ("k_seqs", 0x44, 4), ("k_hs", 0x48, 4), ("k_bas", 0x4C, 4),
    ("opt", 0x50, 4), ("lse", 0x54, 4), ("kv_seq_len", 0x58, 4), ("q_head_num", 0x5C, 4),
    ("v_seqs", 0x60, 4), ("v_hs", 0x64, 4), ("v_bas", 0x68, 4),
    ("d_seqs", 0x6C, 4), ("d_hs", 0x70, 4), ("d_bas", 0x74, 4), ("lse_hs", 0x78, 4),
    ("sink_addr", 0x7C, 8),
]
FWD_SIZE = 0x84  # 132


def asm_forward(q, k, v, softmax_scale=None, causal=True, hip=None):
    """Prebuilt gfx1250 ASM forward. Returns (out, lse) with lse [B, Hq, Sq] fp32.

    lse is NATURAL log, matching what the fused backward and asm_backward()
    expect, so the two chain with no relayout. NEVER RUN.
    """
    import math

    import torch

    batch, seqlen_q, nhead_q, head_dim = q.shape
    seqlen_k, nhead_k = k.shape[1], k.shape[2]
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("the ASM forward requires stride(-1) == 1 on q/k/v")

    out = torch.empty((batch, seqlen_q, nhead_q, head_dim), device=q.device, dtype=q.dtype)
    # The kernel always writes LSE, even when the caller does not want it.
    lse = torch.empty((batch, nhead_q, seqlen_q), device=q.device, dtype=torch.float32)

    def bs(t, elem):
        return (t.stride(0) * elem, t.stride(1) * elem, t.stride(2) * elem)

    q_bas, q_seqs, q_hs = bs(q, BF16)
    k_bas, k_seqs, k_hs = bs(k, BF16)
    v_bas, v_seqs, v_hs = bs(v, BF16)
    d_bas, d_seqs, d_hs = bs(out, BF16)

    hip = hip or HipModule()
    fn = hip.function(FWD_DIR / FWD_CO[causal], FWD_SYM[causal])

    q_tiles = (seqlen_q + SUB_Q - 1) // SUB_Q
    gdx = (q_tiles + 1) // 2 if (FWD_OPT & 0x2) else q_tiles   # double_q halves it
    # remap_xy: the host swaps gdx and gdy so bid.x indexes heads.
    grid = (nhead_q, gdx, batch)

    hip.launch(fn, grid, (BDX, 1, 1), pack_padded(FWD_FIELDS, FWD_SIZE, {
        "d_addr": out.data_ptr(), "q_addr": q.data_ptr(), "k_addr": k.data_ptr(),
        "v_addr": v.data_ptr(), "lse_addr": lse.data_ptr(), "sink_addr": 0,
        "scalar": float(softmax_scale), "q_seq_len": seqlen_q, "kv_seq_len": seqlen_k,
        "q_seqs": q_seqs, "q_ts": SUB_Q * q_seqs, "q_hs": q_hs, "q_bas": q_bas,
        "k_seqs": k_seqs, "k_hs": k_hs, "k_bas": k_bas,
        "v_seqs": v_seqs, "v_hs": v_hs, "v_bas": v_bas,
        "d_seqs": d_seqs, "d_hs": d_hs, "d_bas": d_bas,
        "lse_hs": seqlen_q * FP32, "gqa": nhead_q // nhead_k,
        "opt": FWD_OPT, "lse": 1, "q_head_num": nhead_q,
    }), torch.cuda.current_stream().cuda_stream)
    return out, lse


def selftest() -> int:
    """Exercise the packing without a GPU: sizes, offsets, and byte-stride units."""
    ok = True

    odo = pack_compact(ODO_FIELDS, ODO_SIZE, {n: 0x1122334455667788 if w == 8 else 0x11223344
                                              for n, w in ODO_FIELDS})
    print(f"  odo      packed {len(odo)} B (kernarg 84)      {'ok' if len(odo) == 84 else 'BAD'}")
    ok &= len(odo) == 84

    vals = {n: (0x1122334455667788 if w == 8 else 1) for n, _, w in DQDKDV_FIELDS}
    vals["scalar"], vals["log2e"] = 0.08838835, 1.4426950408889634
    dq = pack_padded(DQDKDV_FIELDS, DQDKDV_SIZE, vals)
    last = max(o + w for _, o, w in DQDKDV_FIELDS)
    print(f"  dqdkdv   packed {len(dq)} B (kernarg 704), last field ends at {last}  "
          f"{'ok' if len(dq) == 704 and last <= 704 else 'BAD'}")
    ok &= len(dq) == 704 and last <= 704
    got = struct.unpack_from("<f", dq, 144)[0]
    print(f"  scalar round-trips as float: {got:.8f}  {'ok' if abs(got - 0.08838835) < 1e-7 else 'BAD'}")
    ok &= abs(got - 0.08838835) < 1e-7

    post = pack_padded(POST_FIELDS, POST_SIZE, {n: 1 for n, _, w in POST_FIELDS})
    tail = bytes(post[0xA0:])
    print(f"  post     packed {len(post)} B (kernarg 208), bytes past 0x9f all zero: "
          f"{'ok' if tail == bytes(len(tail)) else 'BAD'}")
    ok &= len(post) == 208 and tail == bytes(len(tail))

    fwd = pack_padded(FWD_FIELDS, FWD_SIZE, {n: 0 for n, _, _ in FWD_FIELDS})
    last = max(o + w for _, o, w in FWD_FIELDS)
    print(f"  forward  packed {len(fwd)} B (kernarg 132), last field ends at {last}  "
          f"{'ok' if len(fwd) == 132 and last == 132 else 'BAD'}")
    ok &= len(fwd) == 132 and last == 132
    for p in (FWD_DIR / FWD_CO[True], FWD_DIR / FWD_CO[False]):
        print(f"  {p.name:<46} {'present' if p.exists() else 'MISSING'}")
        ok &= p.exists()

    for stem in ("bwd_hd128_odo_bf16", "bwd_hd128_bf16_causal_br_a32_pssk",
                 "bwd_hd128_dq_convert_bf16"):
        p = ASM_DIR / f"{stem}.co"
        print(f"  {stem:<38} {'present' if p.exists() else 'MISSING'}")
        ok &= p.exists()

    print("\nselftest:", "pass" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="check packing and file presence; needs no GPU")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    print("Nothing to do without --selftest; import this module to launch.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
