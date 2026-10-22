###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""fwd_pp: the gfx1250 d128 GQA flash-attention forward k_attn_fwd_pp (8 waves, 2 per SIMD, staggered
ping-pong; hk/notes/fwd_pp_design.md, hk/notes/fwd_pp_impl.md).

lab hk arm (hk/lab/lab.py):  name=hk:<this package dir>#fwd_pp[@<cc.sh output dir>]

    flash_attn_fwd(q, k, v, softmax_scale, causal) -> (o [B,Sq,Hq,128] bf16, lse [B,Hq,Sq] fp32 natural log)

BSHD contiguous bf16 inputs, causal = bottom-right (query i sees keys j <= i + Skv - Sq). One launch on
torch's current stream, routed through the module-level `_launch(name, fn, args)`. The code object is
resolved lazily at the first launch from `HSACO_DIR` (set by lab.py / `set_hsaco_dir`), else
DEFAULT_HSACO_DIR; it is checked against its own metadata (no hidden kernargs, block 256, LDS 327680,
no scratch / spill) before it is used.

Host contract (check_args raises ValueError; no silent fallback): q [B, Sq, Hq, 128], k == v shape
[B, Skv, Hkv, 128], bf16, contiguous, 16-B aligned, one device; G = Hq / Hkv integer with G % 4 == 0
(a workgroup's 4 q heads share one kv head); Sq % 64 == 0, Skv % 64 == 0; causal -> Sq <= Skv;
2 B Sq Hq 128 < 2^31 (O / Q byte offsets are i32 voffsets / soffsets) and 4 B Hq Sq < 2^31; grid < 2^31;
B >= 1, Hq >= 4; softmax_scale finite > 0 with c2 = softmax_scale * log2(e) rounded to f32 a finite normal number
(the kernel's only scale; 0 / denormal / inf would turn masked or real scores into NaN). The kernel output is
deterministic (no atomics).
"""

import math
import os
import struct
import threading

import torch

KERNEL = "k_attn_fwd_pp"
DEFAULT_HSACO_DIR = "/home/lihuzhan/projects/2026_1010__llama3.1_attn/hk/build/fwd_pp"
HSACO_DIR = None  # set by lab.py (attribute or set_hsaco_dir); resolved at the first launch

D = 128  # head dim
QB = 64  # query positions per workgroup
NH = 4  # q heads per workgroup
ST = 64  # kv rows per ring stage
BLOCK = 256  # 8 waves
LDS_BYTES = 327680
VGPR_BUDGET = 512  # 2 waves / SIMD
LOG2E = 1.4426950408889634
I32_LIMIT = 1 << 31
F32_MIN_NORMAL = 2.0**-126

# kernarg list, in order (k_attn_fwd_pp.hip); kinds as hiprt.pack_args takes them
ARGS = (
    ("Q", "ptr"), ("K", "ptr"), ("V", "ptr"), ("O", "ptr"), ("LSE", "ptr"),
    ("o_bytes", "i64"), ("lse_bytes", "i64"),
    ("B", "i32"), ("Sq", "i32"), ("Skv", "i32"), ("Hq", "i32"), ("Hkv", "i32"), ("G", "i32"),
    ("nqb", "i32"), ("nhq4", "i32"), ("nkv_st", "i32"), ("cst", "i32"), ("causal", "i32"),
    ("c2", "f32"),
)  # fmt: skip
KERNARG_BYTES = 104

_LOCK = threading.Lock()
_KERNEL = None  # (hsaco path, hiprt.Kernel)
_DEVICES_CHECKED = set()


def set_hsaco_dir(path):
    """lab.py hook: use the code object in `path` (a cc.sh output dir) from the next launch on."""
    global HSACO_DIR, _KERNEL
    with _LOCK:
        HSACO_DIR = None if path is None else os.path.abspath(path)
        _KERNEL = None


def hsaco_path():
    return os.path.join(HSACO_DIR or DEFAULT_HSACO_DIR, KERNEL + ".hsaco")


def check_code_object(path):
    """Raise unless the code object at `path` is a k_attn_fwd_pp build this module can launch."""
    from . import codeobj

    table = codeobj.kernels(path)
    if KERNEL not in table:
        raise RuntimeError(f"{path}: no kernel {KERNEL} ({sorted(table)})")
    k = table[KERNEL]
    bad = []
    if k["lds"] != LDS_BYTES:
        bad.append(f"LDS {k['lds']} != {LDS_BYTES}")
    if k["scratch"] != 0 or k.get("vgpr_spill", 0) or k.get("sgpr_spill", 0):
        bad.append(f"scratch {k['scratch']} / spills {k.get('vgpr_spill')} {k.get('sgpr_spill')}")
    if not 0 < k["vgpr"] <= VGPR_BUDGET:
        bad.append(f"vgpr {k['vgpr']} outside (0, {VGPR_BUDGET}]")
    if k["max_block"] != BLOCK or k.get("wavefront_size", 32) != 32:
        bad.append(f"max_block {k['max_block']} / wave {k.get('wavefront_size')}")
    if k["kernarg_size"] != KERNARG_BYTES:
        bad.append(f"kernarg {k['kernarg_size']} != {KERNARG_BYTES}")
    if bad:
        raise RuntimeError(f"{path}: {KERNEL} is not launchable here: " + "; ".join(bad))
    return k


def _kernel():
    global _KERNEL
    path = hsaco_path()
    kern = _KERNEL
    if kern is None or kern[0] != path:
        with _LOCK:
            if _KERNEL is None or _KERNEL[0] != path:
                from . import hiprt

                check_code_object(path)
                ks = hiprt.load_hsaco(path, {KERNEL: {"block": BLOCK, "args": list(ARGS)}})
                _KERNEL = (path, ks[KERNEL])
            kern = _KERNEL
    return kern[1]


def check_args(q, k, v, softmax_scale, causal):
    """The kernel's argument contract; returns (B, Sq, Skv, Hq, Hkv). Raises ValueError."""
    for name, t in (("q", q), ("k", k), ("v", v)):
        if not isinstance(t, torch.Tensor):
            raise ValueError(f"{name} must be a torch.Tensor")
        if t.dtype != torch.bfloat16 or t.dim() != 4 or not t.is_contiguous():
            raise ValueError(
                f"{name} must be a contiguous 4-D bfloat16 tensor (BSHD), got {t.dtype} {tuple(t.shape)}"
            )
        if t.device != q.device:
            raise ValueError(f"{name} is on {t.device}, q on {q.device}")
        if t.device.type != "meta" and t.data_ptr() % 16:
            raise ValueError(f"{name} data_ptr not 16-B aligned")
    if not isinstance(causal, bool):
        raise ValueError(f"causal must be a bool, got {causal!r}")
    b, sq, hq, d = q.shape
    skv, hkv = k.shape[1], k.shape[2]
    if d != D or tuple(k.shape) != (b, skv, hkv, D) or tuple(v.shape) != tuple(k.shape):
        raise ValueError(
            f"shapes q {tuple(q.shape)} k {tuple(k.shape)} v {tuple(v.shape)}: need head dim {D}, k == v"
        )
    if b < 1 or hq < NH:
        raise ValueError(f"B {b} / Hq {hq}: need B >= 1 and Hq >= {NH}")
    if hkv < 1 or hq % hkv:
        raise ValueError(f"Hq {hq} not a multiple of Hkv {hkv}")
    g = hq // hkv
    if g % NH:
        raise ValueError(
            f"GQA ratio {g}: k_attn_fwd_pp needs G % 4 == 0 (4 q heads of a workgroup share a kv head)"
        )
    if sq < QB or sq % QB or skv < ST or skv % ST:
        raise ValueError(f"Sq {sq} / Skv {skv} must be positive multiples of 64")
    if causal and sq > skv:
        raise ValueError(f"causal with Sq {sq} > Skv {skv}: rows without a visible key are not supported")
    if 2 * b * sq * hq * D >= I32_LIMIT or 4 * b * hq * sq >= I32_LIMIT:
        raise ValueError("O / LSE byte extents must be < 2^31 (i32 buffer offsets)")
    if 2 * b * skv * hkv * D >= 1 << 47:
        raise ValueError("K / V byte extent beyond the TDM address range")
    if b * (hq // NH) * (sq // QB) >= I32_LIMIT:
        raise ValueError("grid too large")
    if not (isinstance(softmax_scale, (int, float)) and math.isfinite(softmax_scale) and softmax_scale > 0):
        raise ValueError(f"softmax_scale must be finite and > 0, got {softmax_scale!r}")
    c2 = c2_f32(softmax_scale)
    if not (math.isfinite(c2) and c2 >= F32_MIN_NORMAL):
        raise ValueError(
            f"softmax_scale {softmax_scale!r}: c2 = scale * log2(e) in f32 is {c2!r}, not a finite normal"
        )
    return b, sq, skv, hq, hkv


def c2_f32(softmax_scale):
    """c2 = softmax_scale * log2(e) as the kernel receives it (f32); inf when it overflows f32."""
    try:
        return struct.unpack("<f", struct.pack("<f", float(softmax_scale) * LOG2E))[0]
    except OverflowError:
        return math.inf


def plan(q, k, v, o, lse, softmax_scale, causal):
    """(grid, block, kernarg values in ARGS order) of one launch; pure (meta tensors work)."""
    b, sq, skv, hq, hkv = check_args(q, k, v, softmax_scale, causal)
    if tuple(o.shape) != tuple(q.shape) or o.dtype != torch.bfloat16 or not o.is_contiguous():
        raise ValueError("o must be a contiguous bf16 tensor shaped like q")
    if tuple(lse.shape) != (b, hq, sq) or lse.dtype != torch.float32 or not lse.is_contiguous():
        raise ValueError("lse must be a contiguous fp32 tensor [B, Hq, Sq]")
    nqb, nhq4, nkv_st = sq // QB, hq // NH, skv // ST
    o_bytes, lse_bytes = 2 * b * sq * hq * D, 4 * b * hq * sq
    c2 = float(softmax_scale) * LOG2E  # double, rounded to f32 by the packer
    values = [q, k, v, o, lse, o_bytes, lse_bytes, b, sq, skv, hq, hkv, hq // hkv, nqb, nhq4, nkv_st,
              (skv - sq) // ST, int(causal), c2]  # fmt: skip
    return (b * nhq4 * nqb, 1, 1), (BLOCK, 1, 1), values


def _check_device_once(device):
    if device in _DEVICES_CHECKED:
        return
    arch = torch.cuda.get_device_properties(device).gcnArchName
    if "gfx1250" not in arch:
        raise RuntimeError(f"k_attn_fwd_pp is gfx1250 code; device {device} is {arch}")
    _DEVICES_CHECKED.add(device)


def _launch(name, fn, args):
    """Every kernel launch goes through here (lab.py --seg wraps it)."""
    return fn(*args)


def flash_attn_fwd(q, k, v, softmax_scale, causal):
    b, sq, skv, hq, hkv = check_args(q, k, v, softmax_scale, causal)
    if q.device.type != "cuda":
        raise ValueError(f"tensors must be on a GPU, got {q.device}")
    _check_device_once(q.device)
    o = torch.empty_like(q)
    lse = torch.empty((b, hq, sq), dtype=torch.float32, device=q.device)
    grid, block, values = plan(q, k, v, o, lse, softmax_scale, causal)
    kern = _kernel()
    _launch(KERNEL, kern.call, (grid, block, values, torch.cuda.current_stream(q.device)))
    return o, lse
