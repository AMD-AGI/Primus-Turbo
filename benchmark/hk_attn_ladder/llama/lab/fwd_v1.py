###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""HK gfx1250 flash-attention forward v1 (HIP kernel ``k_attn_fwd_v1``): head dim 128, bf16 BSHD, GQA with
G = Hq / Hkv in (4, 8, 16), bottom-right causal (query i sees keys j <= i + Skv - Sq). Full attention
(causal=False) has a kernel path and a CPU proof but is not admitted: no GPU check of it exists (lab runs causal
only). Same call contract as fwd_v0.py:

    o, lse = flash_attn_fwd(q, k, v, softmax_scale=None, causal=True)
        q [B, Sq, Hq, 128], k / v [B, Skv, Hkv, 128] bf16 contiguous on one gfx1250 device
        -> o [B, Sq, Hq, 128] bf16, lse [B, Hq, Sq] fp32 (natural log of the scaled row sums)

Strided layouts (``SUPPORTS_STRIDES``; hk/notes/fwd_v1_strides.md). A code object built with -DHK_STRIDES=1 takes 8
more int64 kernel arguments (``STRIDE_ARGS``: batch and seq element strides of q, k, v, o) and then accepts each of
q, k, v independently in either layout (``layout_of``):
    "bshd"  contiguous [B, S, H, 128]                    strides (S H 128, H 128, 128, 1)
    "sbhd"  [S, B, H, 128] contiguous storage seen as     strides (H 128, B H 128, 128, 1)
            [B, S, H, 128] (x.permute(1, 0, 2, 3); Megatron's q / k / v)
(strides of size-1 dims are ignored, so B == 1 is "bshd" either way: the same memory); anything else raises
ValueError. o is allocated in q's layout ("sbhd": torch.empty(S, B, H, 128).permute(1, 0, 2, 3)); lse stays fp32
[B, Hq, Sq] contiguous. The bits of o / lse do not depend on the layouts. ``SUPPORTS_STRIDES`` (module attribute,
computed on access) says whether the code object the next launch loads is such a build, from its own kernel metadata
(30 arguments / 148 B: no; 38 / 216 B: yes; anything else raises; a cc.sh build's recorded -DHK_STRIDES flag must
agree); without it everything is exactly as before (contiguous BSHD required).

v1 runs v0's work item (64 positions x one head quad) from a PERSISTENT grid (hk/notes/fwd_v1_impl.md):
  * mode "pairs": W = min(#CUs, #units) workgroups; a unit is a zig-zag pair of q blocks of one head (heavy
    block first), plus one single middle block per head when Sq / 64 is odd; units are dealt XCD-major
    (XCC = workgroup id % 8), so the workgroups of one XCD work on one head at a time;
  * mode "items": v0's longest-first grid (W = #items, one item per workgroup), v1's body;
  * mode "auto" (default): "pairs" when #units >= #CUs, else "items".
The schedule is decoded in the kernel (fwd1/idx.h f1::wg_item) with host-computed magic numbers for every
runtime divisor (``fdiv_of``, the mirror of f1::fdiv_of).

lab hk contract (hk/lab/lab.py, arm ``hk:<pkgdir>#fwd_v1[@<hsaco_dir>]``): every launch goes through the
module-level ``_launch(name, fn, args)``; the code object is resolved lazily at the first launch from
``HSACO_DIR`` (set by lab or ``set_hsaco_dir``), else the package attribute ``HSACO_DIR``, else env
``HK_HSACO_DIR``, else ``DEFAULT_HSACO_DIR``. A build directory may hold ``k_attn_fwd_v1.launch.json``
({"mode": "auto" | "pairs" | "items", "W": optional int}), the launch configuration of that variant, so one
lab job can A/B schedules (hk/build/fwd_v1/build_variants.sh writes them).

Production (impl.py, the package default "v1"): impl points the module at the package's pinned ``bin/`` with
``set_hsaco_dir`` before its first launch, so ``DEFAULT_HSACO_DIR`` (a lab path) is never used there, and a
launch.json in ``bin/`` is ignored: production always launches the host default schedule (mode auto), which is
what the validated lsum0 build directory's launch.json says too (the sha256 pin covers the code object only).

Host gate: every argument contract is checked and anything outside the proven range raises (ValueError for
a broken contract, NotImplementedError for a valid but unsupported problem); there is no silent fallback.
``unsupported_reason`` is self-contained (v0's conditions copied, not imported from fwd_v0.py, whose family is
edited independently) and states the same conditions as hk/lab/bounds/fwd_v1_gate.h ``fwd1_admitted``, the gate
of the bounds proof hk/lab/bounds/fwd_v1_check.cpp (which replays the grids ``plan`` launches);
hk/lab/test_ext_fwd1.py (gate_sweep) compiles that header and compares the two on a shape sweep.
"""

import json
import math
import os
import sys
import threading

import torch

from . import codeobj, hiprt

# (ladder copy: no production bin/ next to these lab modules, so no launch.json is ever ignored)
_PROD_BIN_DIR = None

__all__ = ["flash_attn_fwd", "unsupported_reason", "set_hsaco_dir", "plan", "fdiv_of", "MODES", "layout_of",
           "supports_strides"]

KERNEL = "k_attn_fwd_v1"
D = 128
BLOCK = 128  # 4 waves (1 per SIMD)
LDS_BYTES = 327680  # static: one workgroup per CU
QB = 64  # query positions per item
NH = 4  # q heads per item (one head quad)
KV_ST = 64  # kv rows per ring stage
NXCD = 8  # XCC = workgroup id % 8 (measured)
GQA_RATIOS = (4, 8, 16)  # powers of two: the kernel's kv head is a shift by log2 G (f1::hkv_of)
DEFAULT_CUS = 256  # MI455X; the device's multi_processor_count is used when a device is present
MODES = ("auto", "pairs", "items")
MODE_ID = {"items": 0, "pairs": 1}  # f1::MODE_ITEMS / MODE_PAIRS
KARG_BYTES = 148
# kernel arguments in order, with explicit kinds (k_attn_fwd_v1.hip, struct KArgs)
ARGS = (
    ("Q", "ptr"), ("K", "ptr"), ("V", "ptr"), ("O", "ptr"), ("LSE", "ptr"),
    ("o_bytes", "i64"), ("lse_bytes", "i64"),
    ("B", "i32"), ("Sq", "i32"), ("Skv", "i32"), ("Hq", "i32"), ("Hkv", "i32"), ("G", "i32"),
    ("nqb", "i32"), ("nhq4", "i32"), ("nkv_st", "i32"), ("cst", "i32"), ("causal", "i32"),
    ("c2", "f32"), ("c2inv", "f32"), ("W", "i32"), ("mode", "i32"),
    ("mW", "u32"), ("shW", "i32"), ("mNP", "u32"), ("shNP", "i32"),
    ("mH4", "u32"), ("shH4", "i32"), ("mNH", "u32"), ("shNH", "i32"),
)  # fmt: skip
# byte offsets of KArgs (static_assert'ed in the kernel; checked against the code object's metadata)
ARG_OFFSETS = (0, 8, 16, 24, 32, 40, 48) + tuple(56 + 4 * i for i in range(23))
# -DHK_STRIDES=1 builds: these 8 follow ARGS (KArgs sQb .. sOs), element strides (batch, seq) of q, k, v, o
STRIDE_ARGS = (
    ("sQb", "i64"), ("sQs", "i64"), ("sKb", "i64"), ("sKs", "i64"),
    ("sVb", "i64"), ("sVs", "i64"), ("sOb", "i64"), ("sOs", "i64"),
)  # fmt: skip
STRIDE_ARG_OFFSETS = tuple(152 + 8 * i for i in range(8))
KARG_BYTES_STRIDES = 216
LAYOUTS = ("bshd", "sbhd")
TDM_STRIDE_LIMIT = 1 << 48  # the TDM descriptor's row-stride field (elements)
I32_LIMIT = 1 << 31
DEFAULT_HSACO_DIR = "/home/lihuzhan/projects/2026_1010__llama3.1_attn/hk/build/fwd_v1"
LAUNCH_JSON = f"{KERNEL}.launch.json"

HSACO_DIR = None  # set by lab.py / set_hsaco_dir; resolved at the first launch
_KERNEL = None  # (path, hiprt.Kernel, launch cfg, strided)
_STRIDED = {}  # code object path -> ((mtime_ns, size), strided) of supports_strides
_LOCK = threading.Lock()
_DEVICES = {}  # device index -> multi_processor_count (gfx1250 checked)


def set_hsaco_dir(path):
    """Use the code object (and launch.json) in ``path`` (a cc.sh output dir) from the next launch on."""
    global HSACO_DIR, _KERNEL
    with _LOCK:
        HSACO_DIR = None if path is None else os.path.abspath(path)
        _KERNEL = None


def _hsaco_dir():
    if HSACO_DIR:
        return HSACO_DIR
    pkg = sys.modules.get(__package__) if __package__ else None
    d = getattr(pkg, "HSACO_DIR", None) if pkg is not None else None
    return d or os.environ.get("HK_HSACO_DIR") or DEFAULT_HSACO_DIR


def fdiv_of(d):
    """(m, sh) with n // d == (n * m) >> sh for every 0 <= n < 2^31 (mirror of f1::fdiv_of)."""
    d = int(d)
    if not 1 <= d < I32_LIMIT:
        raise ValueError(f"fdiv divisor {d} outside [1, 2^31)")
    lg = 0
    while (1 << lg) < d:
        lg += 1
    m = ((1 << (31 + lg)) + d - 1) // d
    assert 0 < m < (1 << 32)
    return m, 31 + lg


def read_launch_cfg(dirpath):
    """The variant's launch configuration from <dir>/k_attn_fwd_v1.launch.json ({} when absent)."""
    p = os.path.join(dirpath, LAUNCH_JSON)
    if not os.path.isfile(p):
        return {}
    with open(p) as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict) or set(cfg) - {"mode", "W", "note"}:
        raise ValueError(f"{p}: keys must be a subset of mode / W / note, got {cfg!r}")
    if cfg.get("mode", "auto") not in MODES:
        raise ValueError(f"{p}: mode {cfg.get('mode')!r} not in {MODES}")
    if cfg.get("W") is not None and (not isinstance(cfg["W"], int) or cfg["W"] < 1):
        raise ValueError(f"{p}: W must be a positive int, got {cfg['W']!r}")
    return cfg


def launch_cfg(dirpath):
    """The launch configuration used with the code object in ``dirpath``: {} (the host default) for the package's
    pinned ``bin/`` (``hsaco.BIN_DIR``), else ``read_launch_cfg``."""
    if _PROD_BIN_DIR is not None and os.path.realpath(dirpath) == os.path.realpath(_PROD_BIN_DIR):
        return {}
    return read_launch_cfg(dirpath)


def _flag_strides(flags):
    """The HK_STRIDES value a recorded compile command defines (0 when it defines none)."""
    val = 0
    for f in flags:
        if isinstance(f, str) and (f == "-DHK_STRIDES" or f.startswith("-DHK_STRIDES=")):
            v = f.partition("=")[2] or "1"
            if v not in ("0", "1"):
                raise RuntimeError(f"compile flag {f!r}: HK_STRIDES must be 0 or 1")
            val = int(v)
    return val


def _check_code_object(path):
    """Resources and kernarg layout of the code object must be the ones the bounds proof and the launch
    assume (kernel argument offsets: the kernel re-reads arguments through the kernarg segment pointer as
    struct KArgs, so the metadata must match that struct byte for byte). The layout is exactly one of two:
    ARGS (148 B, the default build) or ARGS + STRIDE_ARGS (216 B, -DHK_STRIDES=1); the returned metadata carries
    ``strided`` accordingly. A cc.sh build (meta.json) must also have recorded the matching -DHK_STRIDES flag."""
    meta = codeobj.kernels(path).get(KERNEL)
    if meta is None:
        raise RuntimeError(f"{path}: no kernel {KERNEL}")
    want = {"lds": LDS_BYTES, "scratch": 0, "vgpr_spill": 0, "sgpr_spill": 0, "wavefront_size": 32,
            "max_block": BLOCK, "dynamic_stack": False}  # fmt: skip
    bad = {k: meta[k] for k, v in want.items() if meta[k] != v}
    if bad:
        raise RuntimeError(f"{path}: {KERNEL} resources {bad} != {want}")
    args = meta["args"]
    sizes = {"ptr": 8, "i64": 8, "i32": 4, "u32": 4, "f32": 4}
    got = [(a["offset"], a["size"]) for a in args]
    exp = [(o, sizes[k]) for o, (_, k) in zip(ARG_OFFSETS, ARGS)]
    exp_s = exp + [(o, sizes[k]) for o, (_, k) in zip(STRIDE_ARG_OFFSETS, STRIDE_ARGS)]
    if any(a["value_kind"] not in ("global_buffer", "by_value") for a in args):
        raise RuntimeError(f"{path}: {KERNEL} has hidden kernel arguments {[a['value_kind'] for a in args]}")
    if got == exp and meta["kernarg_size"] == KARG_BYTES:
        strided = False
    elif got == exp_s and meta["kernarg_size"] == KARG_BYTES_STRIDES:
        strided = True
    else:
        raise RuntimeError(f"{path}: {KERNEL} kernarg layout {got} ({meta['kernarg_size']} B) is neither KArgs "
                           f"{exp} ({KARG_BYTES} B) nor KArgs + strides ({KARG_BYTES_STRIDES} B)")  # fmt: skip
    mj = os.path.splitext(path)[0] + ".meta.json"
    if os.path.isfile(mj):  # cc.sh output: the build must have passed and match these bytes
        import hashlib

        with open(mj) as f:
            m = json.load(f)
        with open(path, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        if not m.get("ok", False) or m.get("hsaco_sha256") != sha:
            raise RuntimeError(f"{path}: stale or failed build ({mj}: ok={m.get('ok')}, sha256 mismatch "
                               f"{m.get('hsaco_sha256', '')[:12]} vs {sha[:12]})")  # fmt: skip
        if "flags" in m and _flag_strides(m["flags"]) != int(strided):
            raise RuntimeError(f"{path}: kernarg layout says HK_STRIDES={int(strided)}, but {mj} records the flags "
                               f"{[f for f in m['flags'] if 'HK_STRIDES' in str(f)]}")  # fmt: skip
    return dict(meta, strided=strided)


def _kernel():
    """(hiprt.Kernel, launch cfg, strided) of the resolved build (loaded at the first launch)."""
    global _KERNEL
    d = _hsaco_dir()
    path = os.path.join(d, f"{KERNEL}.hsaco")
    k = _KERNEL
    if k is not None and k[0] == path:
        return k[1], k[2], k[3]
    with _LOCK:
        if _KERNEL is None or _KERNEL[0] != path:
            if not os.path.isfile(path):
                raise RuntimeError(f"{KERNEL}: code object {path} not found (build it with hk/lab/cc.sh)")
            strided = _check_code_object(path)["strided"]
            cfg = launch_cfg(d)
            spec = {KERNEL: {"block": BLOCK, "args": list(ARGS) + (list(STRIDE_ARGS) if strided else [])}}
            _KERNEL = (path, hiprt.load_hsaco(path, spec)[KERNEL], cfg, strided)
        return _KERNEL[1], _KERNEL[2], _KERNEL[3]


def supports_strides():
    """Whether the code object the next launch loads (the resolved code-object directory) was built with
    -DHK_STRIDES=1, read from its own kernel metadata (``_check_code_object``: the kernarg layout is exactly the
    default one or the strided one, and agrees with the build's recorded flags). Never guessed: a missing code object
    or a layout that matches neither raises RuntimeError. Also the module attribute ``SUPPORTS_STRIDES``. Ask it after
    the caller has pointed this module at its code objects (impl does that in ``hk_unavailable_reason``)."""
    path = os.path.join(_hsaco_dir(), f"{KERNEL}.hsaco")
    k = _KERNEL
    if k is not None and k[0] == path:
        return k[3]
    if not os.path.isfile(path):
        raise RuntimeError(f"{KERNEL}: code object {path} not found: cannot tell whether it takes strides")
    st = os.stat(path)
    key = (st.st_mtime_ns, st.st_size)
    hit = _STRIDED.get(path)
    if hit is not None and hit[0] == key:
        return hit[1]
    strided = bool(_check_code_object(path)["strided"])
    _STRIDED[path] = (key, strided)
    return strided


def __getattr__(name):
    # SUPPORTS_STRIDES is computed on access (PEP 562): it always describes the code object this module would load
    if name == "SUPPORTS_STRIDES":
        return supports_strides()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _launch(name, fn, args):
    """Every kernel launch of this module (lab --seg wraps it)."""
    fn(*args)


def n_units(mode, b, nhq4, nqb):
    """Schedule units: zig-zag pairs (+ one single per head for odd nqb) or items (f1::n_units)."""
    return b * nhq4 * ((nqb + 1) // 2) if mode == "pairs" else b * nhq4 * nqb


def unsupported_reason(b, sq, skv, hq, hkv, causal, d=D):
    """Why k_attn_fwd_v1 cannot run this problem (None when it can). The conditions of
    hk/lab/bounds/fwd_v1_gate.h fwd1_admitted() (the bounds proof's gate), in the same order: v0's contract
    (copied from fwd_v0.py at its validated state), causal only, K / V byte extents < 2^31 (the kernel's compact
    op form keeps 31-bit element offsets and every fdiv numerator < 2^31)."""
    if d != D:
        return f"head dim must be {D}, got {d}"
    if min(b, sq, skv, hq, hkv) < 1:
        return f"empty problem: b {b} sq {sq} skv {skv} hq {hq} hkv {hkv}"
    if hq % hkv:
        return f"hq {hq} not a multiple of hkv {hkv}"
    g = hq // hkv
    if g not in GQA_RATIOS:
        return f"G = hq / hkv = {g} not in {GQA_RATIOS} (4 q heads of one kv head per item; kv head = shift by log2 G)"
    if sq % QB or skv % KV_ST:
        return f"sq {sq} and skv {skv} must be multiples of {QB} / {KV_ST}"
    if not causal:
        return "full attention (causal=False) is not admitted: the kernel path is CPU-proven but has no GPU check"
    if sq > skv:
        return f"causal needs sq <= skv (bottom-right: the first sq - skv queries see no key), got {sq} > {skv}"
    if 2 * b * sq * hq * D >= I32_LIMIT or 4 * b * hq * sq >= I32_LIMIT:
        return "O / LSE byte extents must be < 2^31 (int32 buffer offsets)"
    if b * (hq // NH) * (sq // QB) >= I32_LIMIT:
        return "grid too large"
    if 2 * b * skv * hkv * D >= I32_LIMIT:
        return "K / V byte extents must be < 2^31 (31-bit element offsets of the compact TDM op)"
    return None


def layout_of(t):
    """The layout of a 4-D [B, S, H, D] tensor: "bshd" (contiguous), "sbhd" ([S, B, H, D] contiguous storage viewed
    through permute(1, 0, 2, 3): strides (H D, B H D, D, 1)) or None (anything else). Strides of size-1 dims address
    nothing and are ignored ("bshd" is then exactly t.is_contiguous()); a tensor that is both (B == 1: the same
    memory) is "bshd"."""
    if t.dim() != 4:
        return None
    b, s, h, d = t.shape

    def match(want):
        return all(n == 1 or x == y for n, x, y in zip(t.shape, t.stride(), want))

    if match((s * h * d, h * d, d, 1)):
        return "bshd"
    if match((h * d, b * h * d, d, 1)):
        return "sbhd"
    return None


def layout_strides(shape, layout):
    """(batch, seq) element strides of a [B, S, H, D] tensor in ``layout`` (the kernel's sXb, sXs)."""
    b, s, h, d = shape
    if layout == "bshd":
        return s * h * d, h * d
    if layout == "sbhd":
        return h * d, b * h * d
    raise ValueError(f"layout {layout!r} not in {LAYOUTS}")


def _check_args(q, k, v, softmax_scale, causal, strided=False):
    """fwd_v0's contract checks (types, dtype, layout, alignment, shapes, scale), then v1's gate. ``strided``: the
    code object takes strides, so each of q, k, v may be "bshd" or "sbhd" (``layout_of``); else contiguous only."""
    named = {"q": q, "k": k, "v": v}
    for n, t in named.items():
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{n} must be a torch.Tensor")
        if t.dtype != torch.bfloat16:
            raise ValueError(f"{n} must be bfloat16, got {t.dtype}")
        if t.dim() != 4:
            raise ValueError(f"{n} must be 4-D [B, S, H, D], got {tuple(t.shape)}")
        if strided:
            if layout_of(t) is None:
                raise ValueError(f"{n} must be contiguous BSHD or an SBHD-storage view ([S, B, H, D] contiguous, "
                                 f"permute(1, 0, 2, 3)); shape {tuple(t.shape)} strides {t.stride()}")  # fmt: skip
        elif not t.is_contiguous():
            raise ValueError(f"{n} must be contiguous (BSHD), strides {t.stride()}")
        if t.device != q.device:
            raise ValueError(f"all tensors on one device: q on {q.device}, {n} on {t.device}")
        if t.device.type != "meta" and t.data_ptr() % 16:
            raise ValueError(f"{n}.data_ptr() must be 16-B aligned")
    b, sq, hq, d = q.shape
    bk, skv, hkv, dk = k.shape
    if tuple(v.shape) != tuple(k.shape) or bk != b or dk != d:
        raise ValueError(f"shapes q {tuple(q.shape)} k {tuple(k.shape)} v {tuple(v.shape)} do not match")
    if not isinstance(causal, (bool, int)) or int(causal) not in (0, 1):
        raise ValueError(f"causal must be a bool, got {causal!r}")
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)
    scale = float(softmax_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"softmax_scale must be finite and > 0, got {softmax_scale!r}")
    reason = unsupported_reason(b, sq, skv, hq, hkv, int(causal), d)
    if reason is not None:
        raise NotImplementedError(f"{KERNEL}: {reason}")
    return b, sq, skv, hq, hkv, scale, int(causal)


def choose_schedule(b, nhq4, nqb, num_cus, mode="auto", w=None):
    """(mode, W) of the launch: "auto" = pairs when #units >= #CUs, else items; W = min(#CUs, #units)
    (pairs) or #items (items); a given W (>= 1, e.g. from a variant's launch.json) is clamped to #units."""
    if mode not in MODES:
        raise ValueError(f"mode {mode!r} not in {MODES}")
    if not isinstance(num_cus, int) or num_cus < 1:
        raise ValueError(f"num_cus must be a positive int, got {num_cus!r}")
    if mode == "auto":
        mode = "pairs" if n_units("pairs", b, nhq4, nqb) >= num_cus else "items"
    units = n_units(mode, b, nhq4, nqb)
    if w is None:
        w = min(num_cus, units) if mode == "pairs" else units
    if not isinstance(w, int) or isinstance(w, bool) or w < 1:
        raise ValueError(f"W must be a positive int, got {w!r}")
    w = min(w, units)
    return mode, w


def plan(q, k, v, softmax_scale=None, causal=True, mode="auto", num_cus=DEFAULT_CUS, w=None, strided=False):
    """The launch flash_attn_fwd makes for these inputs (works on meta tensors; touches no GPU):
    {"grid", "block", "values" (kernarg values in ARGS order, + STRIDE_ARGS when ``strided``), "outputs" (o, lse),
    "c2", "derived"}. ``strided``: for a -DHK_STRIDES=1 code object (q, k, v in either layout, o in q's)."""
    b, sq, skv, hq, hkv, scale, c = _check_args(q, k, v, softmax_scale, causal, strided)
    g = hq // hkv
    assert g in GQA_RATIOS and g & (g - 1) == 0 and c == 1, (g, c)  # the kernel's hkv_of is a shift by log2 G
    nqb, nhq4, nkv_st = sq // QB, hq // NH, skv // KV_ST
    cst = (skv - sq) // KV_ST if c else 0
    mode, w = choose_schedule(b, nhq4, nqb, num_cus, mode, w)
    if not strided:
        o = torch.empty_like(q)
    elif layout_of(q) == "sbhd":  # o in q's layout: [S, B, H, D] storage, seen as [B, S, H, D]
        o = torch.empty((sq, b, hq, D), dtype=q.dtype, device=q.device).permute(1, 0, 2, 3)
    else:
        o = torch.empty((b, sq, hq, D), dtype=q.dtype, device=q.device)
    lse = torch.empty((b, hq, sq), dtype=torch.float32, device=q.device)
    o_bytes, lse_bytes = 2 * o.numel(), 4 * lse.numel()
    assert o_bytes == 2 * b * sq * hq * D and lse_bytes == 4 * b * hq * sq
    c2 = scale * math.log2(math.e)  # double; packed as fp32 (RNE) into the kernarg
    c2inv = 1.0 / c2  # the lean rescale test's threshold scale (fp32 RNE; any rounding is safe, see impl note)
    mw, shw = fdiv_of(w)
    mnp, shnp = fdiv_of(max(nqb // 2, 1))
    mh4, shh4 = fdiv_of(nhq4)
    mnh, shnh = fdiv_of(b * nhq4)
    values = (q, k, v, o, lse, o_bytes, lse_bytes, b, sq, skv, hq, hkv, g, nqb, nhq4, nkv_st, cst, c, c2, c2inv,
              w, MODE_ID[mode], mw, shw, mnp, shnp, mh4, shh4, mnh, shnh)  # fmt: skip
    assert len(values) == len(ARGS)
    derived = {"G": g, "nqb": nqb, "nhq4": nhq4, "nkv_st": nkv_st, "cst": cst, "mode": mode, "W": w,
               "units": n_units(mode, b, nhq4, nqb), "items": b * nhq4 * nqb}  # fmt: skip
    if strided:
        lays, strides = {}, ()
        for n, t in (("q", q), ("k", k), ("v", v), ("o", o)):
            lay = layout_of(t)
            assert lay in LAYOUTS, (n, t.shape, t.stride())
            sb, ss = layout_strides(tuple(t.shape), lay)
            bt, st, ht, _ = t.shape
            # the kernel addresses element (b, s, h, d) at b sb + s ss + h D + d: both layouts map onto exactly
            # [0, numel) (the true extent its resource / TDM ops assume); the seq stride is the TDM row stride
            # (48-bit field) and every stride fits the 31-bit compact offsets (the extents are < 2^31 B by the gate)
            assert (bt - 1) * sb + (st - 1) * ss + (ht - 1) * D + D - 1 == t.numel() - 1, (n, sb, ss)
            assert D <= sb < I32_LIMIT and D <= ss < I32_LIMIT and ss < TDM_STRIDE_LIMIT, (n, sb, ss)
            lays[n] = lay
            strides += (sb, ss)
        assert lays["o"] == lays["q"], lays
        values += strides
        assert len(values) == len(ARGS) + len(STRIDE_ARGS)
        derived["layouts"] = lays
    return {"grid": (w, 1, 1), "block": (BLOCK, 1, 1), "values": values, "outputs": (o, lse), "c2": c2,
            "derived": derived}


def _device_cus(device):
    if device.type != "cuda":
        raise ValueError(f"{KERNEL} runs on a CUDA/HIP device, got {device}")
    idx = device.index if device.index is not None else torch.cuda.current_device()
    n = _DEVICES.get(idx)
    if n is None:
        props = torch.cuda.get_device_properties(idx)
        if "gfx1250" not in props.gcnArchName:
            raise RuntimeError(f"{KERNEL} is gfx1250 code; device {idx} is {props.gcnArchName}")
        n = int(props.multi_processor_count)
        _DEVICES[idx] = n
    return n


def flash_attn_fwd(q, k, v, softmax_scale=None, causal=True):
    """(o [B, Sq, Hq, 128] bf16, lse [B, Hq, Sq] fp32 natural log) on torch's current stream. With a strided code
    object (``SUPPORTS_STRIDES``) q, k, v may each be "bshd" or "sbhd" (``layout_of``) and o comes in q's layout."""
    cus = _device_cus(q.device)
    kern, cfg, strided = _kernel()
    p = plan(q, k, v, softmax_scale, causal, cfg.get("mode", "auto"), cus, cfg.get("W"), strided=strided)
    stream = torch.cuda.current_stream(q.device)
    _launch("fwd", kern.call, (p["grid"], p["block"], p["values"], stream))
    return p["outputs"]
