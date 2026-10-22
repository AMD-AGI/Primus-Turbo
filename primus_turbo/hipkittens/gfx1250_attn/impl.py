###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Production entry points of the gfx1250 HipKittens attention kernels (head dim 128, GQA, bottom-right
causal, bf16, contiguous BSHD; the Llama-3.1-8B attention).

    hk_unsupported_reason(direction, q, k, v, causal, deterministic=False, softmax_scale=None) -> None | str
    hk_attn_fwd(q, k, v, softmax_scale, causal)                    -> (o [B,Sq,Hq,128] bf16, lse [B,Hq,Sq] fp32)
    hk_attn_bwd(do, q, k, v, o, lse, softmax_scale, causal, deterministic=False) -> (dq, dk, dv)

lse is the natural-log logsumexp [B, Hq, Sq], the FlyDSL gfx1250 kernels' convention, so either family's
forward feeds either family's backward. ``hk_attn_*`` raise NotImplementedError where
``hk_unsupported_reason`` gives a reason; the caller (the HIPKITTENS backend) runs FlyDSL then.

Kernel modules of this package, each with the attention lab's contract (``hk/lab/lab.py``):
``flash_attn_fwd(q, k, v, softmax_scale, causal)`` / ``flash_attn_bwd(do, q, k, v, o, lse, softmax_scale,
causal)``, a module-level ``_launch``, ``HSACO_DIR`` / ``set_hsaco_dir(path)`` resolved at the first
launch, and optionally a gate asked on top of ``limits``: ``unsupported_reason(q_shape, k_shape, v_shape,
dtype, causal)`` or ``unsupported_reason(b, sq, skv, hq, hkv, causal[, d])`` -> None | str (``module_gate``
tells them apart by their parameter names). Selected once, at import:

    forward   PRIMUS_TURBO_HK_FWD = v0 (DEFAULT_FWD) | pp | flydsl      -> fwd_v0 | fwd_pp | none
    backward  PRIMUS_TURBO_HK_BWD = v0 (DEFAULT_BWD) | flydsl           -> bwd_v0 | none

"flydsl" selects no module: that direction is unsupported here and runs FlyDSL (mixed mode). The
backward with deterministic=True is unsupported too (dQ is summed with fp32 atomics).

Code objects. Production: ``bin/<kernel>.hsaco`` of this package, only for kernels that
``manifest.json`` lists for the module (``module`` field) with a sha256 pin matching the file
(``hsaco.unavailable_reason``, cached); the module is pointed at ``bin/``, and the files are hashed
again right before the module's first launch, which is when the module loads them (a mismatch then is
a NotImplementedError and the direction stays unavailable). The modules themselves load by path without
a pin (``hiprt.load_hsaco(path, specs)``), so this re-check is the enforcement; a module that passes
``hsaco.code_object``'s sha256 to ``load_hsaco`` would close the remaining gap. Lab: a search path given
by ``set_hsaco_dir(path)`` (or the ``HSACO_DIR`` attribute the lab sets on this module or its package)
or by env ``HK_HSACO_DIR`` (read at import): one directory, or several separated by ``os.pathsep``
(e.g. the cc.sh output directories of the forward and the backward). Each module uses the first
directory that holds every one of its code objects, unpinned (they must exist); a directory of the path
that does not exist is a reason. Nothing here raises at import: a missing or broken module, a missing,
unpinned or stale code object and an invalid setting are reasons.
"""

import importlib
import inspect
import math
import os
import sys
import threading
from numbers import Real
from pathlib import Path
from typing import Dict, List, Optional

import torch

from . import hsaco, limits

__all__ = [
    "code_object_files",
    "hk_attn_bwd",
    "hk_attn_fwd",
    "hk_unavailable_reason",
    "hk_unsupported_reason",
    "flash_attn_bwd",
    "flash_attn_fwd",
    "set_hsaco_dir",
]

FWD_ENV = "PRIMUS_TURBO_HK_FWD"
BWD_ENV = "PRIMUS_TURBO_HK_BWD"
HSACO_DIR_ENV = "HK_HSACO_DIR"

# The kernel module of each direction when its variable is unset or empty: the one place to change it.
DEFAULT_FWD = "v0"
DEFAULT_BWD = "v0"
# setting -> kernel module of this package; None: no HK kernels for that direction (FlyDSL runs it).
FWD_MODULES = {"pp": "fwd_pp", "v0": "fwd_v0", "flydsl": None}
BWD_MODULES = {"v0": "bwd_v0", "flydsl": None}
HEAD_DIM = limits.HEAD_DIM

# Read once at import, like the FlyDSL kernels' switches: a process uses one configuration.
FWD_SETTING = os.environ.get(FWD_ENV, "")
BWD_SETTING = os.environ.get(BWD_ENV, "")
ENV_HSACO_DIR = os.environ.get(HSACO_DIR_ENV, "") or None

# Lab override of the code-object search path (set_hsaco_dir); None: the package's pinned bin/.
HSACO_DIR = None
# Tensor start alignment the kernels address with (fwd_v0 / fwd_pp raise on less); a misaligned call is
# a reason (q / k / v) or a NotImplementedError of hk_attn_bwd (do / o / lse), so it runs on FlyDSL.
ALIGNMENT = 16

_DIRECTIONS = {"fwd": (FWD_ENV, DEFAULT_FWD, FWD_MODULES), "bwd": (BWD_ENV, DEFAULT_BWD, BWD_MODULES)}
_ENTRY = {"fwd": "flash_attn_fwd", "bwd": "flash_attn_bwd"}
_MODULES = {}  # kernel module name -> module, or why it cannot be used (str)
_REASONS = {}  # direction -> None or why it cannot run on HK at all (setting, module, code objects)
_LAB_DIRS = {}  # kernel module name -> the search-path directory holding its code objects (lab)
_VERIFIED = set()  # kernel modules whose pinned code objects were re-hashed before their first launch
_LOCK = threading.RLock()


# ---------------------------------------------------------------- selection


def _raw_setting(direction):
    return FWD_SETTING if direction == "fwd" else BWD_SETTING


def setting(direction) -> str:
    """The direction's setting, lower-cased: its variable, or the default when unset or empty."""
    return _raw_setting(direction).strip().lower() or _DIRECTIONS[direction][1]


def module_name(direction) -> Optional[str]:
    """The kernel module a direction runs on, or None (flydsl, or an invalid setting)."""
    return _DIRECTIONS[direction][2].get(setting(direction))


def hsaco_dir() -> Optional[str]:
    """The lab code-object search path as given (one directory or several joined by ``os.pathsep``),
    or None (production: the pinned ``bin/``)."""
    if HSACO_DIR:
        return str(HSACO_DIR)
    lab = getattr(sys.modules.get(__package__), "HSACO_DIR", None)
    if lab:
        return str(lab)
    return ENV_HSACO_DIR


def hsaco_dirs() -> List[str]:
    """The lab search path as a list of directories; empty in production."""
    lab = hsaco_dir()
    return [d for d in lab.split(os.pathsep) if d] if lab else []


def set_hsaco_dir(path) -> None:
    """Load code objects from ``path`` (unpinned, lab: a directory, several joined by ``os.pathsep``, or
    a list of them) instead of the pinned ``bin/``; None restores production. Re-points the kernel
    modules already loaded and forgets every cached verdict."""
    global HSACO_DIR
    if isinstance(path, (list, tuple)):
        path = os.pathsep.join(str(p) for p in path)
    with _LOCK:
        dirs = [d for d in str(path).split(os.pathsep) if d] if path else []
        HSACO_DIR = os.pathsep.join(str(Path(d).resolve()) for d in dirs) or None
        _REASONS.clear()
        _LAB_DIRS.clear()
        _VERIFIED.clear()
        for modname, mod in _MODULES.items():
            if not isinstance(mod, str):
                _point_module(modname, mod)


def _reset() -> None:
    """Forget loaded kernel modules and verdicts (tests)."""
    with _LOCK:
        _MODULES.clear()
        _REASONS.clear()
        _GATES.clear()
        _LAB_DIRS.clear()
        _VERIFIED.clear()


def _kernel_names(modname, mod) -> List[str]:
    """The code objects a module launches: the manifest's entries for it, else its ``KERNELS``."""
    return hsaco.kernel_names(modname) or list(getattr(mod, "KERNELS", ()))


def _find_lab_dir(names, dirs) -> Optional[str]:
    """The first directory of ``dirs`` holding every code object in ``names``, or None."""
    for d in dirs:
        if all(hsaco.binary_path(n, Path(d)).is_file() for n in names):
            return d
    return None


def _module_dir(modname, mod) -> str:
    """Where the module loads from: its lab directory (the first one of the search path holding its
    code objects, else the first one), or ``bin/``."""
    dirs = hsaco_dirs()
    if not dirs:
        return str(hsaco.BIN_DIR)
    if modname not in _LAB_DIRS:
        found = _find_lab_dir(_kernel_names(modname, mod), dirs)
        if found is None:
            return dirs[0]
        _LAB_DIRS[modname] = found
    return _LAB_DIRS[modname]


def _point_module(modname, mod) -> None:
    d = _module_dir(modname, mod)
    setter = getattr(mod, "set_hsaco_dir", None)
    if callable(setter):
        setter(d)
    else:
        mod.HSACO_DIR = d


def code_object_files(direction) -> Dict[str, Path]:
    """{code object: file} the direction's module loads (lab directory or ``bin/``), for logs and
    validation records; empty when the direction has no module or it cannot be imported."""
    modname = module_name(direction)
    mod = _load_module(modname) if modname else None
    if mod is None or isinstance(mod, str):
        return {}
    names = _kernel_names(modname, mod)
    if not hsaco_dirs():
        return {n: hsaco.binary_path(n) for n in names}
    d = _find_lab_dir(names, hsaco_dirs()) or hsaco_dirs()[0]
    return {n: hsaco.binary_path(n, Path(d)) for n in names}


def _load_module(modname):
    """The kernel module, or why it cannot be imported (str); imported once."""
    with _LOCK:
        mod = _MODULES.get(modname)
        if mod is None:
            full = f"{__package__}.{modname}"
            try:
                mod = importlib.import_module(full)
            except ModuleNotFoundError as exc:
                if exc.name == full:
                    mod = f"kernel module {modname}.py is not in {Path(__file__).parent}"
                else:
                    mod = f"kernel module {modname} failed to import: {type(exc).__name__}: {exc}"
            except Exception as exc:  # noqa: BLE001 -- a broken module is a reason, never an import error
                mod = f"kernel module {modname} failed to import: {type(exc).__name__}: {exc}"
            _MODULES[modname] = mod
        return mod


def _code_objects_reason(modname, mod) -> Optional[str]:
    dirs = hsaco_dirs()
    if dirs:
        absent = [d for d in dirs if not Path(d).is_dir()]
        if absent:
            return f"code-object directory {', '.join(absent)} ({HSACO_DIR_ENV} / HSACO_DIR) does not exist"
        names = _kernel_names(modname, mod)
        if not names:
            return f"{hsaco.MANIFEST_PATH.name} lists no kernels for {modname} (and it has no KERNELS)"
        if _find_lab_dir(names, dirs) is None:
            return f"{', '.join(names)} not in {hsaco_dir()} (one directory must hold all of them)"
        return None
    names = hsaco.kernel_names(modname)
    if not names:
        return f"{hsaco.MANIFEST_PATH.name} lists no kernels for {modname}"
    return hsaco.unavailable_reason(hsaco.BIN_DIR, names)


def _unavailable(direction) -> Optional[str]:
    env, _, table = _DIRECTIONS[direction]
    choice = setting(direction)
    if choice not in table:
        return f"{env}={_raw_setting(direction)!r} is not one of {sorted(table)}"
    modname = table[choice]
    if modname is None:
        return f"{env}={choice}"
    mod = _load_module(modname)
    if isinstance(mod, str):
        return mod
    if not callable(getattr(mod, _ENTRY[direction], None)):
        return f"kernel module {modname} has no {_ENTRY[direction]}"
    reason = _code_objects_reason(modname, mod)
    if reason is None:
        _point_module(modname, mod)
    return reason


def hk_unavailable_reason(direction) -> Optional[str]:
    """Why ``direction`` ("fwd" / "bwd") cannot run on HK at all (setting, kernel module, code objects),
    or None. Resolved once per direction (``set_hsaco_dir`` starts over); never raises."""
    if direction not in _DIRECTIONS:
        return f"direction must be 'fwd' or 'bwd', got {direction!r}"
    try:
        return _REASONS[direction]
    except KeyError:
        pass
    with _LOCK:
        if direction not in _REASONS:
            try:
                _REASONS[direction] = _unavailable(direction)
            except Exception as exc:  # noqa: BLE001 -- e.g. an unreadable manifest
                _REASONS[direction] = f"HK {direction} setup failed: {type(exc).__name__}: {exc}"
        return _REASONS[direction]


def describe(direction) -> str:
    """One line for the log: the module and where its code objects come from."""
    modname = module_name(direction)
    if hsaco_dirs():
        source = f"{_LAB_DIRS.get(modname) or hsaco_dir()} (unpinned, lab)"
    else:
        source = f"{hsaco.BIN_DIR} (pinned by {hsaco.MANIFEST_PATH.name})"
    return f"{direction}: {modname} ({setting(direction)}), code objects {source}"


# ---------------------------------------------------------------- gate

_SHAPE_GATE = ("b", "sq", "skv", "hq", "hkv", "causal")
_GATES = {}  # kernel module name -> its gate as gate(q_shape, k_shape, v_shape, dtype, causal), or None


def module_gate(mod):
    """The module's ``unsupported_reason`` as ``gate(q_shape, k_shape, v_shape, dtype, causal)``, or None.

    Two forms are accepted, told apart by the parameter names: (q_shape, k_shape, v_shape, dtype, causal),
    FlyDSL's, and (b, sq, skv, hq, hkv, causal[, d]), integers."""
    gate = getattr(mod, "unsupported_reason", None)
    if not callable(gate):
        return None
    try:
        params = list(inspect.signature(gate).parameters)
    except (TypeError, ValueError):
        params = []
    if tuple(params[: len(_SHAPE_GATE)]) != _SHAPE_GATE:
        return gate
    with_d = len(params) > len(_SHAPE_GATE) and params[len(_SHAPE_GATE)] == "d"

    def by_dims(q_shape, k_shape, v_shape, dtype, causal):
        b, sq, hq, d = q_shape
        skv, hkv = k_shape[1], k_shape[2]
        return gate(b, sq, skv, hq, hkv, causal, d) if with_d else gate(b, sq, skv, hq, hkv, causal)

    return by_dims


def _scale(softmax_scale) -> float:
    return float(HEAD_DIM**-0.5 if softmax_scale is None else softmax_scale)


def shape_unsupported_reason(
    direction, q_shape, k_shape, v_shape, dtype, causal, deterministic=False, softmax_scale=None
) -> Optional[str]:
    """The per-call part of the gate (``hk_unavailable_reason`` must be None first): dtype, scale,
    ``limits`` for the selected module, then the module's own ``unsupported_reason`` if it has one."""
    modname = module_name(direction)
    if direction == "bwd" and deterministic:
        return "deterministic=True (the HK backward adds dQ with fp32 atomics)"
    if dtype != torch.bfloat16:
        return f"dtype must be bfloat16, got {dtype}"
    if softmax_scale is not None and (
        not isinstance(softmax_scale, Real)
        or isinstance(softmax_scale, bool)
        or not math.isfinite(softmax_scale)
        or softmax_scale <= 0
    ):
        return f"softmax_scale must be a finite positive number, got {softmax_scale!r}"
    reason = limits.shape_unsupported_reason(modname, q_shape, k_shape, v_shape, bool(causal))
    if reason is not None:
        return reason
    gate = _GATES.get(modname)
    if modname not in _GATES:
        mod = _MODULES.get(modname)
        if mod is not None and not isinstance(mod, str):  # not loaded yet: nothing cached, asked again
            gate = _GATES[modname] = module_gate(mod)
    if gate is not None:
        try:
            reason = gate(tuple(q_shape), tuple(k_shape), tuple(v_shape), dtype, bool(causal))
        except Exception as exc:  # noqa: BLE001 -- a gate that cannot answer declines
            return f"{modname}.unsupported_reason raised {type(exc).__name__}: {exc}"
        if reason is not None:
            return f"{modname}: {reason}"
    return None


def hk_unsupported_reason(
    direction, q, k, v, causal, deterministic=False, softmax_scale=None
) -> Optional[str]:
    """Why HK cannot run this direction ("fwd" / "bwd") of this BSHD call, or None. Never raises.

    Order: the setting, kernel module and code objects (once per process), then q / k / v on one
    device with one dtype, then ``shape_unsupported_reason``.
    """
    reason = hk_unavailable_reason(direction)
    if reason is not None:
        return reason
    if k is None or v is None:
        return "k and v are required"
    if k.device != q.device or v.device != q.device:
        return f"q / k / v must be on one device, got {q.device} / {k.device} / {v.device}"
    if k.dtype != q.dtype or v.dtype != q.dtype:
        return f"q / k / v must share a dtype, got {q.dtype} / {k.dtype} / {v.dtype}"
    misaligned = [name for name, t in (("q", q), ("k", k), ("v", v)) if _misaligned(t)]
    if misaligned:
        return f"{' / '.join(misaligned)} data_ptr not {ALIGNMENT}-byte aligned"
    return shape_unsupported_reason(
        direction, q.shape, k.shape, v.shape, q.dtype, causal, deterministic, softmax_scale
    )


def _misaligned(t) -> bool:
    """Whether a tensor with storage starts off a ``ALIGNMENT``-byte boundary (meta / fake: no)."""
    if t.device.type == "meta":
        return False
    try:
        return t.data_ptr() % ALIGNMENT != 0
    except Exception:  # noqa: BLE001 -- a tensor without storage (e.g. a FakeTensor) has no address
        return False


def _verify_before_first_launch(direction) -> None:
    """Production: hash the module's pinned code objects again just before its first launch (the module
    loads them on that launch), so bytes replaced after the cached check are never loaded.
    NotImplementedError (and the direction stays unavailable) on a mismatch; lab directories: nothing."""
    modname = module_name(direction)
    if modname in _VERIFIED or hsaco_dirs():
        return
    with _LOCK:
        if modname in _VERIFIED:
            return
        reason = hsaco.check_pins(hsaco.BIN_DIR, hsaco.kernel_names(modname))
        if reason is not None:
            _REASONS[direction] = reason
            raise NotImplementedError(f"hk_attn_{direction}: {reason}")
        _VERIFIED.add(modname)


def _check_tensors(**tensors):
    """ValueError unless every tensor is contiguous and on q's device (the kernels address BSHD bytes)."""
    q = tensors["q"]
    for name, t in tensors.items():
        if t.device != q.device:
            raise ValueError(
                f"all tensors must be on one device, got q on {q.device} and {name} on {t.device}"
            )
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous")


# ---------------------------------------------------------------- entry points


def hk_attn_fwd(q, k, v, softmax_scale=None, causal=True):
    """Forward on the selected HK module: (o [B, Sq, Hq, 128] bf16, lse [B, Hq, Sq] fp32 natural log).

    NotImplementedError when ``hk_unsupported_reason("fwd", ...)`` gives a reason, ValueError for
    tensors the kernels cannot address (non-contiguous, on another device)."""
    scale = _scale(softmax_scale)
    reason = hk_unsupported_reason("fwd", q, k, v, causal, softmax_scale=scale)
    if reason is not None:
        raise NotImplementedError(f"hk_attn_fwd: {reason}")
    _check_tensors(q=q, k=k, v=v)
    _verify_before_first_launch("fwd")
    return _MODULES[module_name("fwd")].flash_attn_fwd(q, k, v, scale, bool(causal))


def hk_attn_bwd(do, q, k, v, o, lse, softmax_scale=None, causal=True, deterministic=False):
    """Backward on the selected HK module: (dq, dk, dv) like q / k / v (dk / dv with Hkv heads).

    q / k / v / o / lse are what the forward saw and returned (either family's: same lse convention);
    do must have q's shape. NotImplementedError when ``hk_unsupported_reason("bwd", ...)`` gives a
    reason (deterministic=True always does) or do / o / lse are not 16-byte aligned, ValueError for
    mismatched tensors."""
    scale = _scale(softmax_scale)
    reason = hk_unsupported_reason("bwd", q, k, v, causal, deterministic, softmax_scale=scale)
    if reason is not None:
        raise NotImplementedError(f"hk_attn_bwd: {reason}")
    _check_tensors(q=q, k=k, v=v, do=do, o=o, lse=lse)
    b, sq, hq, _ = q.shape
    for name, t in (("do", do), ("o", o)):
        if t.shape != q.shape or t.dtype != q.dtype:
            raise ValueError(
                f"{name} must be {q.dtype} of q's shape {tuple(q.shape)}, got {t.dtype} {tuple(t.shape)}"
            )
    if lse.dtype != torch.float32 or tuple(lse.shape) != (b, hq, sq):
        raise ValueError(f"lse must be float32 of shape {(b, hq, sq)}, got {lse.dtype} {tuple(lse.shape)}")
    misaligned = [name for name, t in (("do", do), ("o", o), ("lse", lse)) if _misaligned(t)]
    if misaligned:
        raise NotImplementedError(
            f"hk_attn_bwd: {' / '.join(misaligned)} data_ptr not {ALIGNMENT}-byte aligned"
        )
    _verify_before_first_launch("bwd")
    return _MODULES[module_name("bwd")].flash_attn_bwd(do, q, k, v, o, lse, scale, bool(causal))


# The attention lab's arm contract (lab.py hk:<package>, default module impl): the production selection
# as one arm. --seg brackets each kernel module's own _launch; use hk:<package>#<module> for that.
def flash_attn_fwd(q, k, v, softmax_scale=None, causal=True):
    return hk_attn_fwd(q, k, v, softmax_scale, causal)


def flash_attn_bwd(do, q, k, v, o, lse, softmax_scale=None, causal=True):
    return hk_attn_bwd(do, q, k, v, o, lse, softmax_scale, causal, False)
