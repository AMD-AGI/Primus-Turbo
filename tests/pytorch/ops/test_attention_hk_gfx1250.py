###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""HipKittens attention on gfx1250 (MI455X), head dim 128: the HK package's settings, gate, code objects,
manifest and pin tool, the HIPKITTENS backend's routing with its per-direction FlyDSL fallback, and the
numerics of the HIP kernels.

The host-only tests use fake kernel modules, temporary manifests and code-object directories, patched arch
probes and meta / CPU tensors, so they run no GPU kernel (a device-less container with a plugin that makes
torch.cuda report one gfx1250 device runs them). Those that go through FlashAttnFunc need a flydsl that
satisfies the FlyDSL kernels' FLYDSL_REQUIREMENT (the backend's gate is FlyDSL's); the pin-tool test needs
amdclang++ with the gfx1250 target. The numeric tests need a gfx1250 card and the HK kernels of the
direction they test: pinned code objects in bin/, or lab directories in HK_HSACO_DIR (a search path; lab code
objects also need PRIMUS_TURBO_HK_FWD / PRIMUS_TURBO_HK_BWD set explicitly). A direction named in
HK_TEST_REQUIRE (fwd,bwd) fails instead of skipping when it is unavailable, and the tests through
flash_attn_func assert that HK, not the FlyDSL fallback, ran. Run them toy first (``-k toy``); the
production shape needs PRIMUS_TURBO_TEST_LARGE=1. With HK_VALIDATION_RECORD=<file>, a session in which
nothing failed appends one JSON line naming the code objects (sha256) and the checks that passed per
direction (scripts/hk/e2e_mg.sh and the pin tool's --record read it). Their fp32 reference runs on the CPU.
"""

import ast
import contextlib
import datetime
import functools
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import types
from pathlib import Path

import pytest
import torch

import primus_turbo.pytorch.ops.attention.flash_attn_interface as flash_attn_interface
from primus_turbo.pytorch.core.backend import (
    BackendType,
    GlobalBackendManager,
    PrecisionType,
)
from primus_turbo.pytorch.core.utils import is_gfx1250
from primus_turbo.pytorch.kernels.attention import attention_hk_gfx1250_impl as hk_adapter
from primus_turbo.pytorch.kernels.attention import attention_impl
from primus_turbo.pytorch.kernels.attention.attention_impl import (
    resolve_flash_attn_backend,
)
from primus_turbo.pytorch.ops import flash_attn_func
from tests.pytorch.test_utils import compute_snr, pinned_backend_takes

_PKG = "primus_turbo.hipkittens.gfx1250_attn"
# Taken before this module imports the package: importing primus_turbo must not.
_HK_MODULES_AT_IMPORT = sorted(m for m in sys.modules if m == _PKG or m.startswith(_PKG + "."))

from primus_turbo.hipkittens.gfx1250_attn import hsaco, impl, limits  # noqa: E402

# the forward module the package selects by default (impl.DEFAULT_FWD); the selection tests use it
_FWD = impl._DIRECTIONS["fwd"][2][impl.DEFAULT_FWD]

_ROOT = Path(__file__).resolve().parents[3]
D = 128
PROD = (4, 8192, 8192, 32, 8)


def _flydsl_reason():
    """The FlyDSL kernels' version gate, loaded by path (standard library only): importing the kernel
    package here would break test_attention_flydsl_gfx1250's check that nothing imports it early."""
    path = _ROOT / "primus_turbo" / "flydsl" / "attention" / "gfx1250" / "flydsl_version.py"
    spec = importlib.util.spec_from_file_location("_hk_test_flydsl_version", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.flydsl_unavailable_reason()


_FLYDSL_REASON = _flydsl_reason()
needs_flydsl = pytest.mark.skipif(_FLYDSL_REASON is not None, reason=str(_FLYDSL_REASON))
_ON_GFX1250 = torch.cuda.is_available() and is_gfx1250()


def _meta(b, sq, skv, hq, hkv, d=D, dtype=torch.bfloat16):
    q = torch.empty(b, sq, hq, d, dtype=dtype, device="meta")
    k = torch.empty(b, skv, hkv, d, dtype=dtype, device="meta")
    return q, k, torch.empty_like(k)


class _LogRecorder:
    """Stands in for the adapter's logger: records (level, message, once)."""

    def __init__(self):
        self.records = []

    def warning(self, msg, *args, once=False, **kwargs):
        self.records.append(("warning", msg, once))

    def info(self, msg, *args, once=False, **kwargs):
        self.records.append(("info", msg, once))

    def warnings(self):
        return [r for r in self.records if r[0] == "warning"]


# ---------------------------------------------------------------------------
# Fixtures: a clean impl, fake kernel modules, code-object directories, manifests
# ---------------------------------------------------------------------------


@pytest.fixture
def hk(monkeypatch):
    """impl with the default settings, no lab directory and nothing loaded yet."""
    impl._reset()
    monkeypatch.setattr(impl, "FWD_SETTING", "")
    monkeypatch.setattr(impl, "BWD_SETTING", "")
    monkeypatch.setattr(impl, "ENV_HSACO_DIR", None)
    monkeypatch.setattr(impl, "HSACO_DIR", None)
    yield impl
    impl._reset()


def _fake_module(monkeypatch, name, *, fwd=True, bwd=True, gate=None, kernels=None, setter=True):
    """A kernel module of the lab contract, put where impl imports ``<package>.<name>`` from."""
    mod = types.ModuleType(f"{_PKG}.{name}")
    mod.calls, mod.dirs = [], []
    if fwd:

        def flash_attn_fwd(q, k, v, softmax_scale, causal):
            mod.calls.append(("fwd", softmax_scale, causal))
            b, sq, hq, _ = q.shape
            return torch.zeros_like(q), torch.zeros(b, hq, sq, device=q.device)

        mod.flash_attn_fwd = flash_attn_fwd
    if bwd:

        def flash_attn_bwd(do, q, k, v, o, lse, softmax_scale, causal):
            mod.calls.append(("bwd", softmax_scale, causal))
            return torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)

        mod.flash_attn_bwd = flash_attn_bwd
    if setter:
        mod.set_hsaco_dir = mod.dirs.append
    if gate is not None:
        mod.unsupported_reason = gate
    if kernels is not None:
        mod.KERNELS = tuple(kernels)
    monkeypatch.setitem(sys.modules, f"{_PKG}.{name}", mod)
    return mod


def _lab_dir(tmp_path, names):
    d = tmp_path / "lab"
    d.mkdir(exist_ok=True)
    for n in names:
        (d / f"{n}.hsaco").write_bytes(b"fake code object " + n.encode())
    return d


def _available(monkeypatch, hk, tmp_path, fwd=_FWD, bwd="bwd_v0", **kwargs):
    """Fake fwd / bwd modules and a lab directory holding their manifest kernels: both directions run HK."""
    mods = {}
    for name in (fwd, bwd):
        if name is not None:
            mods[name] = _fake_module(monkeypatch, name, **kwargs)
    names = [n for m in mods for n in hsaco.kernel_names(m)]
    hk.set_hsaco_dir(_lab_dir(tmp_path, names))
    return mods


@pytest.fixture
def tmp_manifest(tmp_path):
    """hsaco reads ``tmp_path/manifest.json`` and ``tmp_path/bin`` instead of the package's; returns a writer
    write(kernels) -> manifest dict."""
    old = hsaco.MANIFEST_PATH, hsaco.BIN_DIR
    root = tmp_path / "pkg"
    (root / "bin").mkdir(parents=True)
    hsaco.MANIFEST_PATH, hsaco.BIN_DIR = root / "manifest.json", root / "bin"
    hsaco.reload_manifest()

    def write(kernels, toolchain="test toolchain"):
        m = {"schema": 2, "flags": [], "toolchain": toolchain, "kernels": kernels}
        hsaco.MANIFEST_PATH.write_text(json.dumps(m))
        hsaco.reload_manifest()
        return m

    yield write
    hsaco.MANIFEST_PATH, hsaco.BIN_DIR = old
    hsaco.reload_manifest()


def _entry(module, sha=None):
    return {"module": module, "src": f"src/{module}.hip", "extra_flags": [], "hsaco_sha256": sha}


# ---------------------------------------------------------------------------
# Package: lazy imports, settings
# ---------------------------------------------------------------------------


def test_importing_primus_turbo_does_not_import_the_hk_package():
    """The adapter imports the HK package on its first call, never with ``import primus_turbo``."""
    assert _HK_MODULES_AT_IMPORT == []


def test_importing_the_package_imports_none_of_its_modules():
    code = (
        f"import sys, {_PKG} as p\n"
        "print(sorted(m for m in sys.modules if m.startswith(p.__name__ + '.')))\n"
        "print(hasattr(p, 'HSACO_DIR'))\n"
    )
    run = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=_ROOT, env=_clean_env()
    )
    assert run.stdout.split() == ["[]", "False"], run.stdout + run.stderr


@pytest.mark.parametrize("module", ["hsaco", "limits", "codeobj"])
def test_gate_modules_import_only_the_standard_library(module):
    """setup.py and the pin tool load hsaco / codeobj by path, and limits is the gate's: no torch."""
    path = Path(hsaco.__file__).with_name(f"{module}.py")
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('m', {str(path)!r})\n"
        "spec.loader.exec_module(importlib.util.module_from_spec(spec))\n"
        "print(sorted(m for m in ('torch', 'flydsl', 'primus_turbo') if m in sys.modules))\n"
    )
    run = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True)
    assert run.stdout.strip() == "[]", run.stdout + run.stderr


def _clean_env(**extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PRIMUS_TURBO_HK_", "HK_HSACO"))}
    env["PYTHONPATH"] = os.pathsep.join([str(_ROOT), env.get("PYTHONPATH", "")])
    env.update(extra)
    return env


def test_settings_are_read_once_at_import():
    code = (
        f"import os\nfrom {_PKG} import impl\n"
        "os.environ['PRIMUS_TURBO_HK_FWD'] = 'pp'\n"
        "print(impl.FWD_SETTING, impl.BWD_SETTING, impl.ENV_HSACO_DIR, impl.module_name('fwd'), impl.module_name('bwd'))\n"
    )
    env = _clean_env(PRIMUS_TURBO_HK_FWD="v0", PRIMUS_TURBO_HK_BWD="flydsl", HK_HSACO_DIR="/lab/dir")
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env)
    assert run.stdout.split() == ["v0", "flydsl", "/lab/dir", "fwd_v0", "None"], run.stdout + run.stderr


def test_default_selection(hk):
    assert hk.DEFAULT_FWD == "v0" and hk.DEFAULT_BWD == "v0"
    assert (hk.setting("fwd"), hk.module_name("fwd")) == ("v0", "fwd_v0")
    assert (hk.setting("bwd"), hk.module_name("bwd")) == ("v0", "bwd_v0")


@pytest.mark.parametrize(
    ("direction", "value", "module"),
    [
        ("fwd", "v0", "fwd_v0"),
        ("fwd", " PP ", "fwd_pp"),
        ("fwd", "flydsl", None),
        ("fwd", "  ", "fwd_v0"),
        ("bwd", "V0", "bwd_v0"),
        ("bwd", "FlyDSL", None),
        ("fwd", "v1", None),
        ("bwd", "pp", None),
    ],
)
def test_setting_parsing(monkeypatch, hk, direction, value, module):
    monkeypatch.setattr(hk, "FWD_SETTING" if direction == "fwd" else "BWD_SETTING", value)
    assert hk.module_name(direction) == module


@pytest.mark.parametrize(
    ("direction", "value", "expected"),
    [
        ("fwd", "flydsl", "PRIMUS_TURBO_HK_FWD=flydsl"),
        ("bwd", "FLYDSL", "PRIMUS_TURBO_HK_BWD=flydsl"),
        ("fwd", "v1", "PRIMUS_TURBO_HK_FWD='v1' is not one of"),
        ("bwd", "pp", "PRIMUS_TURBO_HK_BWD='pp' is not one of"),
    ],
)
def test_flydsl_and_invalid_settings_are_reasons(monkeypatch, hk, direction, value, expected):
    monkeypatch.setattr(hk, "FWD_SETTING" if direction == "fwd" else "BWD_SETTING", value)
    reason = hk.hk_unavailable_reason(direction)
    assert reason is not None and expected in reason, reason


def test_unknown_direction_is_a_reason(hk):
    assert "direction must be" in hk.hk_unavailable_reason("both")


# ---------------------------------------------------------------------------
# Kernel modules and code objects
# ---------------------------------------------------------------------------


def test_missing_kernel_module_is_a_reason(monkeypatch, hk):
    monkeypatch.setitem(sys.modules, f"{_PKG}.{_FWD}", None)  # import raises ModuleNotFoundError
    q, k, v = _meta(*PROD)
    reason = hk.hk_unsupported_reason("fwd", q, k, v, True)
    assert reason is not None and _FWD in reason, reason


class _BrokenFinder:
    """A meta-path finder whose one module fails to import (as a module raising at import does)."""

    def __init__(self, name):
        self.name, self.tries = name, 0

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.name:
            return None
        self.tries += 1
        raise AttributeError("module 'hiprt' has no attribute 'renamed'")


def test_module_that_fails_to_import_is_a_reason_resolved_once(monkeypatch, hk):
    finder = _BrokenFinder(f"{_PKG}.bwd_v0")
    monkeypatch.delitem(sys.modules, finder.name, raising=False)
    monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])
    for _ in range(3):
        reason = hk.hk_unavailable_reason("bwd")
        assert "AttributeError" in reason and "renamed" in reason, reason
    assert finder.tries == 1


def test_module_without_its_entry_point_is_a_reason(monkeypatch, hk, tmp_path):
    _fake_module(monkeypatch, _FWD, fwd=False)
    hk.set_hsaco_dir(_lab_dir(tmp_path, hsaco.kernel_names(_FWD)))
    assert "has no flash_attn_fwd" in hk.hk_unavailable_reason("fwd")


def test_lab_directory_needs_the_modules_kernels(monkeypatch, hk, tmp_path):
    mod = _fake_module(monkeypatch, "bwd_v0")
    empty = tmp_path / "empty"
    empty.mkdir()
    hk.set_hsaco_dir(empty)
    reason = hk.hk_unavailable_reason("bwd")
    assert reason is not None and "not in" in reason and str(empty) in reason, reason
    hk.set_hsaco_dir(_lab_dir(tmp_path, hsaco.kernel_names("bwd_v0")))
    assert hk.hk_unavailable_reason("bwd") is None
    assert mod.dirs[-1] == str(tmp_path / "lab")
    hk.set_hsaco_dir(tmp_path / "nowhere")
    assert "does not exist" in hk.hk_unavailable_reason("bwd")


def test_lab_directory_from_the_package_attribute_and_the_environment(monkeypatch, hk, tmp_path):
    """The lab sets HSACO_DIR on the package before importing impl; HK_HSACO_DIR is read at import."""
    mod = _fake_module(monkeypatch, _FWD, setter=False)
    lab = _lab_dir(tmp_path, hsaco.kernel_names(_FWD))
    monkeypatch.setattr(sys.modules[_PKG], "HSACO_DIR", str(lab), raising=False)
    assert hk.hsaco_dir() == str(lab)
    assert hk.hk_unavailable_reason("fwd") is None
    assert mod.HSACO_DIR == str(lab)  # no set_hsaco_dir: the attribute
    monkeypatch.delattr(sys.modules[_PKG], "HSACO_DIR")
    monkeypatch.setattr(hk, "ENV_HSACO_DIR", "/from/env")
    assert hk.hsaco_dir() == "/from/env"


def test_lab_directory_falls_back_to_the_modules_kernel_list(monkeypatch, hk, tmp_path, tmp_manifest):
    tmp_manifest({})
    _fake_module(monkeypatch, "fwd_v0", kernels=["k_new"])
    monkeypatch.setattr(hk, "FWD_SETTING", "v0")
    hk.set_hsaco_dir(_lab_dir(tmp_path, []))
    assert "k_new not in" in hk.hk_unavailable_reason("fwd")
    hk.set_hsaco_dir(_lab_dir(tmp_path, ["k_new"]))
    assert hk.hk_unavailable_reason("fwd") is None


def test_set_hsaco_dir_repoints_loaded_modules(monkeypatch, hk, tmp_path):
    mods = _available(monkeypatch, hk, tmp_path)
    assert hk.hk_unavailable_reason("fwd") is None and hk.hk_unavailable_reason("bwd") is None
    other = tmp_path / "other"
    shutil.copytree(tmp_path / "lab", other)
    hk.set_hsaco_dir(other)
    assert all(m.dirs[-1] == str(other) for m in mods.values())
    hk.set_hsaco_dir(None)  # production: the package's bin/
    assert all(m.dirs[-1] == str(hsaco.BIN_DIR) for m in mods.values())


def test_lab_search_path_gives_each_module_its_own_directory(monkeypatch, hk, tmp_path):
    """HK_HSACO_DIR may join several cc.sh output directories (one per kernel): each module loads from the
    first one holding all of its code objects; a missing directory of the path is a reason."""
    mods = {n: _fake_module(monkeypatch, n) for n in (_FWD, "bwd_v0")}
    dirs = {}
    for n in mods:
        dirs[n] = tmp_path / n
        dirs[n].mkdir()
        for k in hsaco.kernel_names(n):
            (dirs[n] / f"{k}.hsaco").write_bytes(b"x")
    hk.set_hsaco_dir(os.pathsep.join(str(dirs[n]) for n in (_FWD, "bwd_v0")))
    assert hk.hsaco_dirs() == [str(dirs[_FWD]), str(dirs["bwd_v0"])]
    assert hk.hk_unavailable_reason("fwd") is None and hk.hk_unavailable_reason("bwd") is None
    assert mods[_FWD].dirs[-1] == str(dirs[_FWD]) and mods["bwd_v0"].dirs[-1] == str(dirs["bwd_v0"])
    assert str(dirs["bwd_v0"]) in hk.describe("bwd") and "unpinned, lab" in hk.describe("bwd")
    files = hk.code_object_files("bwd")
    assert files == {k: dirs["bwd_v0"] / f"{k}.hsaco" for k in hsaco.kernel_names("bwd_v0")}
    hk.set_hsaco_dir([dirs[_FWD]])  # a list works too; bwd's code objects are not on this path
    assert "not in" in hk.hk_unavailable_reason("bwd") and hk.hk_unavailable_reason("fwd") is None
    hk.set_hsaco_dir(os.pathsep.join([str(dirs[_FWD]), str(tmp_path / "typo")]))
    assert "does not exist" in hk.hk_unavailable_reason("fwd")


def test_pinned_code_objects_are_hashed_again_before_the_first_launch(monkeypatch, hk, tmp_manifest):
    """Production: bytes replaced in bin/ after the (cached) check are refused before the module's first
    launch, and the direction stays unavailable."""
    mod = _fake_module(monkeypatch, "bwd_v0")
    data = b"validated code object"
    tmp_manifest({"k_attn_bwd_v0": _entry("bwd_v0", hashlib.sha256(data).hexdigest())})
    (hsaco.BIN_DIR / "k_attn_bwd_v0.hsaco").write_bytes(data)
    q, k, v = (torch.zeros(s, dtype=torch.bfloat16) for s in ((1, 128, 4, D), (1, 128, 1, D), (1, 128, 1, D)))
    lse = torch.zeros(1, 4, 128)
    assert hk.hk_unavailable_reason("bwd") is None
    (hsaco.BIN_DIR / "k_attn_bwd_v0.hsaco").write_bytes(b"swapped")
    with pytest.raises(NotImplementedError, match="sha256"):
        hk.hk_attn_bwd(q, q, k, v, q, lse, None, True)
    assert mod.calls == [] and "sha256" in hk.hk_unavailable_reason("bwd")
    (hsaco.BIN_DIR / "k_attn_bwd_v0.hsaco").write_bytes(data)
    hk._reset()
    hk.hk_attn_bwd(q, q, k, v, q, lse, None, True)
    (hsaco.BIN_DIR / "k_attn_bwd_v0.hsaco").write_bytes(b"swapped")  # loaded already: not re-read
    hk.hk_attn_bwd(q, q, k, v, q, lse, None, True)
    assert len(mod.calls) == 2


@pytest.mark.parametrize("state", ["unlisted", "unpinned", "unbuilt", "stale", "pinned"])
def test_production_code_objects_must_match_their_pins(monkeypatch, hk, tmp_manifest, state):
    """Without a lab directory, a module runs only when every manifest kernel of it is in bin/ with the
    pinned sha256; the module is then pointed at bin/."""
    mod = _fake_module(monkeypatch, "bwd_v0")
    data = b"validated code object"
    sha = hashlib.sha256(data).hexdigest()
    kernels = {
        "unlisted": {"k_other": _entry("fwd_v0", sha)},
        "unpinned": {"k_attn_bwd_v0": _entry("bwd_v0")},
    }.get(state, {"k_attn_bwd_v0": _entry("bwd_v0", sha)})
    tmp_manifest(kernels)
    if state in ("stale", "pinned"):
        (hsaco.BIN_DIR / "k_attn_bwd_v0.hsaco").write_bytes(data if state == "pinned" else b"rebuilt")
    reason = hk.hk_unavailable_reason("bwd")
    expected = {
        "unlisted": "lists no kernels for bwd_v0",
        "unpinned": "is not pinned",
        "unbuilt": "is not built",
        "stale": "sha256",
    }
    if state == "pinned":
        assert reason is None and mod.dirs == [str(hsaco.BIN_DIR)]
    else:
        assert reason is not None and expected[state] in reason, reason
        assert mod.dirs == []


def test_missing_manifest_is_a_reason(monkeypatch, hk, tmp_path):
    _fake_module(monkeypatch, _FWD)
    old = hsaco.MANIFEST_PATH
    hsaco.MANIFEST_PATH = tmp_path / "missing.json"
    hsaco.reload_manifest()
    try:
        assert "lists no kernels" in hk.hk_unavailable_reason("fwd")
        assert "is missing" in hsaco.unavailable_reason(tmp_path)
    finally:
        hsaco.MANIFEST_PATH = old
        hsaco.reload_manifest()


# ---------------------------------------------------------------------------
# Shape gate
# ---------------------------------------------------------------------------

_ACCEPT = {
    "llama3.1-8b": PROD,
    "toy": (1, 128, 128, 4, 1),
    "sq_lt_skv": (1, 256, 512, 8, 2),
    "odd_batch_heads": (3, 192, 384, 12, 3),
    "sq64": (1, 64, 128, 4, 1),
}


@pytest.mark.parametrize("direction", ["fwd", "bwd"])
@pytest.mark.parametrize("shape", list(_ACCEPT.values()), ids=list(_ACCEPT))
def test_gate_accepts(monkeypatch, hk, tmp_path, direction, shape):
    _available(monkeypatch, hk, tmp_path)
    q, k, v = _meta(*shape)
    assert hk.hk_unsupported_reason(direction, q, k, v, True) is None


@pytest.mark.parametrize(
    ("direction", "shape", "kwargs", "expected"),
    [
        ("fwd", (1, 256, 256, 16, 2), {}, None),  # G 8
        ("fwd", (1, 128, 256, 32, 2), {}, None),  # G 16
        ("fwd", (1, 128, 192, 4, 1), {}, None),  # Skv % 64 suffices forward
        ("bwd", (1, 256, 256, 16, 2), {}, "must be one of (4,)"),
        ("bwd", (1, 128, 192, 4, 1), {}, "seqlen_kv must be a multiple of 128"),
        ("fwd", (1, 256, 256, 8, 4), {}, "multiple of 4"),
        ("fwd", (1, 256, 256, 4, 4), {}, "multiple of 4"),
        ("fwd", (1, 96, 128, 4, 1), {}, "seqlen_q must be a multiple of 64"),
        ("fwd", (1, 128, 160, 4, 1), {}, "seqlen_kv must be a multiple of 64"),
        ("fwd", (1, 256, 128, 4, 1), {}, "seqlen_q <= seqlen_kv"),
        ("bwd", (1, 256, 128, 4, 1), {}, "seqlen_q <= seqlen_kv"),
        ("fwd", (1, 128, 128, 4, 1), {"causal": False}, "causal attention only"),
        ("bwd", (1, 128, 128, 4, 1), {"causal": False}, "causal attention only"),
        ("bwd", (1, 128, 128, 4, 1), {"deterministic": True}, "deterministic=True"),
        ("fwd", (1, 128, 128, 4, 1), {"deterministic": True}, None),  # the forward is deterministic
        ("fwd", (1, 128, 128, 4, 1, 64), {}, "head_dim must be 128"),
        ("fwd", (1, 128, 128, 4, 1, 128, torch.float16), {}, "bfloat16"),
        ("fwd", (0, 128, 128, 4, 1), {}, "no empty dimension"),
        ("bwd", (1, 0, 128, 4, 1), {}, "no empty dimension"),
        ("fwd", (1, 128, 128, 4, 1), {"softmax_scale": 0.0}, "finite positive"),
        ("fwd", (1, 128, 128, 4, 1), {"softmax_scale": float("nan")}, "finite positive"),
        ("bwd", (1, 128, 128, 4, 1), {"softmax_scale": True}, "finite positive"),
        ("fwd", (1, 128, 128, 4, 1), {"softmax_scale": 0.05}, None),
        ("fwd", (64, 65536, 65536, 4, 1), {}, "int32 offsets"),
        ("bwd", (16, 8192, 8192, 32, 8), {}, "int32 offsets"),  # 2 GiB of fp32 dQ accumulator
    ],
)
def test_gate_rules(monkeypatch, hk, tmp_path, direction, shape, kwargs, expected):
    _available(monkeypatch, hk, tmp_path)
    q, k, v = _meta(*shape)
    causal = kwargs.pop("causal", True)
    reason = hk.hk_unsupported_reason(direction, q, k, v, causal, **kwargs)
    if expected is None:
        assert reason is None, reason
    else:
        assert reason is not None and expected in reason, reason


def test_gate_refuses_misaligned_tensors(monkeypatch, hk, tmp_path):
    """The kernels need 16-byte aligned starts (fwd_v0 / fwd_pp raise otherwise): q / k / v that are not
    are a reason, do / o / lse a NotImplementedError of hk_attn_bwd; either way FlyDSL runs the call."""
    mods = _available(monkeypatch, hk, tmp_path)
    shapes = ((1, 128, 4, D), (1, 128, 1, D), (1, 128, 1, D))
    q, k, v = (torch.zeros(s, dtype=torch.bfloat16) for s in shapes)
    shifted = torch.zeros(k.numel() + 1, dtype=torch.bfloat16)[1:].view(k.shape)  # 2 bytes off
    assert shifted.data_ptr() % 16 and shifted.is_contiguous()
    assert "k data_ptr not 16-byte aligned" in hk.hk_unsupported_reason("fwd", q, shifted, v, True)
    assert hk.hk_unsupported_reason("fwd", q, k, v, True) is None
    lse = torch.zeros(1, 4, 128)
    do = torch.zeros(q.numel() + 1, dtype=torch.bfloat16)[1:].view(q.shape)
    with pytest.raises(NotImplementedError, match="do data_ptr not 16-byte aligned"):
        hk.hk_attn_bwd(do, q, k, v, q, lse, None, True)
    assert mods["bwd_v0"].calls == []


def test_module_gate_is_cached_only_once_the_module_is_loaded(monkeypatch, hk, tmp_path):
    seen = []

    def gate(q_shape, k_shape, v_shape, dtype, causal):
        seen.append(q_shape)
        return "the kernel says no"

    q, k, v = _meta(1, 128, 128, 4, 1)
    assert hk.shape_unsupported_reason("bwd", q.shape, k.shape, v.shape, torch.bfloat16, True) is None
    assert "bwd_v0" not in hk._GATES  # not loaded yet: nothing cached
    _available(monkeypatch, hk, tmp_path, gate=gate)
    assert hk.hk_unsupported_reason("bwd", q, k, v, True) == "bwd_v0: the kernel says no" and seen


def test_gate_refuses_mismatched_tensors(monkeypatch, hk, tmp_path):
    _available(monkeypatch, hk, tmp_path)
    q, k, v = _meta(1, 128, 128, 4, 1)
    k_cpu = torch.empty(k.shape, dtype=k.dtype)
    assert "one device" in hk.hk_unsupported_reason("fwd", q, k_cpu, v, True)
    assert "share a dtype" in hk.hk_unsupported_reason("fwd", q, k.float(), v, True)
    assert "must match" in hk.hk_unsupported_reason("fwd", q, k, v[:, :64], True)


@pytest.mark.parametrize("module", ["fwd_v0", "fwd_pp"])
def test_forward_module_choice_picks_its_rules(monkeypatch, hk, tmp_path, module):
    monkeypatch.setattr(hk, "FWD_SETTING", module.split("_")[1])
    _available(monkeypatch, hk, tmp_path, fwd=module, bwd=None)
    assert hk.module_name("fwd") == module
    q, k, v = _meta(1, 96, 128, 4, 1)
    assert hk.hk_unsupported_reason("fwd", q, k, v, True).startswith(module)


def test_module_gate_is_asked_too(monkeypatch, hk, tmp_path):
    seen = []

    def gate(q_shape, k_shape, v_shape, dtype, causal):
        seen.append((q_shape, k_shape, dtype, causal))
        return "the kernel says no" if q_shape[1] == 256 else None

    _available(monkeypatch, hk, tmp_path, gate=gate)
    q, k, v = _meta(1, 128, 256, 4, 1)
    assert hk.hk_unsupported_reason("fwd", q, k, v, True) is None
    assert seen == [((1, 128, 4, D), (1, 256, 1, D), torch.bfloat16, True)]
    q, k, v = _meta(1, 256, 256, 4, 1)
    assert hk.hk_unsupported_reason("bwd", q, k, v, True) == "bwd_v0: the kernel says no"


def test_module_gate_that_raises_declines(monkeypatch, hk, tmp_path):
    def gate(*args):
        raise TypeError("unexpected signature")

    _available(monkeypatch, hk, tmp_path, gate=gate)
    q, k, v = _meta(1, 128, 128, 4, 1)
    assert "raised TypeError" in hk.hk_unsupported_reason("fwd", q, k, v, True)


def test_module_gate_accepts_both_signatures():
    seen = []
    dims = types.SimpleNamespace(unsupported_reason=lambda b, sq, skv, hq, hkv, causal, d=D: seen.append(
        (b, sq, skv, hq, hkv, causal, d)))  # fmt: skip
    shapes = types.SimpleNamespace(unsupported_reason=lambda q_shape, k_shape, v_shape, dtype, causal: seen.append(
        (q_shape, dtype)))  # fmt: skip
    q, k, v = _meta(2, 128, 256, 8, 2)
    impl.module_gate(dims)(q.shape, k.shape, v.shape, torch.bfloat16, True)
    impl.module_gate(shapes)(q.shape, k.shape, v.shape, torch.bfloat16, True)
    assert seen == [(2, 128, 256, 8, 2, True, D), (q.shape, torch.bfloat16)]
    assert impl.module_gate(types.SimpleNamespace()) is None


def test_limits_cover_every_selectable_module():
    modules = {m for t in (impl.FWD_MODULES, impl.BWD_MODULES) for m in t.values() if m}
    assert modules <= set(limits.MODULE_RULES)
    assert "no shape rules" in limits.shape_unsupported_reason("fwd_v9", (1,) * 4, (1,) * 4, (1,) * 4, True)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def test_entry_points_call_the_selected_modules(monkeypatch, hk, tmp_path):
    mods = _available(monkeypatch, hk, tmp_path)
    q, k, v = (torch.zeros(s, dtype=torch.bfloat16) for s in ((1, 128, 4, D), (1, 128, 1, D), (1, 128, 1, D)))
    out, lse = hk.hk_attn_fwd(q, k, v, None, True)
    assert out.shape == q.shape and lse.shape == (1, 4, 128)
    hk.hk_attn_bwd(q, q, k, v, out, lse, 0.05, True)
    assert mods[_FWD].calls == [("fwd", D**-0.5, True)]
    assert mods["bwd_v0"].calls == [("bwd", 0.05, True)]
    # the lab arm contract: the production selection under the lab's names
    hk.flash_attn_fwd(q, k, v, 0.1, True)
    hk.flash_attn_bwd(q, q, k, v, out, lse, 0.1, True)
    assert mods[_FWD].calls[-1] == ("fwd", 0.1, True) and mods["bwd_v0"].calls[-1] == ("bwd", 0.1, True)


def test_entry_points_raise_when_unsupported(monkeypatch, hk, tmp_path):
    _available(monkeypatch, hk, tmp_path)
    q, k, v = (torch.zeros(s, dtype=torch.bfloat16) for s in ((1, 128, 4, D), (1, 128, 1, D), (1, 128, 1, D)))
    lse = torch.zeros(1, 4, 128)
    with pytest.raises(NotImplementedError, match="causal attention only"):
        hk.hk_attn_fwd(q, k, v, None, False)
    with pytest.raises(NotImplementedError, match="deterministic=True"):
        hk.hk_attn_bwd(q, q, k, v, q, lse, None, True, deterministic=True)
    with pytest.raises(ValueError, match="lse must be float32 of shape"):
        hk.hk_attn_bwd(q, q, k, v, q, lse.transpose(1, 2).contiguous(), None, True)
    with pytest.raises(ValueError, match="o must be"):
        hk.hk_attn_bwd(q, q, k, v, q[:, :64].contiguous(), lse, None, True)
    with pytest.raises(ValueError, match="q must be contiguous"):
        hk.hk_attn_fwd(q.transpose(1, 2).contiguous().transpose(1, 2), k, v, None, True)


# ---------------------------------------------------------------------------
# Manifest and pinning
# ---------------------------------------------------------------------------

_KINDS = {"ptr": 8, "i32": 4, "u32": 4, "i64": 8, "u64": 8, "f32": 4, "f64": 8}


def _symbols(name, entry):
    return entry.get("symbols") or {name: entry}


def test_manifest_is_valid():
    m = json.loads(hsaco.MANIFEST_PATH.read_text())
    assert m["schema"] == 2 and m["arch"] == "gfx1250"
    assert "-DKITTENS_CDNA5" in m["flags"] and not any("CDNA4" in f for f in m["flags"])
    assert "--cuda-device-only" in m["flags"] and "--offload-arch=gfx1250" in m["flags"]
    modules = {mod for t in (impl.FWD_MODULES, impl.BWD_MODULES) for mod in t.values() if mod}
    assert {e["module"] for e in m["kernels"].values()} == modules
    assert any(hsaco.is_pinned(e) for e in m["kernels"].values()) <= (m["toolchain"] is not None)
    for name, e in m["kernels"].items():
        assert name.isidentifier() and Path(e["src"]).parts[0] == "src" and e["src"].endswith(".hip"), name
        assert Path(e["src"]).stem == name, f"{name}: -cuid and cc.sh name the code object after the source"
        assert isinstance(e["extra_flags"], list) and all(isinstance(f, str) for f in e["extra_flags"])
        assert e["hsaco_sha256"] is None or hsaco.is_pinned(e), name
        for sym, s in _symbols(name, e).items():
            assert s.get("args") is None or isinstance(s.get("block"), int), sym
            if hsaco.is_pinned(e):
                end = 0
                for a in s["args"]:
                    assert _KINDS[a["kind"]] == a["size"] and a["offset"] >= end, (sym, a)
                    end = a["offset"] + a["size"]
                assert s["kernarg_size"] == end and s["block"] >= 1 and s["vgpr"] > 0, sym
            elif s.get("args"):
                assert all(len(a) == 2 and a[1] in _KINDS for a in s["args"]), sym


def test_every_pinned_kernel_matches_its_code_object_when_built():
    m = hsaco.load_manifest()
    pinned = [n for n, e in m["kernels"].items() if hsaco.is_pinned(e)]
    built = [n for n in pinned if hsaco.binary_path(n).is_file()]
    if not built:
        pytest.skip("no pinned code object is built here")
    assert hsaco.unavailable_reason(hsaco.BIN_DIR, built) is None


def test_hsaco_build_skips_unpinned_kernels(tmp_manifest):
    tmp_manifest({"k_attn_" + _FWD: _entry(_FWD)})
    log = []
    assert hsaco.build(cxx="/nonexistent/amdclang++", log=log.append) is True
    assert any("not pinned yet" in line for line in log), log
    assert not any(hsaco.BIN_DIR.iterdir())


def test_check_pins_reads_the_files_now(tmp_manifest):
    data = b"pinned"
    tmp_manifest({"k_a": _entry("fwd_v0", hashlib.sha256(data).hexdigest())})
    (hsaco.BIN_DIR / "k_a.hsaco").write_bytes(data)
    assert hsaco.unavailable_reason(None, ["k_a"]) is None and hsaco.check_pins(None, ["k_a"]) is None
    (hsaco.BIN_DIR / "k_a.hsaco").write_bytes(b"other")
    assert hsaco.unavailable_reason(None, ["k_a"]) is None  # cached
    assert "sha256" in hsaco.check_pins(None, ["k_a"])


def test_compile_timeout_is_a_build_warning(monkeypatch, tmp_manifest, tmp_path):
    """A compiler that hangs costs the install at most the timeout, then that kernel is not built."""
    data = b"pinned"
    m = tmp_manifest({"k_a": _entry("fwd_v0", hashlib.sha256(data).hexdigest())})
    m["kernels"]["k_a"]["src"] = "src/k_a.hip"
    hsaco.MANIFEST_PATH.write_text(json.dumps(m))
    hsaco.reload_manifest()
    monkeypatch.setattr(hsaco, "HIP_DIR", tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "k_a.hip").write_text("")
    slow = tmp_path / "slow_cxx"
    slow.write_text("#!/bin/sh\nsleep 30 &\nexec sleep 30\n")  # a child holds the pipes too
    slow.chmod(0o755)
    monkeypatch.setenv(hsaco.COMPILE_TIMEOUT_ENV, "0.5")
    log = []
    assert hsaco.build(cxx=str(slow), log=log.append) is False
    assert any("did not finish in 0.5 s" in line for line in log), log
    assert not hsaco.binary_path("k_a").exists()


def test_code_object_helper(tmp_manifest, tmp_path):
    sha = "a" * 64
    tmp_manifest({"k_a": _entry("fwd_v0", sha), "k_b": _entry("fwd_v0")})
    assert hsaco.code_object("k_a") == (hsaco.BIN_DIR / "k_a.hsaco", sha)
    assert hsaco.code_object("k_b", tmp_path) == (tmp_path / "k_b.hsaco", None)
    with pytest.raises(ValueError, match="no sha256 pin"):
        hsaco.code_object("k_b")
    assert hsaco.kernel_names("fwd_v0") == ["k_a", "k_b"] and hsaco.kernel_names("bwd_v0") == []


_TOY_HIP = textwrap.dedent(
    """\
    #include <hip/hip_runtime.h>
    extern "C" __global__ __launch_bounds__(64) void k_toy(float *x, int n, float a) {
        if ((int) threadIdx.x < n) x[threadIdx.x] = a * (float) n;
    }
    """
)


@functools.lru_cache(maxsize=None)
def _gfx1250_compiler_reason():
    """None when amdclang++ compiles device code for gfx1250 (one tiny compile, once per session)."""
    cxx = hsaco.find_compiler()
    if cxx is None:
        return "needs amdclang++ (compile only, no GPU)"
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "probe.hip"
        src.write_text(_TOY_HIP)
        cmd = [cxx, "-x", "hip", "--offload-arch=gfx1250", "--cuda-device-only", "--no-gpu-bundle-output"]
        try:
            r = subprocess.run(
                [*cmd, "-O1", "-c", str(src), "-o", str(Path(tmp) / "probe.hsaco")],
                capture_output=True,
                text=True,
                timeout=300,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            return f"{cxx} cannot compile for gfx1250: {e}"
        if r.returncode != 0:
            return f"{cxx} cannot compile for gfx1250: {r.stderr.strip()[-300:]}"
    return None


def _toy_package(tmp_path, args):
    pkg = tmp_path / "pkg"
    (pkg / "hip" / "src").mkdir(parents=True)
    (pkg / "hip" / "include").mkdir()
    for f in ("hsaco.py", "codeobj.py", "hiprt.py"):
        shutil.copy(Path(hsaco.__file__).with_name(f), pkg / f)
    (pkg / "__init__.py").write_text("")
    (pkg / "hip" / "src" / "k_toy.hip").write_text(_TOY_HIP)
    m = json.loads(hsaco.MANIFEST_PATH.read_text())
    m["toolchain"] = None
    m["kernels"] = {"k_toy": {**_entry("toy"), "src": "src/k_toy.hip", "block": 64, "args": args}}
    (pkg / "manifest.json").write_text(json.dumps(m))
    return pkg


def _pin_tool():
    spec = importlib.util.spec_from_file_location("hk_pin", _ROOT / "tools" / "hk_gfx1250_attn_pin.py")
    pin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pin)
    return pin


def test_pin_tool_writes_pins_only_for_named_validated_kernels(tmp_path, capsys):
    """Without --kernels, or without --against (or --unvalidated), nothing is compiled or written."""
    pin = _pin_tool()
    pkg = _toy_package(tmp_path, [["X", "ptr"], ["n", "i32"], ["a", "f32"]])
    before = (pkg / "manifest.json").read_text()
    assert pin.main(["--package-dir", str(pkg)]) == 1
    assert "--kernels" in capsys.readouterr().err
    assert pin.main(["--package-dir", str(pkg), "--kernels", "k_toy"]) == 1
    assert "--against" in capsys.readouterr().err
    assert (pkg / "manifest.json").read_text() == before and not (pkg / "bin").exists()


def test_pin_tool_reads_the_lab_builds_flags_and_records(tmp_path):
    pin = _pin_tool()
    cmd = [
        "amdclang++",
        "-x",
        "hip",
        "-save-temps=obj",
        "-cuid=k_a",
        "-Wuninitialized",
        "-Wsometimes-uninitialized",
    ]
    ref = tmp_path / "k_a.hsaco"
    ref.write_bytes(b"x")
    (tmp_path / "k_a.flags").write_text(
        "\n".join([*cmd, "-mllvm", "-foo", "-c", "k_a.hip", "-o", "o"]) + "\n"
    )
    assert pin.lab_extra_flags(ref) == ["-mllvm", "-foo"]
    meta = {"ok": True, "flags": [*cmd, "-c", "k_a.hip", "-o", "o"]}
    (tmp_path / "k_a.meta.json").write_text(json.dumps(meta))
    assert pin.lab_extra_flags(ref) == []  # meta.json first
    assert pin.search_dirs(["a" + os.pathsep + "b", "c"]) == [Path("a"), Path("b"), Path("c")]
    sha = "ab" * 32
    rec = {"code_objects": {"k_a": {"direction": "bwd", "sha256": sha}}, "stages": {"bwd": {"full": True}}}
    assert "['large'] not recorded" in pin.record_reason([rec], "k_a", sha)
    other = {"code_objects": {"k_a": {"direction": "bwd", "sha256": sha}}, "stages": {"fwd": {"large": True}}}
    assert "['large'] not recorded" in pin.record_reason([rec, other], "k_a", sha)  # another direction's
    large = {"code_objects": {"k_a": {"direction": "bwd", "sha256": sha}}, "stages": {"bwd": {"large": True}}}
    assert pin.record_reason([rec, large], "k_a", sha) is None  # stages add up over sessions
    assert "no validation record" in pin.record_reason([rec], "k_a", "cd" * 32)


def test_pin_tool_pins_and_checks(tmp_path, capsys):
    """One command compiles the named entry, reads its kernel table and writes sha256, metadata and
    toolchain; --check then matches; --against refuses other bytes, other flags and a failed cc.sh build and
    searches several directories; --record refuses bytes without a passing validation record; an argument
    list that disagrees with the code object is refused and leaves the manifest as it was."""
    reason = _gfx1250_compiler_reason()
    if reason is not None:
        pytest.skip(reason)
    pin = _pin_tool()

    pkg = _toy_package(tmp_path, [["X", "ptr"], ["n", "i32"], ["a", "f32"]])
    assert pin.main(["--package-dir", str(pkg), "--kernels", "k_toy", "--unvalidated"]) == 0
    m = json.loads((pkg / "manifest.json").read_text())
    e = m["kernels"]["k_toy"]
    data = (pkg / "bin" / "k_toy.hsaco").read_bytes()
    assert e["hsaco_sha256"] == hashlib.sha256(data).hexdigest()
    assert m["toolchain"] and "clang" in m["toolchain"]
    assert e["kernarg_size"] == 16 and e["block"] == 64 and e["vgpr"] > 0
    assert [(a["name"], a["kind"], a["offset"], a["size"]) for a in e["args"]] == [
        ("X", "ptr", 0, 8),
        ("n", "i32", 8, 4),
        ("a", "f32", 12, 4),
    ]
    assert pin.main(["--package-dir", str(pkg), "--check"]) == 0
    other, empty = tmp_path / "other", tmp_path / "empty"
    other.mkdir()
    empty.mkdir()
    (other / "k_toy.hsaco").write_bytes(b"something else")
    args = ["--package-dir", str(pkg), "--kernels", "k_toy", "--no-bin"]
    assert pin.main([*args, "--against", str(other)]) == 1
    (other / "k_toy.hsaco").write_bytes(data)
    capsys.readouterr()
    assert pin.main([*args, "--against", f"{empty}{os.pathsep}{other}"]) == 0
    assert str(other / "k_toy.hsaco") in capsys.readouterr().out
    flags = [
        "amdclang++",
        "-cuid=k_toy",
        "-Wuninitialized",
        "-Wsometimes-uninitialized",
        "-mllvm",
        "-x",
        "-c",
    ]
    (other / "k_toy.flags").write_text("\n".join([*flags, "src.hip"]))
    assert pin.main([*args, "--against", str(other)]) == 1
    assert "was built with ['-mllvm', '-x']" in capsys.readouterr().err
    (other / "k_toy.flags").unlink()
    (other / "k_toy.meta.json").write_text(json.dumps({"ok": False}))
    assert pin.main([*args, "--against", str(other)]) == 1
    assert "ok=False" in capsys.readouterr().err
    (other / "k_toy.meta.json").unlink()
    record = tmp_path / "validated.jsonl"
    sha = hashlib.sha256(data).hexdigest()
    entry = {"code_objects": {"k_toy": {"direction": "fwd", "sha256": sha}}}
    record.write_text(json.dumps({**entry, "stages": {"fwd": {"toy": True}}}) + "\n")
    assert pin.main([*args, "--against", str(other), "--record", str(record)]) == 1
    assert "not recorded" in capsys.readouterr().err
    with record.open("a") as f:
        f.write(json.dumps({**entry, "stages": {"fwd": {"toy": True, "full": True, "large": True}}}) + "\n")
    assert pin.main([*args, "--against", str(other), "--record", str(record)]) == 0
    bad = _toy_package(tmp_path / "bad", [["X", "ptr"], ["n", "i64"], ["a", "f32"]])
    before = (bad / "manifest.json").read_text()
    assert pin.main(["--package-dir", str(bad), "--kernels", "k_toy", "--unvalidated"]) == 1
    assert (bad / "manifest.json").read_text() == before and not (bad / "bin").exists()
    assert "size" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# setup.py hook
# ---------------------------------------------------------------------------


def _setup_hook_namespace(project_root):
    """HK_GFX1250_PACKAGES, build_hk_gfx1250_hsaco and hk_gfx1250_package_data of setup.py, executed alone
    with PROJECT_ROOT = project_root."""
    setup_py = _ROOT / "setup.py"
    if not setup_py.is_file():
        pytest.skip("no setup.py next to the tests")
    tree = ast.parse(setup_py.read_text())
    wanted = {"HK_GFX1250_PACKAGES", "build_hk_gfx1250_hsaco", "hk_gfx1250_package_data"}
    nodes = [
        n
        for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name in wanted)
        or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in wanted for t in n.targets))
    ]
    assert len(nodes) == 3
    ns = {"PROJECT_ROOT": Path(project_root), "os": os, "importlib": importlib, "json": json}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(setup_py), "exec"), ns)
    return ns


def test_setup_builds_each_hk_package_and_never_fails(monkeypatch, tmp_path, capsys):
    root = tmp_path / "proj"
    for pkg, body in (
        ("gfx1250_attn", "def build(log):\n    log('built attn')\n"),
        ("gfx1250_mla_bwd", "def build(log):\n    raise RuntimeError('no compiler')\n"),
    ):
        d = root / "primus_turbo" / "hipkittens" / pkg
        d.mkdir(parents=True)
        (d / "hsaco.py").write_text(body)
    attn = root / "primus_turbo" / "hipkittens" / "gfx1250_attn"
    (attn / "manifest.json").write_text(json.dumps({"kernels": {"k_a": {}, "k_b": {}}}))
    (attn.parent / "gfx1250_mla_bwd" / "manifest.json").write_text("{not json")
    ns = _setup_hook_namespace(root)
    assert ns["HK_GFX1250_PACKAGES"]["gfx1250_attn"] == "PRIMUS_TURBO_BUILD_HK_ATTN"
    # the manifest's code objects only (bin/ may hold lab files); an unreadable manifest ships none
    assert ns["hk_gfx1250_package_data"]() == [
        "hipkittens/gfx1250_attn/manifest.json",
        "hipkittens/gfx1250_attn/bin/k_a.hsaco",
        "hipkittens/gfx1250_attn/bin/k_b.hsaco",
        "hipkittens/gfx1250_mla_bwd/manifest.json",
    ]
    assert "no code objects packaged" in capsys.readouterr().out
    monkeypatch.delenv("PRIMUS_TURBO_BUILD_HK_ATTN", raising=False)
    monkeypatch.delenv("PRIMUS_TURBO_BUILD_HK_MLA_BWD", raising=False)
    ns["build_hk_gfx1250_hsaco"](["--offload-arch=gfx950"])
    assert capsys.readouterr().out == ""
    ns["build_hk_gfx1250_hsaco"](["--offload-arch=gfx1250"])
    out = capsys.readouterr().out
    assert "built attn" in out and "gfx1250_mla_bwd code objects not built: RuntimeError" in out
    monkeypatch.setenv("PRIMUS_TURBO_BUILD_HK_ATTN", "0")
    ns["build_hk_gfx1250_hsaco"](["--offload-arch=gfx1250"])
    assert "built attn" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Backend gate and resolution
# ---------------------------------------------------------------------------


def _on_gfx1250(monkeypatch):
    monkeypatch.setattr(attention_impl, "is_gfx1250", lambda: True)
    monkeypatch.setattr(flash_attn_interface, "is_gfx1250", lambda: True)


def _resolve_kwargs(q, k, v, causal=True):
    return dict(q=q, k=k, v=v, causal=causal, window_size=(-1, -1), qkv_format="bshd", needs_backward=True)


def _mla_meta(b=1, s=256, h=2):
    """DeepSeek-V3 MLA head dims (qk 192, v 128): the FLYDSL backend's other gfx1250 family."""
    q = torch.empty(b, s, h, 192, dtype=torch.bfloat16, device="meta")
    return q, torch.empty_like(q), torch.empty(b, s, h, 128, dtype=torch.bfloat16, device="meta")


@contextlib.contextmanager
def _pinned(backend):
    GlobalBackendManager.set_attn_backend(backend, PrecisionType.BF16_FP16_FP32)
    try:
        yield
    finally:
        GlobalBackendManager.set_attn_backend(None, PrecisionType.BF16_FP16_FP32)


@needs_flydsl
@pytest.mark.parametrize("shape", [PROD, (1, 64, 64, 1, 1), (1, 256, 512, 8, 2), (2, 128, 128, 16, 1)])
def test_gfx1250_gate_is_flydsls_head_dim_128_gate(monkeypatch, shape):
    """HIPKITTENS on gfx1250 takes every head-dim-128 call the FlyDSL kernels take, also the ones its HIP
    kernels cannot (G 1, non-causal): those directions run FlyDSL."""
    _on_gfx1250(monkeypatch)
    q, k, v = _meta(*shape)
    for causal in (True, False):
        hk_ok = attention_impl.DenseAttnFwdHipkittensBackend.can_handle(q, k=k, v=v, causal=causal)
        fly_ok = attention_impl.DenseAttnFwdFlydslBackend.can_handle(q, k=k, v=v, causal=causal)
        assert hk_ok is fly_ok is True


@needs_flydsl
@pytest.mark.parametrize(
    ("shape", "kwargs"),
    [
        ((1, 96, 96, 4, 1), {}),
        ((1, 256, 256, 4, 2, 64), {}),
        ((1, 256, 256, 4, 1), {"dropout_p": 0.1}),
        ((1, 256, 256, 4, 1), {"window_size": (128, 0)}),
        ((1, 256, 256, 4, 1), {"sink": torch.empty(4)}),
        ((1, 256, 256, 4, 1, 128, torch.float16), {}),
    ],
    ids=["sq_not_x64", "d64", "dropout", "window", "sink", "fp16"],
)
def test_gfx1250_gate_refuses_what_flydsl_refuses(monkeypatch, shape, kwargs):
    _on_gfx1250(monkeypatch)
    q, k, v = _meta(*shape)
    assert (
        attention_impl.DenseAttnFwdHipkittensBackend.can_handle(q, k=k, v=v, causal=True, **kwargs) is False
    )
    assert pinned_backend_takes(BackendType.HIPKITTENS, **{**_resolve_kwargs(q, k, v), **kwargs}) is False


@needs_flydsl
def test_mla_head_dims_stay_on_flydsl(monkeypatch):
    _on_gfx1250(monkeypatch)
    q, k, v = _mla_meta()
    assert attention_impl.DenseAttnFwdHipkittensBackend.can_handle(q, k=k, v=v, causal=True) is False


@needs_flydsl
def test_hk_is_selectable_but_not_the_default(monkeypatch):
    _on_gfx1250(monkeypatch)
    q, k, v = _meta(*PROD)
    kw = _resolve_kwargs(q, k, v)
    assert attention_impl.HK_GFX1250_DEFAULT is False
    assert resolve_flash_attn_backend(False, None, **kw) == BackendType.FLYDSL
    assert resolve_flash_attn_backend(False, BackendType.HIPKITTENS, **kw) == BackendType.HIPKITTENS
    monkeypatch.setenv("PRIMUS_TURBO_ATTN_BACKEND", "hipkittens")
    assert GlobalBackendManager.get_attn_backend(PrecisionType.BF16_FP16_FP32) == BackendType.HIPKITTENS
    monkeypatch.setattr(attention_impl, "HK_GFX1250_DEFAULT", True)  # the switch to flip later
    assert resolve_flash_attn_backend(False, None, **kw) == BackendType.HIPKITTENS
    assert resolve_flash_attn_backend(False, None, **_resolve_kwargs(*_mla_meta())) == BackendType.FLYDSL


def test_hk_is_no_autotune_candidate_until_it_is_the_default(monkeypatch):
    """Under autotune on gfx1250, HIPKITTENS is timed only once HK_GFX1250_DEFAULT is True and HK runs
    the forward of the call itself (else it would time FlyDSL a second time): its tuning_kwargs raise,
    which AutoKernelDispatcher.tune counts as a backend that cannot run."""
    _on_gfx1250(monkeypatch)
    q, k, v = _meta(*PROD)
    kw = dict(q=q, k=k, v=v, causal=True, softmax_scale=None)
    backend = attention_impl.DenseAttnFwdHipkittensBackend
    with pytest.raises(RuntimeError, match="not an autotune candidate"):
        backend.tuning_kwargs(**kw)
    monkeypatch.setattr(attention_impl, "HK_GFX1250_DEFAULT", True)
    reasons = {"fwd": f"k_attn_{_FWD} is not pinned"}
    monkeypatch.setattr(attention_impl, "hk_gfx1250_unsupported_reason", lambda d, *a, **kw: reasons.get(d))
    with pytest.raises(RuntimeError, match="would run this forward on FlyDSL"):
        backend.tuning_kwargs(**kw)
    reasons.clear()
    assert backend.tuning_kwargs(**kw) == kw
    monkeypatch.setattr(attention_impl, "is_gfx1250", lambda: False)
    monkeypatch.setattr(attention_impl, "HK_GFX1250_DEFAULT", False)
    assert backend.tuning_kwargs(**kw) == kw  # gfx950: as before


def test_gfx950_hipkittens_gate_is_unchanged(monkeypatch):
    """Off gfx1250 the HIPKITTENS gate is the gfx950 one (sbhd only), as before."""
    monkeypatch.setattr(attention_impl, "is_gfx1250", lambda: False)
    calls = []

    def supported(q, k, v, **kwargs):
        calls.append(tuple(q.shape))
        return True, ""

    monkeypatch.setattr(attention_impl, "hipkittens_attn_supported_impl", supported)
    q, k, v = _meta(2, 256, 256, 4, 1)
    assert attention_impl.DenseAttnFwdHipkittensBackend.can_handle(q, k=k, v=v, qkv_format="bshd") is False
    sb = [t.permute(1, 0, 2, 3) for t in _meta(256, 2, 2, 4, 1)]  # [b, s, h, d] views of sbhd bytes
    assert attention_impl.DenseAttnFwdHipkittensBackend.can_handle(sb[0], k=sb[1], v=sb[2], qkv_format="sbhd")
    assert calls == [(256, 2, 4, D)]  # asked on the sbhd view


# ---------------------------------------------------------------------------
# FlashAttnFunc routing
# ---------------------------------------------------------------------------


@needs_flydsl
@pytest.mark.parametrize("deterministic", [False, True])
def test_flash_attn_func_routes_hipkittens_to_the_gfx1250_hk_impls(monkeypatch, deterministic):
    """HIPKITTENS on "gfx1250": the HK impls get contiguous BSHD q/k/v (sbhd bytes in), lse [B, Hq, Sq] and
    the deterministic flag; neither the gfx950 HipKittens impls nor the FlyDSL impls are reached."""
    _on_gfx1250(monkeypatch)
    b, sq, skv, hq, hkv = 2, 128, 256, 8, 2
    calls = []

    def fwd(q, k, v, softmax_scale=None, causal=True):
        calls.append(("fwd", q.is_contiguous(), k.is_contiguous(), v.is_contiguous(), softmax_scale, causal))
        return torch.zeros_like(q), torch.zeros(b, hq, sq)

    def bwd(dout, q, k, v, out, lse, softmax_scale=None, causal=True, deterministic=False):
        calls.append(("bwd", tuple(lse.shape), tuple(out.shape), deterministic))
        return torch.ones_like(q), torch.ones_like(k), torch.ones_like(v)

    def forbidden(*args, **kwargs):
        raise AssertionError("another backend's impl reached")

    monkeypatch.setattr(flash_attn_interface, "flash_attn_hk_gfx1250_forward_impl", fwd)
    monkeypatch.setattr(flash_attn_interface, "flash_attn_hk_gfx1250_backward_impl", bwd)
    for name in (
        "flash_attn_sbhd_hipkittens_forward_impl",
        "flash_attn_sbhd_hipkittens_backward_impl",
        "flash_attn_flydsl_gfx1250_forward_impl",
        "flash_attn_flydsl_gfx1250_backward_impl",
    ):
        monkeypatch.setattr(flash_attn_interface, name, forbidden)
    q = torch.randn(sq, b, hq, D, dtype=torch.bfloat16).permute(1, 0, 2, 3).requires_grad_()
    k = torch.randn(skv, b, hkv, D, dtype=torch.bfloat16).permute(1, 0, 2, 3).requires_grad_()
    v = torch.randn(skv, b, hkv, D, dtype=torch.bfloat16).permute(1, 0, 2, 3).requires_grad_()
    with _pinned(BackendType.HIPKITTENS):
        out = flash_attn_func(q, k, v, causal=True, deterministic=deterministic)
        out.sum().backward()
    assert calls == [
        ("fwd", True, True, True, None, True),
        ("bwd", (b, hq, sq), (b, sq, hq, D), deterministic),
    ]
    assert k.grad.shape == (b, skv, hkv, D) and torch.all(k.grad == 1)


@needs_flydsl
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"dropout_p": 0.1}, "does not implement dropout"),
        ({"window_size": (64, 0)}, "no sliding window"),
        ({"return_attn_probs": True}, "cannot return the softmax matrix"),
    ],
)
def test_direct_apply_refuses_what_the_kernels_lack(monkeypatch, kwargs, match):
    """FlashAttnFunc.apply is reachable without the gate: HIPKITTENS on gfx1250 raises for arguments its
    kernels (and FlyDSL's) do not implement instead of ignoring them."""
    _on_gfx1250(monkeypatch)
    q, k, v = _meta(1, 128, 128, 4, 1)
    args = {**dict(dropout_p=0.0, window_size=(-1, -1), return_attn_probs=False), **kwargs}
    with pytest.raises(ValueError, match=f"gfx1250 hipkittens attention.*{match}"):
        flash_attn_interface.FlashAttnFunc.apply(
            q, k, v, args["dropout_p"], None, True, args["window_size"], None, None, False, False,
            args["return_attn_probs"], False, 1, None, "bshd", BackendType.HIPKITTENS,
        )  # fmt: skip


# ---------------------------------------------------------------------------
# Per-direction fallback inside the custom ops
# ---------------------------------------------------------------------------


class _FakeImpl:
    """Stands in for the HK package's impl: answers ``reasons[direction]`` and records the HK calls."""

    def __init__(self, fwd=None, bwd=None):
        self.reasons = {"fwd": fwd, "bwd": bwd}
        self.calls = []

    def hk_unsupported_reason(self, direction, q, k, v, causal, deterministic=False, softmax_scale=None):
        if direction == "bwd" and deterministic:
            return "deterministic=True (the HK backward adds dQ with fp32 atomics)"
        return self.reasons[direction]

    def describe(self, direction):
        return f"{direction}: fake"

    def hk_attn_fwd(self, q, k, v, softmax_scale, causal):
        self.calls.append(("hk_fwd", softmax_scale, causal))
        return "hk out", "hk lse"

    def hk_attn_bwd(self, do, q, k, v, o, lse, softmax_scale, causal, deterministic):
        self.calls.append(("hk_bwd", softmax_scale, causal, deterministic))
        return "hk grads"


class _FakeFlydsl:
    def __init__(self, calls):
        self.calls = calls

    def flash_attn_fwd(self, q, k, v, softmax_scale, causal):
        self.calls.append(("fly_fwd", softmax_scale, causal))
        return "fly out", "fly lse"

    def flash_attn_bwd(self, do, q, k, v, o, lse, softmax_scale, causal, deterministic):
        self.calls.append(("fly_bwd", softmax_scale, causal, deterministic))
        return "fly grads"


@pytest.fixture
def adapter(monkeypatch):
    """The adapter with a fake HK impl, a fake FlyDSL interface and a recording logger."""
    fake = _FakeImpl()
    log = _LogRecorder()
    monkeypatch.setattr(hk_adapter, "_IMPL", fake)
    monkeypatch.setattr(hk_adapter, "_flydsl_interface", lambda: _FakeFlydsl(fake.calls))
    monkeypatch.setattr(hk_adapter, "logger", log)
    return fake, log


def _run_both(q, k, v, deterministic=False):
    out = hk_adapter._run_forward(q, k, v, 0.088, True)
    grads = hk_adapter._run_backward(q, q, k, v, q, None, 0.088, True, deterministic)
    return out, grads


def test_supported_calls_run_hk(adapter):
    fake, log = adapter
    q, k, v = _meta(*PROD)
    assert _run_both(q, k, v) == (("hk out", "hk lse"), "hk grads")
    assert fake.calls == [("hk_fwd", 0.088, True), ("hk_bwd", 0.088, True, False)]
    assert log.warnings() == [] and [r[2] for r in log.records] == [True, True]


@pytest.mark.parametrize(
    ("fwd", "bwd", "deterministic", "expected"),
    [
        ("seqlen_q must be a multiple of 64", "seqlen_q must be a multiple of 64", False, ("fly", "fly")),
        (None, "k_attn_bwd_v0 is not pinned in manifest.json", False, ("hk", "fly")),
        ("PRIMUS_TURBO_HK_FWD=flydsl", None, False, ("fly", "hk")),
        (None, None, True, ("hk", "fly")),
    ],
    ids=["shape", "bwd_not_pinned", "mixed_fwd_flydsl", "deterministic"],
)
def test_unsupported_directions_run_flydsl_with_one_warning_per_reason(
    adapter, fwd, bwd, deterministic, expected
):
    fake, log = adapter
    fake.reasons = {"fwd": fwd, "bwd": bwd}
    q, k, v = _meta(*PROD)
    for _ in range(3):
        _run_both(q, k, v, deterministic)
    kinds = [c[0].split("_")[0] for c in fake.calls[:2]]
    assert tuple(kinds) == expected
    bwd_call = fake.calls[1]
    assert bwd_call[-1] == (deterministic if expected[1] == "fly" else False)
    warnings = log.warnings()
    assert all(once for _, _, once in warnings)
    reasons = {w[1] for w in warnings}  # the real logger emits each (level, message) once
    assert len(reasons) == sum(e == "fly" for e in expected)
    for r in reasons:
        assert r.startswith("HIPKITTENS attention on gfx1250: ") and "runs on FlyDSL (" in r


def test_hk_entry_point_that_declines_late_runs_flydsl(adapter):
    """hk_attn_* may still raise NotImplementedError before launching (do / o / lse alignment, a pinned code
    object replaced after the gate's check): that call runs on FlyDSL, with a warning."""
    fake, log = adapter

    def declines(*args, **kwargs):
        raise NotImplementedError("hk_attn_bwd: do data_ptr not 16-byte aligned")

    fake.hk_attn_bwd = declines
    q, k, v = _meta(*PROD)
    assert _run_both(q, k, v) == (("hk out", "hk lse"), "fly grads")
    assert [w[1] for w in log.warnings()] == [
        "HIPKITTENS attention on gfx1250: backward runs on FlyDSL (hk_attn_bwd: do data_ptr not 16-byte aligned)"
    ]


def test_hk_package_that_fails_to_import_runs_flydsl(monkeypatch):
    log = _LogRecorder()
    calls = []

    def broken():
        raise ImportError("no module named hiprt")

    monkeypatch.setattr(hk_adapter, "_IMPL", None)
    monkeypatch.setattr(hk_adapter, "_import_impl", broken)
    monkeypatch.setattr(hk_adapter, "_flydsl_interface", lambda: _FakeFlydsl(calls))
    monkeypatch.setattr(hk_adapter, "logger", log)
    q, k, v = _meta(*PROD)
    assert hk_adapter._run_forward(q, k, v, 0.1, True) == ("fly out", "fly lse")
    reason = hk_adapter.hk_gfx1250_unsupported_reason("bwd", q, k, v, True)
    assert "failed to import: ImportError: no module named hiprt" in reason
    assert len(log.warnings()) == 1 and "no module named hiprt" in log.warnings()[0][1]


def test_mixed_mode_through_the_real_impl(monkeypatch, hk, tmp_path):
    """PRIMUS_TURBO_HK_FWD=flydsl with an available backward: forward on FlyDSL, backward on HK."""
    calls = []
    monkeypatch.setattr(hk, "FWD_SETTING", "flydsl")
    mods = _available(monkeypatch, hk, tmp_path, fwd=None)
    monkeypatch.setattr(hk_adapter, "_IMPL", hk)
    monkeypatch.setattr(hk_adapter, "_flydsl_interface", lambda: _FakeFlydsl(calls))
    monkeypatch.setattr(hk_adapter, "logger", _LogRecorder())
    q, k, v = (torch.zeros(s, dtype=torch.bfloat16) for s in ((1, 128, 4, D), (1, 128, 1, D), (1, 128, 1, D)))
    lse = torch.zeros(1, 4, 128)
    assert hk_adapter._run_forward(q, k, v, 0.1, True) == ("fly out", "fly lse")
    hk_adapter._run_backward(q, q, k, v, q, lse, 0.1, True, False)
    assert calls == [("fly_fwd", 0.1, True)] and mods["bwd_v0"].calls == [("bwd", 0.1, True)]
    hk_adapter._run_backward(q, q, k, v, q, lse, 0.1, True, True)
    assert calls[-1] == ("fly_bwd", 0.1, True, True) and len(mods["bwd_v0"].calls) == 1


# ---------------------------------------------------------------------------
# Custom ops: fake impls and torch.compile
# ---------------------------------------------------------------------------


def test_fake_impls_give_the_eager_shapes():
    b, sq, skv, hq, hkv = 2, 128, 256, 8, 2
    q, k, v = _meta(b, sq, skv, hq, hkv)
    out, lse = torch.ops.primus_turbo.flash_attn_hk_gfx1250_forward(q, k, v, 0.1, True)
    assert (out.shape, out.dtype, lse.shape, lse.dtype) == (q.shape, q.dtype, (b, hq, sq), torch.float32)
    lse = torch.empty(b, hq, sq, device="meta")
    dq, dk, dv = torch.ops.primus_turbo.flash_attn_hk_gfx1250_backward(q, q, k, v, q, lse, 0.1, True, False)
    assert (dq.shape, dk.shape, dv.shape) == (q.shape, k.shape, v.shape)


@needs_flydsl
@pytest.mark.parametrize("dynamic", [False, True])
def test_torch_compile_traces_the_hk_custom_ops(monkeypatch, dynamic):
    """torch.compile(fullgraph) of flash_attn_func with HIPKITTENS pinned on meta tensors: the forward and
    backward graphs hold the HK custom op once each (the HK / FlyDSL decision is inside them, never
    traced), with the eager shapes. No kernel runs."""
    from functorch.compile import make_boxed_func
    from torch._dynamo.backends.common import aot_autograd

    _on_gfx1250(monkeypatch)
    b, sq, skv, hq, hkv = 2, 128, 256, 8, 2
    seen = {"fw": [], "bw": []}

    def record(kind):
        def compiler(gm, example_inputs):
            seen[kind] += [str(n.target) for n in gm.graph.nodes if n.op == "call_function"]
            return make_boxed_func(gm.forward)

        return compiler

    q, k, v = (t.requires_grad_() for t in _meta(b, sq, skv, hq, hkv))

    @torch.compile(
        fullgraph=True,
        dynamic=dynamic,
        backend=aot_autograd(fw_compiler=record("fw"), bw_compiler=record("bw")),
    )
    def fn(q, k, v):
        return flash_attn_func(q, k, v, causal=True, return_lse=True)

    torch._dynamo.reset()
    try:
        with _pinned(BackendType.HIPKITTENS):
            out, lse = fn(q, k, v)
            (out.float().sum() + lse.sum()).backward()
    finally:
        torch._dynamo.reset()
    ops = {kind: [t for t in targets if t.startswith("primus_turbo.")] for kind, targets in seen.items()}
    assert ops == {
        "fw": ["primus_turbo.flash_attn_hk_gfx1250_forward.default"],
        "bw": ["primus_turbo.flash_attn_hk_gfx1250_backward.default"],
    }
    assert out.shape == (b, sq, hq, D) and lse.shape == (b, hq, sq)
    assert k.grad.shape == k.shape and q.grad.shape == q.shape


# ---------------------------------------------------------------------------
# Launch plans of the real kernel modules (when present)
# ---------------------------------------------------------------------------


def _real_module(module):
    """The kernel module of the package (not a fake), or skip when it is not written yet."""
    try:
        return importlib.import_module(f"{_PKG}.{module}")
    except ModuleNotFoundError as e:
        if e.name == f"{_PKG}.{module}":
            pytest.skip(f"{module}.py is not in the package yet")
        raise


def _manifest_symbols(module):
    """{kernel symbol: {block, args}} of every manifest entry the module launches."""
    out = {}
    for name, e in hsaco.load_manifest()["kernels"].items():
        if e.get("module") == module:
            out.update(_symbols(name, e))
    return out


def _module_specs(mod):
    """{symbol: {block, args}} the module launches with: SPECS, or KERNEL / BLOCK / ARGS of a one-kernel module."""
    specs = getattr(mod, "SPECS", None)
    if specs is None and all(hasattr(mod, a) for a in ("KERNEL", "BLOCK", "ARGS")):
        specs = {mod.KERNEL: {"block": mod.BLOCK, "args": mod.ARGS}}
    return specs


@pytest.mark.parametrize("module", [_FWD, "fwd_v0", "bwd_v0"])
def test_kernel_module_meets_the_contract(module):
    """The lab / impl contract of a kernel module, its gate in a form impl understands, and its kernel specs
    equal to the manifest's: the pin tool records the arguments the module packs."""
    mod = _real_module(module)
    entry = "flash_attn_fwd" if module.startswith("fwd") else "flash_attn_bwd"
    assert callable(getattr(mod, entry, None)) and callable(getattr(mod, "_launch", None))
    assert callable(getattr(mod, "set_hsaco_dir", None)) or hasattr(mod, "HSACO_DIR")
    gate = impl.module_gate(mod)
    if gate is not None:
        q, k, v = _meta(*PROD)
        assert gate(q.shape, k.shape, v.shape, torch.bfloat16, True) is None
        q, k, v = _meta(1, 96, 128, 4, 1)  # Sq % 64
        assert gate(q.shape, k.shape, v.shape, torch.bfloat16, True) is not None
    specs = _module_specs(mod)
    if specs is None:
        pytest.skip(f"{module} exposes no kernel specs")
    manifest = _manifest_symbols(module)
    assert set(specs) == set(manifest), (sorted(specs), sorted(manifest))
    for sym, spec in specs.items():
        m = manifest[sym]
        assert spec["block"] == m["block"], sym
        got = [(a["name"], a["kind"]) if isinstance(a, dict) else tuple(a) for a in m["args"]]
        assert got == [tuple(a) for a in spec["args"]], sym


def _plan_launches(mod, b, sq, skv, hq, hkv):
    """[(kernel, grid, block, n_args)] of the module's plan, which takes the shapes (b, sq, skv, hq, hkv,
    causal, scale), meta tensors (q, k, v, scale, causal) or meta tensors and outputs (q, k, v, o, lse,
    scale, causal), and returns a dict ({grid, block, values} or {launches}) or (grid, block, values)."""
    params = list(inspect.signature(mod.plan).parameters)
    if params[:3] == ["q", "k", "v"]:
        q, k, v = _meta(b, sq, skv, hq, hkv)
        if params[3:5] == ["o", "lse"]:
            lse = torch.empty(b, hq, sq, dtype=torch.float32, device="meta")
            p = mod.plan(q, k, v, torch.empty_like(q), lse, D**-0.5, True)
        else:
            p = mod.plan(q, k, v, D**-0.5, True)
        grid, block, values = (p["grid"], p["block"], p["values"]) if isinstance(p, dict) else p
        block = block[0] if isinstance(block, tuple) else block
        return [(getattr(mod, "KERNEL", None), grid, block, len(values))]
    p = mod.plan(b, sq, skv, hq, hkv, True, D**-0.5)
    launches = p["launches"] if isinstance(p, dict) else p
    return [(n, g, blk, len(a)) for n, g, blk, a in launches if g is not None]  # grid None: a torch memset


@pytest.mark.parametrize("module", [_FWD, "fwd_v0", "bwd_v0"])
def test_kernel_module_launch_plan(module):
    """A kernel module's ``plan`` for the production shape (no GPU): every kernel it launches is one the
    manifest lists for it, with the manifest's block and argument count, every grid is positive; a shape
    outside the kernels' rules plans no launch."""
    mod = _real_module(module)
    if not callable(getattr(mod, "plan", None)):
        pytest.skip(f"{module} has no plan")
    symbols = _manifest_symbols(module)
    launches = _plan_launches(mod, *PROD)
    assert launches
    for name, grid, block, nargs in launches:
        assert name in symbols, f"{name} is not a manifest kernel of {module} ({sorted(symbols)})"
        assert block == symbols[name]["block"] and all(int(x) >= 1 for x in grid), (name, grid, block)
        assert nargs == len(symbols[name]["args"]), name
    b, sq, skv, hq, hkv = PROD
    with pytest.raises((ValueError, NotImplementedError)):
        _plan_launches(mod, b, sq + 32, skv, hq, hkv)  # Sq % 64


# ---------------------------------------------------------------------------
# Numerics (gfx1250 + HK code objects)
# ---------------------------------------------------------------------------

REQUIRE_ENV = "HK_TEST_REQUIRE"  # fwd,bwd: those directions must run (fail, never skip)
RECORD_ENV = "HK_VALIDATION_RECORD"  # JSON lines file: one line per session in which nothing failed
_REQUIRED = tuple(os.environ.get(REQUIRE_ENV, "").replace(",", " ").split())


def _hk_gpu_reason(direction):
    if not _ON_GFX1250:
        return "needs a gfx1250 card"
    if _FLYDSL_REASON is not None:
        return str(_FLYDSL_REASON)
    reason = impl.hk_unavailable_reason(direction)
    if reason is not None:
        return f"HK {direction} unavailable: {reason}"
    env = impl.FWD_ENV if direction == "fwd" else impl.BWD_ENV
    if impl.hsaco_dirs() and not impl._raw_setting(direction).strip():
        # the default module may not be the kernel under test: lab code objects run on an explicit choice
        return (
            f"lab code objects ({impl.hsaco_dir()}): set {env} explicitly (default {impl.setting(direction)})"
        )
    return None


def _needs_hk(direction):
    """Skip unless HK can run ``direction``; a direction named in HK_TEST_REQUIRE runs (and fails) anyway."""
    reason = _hk_gpu_reason(direction)
    return pytest.mark.skipif(reason is not None and direction not in _REQUIRED, reason=str(reason))


needs_hk_fwd = _needs_hk("fwd")
needs_hk_bwd = _needs_hk("bwd")
# directions whose HK kernels the flash_attn_func tests expect to run
_HK_DIRECTIONS = tuple(d for d in ("fwd", "bwd") if d in _REQUIRED or _hk_gpu_reason(d) is None)
needs_hk_any = pytest.mark.skipif(
    not _HK_DIRECTIONS, reason=f"HK runs neither direction: {_hk_gpu_reason('fwd')}; {_hk_gpu_reason('bwd')}"
)
needs_large = pytest.mark.skipif(
    os.environ.get("PRIMUS_TURBO_TEST_LARGE") != "1",
    reason="Llama-3.1-8B training shape; set PRIMUS_TURBO_TEST_LARGE=1 (about a minute of CPU reference)",
)
MIN_DB = 50.0  # vs the fp32 reference
FLYDSL_MARGIN_DB = 0.5  # HK may trail FlyDSL by at most this

# (b, sq, skv, hq, hkv); toy first: run it alone before anything else on a new build.
_FWD_SHAPES = {
    "toy": (1, 128, 128, 4, 1),
    "gqa4": (2, 256, 256, 8, 2),
    "sq_lt_skv": (1, 256, 512, 8, 2),
    "odd": (3, 192, 384, 12, 3),
    "sq64": (1, 64, 128, 4, 1),
    "gqa8": (1, 256, 256, 16, 2),
    "gqa16": (1, 128, 256, 32, 2),
    "skv_x64": (1, 128, 192, 4, 1),
}
_BWD_SHAPES = {k: _FWD_SHAPES[k] for k in ("toy", "gqa4", "sq_lt_skv", "odd", "sq64")}

# Checks that passed in this session, per direction; the validation record's stages are derived from them:
# toy = the toy reference check, full = every reference shape and the repeat check, large = the production
# shape through flash_attn_func.
_PASSED = {"fwd": set(), "bwd": set()}


def _passed(direction, check):
    _PASSED[direction].add(check)


def _stages(direction):
    shapes = _FWD_SHAPES if direction == "fwd" else _BWD_SHAPES
    done = _PASSED[direction]
    return {
        "toy": "ref:toy" in done,
        "full": {f"ref:{name}" for name in shapes} | {"repeat"} <= done,
        "large": "large" in done,
    }


def _code_objects():
    """{code object: {direction, sha256, path}} of the directions HK can run here."""
    out = {}
    for d in ("fwd", "bwd"):
        if impl.hk_unavailable_reason(d) is not None:
            continue
        for name, path in impl.code_object_files(d).items():
            sha = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            out[name] = {"direction": d, "sha256": sha, "path": str(path)}
    return out


def _record_line(code_objects):
    """One validation record: what ran, on which bytes, and the checks / stages that passed per direction."""
    return {
        "schema": 1,
        "time": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_tag": os.environ.get("RUN_TAG"),
        "settings": {d: impl.setting(d) for d in ("fwd", "bwd")},
        "modules": {d: impl.module_name(d) for d in ("fwd", "bwd")},
        "code_objects": code_objects,
        "checks": {d: sorted(c) for d, c in _PASSED.items()},
        "stages": {d: _stages(d) for d in ("fwd", "bwd")},
    }


@pytest.fixture(scope="module", autouse=True)
def _validation_record(request):
    """With HK_VALIDATION_RECORD set on a gfx1250 card: after this module, if no test of the session failed
    and the code objects are the bytes the session started with, append one JSON line (scripts/hk/e2e_mg.sh
    and tools/hk_gfx1250_attn_pin.py --record read it)."""
    path = os.environ.get(RECORD_ENV)
    before = _code_objects() if path and _ON_GFX1250 else None
    yield
    if before is None or not any(_PASSED.values()) or request.session.testsfailed:
        return
    after = _code_objects()
    msg = f"HK validation record not written: code objects changed during the session ({before} -> {after})"
    if after == before:
        line = _record_line(after)
        with open(path, "a") as f:
            f.write(json.dumps(line, sort_keys=True) + "\n")
        msg = f"HK validation record appended to {path}: stages {line['stages']}"
    capman = request.config.pluginmanager.get_plugin("capturemanager")
    with capman.global_and_fixture_disabled() if capman is not None else contextlib.nullcontext():
        print(f"\n{msg}")


@pytest.fixture
def hk_runs(monkeypatch):
    """Counts the calls of impl.hk_attn_fwd / hk_attn_bwd that returned (the adapter looks them up on impl
    at call time), so a test through flash_attn_func can tell HK from the FlyDSL fallback."""
    runs = {"fwd": 0, "bwd": 0}

    def counted(direction, real):
        def wrapper(*args, **kwargs):
            out = real(*args, **kwargs)
            runs[direction] += 1
            return out

        return wrapper

    monkeypatch.setattr(impl, "hk_attn_fwd", counted("fwd", impl.hk_attn_fwd))
    monkeypatch.setattr(impl, "hk_attn_bwd", counted("bwd", impl.hk_attn_bwd))
    return runs


def _assert_hk_ran(runs, deterministic=False):
    """HK ran exactly the directions it is expected to (deterministic=True: the backward is FlyDSL's)."""
    expected = {d: d in _HK_DIRECTIONS and not (d == "bwd" and deterministic) for d in ("fwd", "bwd")}
    assert {d: runs[d] > 0 for d in runs} == expected, (runs, _HK_DIRECTIONS)
    return [d for d, ran in expected.items() if ran]


@pytest.mark.skipif(not _REQUIRED, reason=f"{REQUIRE_ENV} not set")
def test_required_hk_directions_are_available():
    """HK_TEST_REQUIRE=fwd,bwd: those directions must be able to run (checked first, with the reason)."""
    assert set(_REQUIRED) <= {"fwd", "bwd"}, f"{REQUIRE_ENV}={_REQUIRED}: use fwd and / or bwd"
    for d in _REQUIRED:
        assert _hk_gpu_reason(d) is None, _hk_gpu_reason(d)


def test_validation_records_are_what_the_readers_expect(monkeypatch):
    """The lines this module appends are read by the pin tool's record_reason (also e2e_mg.sh's pre-flight):
    a code object counts as validated once its direction's full and large stages are recorded, over one
    or several sessions on the same bytes."""
    for d in ("fwd", "bwd"):
        monkeypatch.setitem(_PASSED, d, set())
    sha = "ab" * 32
    co = {"k_attn_fwd_v0": {"direction": "fwd", "sha256": sha, "path": "/lab/k_attn_fwd_v0.hsaco"}}
    _passed("fwd", "ref:toy")
    toy = json.loads(json.dumps(_record_line(co)))
    assert toy["stages"] == {
        "fwd": {"toy": True, "full": False, "large": False},
        "bwd": {"toy": False, "full": False, "large": False},
    }
    for name in _FWD_SHAPES:
        _passed("fwd", f"ref:{name}")
    _passed("fwd", "repeat")
    full = json.loads(json.dumps(_record_line(co)))
    assert full["stages"]["fwd"] == {"toy": True, "full": True, "large": False}
    pin = _pin_tool()
    assert "['large'] not recorded" in pin.record_reason([toy, full], "k_attn_fwd_v0", sha)
    monkeypatch.setitem(_PASSED, "fwd", {"large"})  # the LARGE session alone
    large = json.loads(json.dumps(_record_line(co)))
    assert pin.record_reason([toy, full, large], "k_attn_fwd_v0", sha) is None
    assert "no validation record" in pin.record_reason([toy, full, large], "k_attn_fwd_v0", "cd" * 32)
    assert "no validation record" in pin.record_reason([toy, full, large], "k_attn_bwd_v0", sha)


def _inputs(b, sq, skv, hq, hkv, seed=0):
    """q, k, v, dout: bf16 BSHD on the CPU."""
    g = torch.Generator(device="cpu").manual_seed(seed)

    def make(s, h):
        return torch.randn(b, s, h, D, generator=g).to(torch.bfloat16)

    return make(sq, hq), make(skv, hkv), make(skv, hkv), make(sq, hq)


def _reference(q, k, v, dout):
    """fp32 bottom-right causal attention and its gradients on the CPU: out, lse, dq, dk, dv."""
    q, k, v = (t.detach().float().requires_grad_() for t in (q, k, v))
    sq, skv, g = q.shape[1], k.shape[1], q.shape[2] // k.shape[2]
    s = torch.einsum("bqhd,bkhd->bhqk", q, k.repeat_interleave(g, 2)) / math.sqrt(D)
    s = s.masked_fill(torch.ones(sq, skv, dtype=torch.bool).triu(skv - sq + 1), float("-inf"))
    lse = torch.logsumexp(s, -1)
    out = torch.einsum("bhqk,bkhd->bqhd", s.softmax(-1), v.repeat_interleave(g, 2))
    out.backward(dout.float())
    return out.detach(), lse.detach(), q.grad, k.grad, v.grad


def _snr(ref, x):
    x = x.float().cpu()
    assert torch.isfinite(x).all(), "non-finite values"
    return compute_snr(ref, x)


def _flydsl():
    from primus_turbo.pytorch.kernels.attention import (
        attention_flydsl_gfx1250_impl as fly,
    )

    return fly


def _compare(names, hk_out, fly_out, ref):
    for name, h, f, r in zip(names, hk_out, fly_out, ref):
        assert h.shape == r.shape, (name, h.shape, r.shape)
        hk_db, fly_db = _snr(r, h), _snr(r, f)
        assert hk_db >= MIN_DB, f"{name}: {hk_db:.1f} dB vs fp32"
        assert hk_db >= fly_db - FLYDSL_MARGIN_DB, f"{name}: HK {hk_db:.2f} dB, FlyDSL {fly_db:.2f} dB"


@needs_hk_fwd
@pytest.mark.parametrize("name", list(_FWD_SHAPES))
def test_hk_forward_matches_reference(name):
    q, k, v, dout = _inputs(*_FWD_SHAPES[name])
    ref = _reference(q, k, v, dout)
    q, k, v = (t.cuda() for t in (q, k, v))
    hk_out = impl.hk_attn_fwd(q, k, v, None, True)
    fly_out = _flydsl().flash_attn_flydsl_gfx1250_forward_impl(q, k, v, None, True)
    _compare(("out", "lse"), hk_out, fly_out, ref[:2])
    _passed("fwd", f"ref:{name}")


@needs_hk_bwd
@pytest.mark.parametrize("name", list(_BWD_SHAPES))
def test_hk_backward_matches_reference(name):
    """The backward on FlyDSL's forward (o, lse): dq / dk / dv vs the fp32 reference and FlyDSL's backward."""
    q, k, v, dout = _inputs(*_BWD_SHAPES[name])
    ref = _reference(q, k, v, dout)
    q, k, v, dout = (t.cuda() for t in (q, k, v, dout))
    fly = _flydsl()
    o, lse = fly.flash_attn_flydsl_gfx1250_forward_impl(q, k, v, None, True)
    hk_grads = impl.hk_attn_bwd(dout, q, k, v, o, lse, None, True)
    fly_grads = fly.flash_attn_flydsl_gfx1250_backward_impl(dout, q, k, v, o, lse, None, True)
    _compare(("dq", "dk", "dv"), hk_grads, fly_grads, ref[2:])
    _passed("bwd", f"ref:{name}")


@needs_hk_fwd
def test_hk_forward_repeats_bitwise():
    q, k, v, _ = (t.cuda() for t in _inputs(2, 1024, 1024, 32, 8))
    first = impl.hk_attn_fwd(q, k, v, None, True)
    for _ in range(3):
        for a, b in zip(first, impl.hk_attn_fwd(q, k, v, None, True)):
            assert torch.equal(a, b)
    _passed("fwd", "repeat")


@needs_hk_bwd
def test_hk_backward_repeats():
    """dK and dV bitwise; dQ is summed with fp32 atomics in no fixed order, so to rounding only."""
    q, k, v, dout = (t.cuda() for t in _inputs(2, 1024, 1024, 32, 8))
    o, lse = _flydsl().flash_attn_flydsl_gfx1250_forward_impl(q, k, v, None, True)
    first = impl.hk_attn_bwd(dout, q, k, v, o, lse, None, True)
    for _ in range(3):
        dq, dk, dv = impl.hk_attn_bwd(dout, q, k, v, o, lse, None, True)
        assert torch.equal(dk, first[1]) and torch.equal(dv, first[2])
        assert compute_snr(first[0].float().cpu(), dq.float().cpu()) > 70.0
    _passed("bwd", "repeat")


@needs_hk_any
@pytest.mark.parametrize("name", ["toy", "odd"])
@pytest.mark.parametrize("deterministic", [False, True], ids=["hk_bwd", "flydsl_bwd"])
def test_flash_attn_func_hipkittens_end_to_end(name, deterministic, hk_runs):
    """flash_attn_func with HIPKITTENS pinned, sbhd bytes in: matches the reference, and HK ran every
    direction it can (deterministic=True: FlyDSL's backward on HK's forward)."""
    q, k, v, dout = _inputs(*_FWD_SHAPES[name])
    ref = _reference(q, k, v, dout)
    q_, k_, v_ = (t.transpose(0, 1).contiguous().transpose(0, 1).cuda().requires_grad_() for t in (q, k, v))
    with _pinned(BackendType.HIPKITTENS):
        out, lse = flash_attn_func(q_, k_, v_, causal=True, return_lse=True, deterministic=deterministic)
    out.backward(dout.cuda())
    for check, x, r in zip(("out", "lse", "dq", "dk", "dv"), (out, lse, q_.grad, k_.grad, v_.grad), ref):
        assert _snr(r, x) >= MIN_DB, check
    for d in _assert_hk_ran(hk_runs, deterministic):
        _passed(d, f"e2e:{name}{':det' if deterministic else ''}")


@needs_hk_any
@needs_large
def test_llama31_8b_training_shape(hk_runs):
    """b4 s8192 hq32 hkv8 through flash_attn_func (HIPKITTENS): HK ran every direction it can; every output
    finite; batch 0, kv head 0 (q heads 0-3, whose dk / dv depend on nothing else) against the CPU
    reference and FlyDSL."""
    b, s, hq, hkv = 4, 8192, 32, 8
    q, k, v, dout = _inputs(b, s, s, hq, hkv)
    qc, kc, vc = (t.cuda().requires_grad_() for t in (q, k, v))
    with _pinned(BackendType.HIPKITTENS):
        out, lse = flash_attn_func(qc, kc, vc, causal=True, return_lse=True)
    out.backward(dout.cuda())
    ran = _assert_hk_ran(hk_runs)
    got = (out, lse, qc.grad, kc.grad, vc.grad)
    for x in got:
        assert torch.isfinite(x).all()
    fly = _flydsl()
    qf, kf, vf, df = (t.cuda() for t in (q, k, v, dout))
    fo, fl = fly.flash_attn_flydsl_gfx1250_forward_impl(qf, kf, vf, None, True)
    fg = fly.flash_attn_flydsl_gfx1250_backward_impl(df, qf, kf, vf, fo, fl, None, True)
    g = hq // hkv
    ref = _reference(q[:1, :, :g], k[:1, :, :1], v[:1, :, :1], dout[:1, :, :g])

    def sub(t, name):
        return t[:1, :g] if name == "lse" else t[:1, :, :g] if name in ("out", "dq") else t[:1, :, :1]

    names = ("out", "lse", "dq", "dk", "dv")
    _compare(
        names, [sub(x, n) for x, n in zip(got, names)], [sub(x, n) for x, n in zip((fo, fl, *fg), names)], ref
    )
    for d in ran:
        _passed(d, "large")
