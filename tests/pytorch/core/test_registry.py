###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``primus_turbo.arch`` (architecture names), ``primus_turbo.registry`` (the kernel registry) and
``primus_turbo.capabilities``. Host-only: the same on every GPU."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import primus_turbo
from primus_turbo import arch, capabilities, registry

pytestmark = pytest.mark.gfx1250

_PKG_DIR = Path(primus_turbo.__file__).resolve().parent

_ENTRY = """{
        "name": "demo.flydsl.x",
        "summary": "a kernel",
        "arch": ("gfx1250",),
        "backend": "flydsl",
        "dtype": ("bf16",),
        "kernel": "primus_turbo.flydsl.demo.demo_kernel_gfx1250",
        "tests": ("tests/test_demo_gfx1250.py",),
    }"""


def _package(tmp_path, registry_source: str, modules=()) -> Path:
    """A package directory with one registry file (``demo/demo_registry.py``) and empty modules."""
    pkg = tmp_path / "primus_turbo"
    (pkg / "demo").mkdir(parents=True)
    (pkg / "demo" / "demo_registry.py").write_text(registry_source)
    for module in modules:
        path = pkg.joinpath(*module.split(".")[1:]).with_suffix(".py")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
    return pkg


@pytest.mark.parametrize(
    "name, expected",
    [
        ("gfx1250", "gfx1250"),
        ("gfx1250:sramecc+:xnack-", "gfx1250"),
        ("GFX950:sramecc+", "gfx950"),
        ("", ""),
        (None, ""),
    ],
)
def test_normalize_arch(name, expected):
    assert arch.normalize_arch(name) == expected


def test_device_arch_is_a_bare_gfx_id():
    # "" where this process has no ROCm device; on a device, an ID without target features.
    name = arch.device_arch()
    assert name == arch.normalize_arch(name)
    assert name == "" or name.startswith("gfx")


def test_every_architecture_has_products():
    assert set(arch.PRODUCTS) == set(arch.KNOWN_ARCHS)


def test_backends_are_the_backend_types():
    from primus_turbo.pytorch.core.backend import BackendType

    assert set(registry.BACKENDS) == {b.name.lower() for b in BackendType}


def test_the_registry_of_this_package_reads_clean():
    entries, problems = registry.load()
    assert not problems, problems
    assert entries and {k.op for k in entries} >= {"attention", "grouped_gemm", "fused_mega_moe", "deep_ep"}
    for entry in entries:
        # not imported (a kernel package may refuse to import off its architecture): the file is there
        for module in entry.modules():
            assert registry.module_file(module) is not None, f"{entry.name}: no module {module}"
        assert entry.file.endswith(f"/{entry.op}{registry.FILE_SUFFIX}") and entry.line > 0


def test_kernels_filters():
    assert registry.kernels(arch="gfx1250:sramecc+:xnack-") == registry.kernels(arch="gfx1250")
    assert registry.kernels(arch="gfx000") == []
    names = [k.name for k in registry.kernels(arch="gfx1250", op="attention")]
    assert "attention.flydsl.mla" in names and "mega_moe.bf16" not in names
    (mla,) = registry.kernels(name="attention.flydsl.mla")
    assert mla.op == "attention" and mla.backend == "flydsl"


def test_owners():
    def names(*paths):
        return [k.name for k in registry.owners(paths)]

    assert names("primus_turbo/flydsl/attention/gfx1250_mla/fwd_kernel.py") == ["attention.flydsl.mla"]
    assert names("primus_turbo/flydsl/grouped_gemm/grouped_gemm_bf16_kernel_gfx1250.py") == [
        "grouped_gemm.flydsl.bf16"
    ]
    assert names("csrc/kernels/attention/hipkittens/gfx1250_mla/src/k_delta.hip") == [
        "attention.hipkittens.mla"
    ]
    # a shared frontend that holds the gate of an entry belongs to it: a change there runs its tests
    assert names("primus_turbo/pytorch/kernels/grouped_gemm/grouped_gemm_impl.py") == [
        "grouped_gemm.flydsl.bf16"
    ]
    assert names("primus_turbo/pytorch/kernels/attention/attention_impl.py") == [
        "attention.flydsl.mha",
        "attention.flydsl.mla",
    ]
    assert names("primus_turbo/flydsl/deep_ep/notify_kernel.py") == ["deep_ep.flydsl"]
    assert names("primus_turbo/lib/libprimus_turbo_kernels.so") == []
    assert names("primus_turbo/lib/hk_gfx1250_mla/k_delta.hsaco") == ["attention.hipkittens.mla"]
    assert names("tests/pytorch/ops/test_mega_bf16_gfx1250_gemm.py") == ["mega_moe.bf16"]
    assert names("tests/pytorch/ops/test_attention.py") == ["attention.flydsl.sparse_mla"]
    # a module claim covers that module only, not what merely starts with its name
    assert names("primus_turbo/flydsl/attention/gfx1250_mla_other/x.py", "README.md") == []


def test_has():
    assert capabilities.has("mega_moe.bf16", arch="gfx1250")
    assert capabilities.has("mega_moe.bf16", arch="gfx1250:sramecc+:xnack-")
    assert not capabilities.has("mega_moe.bf16", arch="gfx000")
    assert not capabilities.has("mega_moe.bf16", arch="gfx950")
    assert not capabilities.has("no.such.feature", arch="gfx1250")
    # a sub-feature: what an entry says it provides
    assert capabilities.has("grouped_gemm.flydsl.bf16.wgrad_accum", arch="gfx1250")
    assert not capabilities.has("grouped_gemm.flydsl.bf16.no_such_part", arch="gfx1250")


def test_features():
    table = capabilities.features("gfx1250")
    entries = registry.kernels(arch="gfx1250")
    assert set(table) == {name for k in entries for name in k.feature_names()}
    # every feature whose modules are part of the source is there; one that also needs build products
    # (the HipKittens code objects) is there exactly when they were built
    for entry in entries:
        built = all(any(_PKG_DIR.glob(pattern)) for pattern in entry.products)
        for name in entry.feature_names():
            assert table[name] == built, name
    assert capabilities.features("gfx000") == {}


def test_the_names_primus_asks_for():
    """Primus (primus/core/utils/turbo_capability.py, the MI455X launch scripts) asks for these by
    name; renaming an entry switches the feature off there without an error."""
    known = set(capabilities.features("gfx1250"))
    assert known >= {
        "attention.flydsl.mha",
        "attention.flydsl.mla",
        "attention.flydsl.sparse_mla",
        "attention.hipkittens.mla",
        "mega_moe.bf16",
        "grouped_gemm.flydsl.bf16.wgrad_accum",
        "deep_ep.flydsl",
    }


def test_what_older_primus_releases_probe_is_still_there():
    """Primus releases that predate primus_turbo.capabilities look for two directories and import one
    function from one module path. Whoever moves one of them switches a feature off there without an
    error (keep an alias or a stub for a release)."""
    flydsl = _PKG_DIR / "flydsl"
    assert (flydsl / "attention" / "gfx1250").is_dir()
    assert (flydsl / "mega" / "gfx1250").is_dir()
    old = flydsl / "grouped_gemm" / "grouped_gemm_bf16_kernel_gfx1250.py"
    assert old.is_file() and "grouped_gemm_bf16_variable_k_accum_supported" in old.read_text()


def test_a_missing_module_or_product_reads_absent(tmp_path):
    pkg = _package(tmp_path, f"KERNELS = ({_ENTRY},)")
    (entry,), problems = registry.load(pkg)
    assert not problems
    assert registry.find("demo.flydsl.x", "gfx1250", pkg)[0] == entry
    assert registry.find("demo.flydsl.x", "gfx950", pkg) is None
    assert not registry.present(entry, pkg)
    pkg = _package(tmp_path / "with_module", f"KERNELS = ({_ENTRY},)", modules=[entry.kernel])
    assert registry.present(entry, pkg)
    needs_build = _ENTRY.replace('"tests"', '"products": ("lib/demo/*.hsaco",),\n        "tests"')
    pkg = _package(tmp_path / "with_products", f"KERNELS = ({needs_build},)", modules=[entry.kernel])
    (entry,), problems = registry.load(pkg)
    assert not problems and not registry.present(entry, pkg)
    (pkg / "lib" / "demo").mkdir(parents=True)
    (pkg / "lib" / "demo" / "k.hsaco").write_text("")
    assert registry.present(entry, pkg)


def _one(old: str = "", new: str = "") -> str:
    """A registry file with the demo entry, ``old`` replaced by ``new`` in it."""
    return f"KERNELS = ({_ENTRY.replace(old, new) if old else _ENTRY},)"


@pytest.mark.parametrize(
    "source, expected",
    [
        ("import os\nKERNELS = ()", "data only"),
        ("KERNELS = ()\nOTHER = 1", "data only"),
        ("KERNELS = tuple()", "data only"),
        ("KERNELS = (make(),)", "an entry is a literal"),
        ("KERNELS = ({['a']: 1},)", "an entry is a literal"),
        ("KERNELS = ({[1, 2]},)", "an entry is a literal"),
        ("3\nKERNELS = ()", "data only"),
        (
            _one('"summary": "a kernel",', '"summary": "a kernel", "summary": "again",'),
            "field 'summary' is given twice",
        ),
        (_one('"tests"', '"products": ("/abs/*.hsaco",), "tests"'), "a pattern relative to the package"),
        (_one('"tests"', '"products": ("lib/../../x",), "tests"'), "a pattern relative to the package"),
        ("KERNELS = ('x',)", "an entry is a dict literal"),
        (f"KERNELS = ({_ENTRY}, {_ENTRY})", "registered twice for gfx1250"),
        (_one("gfx1250", "gfx1251"), "arch 'gfx1251' is not one of"),
        (_one("flydsl", "cuda"), "backend 'cuda' is not one of"),
        (_one("bf16", "bf17"), "dtype 'bf17' is not one of"),
        (_one('("bf16",)', '"bf16"'), "'dtype' must be a tuple"),
        (_one("demo.flydsl.x", "Demo X"), "lower-case words joined by '.'"),
        (_one("summary", "sumary"), "unknown field 'sumary'"),
        (_one('("tests/test_demo_gfx1250.py",)', "()"), "'tests' must be a tuple"),
        (_one('("gfx1250",)', '"gfx1250"'), "'arch' must be a tuple"),
        (_one("primus_turbo.flydsl", "other_pkg.flydsl"), "is not a module of"),
        (_one('"tests"', '"gates": ("no_colon",), "tests"'), "write '<module of primus_turbo>:<symbol>'"),
        (_one('"tests"', '"provides": {"part": "not a symbol"}, "tests"'), "'provides' must be a dict"),
        (
            "KERNELS = ({'name': 'a.b', 'summary': 's', 'arch': ('gfx1250',), 'backend': 'flydsl', "
            "'dtype': ('bf16',), 'tests': ('t.py',)},)",
            "an entry names its code",
        ),
    ],
)
def test_what_is_wrong_in_a_registry_file_is_reported(tmp_path, source, expected):
    _, problems = registry.load(_package(tmp_path, source))
    assert any(expected in p.message for p in problems), [p.message for p in problems]
    assert all(p.file == "primus_turbo/demo/demo_registry.py" for p in problems)


def test_an_entry_with_a_problem_answers_no_query(tmp_path):
    source = (
        f"KERNELS = ({_ENTRY}, {_ENTRY.replace('demo.flydsl.x', 'demo.flydsl.y').replace('bf16', 'bf17')})"
    )
    pkg = _package(tmp_path, source)
    entries, problems = registry.load(pkg)
    assert [k.valid for k in entries] == [True, False] and len(problems) == 1
    assert [k.name for k in registry.kernels(pkg_dir=pkg)] == ["demo.flydsl.x"]
    assert registry.find("demo.flydsl.y", "gfx1250", pkg) is None


def test_products_own_what_their_pattern_matches(tmp_path):
    needs_build = _ENTRY.replace('"tests"', '"products": ("lib/demo/*.hsaco", "lib/*.so"), "tests"')
    (entry,), problems = registry.load(_package(tmp_path, f"KERNELS = ({needs_build},)"))
    assert not problems
    assert entry.owns("primus_turbo/lib/demo/k.hsaco") and entry.owns("primus_turbo/lib/libdemo.so")
    assert not entry.owns("primus_turbo/lib/other/k.hsaco") and not entry.owns("primus_turbo/lib/demo/k.txt")
    assert len({entry, entry}) == 1  # usable in a set


def test_an_entry_without_dtype(tmp_path):
    # an operation that computes in no format of its own (a dispatch / combine kernel) leaves dtype out
    (entry,), problems = registry.load(_package(tmp_path, _one('"dtype": ("bf16",),', "")))
    assert not problems and entry.dtype == ()


def test_the_same_feature_per_architecture(tmp_path):
    other = _ENTRY.replace('("gfx1250",)', '("gfx950",)').replace("_gfx1250", "_gfx950")
    entries, problems = registry.load(_package(tmp_path, f'"""doc"""\nKERNELS = ({_ENTRY}, {other})'))
    assert not problems and [k.arch for k in entries] == [("gfx1250",), ("gfx950",)]


def test_default_arch_is_the_current_device(monkeypatch):
    monkeypatch.setattr(registry, "device_arch", lambda device=None: "gfx1250")
    assert capabilities.has("mega_moe.bf16")
    assert capabilities.features()["mega_moe.bf16"]
    monkeypatch.setattr(registry, "device_arch", lambda device=None: "")
    assert not capabilities.has("mega_moe.bf16")
    assert capabilities.features() == {}


def test_capabilities_cli(capsys):
    assert capabilities._main(["--arch", "gfx1250", "mega_moe.bf16", "attention.flydsl.mha"]) == 0
    assert capsys.readouterr().out.split() == ["True", "True"]
    assert capabilities._main(["--arch", "gfx1250", "mega_moe.bf16", "no.such.feature"]) == 1
    assert capsys.readouterr().out.split() == ["True", "False"]
    assert capabilities._main(["--arch", "gfx1250"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines == [f"{feature} {answer}" for feature, answer in capabilities.features("gfx1250").items()]


def test_registry_cli(capsys):
    assert registry._main(["--arch", "gfx1250", "--op", "fused_mega_moe"]) == 0
    assert capsys.readouterr().out.split()[:3] == ["mega_moe.bf16", "gfx1250", "flydsl"]
    changed = ["./primus_turbo/flydsl/mega/gfx1250/prims.py", "tests/pytorch/ops/test_grouped_gemm.py"]
    assert registry._main(["--owning", *changed, "--tests"]) == 0
    tests = capsys.readouterr().out.split()
    assert "tests/pytorch/ops/test_mega_bf16_gfx1250_prims.py" in tests
    assert "tests/pytorch/ops/test_grouped_gemm_flydsl_gfx1250_keys.py" in tests
    assert len(tests) == len(set(tests)) and not any("attention" in t for t in tests)
    assert registry._main(["--name", "mega_moe.bf16", "--benchmarks"]) == 0
    assert capsys.readouterr().out.split() == ["benchmark/ops/training/bench_mega_moe_gfx1250.py"]
    # nothing owns these: said on stderr with exit code 1, so `pytest $(... --tests)` does not run everything
    assert registry._main(["--owning", "README.md", "docs/env.md", "--tests"]) == 1
    captured = capsys.readouterr()
    assert captured.out == "" and "no registry entry owns" in captured.err
    assert registry._main(["--name", "attention.flydsl.mla", "--json"]) == 0
    (entry,) = json.loads(capsys.readouterr().out)
    assert entry["kernel"] == "primus_turbo.flydsl.attention.gfx1250_mla" and entry["op"] == "attention"


def test_the_query_imports_neither_torch_nor_a_kernel():
    # A launch script asks before anything else is set up; it must stay cheap and unable to fail
    # on a kernel module's import.
    code = (
        "import sys; from primus_turbo import capabilities; "
        "assert capabilities.has('mega_moe.bf16', arch='gfx1250'); "
        "bad = [m for m in sys.modules if m == 'torch' or m.startswith(('flydsl', 'primus_turbo.flydsl', "
        "'primus_turbo.pytorch'))]; "
        "assert not bad, bad"
    )
    root = os.path.dirname(_PKG_DIR)
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=root)
    assert run.returncode == 0, run.stderr
