###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``tools/conventions``: every rule fires on what it is for and stays quiet otherwise, the
baseline only shrinks, and this repository passes. Host-only: the same on every GPU."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.gfx1250

_REPO = Path(__file__).resolve().parents[3]
_TOOL = _REPO / "tools" / "conventions"
if not _TOOL.is_dir():
    pytest.skip("no tools/conventions next to the tests", allow_module_level=True)
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.conventions import core  # noqa: E402

core.load_rules()

_REGISTRY = "primus_turbo/pytorch/kernels/demo/demo_registry.py"
_KERNEL = "primus_turbo/flydsl/demo/demo_bf16_kernel_gfx1250.py"
_TEST = "tests/pytorch/ops/test_demo_gfx1250.py"

_ENTRY = """KERNELS = (
    {
        "name": "demo.flydsl.bf16",
        "summary": "a kernel",
        "arch": ("gfx1250",),
        "backend": "flydsl",
        "dtype": ("bf16",),
        "kernel": "primus_turbo.flydsl.demo.demo_bf16_kernel_gfx1250",
        "gates": ("primus_turbo.flydsl.demo.demo_bf16_kernel_gfx1250:demo_supported",),
        "provides": {"accum": "demo_accum_supported"},
        "tests": ("tests/pytorch/ops/test_demo_gfx1250.py",),
    },
)
"""

# A tree that follows every rule; each case below breaks one thing in it.
_GOOD = {
    _REGISTRY: _ENTRY,
    _KERNEL: "def demo_supported():\n    return True\n\n\ndemo_accum_supported = demo_supported\n",
    _TEST: "def test_demo():\n    pass\n",
    "primus_turbo/flydsl/demo/__init__.py": "",
}


def _check(tmp_path, changes=None, rules=None, baseline=()):
    """Findings of a tree made of ``_GOOD`` with ``changes`` ({path: text, or None to drop it})."""
    files = {**_GOOD, **(changes or {})}
    for path, text in files.items():
        if text is not None:
            (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / path).write_text(text)
    return core.run(core.Tree(tmp_path), rules or sorted(core.RULES), list(baseline))


def _messages(result):
    return [f"{f.rule} {f.path}: {f.message}" for f in result.new]


def _only(result, rule_id, *fragments):
    """Exactly one finding, of that rule, whose text holds every fragment."""
    messages = _messages(result)
    assert len(messages) == 1 and messages[0].startswith(rule_id + " "), messages
    for fragment in fragments:
        assert fragment in messages[0], messages[0]


def test_a_tree_that_follows_the_rules_passes(tmp_path):
    result = _check(tmp_path)
    assert result.ok and not result.known, _messages(result)


def test_arch001_tagged_code_must_be_registered(tmp_path):
    stray = {
        "primus_turbo/flydsl/demo/gfx1250_new/a_kernel.py": "",
        "primus_turbo/flydsl/demo/gfx1250_new/b_kernel.py": "",
    }
    result = _check(tmp_path, stray, rules=["ARCH001"])
    _only(
        result,
        "ARCH001",
        "primus_turbo/flydsl/demo/gfx1250_new (and 1 more file(s) below it)",
        "no kernel registry",
    )
    # csrc counts, build products do not
    built = {"csrc/kernels/demo/k_gfx1250.cu": "", "primus_turbo/lib/hk_gfx1250_x/k.hsaco": ""}
    result = _check(tmp_path / "csrc", built)
    _only(result, "ARCH001", "csrc/kernels/demo/k_gfx1250.cu")


def test_arch001_an_entry_for_another_architecture_does_not_own_it(tmp_path):
    result = _check(tmp_path, {_REGISTRY: _ENTRY.replace('("gfx1250",)', '("gfx950",)')}, rules=["ARCH001"])
    _only(result, "ARCH001", _KERNEL)


def test_arch002_the_tag_is_a_whole_segment(tmp_path):
    result = _check(
        tmp_path, {"tests/pytorch/ops/test_demo_gfx1250x.py": "", "csrc/kernels/Gfx1250/k.cu": ""}
    )
    assert sorted(m.split(":")[0] for m in _messages(result)) == [
        "ARCH002 csrc/kernels/Gfx1250/k.cu",
        "ARCH002 tests/pytorch/ops/test_demo_gfx1250x.py",
    ]


def test_arch003_no_product_name(tmp_path):
    result = _check(tmp_path, {"benchmark/ops/bench_demo_mi455x.py": ""})
    _only(result, "ARCH003", "benchmark/ops/bench_demo_mi455x.py", "gfx1250")


def test_reg001_a_registry_file_is_data_named_after_its_directory(tmp_path):
    result = _check(tmp_path, {_REGISTRY: "import os\n" + _ENTRY}, rules=["REG001"])
    _only(result, "REG001", _REGISTRY, "data only")
    result = _check(
        tmp_path, {_REGISTRY: _ENTRY, "primus_turbo/pytorch/kernels/demo/other_registry.py": "KERNELS = ()"}
    )
    _only(result, "REG001", "other_registry.py", "primus_turbo/pytorch/kernels/demo/demo_registry.py")


@pytest.mark.parametrize(
    "changes, fragment",
    [
        ({_KERNEL: None, "primus_turbo/flydsl/demo/demo_bf16_kernel.py": ""}, "'kernel': no module"),
        ({_KERNEL: "demo_accum_supported = 1\n"}, "does not define demo_supported"),
        ({_KERNEL: "def demo_supported():\n    pass\n"}, "'provides' 'accum'"),
        ({_TEST: None, "tests/pytorch/ops/test_demo_gfx1250_b.py": ""}, "'tests': no file"),
        (
            {_REGISTRY: _ENTRY.replace('_gfx1250.py",)', '_gfx1250.py::test_other",)')},
            "does not define test_other",
        ),
        (
            {_REGISTRY: _ENTRY.replace('"tests"', '"sources": ("csrc/kernels/demo/gfx1250",), "tests"')},
            "'sources': no path",
        ),
        (
            {
                _REGISTRY: _ENTRY.replace(
                    '"tests"', '"benchmarks": ("benchmark/bench_demo_gfx1250.py",), "tests"'
                )
            },
            "'benchmarks': no file",
        ),
    ],
)
def test_reg002_what_an_entry_names_exists(tmp_path, changes, fragment):
    result = _check(tmp_path, changes, rules=["REG002"])
    _only(result, "REG002", _REGISTRY, "'demo.flydsl.bf16'", fragment)


def test_reg003_listed_tests_run_on_the_architecture(tmp_path):
    shared = "tests/pytorch/ops/test_demo.py"
    listed = {_REGISTRY: _ENTRY.replace('"tests": (', f'"tests": ("{shared}", ')}
    one = {_REGISTRY: _ENTRY.replace('"tests": (', f'"tests": ("{shared}::test_x", ')}
    plain = "def test_x():\n    pass\n"
    other_marked = (
        "import pytest\n\n\n@pytest.mark.gfx1250\ndef test_y():\n    pass\n\n\n"
        "def test_x():\n    pass  # not pytest.mark.gfx1250\n"
    )
    test_marked = "import pytest\n\n\n@pytest.mark.gfx1250  # host-only\ndef test_x():\n    pass\n"
    file_marked = "import pytest\n\npytestmark = [pytest.mark.gfx1250, pytest.mark.slow]\n\n\n" + plain
    # the whole file is listed: the file has to opt in, a mark on one of its tests is not enough
    for i, source in enumerate((plain, other_marked, test_marked)):
        _only(
            _check(tmp_path / f"file{i}", {**listed, shared: source}), "REG003", shared, "the file as a whole"
        )
    assert _check(tmp_path / "file_ok", {**listed, shared: file_marked}).ok
    # one test is listed: that test, or the file, has to opt in; a mark on another test or in a comment is not it
    for i, source in enumerate((plain, other_marked)):
        _only(
            _check(tmp_path / f"one{i}", {**one, shared: source}),
            "REG003",
            f"{shared}::test_x",
            "skipped on gfx1250",
        )
    assert _check(tmp_path / "one_ok", {**one, shared: test_marked}).ok
    assert _check(tmp_path / "one_file_ok", {**one, shared: file_marked}).ok


def test_reg004_tagged_tests_and_benchmarks_are_listed(tmp_path):
    result = _check(
        tmp_path,
        {
            "tests/pytorch/ops/test_other_gfx1250.py": "",
            "benchmark/ops/bench_demo_gfx1250.py": "",
            # not tests: a package file and a helper module
            "tests/gfx1250/__init__.py": "",
            "tests/pytorch/ops/demo_gfx1250_utils.py": "",
        },
    )
    assert _messages(result) == [
        "REG004 benchmark/ops/bench_demo_gfx1250.py: carries the gfx1250 tag but no kernel registry entry for "
        "gfx1250 lists it under 'benchmarks'",
        "REG004 tests/pytorch/ops/test_other_gfx1250.py: carries the gfx1250 tag but no kernel registry entry for "
        "gfx1250 lists it under 'tests'",
    ]


def test_name001_a_kernel_file_ends_with_its_role(tmp_path):
    files = {
        "primus_turbo/flydsl/demo/demo_fwd.py": "",  # no role
        "primus_turbo/triton/loose.py": "",  # at the root of a layer
        "primus_turbo/flydsl/demo/demo_bf16_kernel_v2.py": "",  # something after the role that is no architecture
        # fine: a role, a role and an architecture tag, a kernel package, a shared directory
        "primus_turbo/pytorch/kernels/demo/demo_fp8_impl.py": "",
        "primus_turbo/triton/demo/demo_heuristic_gfx950.py": "",
        "primus_turbo/flydsl/demo/pkg/anything.py": "",
        "primus_turbo/flydsl/utils/anything.py": "",
    }
    files["primus_turbo/flydsl/demo/gfx950.py"] = ""  # nothing but an architecture tag
    result = _check(tmp_path, files, rules=["NAME001"])
    assert sorted(f.path for f in result.new) == [
        "primus_turbo/flydsl/demo/demo_bf16_kernel_v2.py",
        "primus_turbo/flydsl/demo/demo_fwd.py",
        "primus_turbo/flydsl/demo/gfx950.py",
        "primus_turbo/triton/loose.py",
    ]


def test_reg002_names_that_are_defined_in_other_ways(tmp_path):
    kernel = "demo_supported, demo_accum_supported = make()\n"
    test = "import pytest\n\n\n@pytest.mark.parametrize('n', [1, 2])\ndef test_demo(n):\n    pass\n"
    listed = _ENTRY.replace('_gfx1250.py",)', '_gfx1250.py::test_demo[2]",)')
    assert _check(tmp_path, {_KERNEL: kernel, _TEST: test, _REGISTRY: listed}).ok


def test_a_tree_walked_without_git_keeps_its_source_directories(tmp_path):
    files = {
        "tests/pytorch/dist/test_new_gfx1250.py": "",  # dist/ below the top is source
        "build/lib/primus_turbo/flydsl/demo/gfx1250_stale/k.py": "",  # build output at the top is not
        "primus_turbo/flydsl/demo/__pycache__/x_gfx1250.cpython-312.pyc": "",
    }
    _only(_check(tmp_path, files), "REG004", "tests/pytorch/dist/test_new_gfx1250.py")


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_tracked_leaves_out_what_git_does_not_track(tmp_path):
    _check(tmp_path)
    git = ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    wip = tmp_path / "tests" / "pytorch" / "ops" / "test_wip_gfx1250.py"
    wip.write_text("")
    # by default a new file is checked before it is staged; the commit hook and CI look at what git tracks
    run = _tool("--root", str(tmp_path))
    assert run.returncode == 1 and "test_wip_gfx1250.py: REG004" in run.stdout
    assert _tool("--root", str(tmp_path), "--tracked").returncode == 0
    subprocess.run([*git, "add", "-A"], check=True)
    assert _tool("--root", str(tmp_path), "--tracked").returncode == 1


def test_the_baseline_only_shrinks(tmp_path):
    old = "primus_turbo/flydsl/demo/demo_fwd.py"
    result = _check(tmp_path, {old: ""}, baseline=[("NAME001", old)])
    assert result.ok and [f.path for f in result.known] == [old]
    # a listed file that was fixed (here: gone) must leave the list
    result = _check(tmp_path / "fixed", baseline=[("NAME001", old)])
    assert not result.ok and not result.new and result.stale == [("NAME001", old)]
    # only a rule declared with baseline=True can be listed: the line neither hides the finding nor may stay
    stray = "primus_turbo/flydsl/demo/gfx1250_new/a_kernel.py"
    result = _check(tmp_path / "strict", {stray: ""}, baseline=[("ARCH001", stray), ("NOPE001", "x.py")])
    assert [f.rule for f in result.new] == ["ARCH001"]
    assert result.stale == [("ARCH001", stray), ("NOPE001", "x.py")]
    # a baseline line of a rule that was not selected is left alone
    assert _check(tmp_path / "subset", rules=["ARCH001"], baseline=[("NAME001", old)]).ok


def _tool(*args):
    return subprocess.run([sys.executable, str(_TOOL), *args], capture_output=True, text=True)


def test_cli_on_a_tree(tmp_path):
    old = "primus_turbo/flydsl/demo/demo_fwd.py"
    _check(tmp_path, {old: ""})
    run = _tool("--root", str(tmp_path))
    assert run.returncode == 1 and f"{old}: NAME001 " in run.stdout and "--explain NAME001" in run.stdout
    run = _tool("--root", str(tmp_path), "--format", "json")
    report = json.loads(run.stdout)
    assert run.returncode == 1 and not report["ok"] and [f["path"] for f in report["new"]] == [old]
    run = _tool("--root", str(tmp_path), "--format", "github")
    assert f"::error file={old},title=NAME001 " in run.stdout
    # the baseline is not extended by --update-baseline alone
    assert _tool("--root", str(tmp_path), "--update-baseline").returncode == 1
    baseline = tmp_path / "tools" / "conventions" / "baseline.txt"
    assert core.read_baseline(baseline) == []
    assert _tool("--root", str(tmp_path), "--update-baseline", "--accept-new").returncode == 0
    assert core.read_baseline(baseline) == [("NAME001", old)]
    assert _tool("--root", str(tmp_path)).returncode == 0
    # fixed: the line is stale until --update-baseline takes it out
    (tmp_path / old).unlink()
    run = _tool("--root", str(tmp_path))
    assert run.returncode == 1 and "no longer breaks NAME001" in run.stdout
    assert _tool("--root", str(tmp_path), "--update-baseline").returncode == 0
    assert core.read_baseline(baseline) == []


def test_cli_rules():
    run = _tool("--list-rules")
    listed = [line.split()[0] for line in run.stdout.splitlines()]
    assert run.returncode == 0 and listed == sorted(core.RULES)
    for rule_id, chosen in core.RULES.items():
        run = _tool("--explain", rule_id)
        # every rule says why it exists and how to fix a finding
        assert run.returncode == 0 and "Why:" in run.stdout and "Fix:" in run.stdout, rule_id
        assert chosen.title in run.stdout
    assert _tool("--explain", "NOPE001").returncode == 2
    assert _tool("--accept-new").returncode == 2  # only with --update-baseline


def test_a_directory_that_is_not_a_source_tree_is_refused(tmp_path):
    (tmp_path / "tests").mkdir()
    run = _tool("--root", str(tmp_path))
    assert run.returncode == 2 and "no primus_turbo/" in run.stderr


def test_this_repository_follows_its_conventions():
    if not (_REPO / "primus_turbo" / "registry.py").is_file():
        # the release job tests the installed package with the source directory removed
        pytest.skip("no primus_turbo/ source next to the tests")
    run = _tool("--tracked")
    assert run.returncode == 0, run.stdout + run.stderr
