###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""FlyDSL kernel-level autotune: the switch, the selection rules, entries and compiles.

Launches are host callables and benches are fakes with fixed per-config times, so every pick
here is deterministic; only the op-level cases at the bottom touch a GPU kernel.
"""

import gc
import weakref

import pytest
import torch

from primus_turbo.flydsl.autotune import (
    Compile,
    NoWorkingConfig,
    Op,
    Race,
    Refine,
    policies,
    staged,
)
from primus_turbo.flydsl.autotune import runtime as rt
from primus_turbo.flydsl.autotune.engine import Entry, Space, Tuner
from primus_turbo.flydsl.autotune.policies import (
    RAISE,
    Bench,
    Check,
    Fastest,
    Finite,
    Point,
    RealArgs,
    Reproducible,
    Stage,
)
from primus_turbo.pytorch.core.backend import GlobalBackendManager


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("PRIMUS_TURBO_AUTO_TUNE", raising=False)
    GlobalBackendManager.set_auto_tune(None)
    rt.force(None)
    rt.clear()
    yield
    GlobalBackendManager.set_auto_tune(None)
    rt.force(None)
    rt.clear()


class FakeBench(Bench):
    """Per-config times; ``launches`` are the configs themselves (see ``ident``)."""

    def __init__(self, times, per_point=None):
        self.t, self.per_point, self.calls = times, per_point, 0

    def times(self, launches, points):
        self.calls += 1
        if self.per_point is not None:
            return [list(self.per_point[l.cfg]) for l in launches]
        return [[self.t[l.cfg]] * len(points) for l in launches]


class Launch:
    def __init__(self, cfg, log=None):
        self.cfg, self.log = cfg, log

    def __call__(self, *args):
        if self.log is not None:
            self.log.append(self.cfg)


def ident(built=None, fail=()):
    def build(cfg):
        if cfg in fail:
            raise RuntimeError(f"cannot build {cfg}")
        if built is not None:
            built.append(cfg)
        return Launch(cfg)

    return build


class RejectSome(Check):
    def __init__(self, bad):
        self.bad = set(bad)

    def check(self, launch, points, ref=None):
        return launch.cfg not in self.bad


def one_point():
    return RealArgs(Point(()))


def space(configs, bench, select, check=None, **kw):
    return Space.of(configs, bench=bench, select=select, check=check, **kw)


@pytest.fixture
def capturing(monkeypatch):
    state = {"on": False}
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: state["on"])
    yield state
    state["on"] = False  # the teardown's runtime.clear() is refused under capture


# ── the switch ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("env, expect", [(None, False), ("0", False), ("1", True)])
def test_switch_follows_env(monkeypatch, env, expect):
    if env is not None:
        monkeypatch.setenv("PRIMUS_TURBO_AUTO_TUNE", env)
    assert rt.enabled() is expect


def test_backend_switch_and_force_override_env(monkeypatch):
    monkeypatch.setenv("PRIMUS_TURBO_AUTO_TUNE", "1")
    GlobalBackendManager.set_auto_tune(False)
    assert rt.enabled() is False
    monkeypatch.setenv("PRIMUS_TURBO_AUTO_TUNE", "0")
    GlobalBackendManager.set_auto_tune(True)
    assert rt.enabled() is True
    rt.force(False)
    assert rt.enabled() is False


# ── Tuner: off, capture, cache, selection ───────────────────────────────────


def test_off_returns_base_without_building():
    built = []
    t = Tuner("test.op")
    res = t.pick("k", space(["a", "b"], FakeBench({}), Fastest()), probe=one_point(), build=ident(built))
    assert res.config == "a" and res.tuned is False and built == [] and t.cache == {}


def test_capture_returns_base_then_tunes(monkeypatch):
    rt.force(True)
    t = Tuner("test.op")
    sp = space(["a", "b"], FakeBench({"a": 2.0, "b": 1.0}), Fastest())
    monkeypatch.setattr(rt, "blocked", lambda: "CUDA-graph capture")
    res = t.pick("k", sp, probe=one_point(), build=ident())
    assert res.config == "a" and res.tuned is None and t.cache == {}
    monkeypatch.setattr(rt, "blocked", lambda: None)
    assert t.pick("k", sp, probe=one_point(), build=ident()).config == "b"


def test_pick_is_cached():
    rt.force(True)
    bench = FakeBench({"a": 3.0, "b": 1.0, "c": 2.0})
    t = Tuner("test.op")
    sp = space(["a", "b", "c"], bench, Fastest())
    res = t.pick("k", sp, probe=one_point(), build=ident())
    assert res.config == "b" and res.tuned and res.launch.cfg == "b" and len(res.built) == 3
    built = []
    hit = t.pick("k", sp, probe=one_point(), build=ident(built))
    assert hit.config == "b" and hit.launch is None and built == [] and bench.calls == 1


@pytest.mark.parametrize(
    "configs, times, select, want",
    [
        (["a", "b", "c"], {"a": 3.0, "b": 1.0, "c": 2.0}, Fastest(), "b"),
        (["a", "b"], {"a": 1.0, "b": 1.0}, Fastest(), "a"),
        (["a", "b"], {"a": 1.05, "b": 1.0}, Fastest(charge=lambda c: 1.1 if c == "b" else 1.0), "a"),
        (["a", "b"], {"a": 1.0, "b": 0.96}, Fastest(margin=0.95), "a"),
        (["a", "b", "c"], {"a": 1.0, "b": 0.96, "c": 0.9}, Fastest(margin=0.95), "c"),
        (["a", "c", "d"], {"a": 1.0, "c": 0.9, "d": 0.88}, Fastest(margin=0.95), "d"),
        (["a", "b"], {"a": [1.0, 1.0], "b": [0.5, 0.99]}, Fastest(margin=0.95), "b"),
        (["a", "b"], {"a": [1.0, 1.0], "b": [0.5, 0.99]}, Fastest(margin=0.95, over="worst"), "a"),
    ],
    ids=[
        "fastest",
        "tie_goes_to_earlier",
        "charge",
        "margin_keeps_the_first",
        "margin_beaten",
        "margin_fastest_challenger",
        "margin_geomean",
        "margin_worst",
    ],
)
def test_selection(configs, times, select, want):
    rt.force(True)
    if isinstance(times[configs[0]], list):
        bench, probe = FakeBench(None, times), RealArgs(*[Point(())] * len(times[configs[0]]))
    else:
        bench, probe = FakeBench(times), one_point()
    assert (
        Tuner("test.op").pick("k", space(configs, bench, select), probe=probe, build=ident()).config == want
    )


def test_failed_build_and_check_leave_the_race():
    rt.force(True)
    sp = space(
        ["a", "b", "c", "d"],
        FakeBench({"a": 3.0, "b": 1.0, "c": 0.5, "d": 2.0}),
        Fastest(),
        RejectSome({"c"}),
    )
    assert Tuner("test.op").pick("k", sp, probe=one_point(), build=ident(fail=("b",))).config == "d"


def test_on_empty():
    rt.force(True)
    sp = space(["a", "b"], FakeBench({}), Fastest(), on_empty="raise")
    with pytest.raises(RuntimeError):
        Tuner("test.op").pick("k", sp, probe=one_point(), build=ident(fail=("a", "b")))
    sp = space(["a", "b"], FakeBench({}), Fastest(), on_empty="base")
    t = Tuner("test.op")
    assert t.pick("k", sp, probe=one_point(), build=ident(fail=("a", "b"))).config == "a"
    assert set(t.errors) == {("k", "a"), ("k", "b")}


def test_first_valid_off_skips_a_base_that_does_not_hold():
    built = []
    t = Tuner("test.op", off="first_valid")
    sp = space(["a", "b", "c", "d"], FakeBench({"c": 2.0, "d": 1.0}), Fastest(), check=RejectSome(["b"]))
    res = t.pick("k", sp, probe=one_point(), build=ident(built, fail=("a",)))
    assert res.config == "c" and res.tuned is False and res.launch.cfg == "c" and built == ["b", "c"]
    assert t.cache == {} and rt.report() == []
    assert t.pick("k", sp, probe=one_point(), build=ident(built)).config == "c" and built == ["b", "c"]
    rt.force(True)
    assert t.pick("k", sp, probe=one_point(), build=ident(fail=("a",))).config == "d"


def test_clear_forgets_picks_entries_and_runs_hooks():
    rt.force(True)
    side = {"x": 1}
    rt.register(clear_hook=side.clear)
    t = Tuner("test.op")
    sp = space(["a", "b"], FakeBench({"a": 2.0, "b": 1.0}), Fastest())
    t.pick("k", sp, probe=one_point(), build=ident(fail=("a",)))
    t.entry("k", lambda: Entry("cfg", None))
    assert t.cache and t.errors and t.entries and rt.report()
    rt.clear()
    assert t.cache == {} and t.errors == {} and t.entries == {} and side == {} and rt.report() == []


# ── stages ──────────────────────────────────────────────────────────────────


def test_stages_carry_the_previous_pick():
    rt.force(True)
    timed = []

    class Recording(FakeBench):
        def times(self, launches, points):
            timed.append([l.cfg for l in launches])
            return super().times(launches, points)

    times = {"a": 3.0, "b": 2.0, "b1": 1.9, "b2": 1.0}
    sp = Space.staged(
        Stage(lambda _p: ["a", "b"], Recording(times), Fastest()),
        Stage(lambda prev: [prev + "1", prev + "2"], Recording(times), Fastest(margin=0.95)),
    )
    # stage 2 leads with stage 1's pick b; b1 is inside the margin, b2 is not.
    assert Tuner("test.op").pick("k", sp, probe=one_point(), build=ident()).config == "b2"
    assert timed == [["a", "b"], ["b", "b1", "b2"]]


def test_reference_stage_takes_the_first_that_builds():
    rt.force(True)
    built = []
    bench = FakeBench({"r1": 9.0, "r2": 9.0, "x": 1.0})
    sp = Space.staged(
        Stage(lambda _p: ["r0", "r1", "r2"], None, check=Finite(n=None)),
        Stage(lambda ref: [ref, "x"], bench, Fastest(margin=0.95)),
        base="r0",
    )
    # r0 does not build: r1 is the reference (r2 is never built), and x then races against it.
    res = Tuner("test.op").pick("k", sp, probe=one_point(), build=ident(built, fail=("r0",)))
    assert res.config == "x" and bench.calls == 1 and built == ["r1", "x"]


# ── failures ────────────────────────────────────────────────────────────────


# A key with no working config raises again without a second search, until clear().
@pytest.mark.parametrize("off", ["base", "first_valid"])
def test_no_working_config_is_remembered_until_clear(off):
    rt.force(off == "base")
    tried = []
    t = Tuner("test.op", off=off)
    sp = Space.staged(Stage(lambda _p: ["r0", "r1"], None, fallback=RAISE), base="r0")

    def build(cfg):
        tried.append(cfg)
        raise RuntimeError(f"cannot build {cfg}")

    for _ in range(2):
        with pytest.raises(NoWorkingConfig):
            t.pick("k", sp, probe=one_point(), build=build)
    assert tried == ["r0", "r1"]
    rt.clear()
    with pytest.raises(NoWorkingConfig):
        t.pick("k", sp, probe=one_point(), build=build)
    assert tried == ["r0", "r1"] * 2


# Candidates that build but all fail on this call's data say nothing about the candidates: the
# search raises without remembering it, and the next call searches again.
def test_data_dependent_failure_is_not_remembered():
    class DataCheck(Check):
        bad = True

        def check(self, launch, points, ref=None):
            return not self.bad

    rt.force(True)
    t, check = Tuner("test.op"), DataCheck()
    sp = space(["a", "b"], FakeBench({"a": 2.0, "b": 1.0}), Fastest(), check=check)
    with pytest.raises(NoWorkingConfig) as ei:
        t.pick("k", sp, probe=one_point(), build=ident())
    assert ei.value.data_dependent and t.cache == {}
    check.bad = False
    res = t.pick("k", sp, probe=one_point(), build=ident())
    assert (res.config, res.tuned, t.cache) == ("b", True, {"k": "b"})


# ── reproducibility ─────────────────────────────────────────────────────────


@pytest.fixture
def host_sync(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)


class Writes:
    """A launch that writes ``value(call)`` over its output; ``calls`` counts the launches."""

    def __init__(self, out, value):
        self.out, self.value, self.calls = out, value, 0

    def __call__(self):
        self.calls += 1
        self.out.fill_(self.value(self.calls))


def _repro_point():
    out = torch.zeros(8)
    return out, Point((), out)


def test_reproducible_keeps_a_candidate_that_repeats(host_sync):
    out, p = _repro_point()
    launch = Writes(out, lambda n: 1.0)
    assert Reproducible(reps=5, flush_mb=1).check(launch, [p])
    # The finite sample, the run whose bytes the repeats compare against, then the repeats.
    assert launch.calls == 1 + 1 + 5
    assert policies._FLUSH[(str(out.device), 1)].numel() == 1 << 20


def test_reproducible_rejects_a_candidate_that_moves(host_sync):
    out, p = _repro_point()
    moves = Writes(out, lambda n: 2.0 if n == 5 else 1.0)
    assert not Reproducible(reps=8, flush_mb=1).check(moves, [p])
    assert moves.calls == 5
    nan = Writes(out, lambda n: float("nan"))
    assert not Reproducible(reps=8, flush_mb=1).check(nan, [p])
    assert nan.calls == 1


def test_reproducible_runs_warm_when_the_flush_does_not_fit(host_sync, monkeypatch):
    tried, warned = [], []

    def oom(device, mb):
        tried.append(mb)
        raise torch.OutOfMemoryError("no room")

    monkeypatch.setattr(policies, "_alloc_flush", oom)
    monkeypatch.setattr(policies.logger, "warning", lambda msg, **kw: warned.append(msg))
    out, p = _repro_point()
    check = Reproducible(reps=4, flush_mb=3)
    assert check.check(Writes(out, lambda n: 1.0), [p])
    assert not check.check(Writes(out, lambda n: float(n == 4)), [p])
    # One attempt per device and size, kept as "warm" until clear().
    assert tried == [3] and policies._FLUSH[(str(out.device), 3)] is None and len(warned) == 1
    rt.clear()
    assert policies._FLUSH == {}


def test_reproducible_is_a_race_check():
    c = Race(FakeBench({}), check="reproducible").check
    assert type(c) is Reproducible and (c.reps, c.flush_mb, c.n) == (64, 512, 1024)
    assert type(Race(FakeBench({}), check="reproducible").checked_by("finite").check) is Finite


class _CfgBench(Bench):
    def __init__(self, time):
        self.time = time

    def times(self, launches, points):
        return [[self.time(l.cfg)] * len(points) for l in launches]


def _mx_fake_pick(tuner, unstable, time, build, keys):
    """The mxfp8 search's ``pick``, run on ``tuner`` with fake timing: a stage held to
    reproducible output rejects the configs ``unstable(cfg)`` names, a finite one keeps all."""

    class Unstable(Check):
        def check(self, launch, points, ref=None):
            return not unstable(launch.cfg)

    def pick(key, space):
        keys.append(key)
        stages = []
        for st in space.stages:
            assert type(st.check) in (Reproducible, Finite)
            check = Unstable() if type(st.check) is Reproducible else st.check
            stages.append(Stage(st.configs, _CfgBench(time), st.select, check, on_empty=st.on_empty))
        return tuner.pick(key, Space(stages, space.base), probe=one_point(), build=build)

    return pick


def _mx_time(cfg):
    # The (256, 8, 8) tile at group_n=8 is the fastest; the non-temporal C store never pays.
    bm, gm, xcd, gn, ks, aux = cfg
    return (1.0 if gm == 8 else 1.2) * {0: 1.0, 4: 0.95, 8: 0.9}[gn] * (1.5 if aux else 1.0)


def _warned_unreproducible(recwarn):
    return any("no reproducible config" in str(w.message) for w in recwarn)


def test_mxfp8_tune_widens_the_band_when_the_seed_width_does_not_reproduce(host_sync, recwarn):
    from primus_turbo.flydsl.gemm import gemm_mxfp8_kernel as mx

    rt.force(True)
    keys = []
    pick = _mx_fake_pick(Tuner("test.mxfp8"), lambda c: c[3] == 0, _mx_time, ident(), keys)
    res = mx._mx_search(pick, ("k",), 1024, 4096, 4096, 1)
    # Stage 1 at every other width, and the band step stays off the seed width it started from.
    assert res.config == (256, 8, 8, 8, 1, 0) and res.tuned
    assert keys == [("k",), ("k", "any-width")] and not _warned_unreproducible(recwarn)


def test_mxfp8_band_step_keeps_the_fallback_band():
    from primus_turbo.flydsl.gemm import gemm_mxfp8_kernel as mx

    band = mx._mx_space([], 4096, 1, mx._MX_RACES, skip_gn=(0,)).stages[1]
    prev = (256, 8, 8, 8, 1, 0)
    assert band.configs(prev) == [prev, (256, 8, 8, 4, 1, 0)]
    assert band.on_empty == "base"


def test_mxfp8_tune_warns_and_races_finite_when_nothing_reproduces(host_sync):
    from primus_turbo.flydsl.gemm import gemm_mxfp8_kernel as mx

    rt.force(True)
    keys = []
    pick = _mx_fake_pick(Tuner("test.mxfp8"), lambda c: True, _mx_time, ident(), keys)
    with pytest.warns(RuntimeWarning, match=r"no reproducible config for M=1024 N=4096 K=4096"):
        res = mx._mx_search(pick, ("k",), 1024, 4096, 4096, 1)
    assert res.config == (256, 8, 8, 8, 1, 0) and res.tuned
    assert keys == [("k",), ("k", "any-width"), ("k", "unchecked")]


def test_mxfp8_tune_falls_back_when_no_wider_band_builds(host_sync):
    from primus_turbo.flydsl.gemm import gemm_mxfp8_kernel as mx

    rt.force(True)
    keys = []

    def build(cfg):
        if cfg[3] != 0:
            raise RuntimeError(f"cannot build {cfg}")
        return Launch(cfg)

    pick = _mx_fake_pick(Tuner("test.mxfp8"), lambda c: True, _mx_time, build, keys)
    with pytest.warns(RuntimeWarning, match="no reproducible config"):
        res = mx._mx_search(pick, ("k",), 1024, 4096, 4096, 1)
    assert res.config == (256, 8, 8, 0, 1, 0)
    assert keys == [("k",), ("k", "any-width"), ("k", "unchecked")]


def test_mxfp8_tune_raises_when_nothing_builds(host_sync, recwarn):
    from primus_turbo.flydsl.gemm import gemm_mxfp8_kernel as mx

    rt.force(True)
    keys = []

    def never(cfg):
        raise RuntimeError(f"cannot build {cfg}")

    pick = _mx_fake_pick(Tuner("test.mxfp8"), lambda c: False, _mx_time, never, keys)
    with pytest.raises(NoWorkingConfig) as ei:
        mx._mx_search(pick, ("k",), 1024, 4096, 4096, 1)
    assert not ei.value.data_dependent and keys == [("k",)] and not _warned_unreproducible(recwarn)


# ── entries ─────────────────────────────────────────────────────────────────


class Jit:
    """A raw launch that counts its calls; ``compile`` calls it once, as flyc.compile does."""

    def __init__(self):
        self.raw_calls, self.compiles, self.compiled_calls = 0, 0, 0

    def __call__(self, *args):
        self.raw_calls += 1

    def compiled(self, *args):
        self.compiled_calls += 1

    def compile(self, raw, args):
        assert raw is self
        self.compiles += 1
        raw(*args)
        return self.compiled, True


@pytest.mark.parametrize("first", ["capture", "eager"])
def test_entry_calls_raw_once_over_capture_and_eager(capturing, first):
    rt.force(False)
    t, jit = Tuner("test.op"), Jit()
    for step in (first, "capture" if first == "eager" else "eager", first):
        capturing["on"] = step == "capture"
        cfg = t.pick(
            "k", space(["a", "b"], FakeBench({}), Fastest()), probe=one_point(), build=ident()
        ).config
        t.entry("k", lambda c=cfg: Entry(c, jit, tuned=False)).run((), jit.compile)
    assert (jit.raw_calls, jit.compiles, jit.compiled_calls) == (1, 1, 2)


def test_entry_built_by_the_search_never_compiles(capturing):
    rt.force(True)
    t, jit, log = Tuner("test.op"), Jit(), []
    sp = space(["a", "b"], FakeBench({"a": 2.0, "b": 1.0}), Fastest())
    res = t.pick("k", sp, probe=one_point(), build=lambda c: Launch(c, log))
    t.entry("k", lambda: Entry(res.config, jit, compiled=res.launch))
    for on in (True, False, True):
        capturing["on"] = on
        t.entry("k", lambda: pytest.fail("cached entry remade")).run((), jit.compile)
    assert (log, jit.raw_calls, jit.compiles) == (["b"] * 3, 0, 0)


def test_untuned_entry_is_replaced_once_enabled():
    rt.force(False)
    t, made = Tuner("test.op"), []

    def make(tuned):
        made.append(tuned)
        return Entry("cfg", None, tuned=tuned)

    t.entry("k", lambda: make(False))
    assert t.hit("k") is t.entry("k", lambda: make(False)) and made == [False]
    rt.force(True)
    assert t.hit("k") is None
    e = t.entry("k", lambda: make(True))
    assert e.tuned and t.hit("k") is e and made == [False, True]
    rt.force(False)
    assert t.entry("k", lambda: make(False)) is e


# A first call under capture takes base provisionally; every captured call after it reuses that
# entry instead of building it again, and the first eager call tunes.
def test_capture_reuses_a_provisional_entry(capturing):
    rt.force(True)
    t, made = Tuner("test.op"), []
    sp = space(["a", "b"], FakeBench({"a": 2.0, "b": 1.0}), Fastest())

    def make():
        res = t.pick("k", sp, probe=one_point(), build=ident())
        made.append((res.config, res.tuned))
        return Entry(res.config, None, tuned=res.tuned)

    capturing["on"] = True
    e = t.entry("k", make)
    assert t.entry("k", make) is e and made == [("a", None)]
    capturing["on"] = False
    assert t.entry("k", make).config == "b" and made == [("a", None), ("b", True)]
    capturing["on"] = True
    assert t.entry("k", make).config == "b" and len(made) == 2


def test_derived_follows_its_base_entry():
    rt.force(False)
    t, made = Tuner("test.op"), []

    def acc(e):
        made.append(e)
        return Entry(e.config, "acc")

    base = t.entry("k", lambda: Entry("cfg", None, tuned=False))
    d = t.derived("k", "acc", base, acc)
    assert t.derived("k", "acc", base, acc) is d and made == [base]
    rt.force(True)
    retuned = t.entry("k", lambda: Entry("cfg", None))
    d2 = t.derived("k", "acc", retuned, acc)
    assert d2 is not d and made[-1] is retuned and t.derived("k", "acc", retuned, acc) is d2


# A graph captured on a provisional (or untuned) entry and its derived one keeps launching their
# code after an eager call replaces them, so both stay alive until clear().
@pytest.mark.parametrize("first", ["capture", "off"])
def test_replaced_entries_stay_alive_until_clear(capturing, first):
    rt.force(first == "capture")
    capturing["on"] = first == "capture"
    t = Tuner("test.op")
    sp = space(["a", "b"], FakeBench({"a": 2.0, "b": 1.0}), Fastest())

    def make():
        res = t.pick("k", sp, probe=one_point(), build=ident())
        return Entry(res.config, Launch(res.config), tuned=res.tuned)

    def acc(e):
        return Entry(e.config, Launch(e.config + "+acc"), tuned=e.tuned)

    old = t.entry("k", make)
    refs = [weakref.ref(old.raw), weakref.ref(t.derived("k", True, old, acc).raw)]
    del old
    capturing["on"] = False
    rt.force(True)
    new = t.entry("k", make)
    assert new.config == "b" and t.derived("k", True, new, acc).raw.cfg == "b+acc"
    gc.collect()
    assert all(r() is not None for r in refs)
    rt.clear()
    gc.collect()
    assert all(r() is None for r in refs)


# ── Compile and Op ──────────────────────────────────────────────────────────


class _FakeJit(Jit):
    pass


@pytest.fixture
def fake_flyc(monkeypatch):
    """``Compile`` on host launches: ``_FakeJit`` stands in for ``@flyc.jit`` and flyc.compile
    calls it once on the arguments it is handed."""
    from primus_turbo.flydsl.autotune import op as op_mod

    compiled = []

    def compile(launch, *args):
        launch(*args)
        compiled.append((launch, args))
        return launch.compiled

    monkeypatch.setattr(op_mod, "JitFunction", _FakeJit)
    monkeypatch.setattr(op_mod.flyc, "compile", compile)
    monkeypatch.setattr(op_mod.flyc, "from_torch_tensor", lambda t: ("static", tuple(t.shape)))
    return compiled


def test_compile_modes(fake_flyc):
    x = torch.zeros(4)
    jit = _FakeJit()
    assert Compile()(jit, (x, 0)) == (jit.compiled, True)
    assert Compile()(jit, (x, 0)) == (jit.compiled, True) and len(fake_flyc) == 2
    once = Compile(once=True)
    assert once(jit, (x, 0)) == (jit.compiled, True)
    assert once(jit, (torch.ones(4), 1)) == (jit.compiled, False) and len(fake_flyc) == 3
    assert once(jit, (torch.ones(8), 1))[1] and len(fake_flyc) == 4
    run, ran = Compile(raw=True, hints={"h": 1})(jit, ())
    assert not ran and jit.raw_calls == 4
    run()
    assert jit.raw_calls == 5 and len(fake_flyc) == 4


def test_static_compile_race_and_entry_share_one_compile(fake_flyc):
    out = torch.full((4,), 7.0)
    args = (torch.zeros(4), torch.zeros(4), out, 0)
    jit, static = _FakeJit(), Compile(static_layout=True)
    timed = static.race(jit, args)
    ((_launch, cargs),) = fake_flyc
    assert timed == jit.compiled and cargs[2] == ("static", (4,)) and out.eq(7.0).all()
    assert static(jit, args) == (jit.compiled, False) and len(fake_flyc) == 1


@pytest.mark.parametrize("enabled", [False, True])
def test_op_entry_races_on_a_miss_only(enabled):
    rt.force(enabled)
    bench = FakeBench({"a": 2.0, "b": 1.0, "c": 3.0})
    o, race, made, picks = Op("test.op"), Race(bench), [], []

    def entry_of(cfg, tuned=True):
        made.append(cfg)
        return Entry(cfg, Launch(cfg), tuned=tuned)

    def pick():
        picks.append(1)
        build = o.build(entry_of)
        sp = staged(Refine(lambda _p: ["a", "b", "c"], race), base="a")
        return build.winner(o.pick("k", sp, probe=one_point(), build=build))

    e = o.entry("k", pick)
    assert e.config == ("b" if enabled else "a") and e.tuned is enabled
    assert o.entry("k", pick) is e and len(picks) == 1
    assert made == (["a", "b", "c"] if enabled else ["a"]) and bench.calls == int(enabled)


# ── jit ops: flydsl's own autotuner, guarded for capture ────────────────────


def _jit_fn(tmp_path, monkeypatch, times, calls):
    from primus_turbo.flydsl.autotune.autotuner import Config, autotune

    monkeypatch.setenv("FLYDSL_AUTOTUNE_CACHE_DIR", str(tmp_path))

    def bench(f, warmup, rep):
        f()
        return times[calls[-1]]

    @autotune([Config(v=1), Config(v=2), Config(v=3)], key=["n"], do_bench=bench)
    def fn(n, v):
        calls.append(v)

    return fn


def test_jit_capture_runs_the_first_config_and_caches_nothing(tmp_path, monkeypatch, capturing):
    calls = []
    fn = _jit_fn(tmp_path, monkeypatch, {1: 3.0, 2: 1.0, 3: 2.0}, calls)
    capturing["on"] = True
    fn(8)
    assert calls == [1] and fn.cache == {} and not (tmp_path / "fn.json").exists()
    capturing["on"] = False
    fn(8)
    assert calls == [1, 1, 2, 3, 2] and len(fn.cache) == 1 and (tmp_path / "fn.json").exists()
    capturing["on"] = True
    fn(8)
    assert calls[-1] == 2


def test_jit_switch_does_not_gate_tuning(tmp_path, monkeypatch):
    rt.force(False)
    calls = []
    fn = _jit_fn(tmp_path, monkeypatch, {1: 3.0, 2: 1.0, 3: 2.0}, calls)
    fn(8)
    assert calls[-1] == 2 and rt.report() == []


# ── op level ────────────────────────────────────────────────────────────────


requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _fp8_gemm(M, N, K, layout="NT", out=None):
    from primus_turbo.flydsl.gemm.gemm_fp8_kernel import gemm_fp8_tensorwise_flydsl_kernel

    trans_a, trans_b = layout[0] == "T", layout[1] == "T"
    a = torch.randn((K, M) if trans_a else (M, K), device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn((N, K) if trans_b else (K, N), device="cuda").to(torch.float8_e4m3fn)
    s = torch.ones(1, device="cuda")
    am = a.float().t() if trans_a else a.float()
    bm = b.float().t() if trans_b else b.float()
    ref = (am @ bm).to(torch.bfloat16)
    beta = 0.0 if out is None else 1.0

    def run():
        return gemm_fp8_tensorwise_flydsl_kernel(
            a, s, b, s, trans_a=trans_a, trans_b=trans_b, beta=beta, out=out
        )

    return run, ref


def _fp8_wgrad(G=2, m=512, N=512, K=512):
    from primus_turbo.flydsl.grouped_gemm.grouped_gemm_fp8_kernel import (
        grouped_gemm_fp8_variable_k_tensorwise_flydsl_kernel,
    )

    g = torch.Generator(device="cuda").manual_seed(5)
    dy = torch.randn(G * m, N, device="cuda", generator=g).to(torch.float8_e4m3fn)
    x = torch.randn(G * m, K, device="cuda", generator=g).to(torch.float8_e4m3fn)
    s = torch.ones(1, device="cuda")
    go = torch.arange(0, G * m + 1, m, device="cuda", dtype=torch.int64)
    ref = torch.stack(
        [dy[i * m : (i + 1) * m].float().t() @ x[i * m : (i + 1) * m].float() for i in range(G)]
    )
    return lambda: grouped_gemm_fp8_variable_k_tensorwise_flydsl_kernel(dy, x, s, s, go), ref.bfloat16()


def _grouped_fp8(G=4, m=256, N=512, K=512):
    from primus_turbo.flydsl.grouped_gemm.grouped_gemm_fp8_kernel import (
        grouped_gemm_fp8_tensorwise_flydsl_kernel,
    )

    g = torch.Generator(device="cuda").manual_seed(7)
    a = torch.randn(G * m, K, device="cuda", generator=g).to(torch.float8_e4m3fn)
    b = torch.randn(G, N, K, device="cuda", generator=g).to(torch.float8_e4m3fn)
    s = torch.ones(1, device="cuda")
    go = torch.arange(0, G * m + 1, m, device="cuda", dtype=torch.int64)
    return lambda: grouped_gemm_fp8_tensorwise_flydsl_kernel(a, b, s, s, go, trans_b=True)


def _mxfp8_gemm(M=1024, N=1024, K=2048):
    from primus_turbo.flydsl.gemm.gemm_mxfp8_kernel import gemm_mxfp8_flydsl_kernel
    from primus_turbo.pytorch.core.low_precision import check_mxfp8_support

    ok, why = check_mxfp8_support()
    if not ok:
        pytest.skip(why)
    g = torch.Generator(device="cuda").manual_seed(9)
    a = torch.randn(M, K, device="cuda", generator=g).to(torch.float8_e4m3fn)
    b = torch.randn(N, K, device="cuda", generator=g).to(torch.float8_e4m3fn)
    sa = torch.full((M, K // 32), 127, device="cuda", dtype=torch.uint8)
    sb = torch.full((N, K // 32), 127, device="cuda", dtype=torch.uint8)
    return lambda: gemm_mxfp8_flydsl_kernel(a, sa, b, sb)


def _mxfp4_gemm(M=2048, N=2048, K=4096):
    from primus_turbo.flydsl.gemm.gemm_mxfp4_kernel import gemm_mxfp4_flydsl_kernel

    g = torch.Generator(device="cuda").manual_seed(11)
    a, b = (torch.randint(0, 256, (d, K // 2), device="cuda", dtype=torch.uint8, generator=g) for d in (M, N))
    sa, sb = (
        torch.randint(124, 130, (d, K // 32), device="cuda", dtype=torch.uint8, generator=g) for d in (M, N)
    )
    return lambda: gemm_mxfp4_flydsl_kernel(a, sa, b, sb)


@pytest.fixture
def raw_calls(monkeypatch):
    """Top-level calls per ``@flyc.jit`` launch; flyc.compile makes one itself."""
    from flydsl._mlir import ir
    from flydsl.compiler.jit_function import JitFunction

    calls = {}
    # Holds every counted launch: one freed mid-test (a clear() drops it) could hand its id
    # to a new launch, which would then look like a second raw call.
    counted_launches = []
    call = JitFunction.__call__

    def counted(self, *args, **kwargs):
        if ir.Context.current is None:
            if id(self) not in calls:
                counted_launches.append(self)
            calls[id(self)] = calls.get(id(self), 0) + 1
        return call(self, *args, **kwargs)

    monkeypatch.setattr(JitFunction, "__call__", counted)
    return calls


def _check(out, ref):
    torch.testing.assert_close(out.float(), ref.float(), atol=1.0, rtol=2e-2)


def _captured(run):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        with torch.cuda.graph(g, stream=s):
            out = run()
    torch.cuda.current_stream().wait_stream(s)
    g.replay()
    torch.cuda.synchronize()
    return out


# NT 1024^2 takes an 8-wave base; NT 4096^3 and TN 1024x512 a whole-loop one, whose launch
# must never be compiled twice.
OP_CASES = [
    ((1024, 1024, 2048), "NT", "gemm_fp8_tensorwise.nt"),
    ((4096, 4096, 4096), "NT", "gemm_fp8_tensorwise.nt"),
    ((1024, 512, 1024), "TN", "gemm_fp8_tensorwise.tn4"),
]


@requires_gpu
@pytest.mark.parametrize("shape, layout, op", OP_CASES)
def test_op_untuned_entry_is_tuned_once_enabled(shape, layout, op):
    run, ref = _fp8_gemm(*shape, layout)
    _check(run(), ref)
    _check(run(), ref)
    assert rt.report() == []
    rt.force(True)
    _check(run(), ref)
    _check(run(), ref)
    assert [r["op"] for r in rt.report()] == [op]


@requires_gpu
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("shape, layout, op", OP_CASES)
def test_op_capture_takes_base(shape, layout, op, enabled):
    rt.force(enabled)
    run, ref = _fp8_gemm(*shape, layout)
    _check(_captured(run), ref)
    assert rt.report() == []
    _check(run(), ref)
    assert [r["op"] for r in rt.report()] == ([op] if enabled else [])


# A launch is called raw by its compile only: every later run, eager or captured, takes the
# compiled object.
@requires_gpu
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("shape, layout, op", OP_CASES)
def test_op_launch_is_called_raw_once(shape, layout, op, enabled, raw_calls):
    rt.force(enabled)
    run, ref = _fp8_gemm(*shape, layout)
    for step in (_captured, lambda r: r(), _captured, lambda r: r()):
        _check(step(run), ref)
    assert all(n == 1 for n in raw_calls.values()), sorted(raw_calls.values())


# Once an op has run a shape, no later call -- eager or captured -- calls a launch raw.
@requires_gpu
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    "make", [_grouped_fp8, lambda: _fp8_wgrad()[0], _mxfp8_gemm], ids=["np", "wgrad", "mxfp8"]
)
def test_op_steady_state_never_calls_raw(make, enabled, raw_calls):
    rt.force(enabled)
    run = make()
    run()
    torch.cuda.synchronize()
    raw_calls.clear()
    for step in (lambda r: r(), _captured, lambda r: r()):
        step(run)
    torch.cuda.synchronize()
    assert raw_calls == {}


@requires_gpu
@pytest.mark.parametrize("enabled", [False, True])
def test_op_accumulate_folds_in_once(enabled):
    rt.force(enabled)
    out = torch.zeros(4096, 4096, device="cuda", dtype=torch.bfloat16)
    run, ref = _fp8_gemm(4096, 4096, 4096, "NT", out=out)
    for n, step in enumerate((lambda r: r(), _captured, lambda r: r()), 1):
        step(run)
        torch.cuda.synchronize()
        torch.testing.assert_close(out.float(), n * ref.float(), atol=n * 1.0, rtol=2e-2)


# A clear() drops the compiles with the picks, so the re-tune compiles each launch again: a
# second raw call, and with a fresh flydsl cache a second trace.
@requires_gpu
@pytest.mark.parametrize("shape, layout, op", OP_CASES)
def test_op_retunes_after_clear(shape, layout, op, raw_calls, monkeypatch, tmp_path):
    from primus_turbo.flydsl.autotune import op as op_mod

    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path))
    rt.force(True)
    run, ref = _fp8_gemm(*shape, layout)
    _check(run(), ref)
    rt.clear()
    assert op_mod._COMPILED_ONCE == {}
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path / "retrace"))
    _check(_captured(run), ref)
    _check(run(), ref)
    assert [r["op"] for r in rt.report()] == [op]
    # The TN whole-loop factory is not cached, so its re-tune builds fresh launches instead.
    again = 1 if op == "gemm_fp8_tensorwise.tn4" else 2
    assert raw_calls and max(raw_calls.values()) == again, sorted(raw_calls.values())


# flydsl snapshots every global a launch reads (a dict, list or set by value) on its first call,
# rejects a later raw call once that value changed, and folds the snapshot into the disk-cache
# key. A mutable one would break re-tunes and tie the key to whatever the process traced before,
# so no launch may read one; the whole-loop launches below must be among those checked.
_WHOLE_LOOP_LAUNCHES = {"launch_dense_wave4", "launch_dense_tn_wave4", "launch_grouped_tn_wgrad_4wave"}


@requires_gpu
def test_no_launch_reads_a_mutable_global():
    from flydsl.compiler.jit_function import JitFunction

    from primus_turbo.pytorch.core.low_precision import check_mxfp4_support

    rt.force(True)
    for run, ref in (_fp8_gemm(4096, 4096, 4096, "NT"), _fp8_gemm(1024, 512, 1024, "TN"), _fp8_wgrad()):
        _check(run(), ref)
    want = set(_WHOLE_LOOP_LAUNCHES)
    if check_mxfp4_support()[0]:
        _mxfp4_gemm()()
        want.add("launch_mxfp4_fused")
    torch.cuda.synchronize()
    # type(), not isinstance(): gc also hands back weakref proxies, and a dead one raises on __class__.
    launches = [o for o in gc.get_objects() if issubclass(type(o), JitFunction)]
    assert want <= {o.func.__name__ for o in launches}
    mutable = sorted(
        f"{o.func.__qualname__} reads {mod}.{name}"
        for o in launches
        for name, mod, g in o._get_global_refs(None)
        if isinstance(g.get(name), (dict, list, set, bytearray))
    )
    assert mutable == []
