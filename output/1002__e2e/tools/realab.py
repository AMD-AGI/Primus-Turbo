#!/usr/bin/env python3
"""realab -- op-level A/B of the gfx1250 attention champions vs aiter ASM on REAL training q/k/v, fwd AND bwd,
ONE condition per process (output/1002__e2e, 2026-10-02, A0 heliosr-1b114-c07-1, container fa-repro).
Run it through tools/realab.sh (lock, KFD check, fresh JIT cache, dmesg), never bare on the card.

Inputs, all prod shape b4 s8192 hq32 hkv8 d128 bf16, causal, BSHD:
  L00 L01 L02 L08 L16 L31  /home/lihuzhan/_prof_dump/qkv_call0{672,673,674,680,688,703}.pt: q/k/v of Llama-3.1-8B
                           (32L) layers 0/1/2/8/16/31 at training step 43, [B,S,H,D] bf16 (the dump's lse is unused)
  randn                    q, k, v, dO = the first four draws of a cuda Generator seeded 0, i.e. bit-identical to
                           make_inputs("prod", seed=0) of both op-evolve jobs (their ruler-1 input)
  bwd                      o/lse = the ASM forward on the same q/k/v (a bf16 kernel on the card, not an fp32
                           reference); dO = the randn set's dO (x RA_DO_SCALE) for EVERY set: fixed and seeded

Arms = the callables of the Llama e2e (output/0927__b0/e2e/attn_backends/e2e_attn/arms.py), so this times exactly
what the e2e runs (E2E_FLY_TREES is set by this script):
  fwd  asm     aiter fmha_fwd_with_sink_asm
       fly     FlyDSL r16 (= r13ns)                 RA_FWD  default e2e/arms/fwd_r16
  bwd  asm     hand-launched aiter ASM bwd (dkdv_heads=q, scratch reuse) + host GQA sum
       s6      FlyDSL s6, re-pinned flydsl 0.3.4.1  RA_S6   default e2e/arms/bwd_s6_0341
       r29     FlyDSL r29, re-pinned 0.3.4.1        RA_R29  default e2e/arms/bwd_r29_0341
  A/A          RA_FWD_AA / RA_S6_AA -> labels fly_aa / s6_aa (byte copies in another directory = noise floor)
  RA_FWD_ARMS / RA_BWD_ARMS = JSON {label: {"path": dir, "env": {K: V}}} replace the FlyDSL arm lists; "env" is
  applied while that tree is IMPORTED (e.g. {"FLY_BWD_SIDE_STREAM": "0"}, read at import) and restored after.
  Every FlyDSL tree must pin flydsl 0.3.4.1 (one flydsl per process).

Conditions (RA_COND, one per process):
  blk   the op-evolve ruler (job benchmark.py; fwd-nospec/tools/ab.py AB_COND=blk): per set and direction,
        palindromic rounds over arms (even count), a round = RA_LEAD (4) untimed + RA_BLOCK (9) timed SAME-arm
        calls back to back, 256 MB L2 flush before every call outside the events. Defaults: fwd 108 / bwd 54
        timed calls per arm per set (RA_BLK_ITERS_FWD / _BWD).
  gb    training operating point (ab.py AB_COND=gb): before EVERY timed call a burst of RA_GB_NG (10) bf16
        32768x4096x14336 F.linear with the IMAGE hipBLASLt library (~25 ms, sclk ~1300 MHz), then the call with
        no sync in between; palindromic rounds of RA_GB_N (5) calls per arm, RA_GB_ROUNDS (4) rounds.
        hipBLASLt is initialised with the image library BEFORE any tree's _env.py runs (the fwd tree re-points
        HIPBLASLT_TENSILE_LIBPATH at the host library); the env is ASSIGNED again after the trees are loaded, and
        the burst time is checked (> RA_GB_MAXMS ms = wrong library -> ruler void, exit 4).
  corr  first launch + correctness pass only.

Per process: arms -> image BLAS env restored -> sets -> first launch of every arm (randn) -> correctness pass on
every set (NaN-poisoned allocator; finite outputs; FlyDSL vs ASM SQNR; s6 vs r29; A/A bitwise) -> continuous
warm-up per arm (RA_WARM s) -> timed loop (RA_SETS order) -> optional kineto pass (RA_KINETO=1, ranges
P::<kind>_<set>::<arm>::<rep> for output/0927__b0/profile/tools/opana.py) -> RESULT/RATIO lines, SUMMARY table,
RA_JSON (rewritten after every set). An arm with a non-finite output on any set is dropped from timing (exit 3).

sclk witness: hwmon freq1_input of the gfx1250 card (card1 on A0; RA_SCLK_FILE overrides), sampled every 1 ms by
a thread while timing. Per call: sclk = median inside the call's GPU window [end - ms, end], sclk_pre = the 3 ms
before it (gb: the end of the burst), sclk_post = one read right after; pre_ms = flush (blk) / burst (gb) time.

Card rules kept here: no fp32 reference anywhere (only arm-vs-arm comparisons); one shape per process; every arm
loaded once and kept for the whole process (job benchmark.py r15); no autotune; no GEMM before the image hipBLASLt
env is asserted; ARCH/FLYDSL_GPU_ARCH=gfx1250; a fresh FLYDSL_RUNTIME_CACHE_DIR (made here if unset). Sharing one
JIT disk cache between the trees of this process is safe: s6_0341 and r29_0341 share only the launch_delta key and
that code object is identical (output/1002__e2e/isa/jitkey_r29_vs_s6.log); A/A copies share keys by construction.
s6_0341 runs the same code objects as the card-proven 0.3.2 s6 (isa/isa_compare.txt; realab.sh re-checks the ISA).

  RA_MOCK=1   CPU-only logic test: no torch.cuda, no flydsl/aiter, sleeping fake arms (RA_MOCK_NAN=<arm> makes
              that arm non-finite on L16, RA_MOCK_REALDUMPS=1 parses the real dump files on the CPU first).
  --sum a.json [b.json ...]   stdlib-only summary of finished runs (host python3, no torch).
"""
import bisect
import contextlib
import hashlib
import json
import math
import os
import socket
import statistics as st
import sys
import threading
import time
from pathlib import Path

PT = Path("/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo")
E2E = PT / "output" / "0927__b0" / "e2e"
OUT = PT / "output" / "1002__e2e"
IMAGE_BLAS = "/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/gfx1250"
PROD = (4, 8192, 32, 8, 128)                            # b, s, hq, hkv, d
FLOP_PROD = {"fwd": 2.199292e12, "bwd": 5.498229e12}    # HANDOFF-A0 section 1 (causal)
REAL_CALLS = (672, 673, 674, 680, 688, 703)             # call % 32 = layer 0, 1, 2, 8, 16, 31
KEYFILES = ("kernels.py", "impl.py", "_env.py", "flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py",
            "flydsl_fwd/fmha_fwd_prefill_a16w16_m32x2.py")
KINETO_KIND = {("blk", "fwd"): ("blk", 13), ("blk", "bwd"): ("bblk", 7),   # opana drops the first 4 / 2
               ("gb", "fwd"): ("gb", 5), ("gb", "bwd"): ("bgb", 5)}
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def _f(k, d):
    return float(os.environ.get(k, d))


def _i(k, d):
    return int(os.environ.get(k, d))


def geomean(xs):
    xs = [x for x in xs if x and x > 0 and math.isfinite(x)]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


# ------------------------------------------------------------------------------------------------ config
def arm_specs():
    fwd = {"fly": {"path": os.environ.get("RA_FWD", str(E2E / "arms" / "fwd_r16"))}}
    if os.environ.get("RA_FWD_AA"):
        fwd["fly_aa"] = {"path": os.environ["RA_FWD_AA"]}
    bwd = {"s6": {"path": os.environ.get("RA_S6", str(E2E / "arms" / "bwd_s6_0341"))},
           "r29": {"path": os.environ.get("RA_R29", str(E2E / "arms" / "bwd_r29_0341"))}}
    if os.environ.get("RA_S6_AA"):
        bwd["s6_aa"] = {"path": os.environ["RA_S6_AA"]}
    if os.environ.get("RA_FWD_ARMS"):
        fwd = json.loads(os.environ["RA_FWD_ARMS"])
    if os.environ.get("RA_BWD_ARMS"):
        bwd = json.loads(os.environ["RA_BWD_ARMS"])
    for lb in list(fwd) + list(bwd):
        if lb == "asm" or not lb or any(c in lb for c in ":,/ "):
            raise SystemExit(f"bad arm label {lb!r}")
    return fwd, bwd


def arm_order(var, specs):
    default = ["asm"] + list(specs)
    want = [a for a in os.environ.get(var, "").split(",") if a]
    o = [a for a in want if a in default] or list(default)       # tolerate absent optional arms
    o = list(dict.fromkeys(o))
    return o + [a for a in default if a not in o]


class Cfg:
    def __init__(self):
        self.mock = os.environ.get("RA_MOCK", "0") == "1"
        self.cond = os.environ.get("RA_COND", "blk")
        if self.cond not in ("blk", "gb", "corr"):
            raise SystemExit(f"RA_COND={self.cond!r}: blk | gb | corr")
        stamp = time.strftime("%m%d_%H%M%S")
        self.json = os.environ.get("RA_JSON") or str(OUT / "runs" / f"realab_{self.cond}_{stamp}.json")
        self.trace = os.environ.get("RA_TRACE") or str(OUT / "traces" / f"realab_{self.cond}_{stamp}.trace.json")
        self.kineto = os.environ.get("RA_KINETO", "1") == "1" and self.cond != "corr"
        self.warm = _f("RA_WARM", "3")
        self.block, self.lead = _i("RA_BLOCK", "9"), _i("RA_LEAD", "4")
        self.iters = {"fwd": _i("RA_BLK_ITERS_FWD", "108"), "bwd": _i("RA_BLK_ITERS_BWD", "54")}
        self.gb_n, self.gb_rounds, self.gb_ng = _i("RA_GB_N", "5"), _i("RA_GB_ROUNDS", "4"), _i("RA_GB_NG", "10")
        self.gb_maxms = _f("RA_GB_MAXMS", "80")
        self.kin_reps = _i("RA_KINETO_REPS", "2")
        self.sqnr_warn = _f("RA_SQNR_WARN", "40")
        self.do_scale = _f("RA_DO_SCALE", "1.0")
        self.deadline = _f("RA_DEADLINE_S", "1e9")
        self.sets = [s for s in os.environ.get("RA_SETS", "").split(",") if s]
        self.dirs = [d for d in os.environ.get("RA_DIRS", "fwd,bwd").split(",") if d in ("fwd", "bwd")]
        self.fwd_arms, self.bwd_arms = arm_specs()
        if "fwd" not in self.dirs:
            self.fwd_arms = {}
        if "bwd" not in self.dirs:
            self.bwd_arms = {}
        self.order = {"fwd": arm_order("RA_ORDER_FWD", self.fwd_arms),
                      "bwd": arm_order("RA_ORDER_BWD", self.bwd_arms)}
        self.shape = PROD if not self.mock else (1, 128, 8, 2, 32)
        self.scale = self.shape[4] ** -0.5

    def flop(self, d):
        if self.shape == PROD:
            return FLOP_PROD[d]
        b, s, hq, _, dd = self.shape
        return 2.0 * b * hq * s * s * dd * (2.5 if d == "bwd" else 1.0)


# ------------------------------------------------------------------------------------------------ sclk
def find_sclk_file():
    p = os.environ.get("RA_SCLK_FILE")
    if p:
        return p
    import glob
    for c in sorted(glob.glob("/sys/class/drm/card*/device/hwmon/hwmon*/freq1_input")):
        if os.path.exists(c.split("/hwmon/")[0] + "/pp_dpm_sclk"):
            return c
    for c in sorted(glob.glob("/sys/class/drm/card*/device/pp_dpm_sclk")):
        return c
    return None


class Sclk:
    """MHz of the card's shader clock. hwmon freq1_input (Hz, the value tools/clksamp.sh logs) or pp_dpm_sclk."""

    def __init__(self, path, period=0.001):
        self.path, self.period = path, period
        self.T, self.V = [], []            # appended V first, then T: a reader that snapshots len(T) is safe
        self.on = False
        self.fd = None
        if path:
            try:
                self.fd = os.open(path, os.O_RDONLY)
            except OSError:
                self.fd = None
        if self.fd is not None:
            threading.Thread(target=self._loop, daemon=True).start()

    @staticmethod
    def _parse(raw):
        s = raw.decode(errors="replace")
        if "Mhz" in s:                     # pp_dpm_sclk: "1: 2355Mhz *"
            for line in s.splitlines():
                if line.rstrip().endswith("*"):
                    return int(line.split(":")[1].strip().split("Mhz")[0])
            return -1
        v = int(s.split()[0])
        return v // 1000000 if v > 100000 else v

    def read(self):
        if self.fd is None:
            return -1
        try:
            return self._parse(os.pread(self.fd, 256, 0))
        except (OSError, ValueError, IndexError):
            return -1

    def _loop(self):
        try:
            fd = os.open(self.path, os.O_RDONLY)
        except OSError:
            return
        while True:
            if self.on:
                try:
                    v = self._parse(os.pread(fd, 256, 0))
                    self.V.append(v)
                    self.T.append(time.perf_counter())
                except (OSError, ValueError, IndexError):
                    pass
            time.sleep(self.period)

    def window(self, a, b):
        n = len(self.T)
        i = bisect.bisect_left(self.T, a, 0, n)
        j = bisect.bisect_right(self.T, b, 0, n)
        return [v for v in self.V[i:j] if v > 0]


# ------------------------------------------------------------------------------------------------ backends
class Card:
    mock = False

    def __init__(self, torch):
        self.torch, self.dev = torch, "cuda"

    def sync(self):
        self.torch.cuda.synchronize()

    def event(self):
        return self.torch.cuda.Event(enable_timing=True)


class MockEvent:
    t = 0.0

    def record(self):
        self.t = time.perf_counter()

    def synchronize(self):
        pass

    def elapsed_time(self, end):
        return (end.t - self.t) * 1e3


class Mock:
    mock = True

    def __init__(self, torch):
        self.torch, self.dev = torch, "cpu"

    def sync(self):
        pass

    def event(self):
        return MockEvent()


@contextlib.contextmanager
def env_overrides(kv):
    old = {k: os.environ.get(k) for k in kv}
    os.environ.update({k: str(v) for k, v in kv.items()})
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def tree_info(path):
    p = Path(path).resolve()
    h = hashlib.md5()
    for f in sorted(p.rglob("*.py")):
        if "__pycache__" in f.parts:
            continue
        h.update(str(f.relative_to(p)).encode())
        h.update(f.read_bytes())
    files = {kf: hashlib.md5((p / kf).read_bytes()).hexdigest() for kf in KEYFILES if (p / kf).is_file()}
    return {"path": str(p), "md5_tree": h.hexdigest()[:8], "files": files}


def load_arms(cfg):
    """The e2e callables. Every tree is imported here, once, and kept for the whole process."""
    for lb, sp in list(cfg.fwd_arms.items()) + list(cfg.bwd_arms.items()):
        p = Path(sp["path"])
        if not (p / "impl.py").is_file():
            raise SystemExit(f"arm {lb}: no impl.py under {p}")
        if "flydsl0341" not in (p / "_env.py").read_text():
            raise SystemExit(f"arm {lb}: {p}/_env.py does not pin flydsl 0.3.4.1 (one flydsl per process)")
    trees = {}
    for lb, sp in cfg.fwd_arms.items():
        trees.setdefault(lb, {})["fwd"] = str(Path(sp["path"]).resolve())
    for lb, sp in cfg.bwd_arms.items():
        trees.setdefault(lb, {})["bwd"] = str(Path(sp["path"]).resolve())
    os.environ["E2E_FLY_TREES"] = json.dumps(trees)
    sys.path.insert(0, str(E2E / "attn_backends"))
    from e2e_attn import arms as A
    A.FLY_TREES.update(trees)   # what arms.py does with E2E_FLY_TREES at import; also right if it was imported earlier
    for lb, t in trees.items():
        assert A.FLY_TREES.get(lb) == t, (lb, A.FLY_TREES.get(lb), t)
    fw, bw = {}, {}
    log("LOAD asm fwd (aiter) + asm bwd (hand-launched ASM + GQA sum)")
    fw["asm"] = A.get_fwd("asm")
    bw["asm"] = A.get_bwd("asm")
    for lb, sp in cfg.fwd_arms.items():
        log(f"LOAD fwd {lb} <- {trees[lb]['fwd']} env={sp.get('env') or {}}")
        with env_overrides(sp.get("env") or {}):
            fw[lb] = A.get_fwd(lb)
    for lb, sp in cfg.bwd_arms.items():
        log(f"LOAD bwd {lb} <- {trees[lb]['bwd']} env={sp.get('env') or {}}")
        with env_overrides(sp.get("env") or {}):
            bw[lb] = A.get_bwd(lb)
    info = {"versions": A.check_versions(), "e2e_arms_py": A.__file__,
            "trees": {f"{d}:{lb}": tree_info(sp["path"])
                      for d, specs in (("fwd", cfg.fwd_arms), ("bwd", cfg.bwd_arms)) for lb, sp in specs.items()}}
    import aiter.ops.mha as mha
    info["aiter_mha"] = mha.__file__
    return fw, bw, info


MOCK_MS = {"fwd": {"asm": 0.40, "fly": 0.44}, "bwd": {"asm": 1.00, "s6": 0.96, "r29": 1.20}}


def mock_arms(cfg, torch, holder):
    nan_arm = os.environ.get("RA_MOCK_NAN", "")

    def base(lb):
        return lb[:-3] if lb.endswith("_aa") else lb

    def fwd(lb):
        ms, eps = MOCK_MS["fwd"].get(base(lb), 0.5), (0.0 if lb == "asm" else 1e-3)

        def f(q, k, v, scale):
            time.sleep(ms * 1e-3)
            o = (q.float() * (1.0 + eps)).to(q.dtype)
            lse = torch.full((q.shape[0], q.shape[2], q.shape[1]), eps, dtype=torch.float32)
            if lb == nan_arm and q.data_ptr() == holder.get("nan_ptr"):
                o[0, 0, 0, 0] = float("nan")
            return o, lse
        return f

    def bwd(lb):
        ms = MOCK_MS["bwd"].get(base(lb), 1.1)
        eq = {"asm": 0.0, "s6": 2e-3, "r29": 3e-3}.get(base(lb), 4e-3)

        def f(do, q, k, v, o, lse, scale):
            time.sleep(ms * 1e-3)
            dq = (do.float() * 0.5 * (1.0 + eq)).to(do.dtype)
            dk, dv = (k.float() * 0.1).to(k.dtype), (v.float() * 0.2).to(v.dtype)
            if lb == nan_arm and q.data_ptr() == holder.get("nan_ptr"):
                dk[0, 0, 0, 0] = float("nan")
            return dq, dk, dv
        return f

    fw = {"asm": fwd("asm"), **{lb: fwd(lb) for lb in cfg.fwd_arms}}
    bw = {"asm": bwd("asm"), **{lb: bwd(lb) for lb in cfg.bwd_arms}}
    return fw, bw, {"versions": "mock", "trees": {}}


# ------------------------------------------------------------------------------------------------ inputs
def load_sets(cfg, card, torch):
    bf = torch.bfloat16
    want = set(cfg.sets) if cfg.sets else None
    b, s, hq, hkv, d = cfg.shape
    sets = {}
    real_dumps = not cfg.mock or os.environ.get("RA_MOCK_REALDUMPS") == "1"
    ddir = Path(os.environ.get("RA_DUMP_DIR", "/home/lihuzhan/_prof_dump"))
    for c in REAL_CALLS:
        name = f"L{c % 32:02d}"
        if want is not None and name not in want:
            continue
        if not real_dumps:
            g = torch.Generator().manual_seed(c)
            sets[name] = tuple((torch.randn(*shp, generator=g) * 3).to(bf)
                               for shp in ((b, s, hq, d), (b, s, hkv, d), (b, s, hkv, d)))
            continue
        f = ddir / f"qkv_call{c:04d}.pt"
        t = torch.load(str(f), map_location="cpu", weights_only=True)
        assert int(t["call"]) == c, (f, t["call"])
        assert abs(float(t["scale"]) - PROD[4] ** -0.5) < 1e-9, (f, t["scale"])
        P = PROD
        for n, shp in (("q", (P[0], P[1], P[2], P[4])), ("k", (P[0], P[1], P[3], P[4])), ("v", (P[0], P[1], P[3], P[4]))):
            x = t[n]
            assert tuple(x.shape) == shp and x.dtype == bf, f"{f.name}: {n} {tuple(x.shape)} {x.dtype}"
        if cfg.mock:   # parse check only, then cut down to the mock shape
            q, k, v = t["q"][:b, :s, :hq, :d], t["k"][:b, :s, :hkv, :d], t["v"][:b, :s, :hkv, :d]
        else:
            q, k, v = t["q"], t["k"], t["v"]
        sets[name] = tuple(x.to(card.dev).contiguous() for x in (q, k, v))
        del t, q, k, v
    g = torch.Generator(device=card.dev).manual_seed(0)

    def rnd(*shp):
        return torch.randn(*shp, generator=g, device=card.dev, dtype=torch.float32).to(bf)

    q, k, v, do = rnd(b, s, hq, d), rnd(b, s, hkv, d), rnd(b, s, hkv, d), rnd(b, s, hq, d)
    if want is None or "randn" in want:
        sets["randn"] = (q, k, v)
    if cfg.do_scale != 1.0:
        do = (do.float() * cfg.do_scale).to(bf)
    return sets, do


# ------------------------------------------------------------------------------------------------ checks
def finite(*ts):
    return all(bool(t.isfinite().all()) for t in ts)


def bitwise(a, b, torch):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype in (torch.bfloat16, torch.float16):
        return torch.equal(a.view(torch.int16), b.view(torch.int16))
    if a.dtype == torch.float32:
        return torch.equal(a.view(torch.int32), b.view(torch.int32))
    return torch.equal(a, b)


def sqnr_db(ref, got):
    """10 log10(sum ref^2 / sum (ref - got)^2), one batch row at a time (bounded temporaries)."""
    num = den = 0.0
    r2, g2 = ref.reshape(ref.shape[0], -1), got.reshape(got.shape[0], -1)
    for i in range(r2.shape[0]):
        rf, gf = r2[i].float(), g2[i].float()
        num += float(rf.square().sum())
        den += float((rf - gf).square().sum())
    if den == 0.0:
        return math.inf
    return 10.0 * math.log10(num / den) if num > 0 else -math.inf


def cmp(ref, got, torch):
    if bitwise(ref, got, torch):
        return "bitwise"
    if not finite(ref, got):
        return "nonfinite"
    return round(sqnr_db(ref, got), 2)


def fmtv(key, v):
    if isinstance(v, dict):
        return " ".join(f"{t}={fmtv(key, x)}" for t, x in v.items())
    if isinstance(v, (bool, str)):
        return str(v)
    return f"{v:.3g}" if "maxabs" in key else f"{v:.2f}dB"


def poison(card, torch, nbytes):
    """Free NaN-filled blocks of the output sizes into the caching allocator, so that an element an arm never writes
    reads back as NaN (bf16 NaN 0x7FC0 is also NaN when the block is reused as fp32). Best effort; the gate is the
    isfinite check."""
    if card.mock:
        return
    blocks = [torch.full((n // 2,), float("nan"), device=card.dev, dtype=torch.bfloat16) for n in nbytes]
    del blocks


def nb(shape, elem):
    return math.prod(shape) * elem


def correctness(cfg, card, torch, fw, bw, sets, do, R):
    """Every arm on every set once (all arms run first, then every comparison, so arm order does not matter).
    The ASM forward's o/lse are kept as that set's bwd inputs. Comparisons are arm vs arm only (no fp32 reference):
    FlyDSL vs ASM SQNR; A/A copy vs its original bitwise; s6 vs r29 (dk/dv expected bitwise, 0930__bwd/REPORT.md)."""
    b, s, hq, hkv, d = cfg.shape
    fwd_nb = [nb((b, s, hq, d), 2), nb((b, hq, s), 4)] * 2
    bwd_nb = [nb((b, s, hq, d), 2), nb((b, s, hkv, d), 2), nb((b, s, hkv, d), 2), nb((b, hq, s), 4)] * 2
    ol, bad, warn = {}, {"fwd": set(), "bwd": set()}, []
    for name in list(sets):
        q, k, v = sets[name]
        c = R["corr"].setdefault(name, {"fwd": {}, "bwd": {}})
        outs = {}
        for a in ["asm"] + [x for x in cfg.order["fwd"] if x != "asm"]:
            poison(card, torch, fwd_nb)
            outs[a] = fw[a](q, k, v, cfg.scale)
            card.sync()
        o0, l0 = outs["asm"]
        for a, (o, lse) in outs.items():
            row = {"finite": finite(o, lse)}
            if a != "asm":
                row["o_vs_asm"] = cmp(o0, o, torch)
                row["lse_maxabs_vs_asm"] = (round(float((lse.float() - l0.float()).abs().max()), 6)
                                            if lse.shape == l0.shape else f"shape{tuple(lse.shape)}")
                base = a[:-3] if a.endswith("_aa") else None
                if base in outs:
                    row["vs_" + base] = ("bitwise" if bitwise(outs[base][0], o, torch)
                                         and bitwise(outs[base][1], lse, torch) else "DIFF")
                if isinstance(row["o_vs_asm"], float) and row["o_vs_asm"] < cfg.sqnr_warn:
                    warn.append(f"{name} fwd {a} o {row['o_vs_asm']} dB")
            c["fwd"][a] = row
            if not row["finite"]:
                bad["fwd"].add(a)
            print(f"CORR set={name} dir=fwd arm={a} " + " ".join(f"{k_}={fmtv(k_, v_)}" for k_, v_ in row.items()),
                  flush=True)
        if not c["fwd"]["asm"]["finite"]:
            print(f"CORR FAIL set={name}: the ASM forward is non-finite -> set dropped (no bwd inputs)", flush=True)
            sets.pop(name)
            continue
        ol[name] = (o0, l0)
        del outs, o0, l0
        if "bwd" not in cfg.dirs:
            continue
        o, lse = ol[name]
        bo = {}
        for a in ["asm"] + [x for x in cfg.order["bwd"] if x != "asm"]:
            poison(card, torch, bwd_nb)
            bo[a] = bw[a](do, q, k, v, o, lse, cfg.scale)
            card.sync()
        for a, g in bo.items():
            row = {"finite": finite(*g)}
            if a != "asm":
                row["vs_asm"] = {t: cmp(x0, x, torch) for t, x0, x in zip(("dq", "dk", "dv"), bo["asm"], g)}
                refs = ([a[:-3]] if a.endswith("_aa") and a[:-3] in bo else []) + (["r29"] if a == "s6" and "r29" in bo else [])
                for ref in refs:
                    row["vs_" + ref] = {t: cmp(x0, x, torch) for t, x0, x in zip(("dq", "dk", "dv"), bo[ref], g)}
                for t, v_ in row["vs_asm"].items():
                    if isinstance(v_, float) and v_ < cfg.sqnr_warn:
                        warn.append(f"{name} bwd {a} {t} {v_} dB")
                if "vs_r29" in row and row["vs_r29"]["dq"] == "bitwise":
                    warn.append(f"{name} bwd s6 dq bitwise == r29 dq: possible JIT cache collision between the trees")
            c["bwd"][a] = row
            if not row["finite"]:
                bad["bwd"].add(a)
            print(f"CORR set={name} dir=bwd arm={a} finite={row['finite']}"
                  + "".join(f" | {k_} {fmtv(k_, v_)}" for k_, v_ in row.items() if k_ != "finite"), flush=True)
        del bo
    for w in warn:
        print(f"CORR WARN {w}", flush=True)
    return ol, bad, warn


# ------------------------------------------------------------------------------------------------ timing
def new_rec():
    return {"ms": [], "sclk": [], "sclk_pre": [], "sclk_post": [], "pre_ms": [], "t": []}


def timed(sclk, ev, fn, pre):
    evb, ev0, ev1 = ev
    evb.record()
    pre()
    ev0.record()
    fn()
    ev1.record()
    ev1.synchronize()
    t1 = time.perf_counter()
    post = sclk.read()
    ms = ev0.elapsed_time(ev1)
    ts = t1 - ms * 1e-3
    win, prev = sclk.window(ts, t1), sclk.window(ts - 0.003, ts)
    return (ms, st.median(win) if win else post, st.median(prev) if prev else -1, post,
            evb.elapsed_time(ev0), time.time())


def add(rec, r):
    for k, v in zip(("ms", "sclk", "sclk_pre", "sclk_post", "pre_ms", "t"), r):
        rec[k].append(round(v, 5) if k != "t" else round(v, 4))


def make_calls(cfg, fw, bw, sets, ol, do):
    calls = {"fwd": {}, "bwd": {}}
    for name, (q, k, v) in sets.items():
        for a in cfg.order["fwd"]:
            calls["fwd"].setdefault(a, {})[name] = (lambda f=fw[a], q=q, k=k, v=v: f(q, k, v, cfg.scale))
        if name in ol:
            o, lse = ol[name]
            for a in cfg.order["bwd"]:
                calls["bwd"].setdefault(a, {})[name] = (
                    lambda f=bw[a], q=q, k=k, v=v, o=o, lse=lse: f(do, q, k, v, o, lse, cfg.scale))
    return calls


def warmup(cfg, card, calls, name, order):
    for d in cfg.dirs:
        for a in order[d]:
            fn = calls[d][a][name]
            t_end, i = time.perf_counter() + cfg.warm, 0
            while time.perf_counter() < t_end:
                fn()
                i += 1
                if i % 8 == 0:          # bounded queue depth (s6's record_stream defers ~1 GiB per queued call)
                    card.sync()
            card.sync()
            log(f"WARM {d} {a}: {i} calls in {cfg.warm:.1f}s on {name}")


def run_timed(cfg, card, sclk, calls, pre, names, order, timing, dump):
    ev = (card.event(), card.event(), card.event())
    sclk.on = True
    try:
        for name in names:
            if time.time() - T0 > cfg.deadline:
                log(f"DEADLINE {cfg.deadline:.0f}s reached: no timing from set {name} on")
                break
            for d in cfg.dirs:
                labels = order[d]
                if len(labels) < 2 or name not in calls[d][labels[0]]:
                    continue
                rec = timing.setdefault(name, {}).setdefault(d, {a: new_rec() for a in labels})
                if cfg.cond == "blk":
                    rounds, lead, block = -(-cfg.iters[d] // cfg.block), cfg.lead, cfg.block
                else:
                    rounds, lead, block = cfg.gb_rounds, 0, cfg.gb_n
                rounds += rounds % 2                       # even: every arm the same mean position
                t = time.time()
                for r in range(rounds):
                    for a in (labels if r % 2 == 0 else labels[::-1]):
                        fn = calls[d][a][name]
                        for _ in range(lead):
                            timed(sclk, ev, fn, pre)
                        for _ in range(block):
                            add(rec[a], timed(sclk, ev, fn, pre))
                med = {a: st.median(rec[a]["ms"]) for a in labels}
                log(f"TIMED {cfg.cond} {d} {name}: {rounds} rounds x {block} (+{lead} lead) in {time.time() - t:.1f}s  "
                    + " ".join(f"{a}={med[a]:.4f}" for a in labels))
            dump()
    finally:
        sclk.on = False


def kineto_pass(cfg, card, torch, calls, pre, names, order):
    from torch.profiler import ProfilerActivity, profile, record_function
    acts = [ProfilerActivity.CPU] + ([] if card.mock else [ProfilerActivity.CUDA])
    with profile(activities=acts) as prof:
        for rep in range(cfg.kin_reps):
            for name in names:
                for d in cfg.dirs:
                    kind, n = KINETO_KIND[(cfg.cond, d)]
                    labels = order[d] if rep % 2 == 0 else order[d][::-1]
                    for a in labels:
                        fn = calls[d].get(a, {}).get(name)
                        if fn is None:
                            continue
                        card.sync()
                        for _ in range(n):
                            pre()
                            with record_function(f"P::{kind}_{name}::{a}::{rep}"):
                                fn()
                        card.sync()
    Path(cfg.trace).parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(cfg.trace)


# ------------------------------------------------------------------------------------------------ report
def ratio_keys(d, labels):
    keys = [(a, "asm") for a in labels if a != "asm"]
    keys += [(a, a[:-3]) for a in labels if a.endswith("_aa") and a[:-3] in labels]
    if d == "bwd" and "s6" in labels and "r29" in labels:
        keys.append(("s6", "r29"))
    return keys


def stat_row(cfg, d, r):
    ms = sorted(r["ms"])
    med = st.median(ms)
    q = st.quantiles(ms, n=4) if len(ms) >= 4 else [ms[0], med, ms[-1]]

    def m(key):
        xs = [x for x in r[key] if x is not None and x > 0]
        return round(st.median(xs), 3) if xs else -1

    return {"n": len(ms), "median_ms": med, "min_ms": ms[0], "max_ms": ms[-1], "q1_ms": q[0], "q3_ms": q[2],
            "tflops": cfg.flop(d) / (med * 1e-3) / 1e12, "sclk": m("sclk"), "sclk_pre": m("sclk_pre"),
            "sclk_post": m("sclk_post"), "pre_ms": m("pre_ms")}


def compute_stats(cfg, timing, order):
    stats, ratios = {}, {}
    for name, dd in timing.items():
        for d, arms in dd.items():
            sd = {a: stat_row(cfg, d, r) for a, r in arms.items() if r["ms"]}
            if not sd:
                continue
            stats.setdefault(name, {})[d] = sd
            rr = {}
            for a, b_ in ratio_keys(d, [a for a in order[d] if a in sd]):
                rr[f"{a}/{b_}"] = sd[a]["median_ms"] / sd[b_]["median_ms"]
                rr[f"{a}/{b_}@min"] = sd[a]["min_ms"] / sd[b_]["min_ms"]
            ratios.setdefault(name, {})[d] = rr
    return stats, ratios


def table_lines(title, stats, ratios, order):
    out = [f"== SUMMARY {title}  median ms of CUDA-event times; ratio = arm/ref (<1: arm faster); "
           f"sclk = MHz median in the call window"]
    for d in ("fwd", "bwd"):
        names = [n for n in stats if d in stats[n]]
        if not names:
            continue
        arms = [a for a in order.get(d, []) if any(a in stats[n][d] for n in names)]
        rkeys = []
        for n in names:
            for k in ratios.get(n, {}).get(d, {}):
                if "@" not in k and k not in rkeys:
                    rkeys.append(k)
        out.append(f"{d} {'set':6s} " + " ".join(f"{a:>8s}" for a in arms) + " |"
                   + "".join(f" {k:>10s}" for k in rkeys) + " | sclk " + "/".join(arms))
        for n in names:
            s, r = stats[n][d], ratios.get(n, {}).get(d, {})
            out.append(f"{d} {n:6s} " + " ".join(f"{s[a]['median_ms']:8.4f}" if a in s else f"{'-':>8s}" for a in arms)
                       + " |" + "".join(f" {r[k]:10.4f}" if k in r else f" {'-':>10s}" for k in rkeys)
                       + " | " + "/".join(f"{s[a]['sclk']:.0f}" if a in s else "-" for a in arms))
        real = [n for n in names if n.startswith("L")]
        if real:
            gm = {k: geomean([ratios[n][d][k] for n in real if k in ratios.get(n, {}).get(d, {})]) for k in rkeys}
            out.append(f"{d} {'real' + str(len(real)):6s} " + " ".join(f"{'':8s}" for _ in arms) + " |"
                       + "".join(f" {gm[k]:10.4f}" for k in rkeys) + " | geomean over the real sets")
    out.append("== END SUMMARY")
    return out


OP_KEYS = ((("bwd", "asm"), "asm_bwd"), (("bwd", "s6"), "s6"), (("bwd", "r29"), "r29"),
           (("fwd", "asm"), "asm_fwd"), (("fwd", "fly"), "r16"))


def op_strings(stats):
    """The OP=... string of output/1002__e2e/E2E-PLAN.md section 6 (analyze.sh): per arm, the mean and the median
    over the real sets of the per-set median ms (asm bwd includes the host GQA sum, as in the e2e)."""
    real = [n for n in stats if n.startswith("L")]
    out = {}
    for stat, fn in (("mean", st.mean), ("median", st.median)):
        parts = []
        for (d, a), key in OP_KEYS:
            xs = [stats[n][d][a]["median_ms"] for n in real if a in stats[n].get(d, {})]
            if xs:
                parts.append(f"{key}={fn(xs):.4f}")
        out[stat] = ",".join(parts)
    return out, len(real)


def report(cfg, R, timing):
    stats, ratios = compute_stats(cfg, timing, R["order"])
    R["stats"], R["ratios"] = stats, ratios
    for name in stats:
        for d, sd in stats[name].items():
            for a, s in sd.items():
                print(f"RESULT cond={cfg.cond} dir={d} set={name} arm={a} n={s['n']} median_ms={s['median_ms']:.4f} "
                      f"min_ms={s['min_ms']:.4f} max_ms={s['max_ms']:.4f} tflops={s['tflops']:.1f} sclk={s['sclk']:.0f} "
                      f"sclk_pre={s['sclk_pre']:.0f} sclk_post={s['sclk_post']:.0f} pre_ms={s['pre_ms']:.2f} "
                      f"order={','.join(R['order'][d])}", flush=True)
            rr = ratios[name][d]
            print(f"RATIO cond={cfg.cond} dir={d} set={name} "
                  + " ".join(f"{k}={v:.4f}" for k, v in rr.items() if "@" not in k)
                  + "  (min-based: " + " ".join(f"{k[:-4]}={v:.4f}" for k, v in rr.items() if "@" in k) + ")",
                  flush=True)
    print("\n".join(table_lines(f"cond={cfg.cond}", stats, ratios, R["order"])), flush=True)
    ops, nreal = op_strings(stats)
    R["op_strings"] = ops
    for stat, v in ops.items():
        print(f'OP_STRING cond={cfg.cond} stat={stat}_of_{nreal}_real_sets OP="{v}"', flush=True)


def write_json(path, R):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(R, indent=1))
    os.replace(tmp, p)


# ------------------------------------------------------------------------------------------------ main
def measure():
    cfg = Cfg()
    if not cfg.mock:
        # ASSIGNED, not setdefault, before torch is imported: the image hipBLASLt library, never the host one.
        os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "1"
        os.environ["HIPBLASLT_TENSILE_LIBPATH"] = IMAGE_BLAS
        # gfx1250 is wave32; FlyDSL must not guess the arch (card-safety #10). realab_run.sh passes the same values.
        os.environ["ARCH"] = os.environ["FLYDSL_GPU_ARCH"] = "gfx1250"
        # a FRESH JIT cache per process (h46/h72: the 0.3.4.1 key can miss module constants -> stale binary)
        cd = os.environ.get("FLYDSL_RUNTIME_CACHE_DIR")
        if not cd:
            import tempfile
            os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = tempfile.mkdtemp(prefix="flycache_realab.", dir="/tmp")
        elif Path(cd).is_dir() and any(Path(cd).iterdir()):
            log(f"WARNING FLYDSL_RUNTIME_CACHE_DIR={cd} is not empty: stale-binary hazard (h46/h72); use a fresh dir")
    import torch
    torch.set_grad_enabled(False)
    card = Mock(torch) if cfg.mock else Card(torch)
    R = {"tool": "realab", "cond": cfg.cond, "mock": cfg.mock, "host": socket.gethostname(), "started": T0,
         "json": cfg.json, "trace": cfg.trace if cfg.kineto else None, "shape": cfg.shape, "scale": cfg.scale,
         "order": cfg.order, "arms": {"fwd": cfg.fwd_arms, "bwd": cfg.bwd_arms},
         "params": {k: getattr(cfg, k) for k in ("warm", "block", "lead", "iters", "gb_n", "gb_rounds", "gb_ng",
                                                 "gb_maxms", "kin_reps", "do_scale", "deadline", "dirs")},
         "env": {k: os.environ.get(k) for k in ("TORCH_BLAS_PREFER_HIPBLASLT", "HIPBLASLT_TENSILE_LIBPATH",
                                                "FLYDSL_RUNTIME_CACHE_DIR", "FLYDSL_RUNTIME_ENABLE_CACHE",
                                                "AMD_SERIALIZE_KERNEL", "HIP_VISIBLE_DEVICES",
                                                "FLY_BWD_SIDE_STREAM", "FLY_BWD_RECORD_STREAM")},
         "phases": {}, "corr": {}, "excluded": {}, "warn": [], "raw": {}}
    phase_t = [time.time()]

    def phase(name):
        now = time.time()
        R["phases"][name] = round(now - phase_t[0], 1)
        phase_t[0] = now
        log(f"PHASE {name} done ({R['phases'][name]:.1f}s)")

    def dump():
        R["finished"] = time.time()
        write_json(cfg.json, R)

    log(f"realab cond={cfg.cond} mock={cfg.mock} shape={cfg.shape} dirs={cfg.dirs} order={cfg.order} json={cfg.json}")
    sclk = Sclk(os.environ.get("RA_SCLK_FILE") or (None if cfg.mock else find_sclk_file()))
    R["sclk_file"] = sclk.path
    log(f"sclk witness {sclk.path} now {sclk.read()} MHz")
    if not cfg.mock:
        pr = torch.cuda.get_device_properties(0)
        R["device"] = {"name": pr.name, "arch": pr.gcnArchName, "cus": pr.multi_processor_count,
                       "torch": torch.__version__, "hip": torch.version.hip}
        log(f"DEVICE {R['device']}")
        if "gfx1250" not in pr.gcnArchName:
            raise SystemExit(f"not a gfx1250: {pr.gcnArchName}")

    # gb: hipBLASLt is initialised with the image library BEFORE any tree's _env.py can re-point it
    pre = (lambda: None)
    if cfg.cond == "gb":
        if card.mock:
            pre = (lambda: time.sleep(0.002))
        else:
            import torch.nn.functional as Fn
            assert os.environ["HIPBLASLT_TENSILE_LIBPATH"] == IMAGE_BLAS
            torch.manual_seed(1234)
            xg = torch.randn(32768, 4096, device="cuda", dtype=torch.bfloat16)
            wg = torch.randn(14336, 4096, device="cuda", dtype=torch.bfloat16) * 0.02
            for _ in range(2):
                Fn.linear(xg, wg)
            torch.cuda.synchronize()
            ng = cfg.gb_ng

            def pre():
                torch.cuda.synchronize()
                for _ in range(ng):
                    Fn.linear(xg, wg)
            log(f"GEMM burst ready: {ng} x bf16 32768x4096x14336, preferred_blas_library="
                f"{torch.backends.cuda.preferred_blas_library()}")
    holder = {}
    fw, bw, info = mock_arms(cfg, torch, holder) if cfg.mock else load_arms(cfg)
    R["info"] = info
    if not cfg.mock:
        # the fwd tree's _env.py ASSIGNS HIPBLASLT_TENSILE_LIBPATH = ~/.local/hipblaslt-gfx1250: undo it
        before = os.environ.get("HIPBLASLT_TENSILE_LIBPATH")
        os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "1"
        os.environ["HIPBLASLT_TENSILE_LIBPATH"] = IMAGE_BLAS
        log(f"BLAS env restored: HIPBLASLT_TENSILE_LIBPATH {before} -> {IMAGE_BLAS}")
        log(f"VERSIONS {info['versions']}  aiter {info['aiter_mha']}")
        for key, ti in info["trees"].items():
            log(f"TREE {key} {ti['path']} md5_tree={ti['md5_tree']} "
                + " ".join(f"{k}={v[:8]}" for k, v in ti["files"].items()))
    phase("load_arms")

    sets, do = load_sets(cfg, card, torch)
    if "L16" in sets:
        holder["nan_ptr"] = sets["L16"][0].data_ptr()
    log(f"SETS {list(sets)} dO=randn draw 4 x {cfg.do_scale}")
    phase("load_sets")

    # first launch of every arm (JIT compile for FlyDSL), randn first, one arm at a time
    first = sets["randn"] if "randn" in sets else next(iter(sets.values()))
    ol0 = None
    for a in ["asm"] + [x for x in cfg.order["fwd"] if x != "asm"]:
        t = time.time()
        log(f"FIRST fwd {a} launching")
        r_ = fw[a](*first, cfg.scale)
        card.sync()
        if a == "asm":
            ol0 = r_
        del r_
        log(f"FIRST fwd {a} ok ({time.time() - t:.1f}s)")
    if "bwd" in cfg.dirs:
        for a in ["asm"] + [x for x in cfg.order["bwd"] if x != "asm"]:
            t = time.time()
            log(f"FIRST bwd {a} launching")
            r_ = bw[a](do, *first, *ol0, cfg.scale)
            card.sync()
            del r_
            log(f"FIRST bwd {a} ok ({time.time() - t:.1f}s)")
    del ol0
    phase("first_launch")

    ol, bad, warn = correctness(cfg, card, torch, fw, bw, sets, do, R)
    R["excluded"] = {d: sorted(v) for d, v in bad.items()}
    R["warn"] = warn
    order = {d: [a for a in cfg.order[d] if a not in bad[d]] for d in ("fwd", "bwd")}
    for d in ("fwd", "bwd"):
        if bad[d]:
            print(f"EXCLUDED dir={d} arms={sorted(bad[d])}: non-finite output on a set -> not timed", flush=True)
        if "asm" in bad[d]:
            order[d] = []
    R["order_timed"] = order
    phase("correctness")
    dump()

    rc = 3 if (bad["fwd"] or bad["bwd"]) else 0
    timing = R["raw"]
    if cfg.cond != "corr":
        if cfg.cond == "gb" and not card.mock:
            assert os.environ["HIPBLASLT_TENSILE_LIBPATH"] == IMAGE_BLAS
            e0, e1, bms = card.event(), card.event(), []
            for _ in range(3):
                card.sync()
                e0.record()
                pre()
                e1.record()
                e1.synchronize()
                bms.append(e0.elapsed_time(e1))
            R["burst_ms"] = bms
            log(f"BURST {cfg.gb_ng} GEMMs: {' '.join(f'{x:.1f}' for x in bms)} ms; sclk now {sclk.read()} MHz")
            if st.median(bms) > cfg.gb_maxms:
                print(f"RULER VOID: GEMM burst {st.median(bms):.1f} ms > {cfg.gb_maxms} ms -- not the image "
                      f"hipBLASLt library? HIPBLASLT_TENSILE_LIBPATH={os.environ.get('HIPBLASLT_TENSILE_LIBPATH')}",
                      flush=True)
                dump()
                return 4
        calls = make_calls(cfg, fw, bw, sets, ol, do)
        warmup(cfg, card, calls, "randn" if "randn" in sets else next(iter(sets)), order)
        phase("warmup")
        if cfg.cond == "blk":
            flush = (None if card.mock else
                     torch.empty(256 * 1024 * 1024 // 4, device=card.dev, dtype=torch.float32))
            pre = (lambda: None) if flush is None else flush.zero_
        run_timed(cfg, card, sclk, calls, pre, list(sets), order, timing, dump)
        phase("timed")
        report(cfg, R, timing)
        dump()
        if cfg.kineto and time.time() - T0 < cfg.deadline:
            try:
                kineto_pass(cfg, card, torch, calls, pre, list(sets), order)
                log(f"KINETO trace {cfg.trace}")
            except Exception as exc:          # the CUDA-event results are already on disk
                R["kineto_error"] = repr(exc)
                log(f"KINETO failed: {exc!r}")
            phase("kineto")
        elif cfg.kineto:
            log("KINETO skipped: deadline")
    dump()
    log(f"REALAB_DONE cond={cfg.cond} rc={rc} excluded={R['excluded']} warn={len(warn)} json={cfg.json}")
    return rc


def summarize(paths):
    """Stdlib only: per-run tables and the per-set ratio across runs (columns = runs)."""
    docs = []
    for p in paths:
        d = json.loads(Path(p).read_text())
        if "stats" not in d:
            print(f"# {p}: no stats (cond={d.get('cond')}, excluded={d.get('excluded')})")
            continue
        d["_label"] = Path(p).stem.replace("realab_", "")
        docs.append(d)
    for d in docs:
        print("\n".join(table_lines(d["_label"], d["stats"], d["ratios"], d["order"])))
        ops, nreal = op_strings(d["stats"])
        for stat, v in ops.items():
            print(f'OP_STRING {d["_label"]} stat={stat}_of_{nreal}_real_sets OP="{v}"')
        print(f"   phases {d.get('phases')}  burst_ms {d.get('burst_ms', '-')}  excluded {d.get('excluded')}  "
              f"warn {len(d.get('warn', []))}\n")
    if not docs:
        return 1
    keys = []
    for d in docs:
        for n in d["ratios"]:
            for dd in d["ratios"][n].values():
                keys += [k for k in dd if "@" not in k and k not in keys]
    names = []
    for d in docs:
        names += [n for n in d["stats"] if n not in names]
    w = max(10, max(len(d["_label"]) for d in docs))
    print("== CROSS ratio per set across runs (median-based)")
    print(f"{'ratio':12s} {'set':8s} " + " ".join(f"{d['_label']:>{w}s}" for d in docs))
    for k in keys:
        for n in names + ["real"]:
            vals = []
            for d in docs:
                if n == "real":
                    real = [m for m in d["ratios"] if m.startswith("L")]
                    vals.append(geomean([d["ratios"][m][dd][k] for m in real for dd in d["ratios"][m]
                                         if k in d["ratios"][m][dd]]))
                else:
                    v = [d["ratios"].get(n, {}).get(dd, {}).get(k) for dd in ("fwd", "bwd")]
                    v = [x for x in v if x is not None]
                    vals.append(v[0] if v else float("nan"))
            if all(math.isnan(x) for x in vals):
                continue
            lab = "real(gm)" if n == "real" else n
            print(f"{k:12s} {lab:8s} " + " ".join(f"{x:>{w}.4f}" for x in vals))
    print("== END CROSS")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--sum":
        sys.exit(summarize(sys.argv[2:]))
    sys.exit(measure())
