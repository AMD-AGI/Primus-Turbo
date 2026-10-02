"""gb ruler: time attention arms at the TRAINING operating point (shared by both gfx1250 attention jobs).

Installed as `job_context/op/gbruler.py`, next to `benchmark.py`, in BOTH op-evolve jobs
(gfx1250-flydsl-attn-bwd-20260917-115934 and gfx1250-flydsl-attn-fwd-b0-20260927), byte-identical in both.
Design, evidence and rollout: /home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/1002__oe/RULER.md.

WHY. The jobs' blocked ruler (lead 4 + block 9 per arm, 256 MB L2 flush, palindromic rounds) times every arm at
the card's steady clock (~1.45-1.6 GHz on A0) with nothing in front of the call; inside Llama-3.1-8B training every
attention call follows bf16 GEMMs. Same-process op A/B on A0 (output/1002__e2e/RESULT-realab.md, 6 real layers)
next to the e2e run of the same trees (output/1002__e2e/RESULT-e2e.md, per-layer CUDA events), 2026-10-02:
                          blocked    gb (this)    e2e
    bwd s6 / r29           0.806      0.774      0.754    <- the kind of ratio acceptance ranks (FlyDSL vs FlyDSL)
    bwd r29 / ASM          1.206      1.341      1.290
    bwd s6 / ASM           0.972      1.038      0.974
    fwd r16 / ASM          1.081      1.349      1.263
gb is the closer ruler in three of four, including the FlyDSL-vs-FlyDSL ranking and the fwd gap; it is pessimistic
for s6 vs ASM (~6%): the e2e step runs at ~1.5 GHz median, between the two rulers' ~1.45-1.5 / ~1.28 GHz.

METHOD (identical to output/1002__e2e/tools/realab.py RA_COND=gb and output/0927__b0/fwd-nospec/tools/ab.py
AB_COND=gb, which produced those numbers):
  burst      before EVERY timed call: torch.cuda.synchronize(), then --gb-ng (10) x F.linear of bf16
             x[32768,4096] by w[14336,4096]^T (~21.5 ms on A0), then the call with no sync in between. The CUDA
             events time the call only; the burst's own time is recorded as gb_burst_ms.
  library    the IMAGE hipBLASLt library (IMAGE_BLAS). preimport() ASSIGNS the env before torch is imported,
             the burst's warm-up initialises hipBLASLt BEFORE any arm is loaded (an arm's _env.py may re-point
             HIPBLASLT_TENSILE_LIBPATH: the fwd trees assign the host library, with which this GEMM runs at
             ~80 TF/s, the clock stays ~2.1 GHz and the ruler is void), and the env is assigned again after
             the arms are loaded.
  void       3 timed bursts right after the arms are loaded, and every timed call's own burst: a median above
             --gb-max-burst-ms (80) means the image library is not in use. Then no RESULT row is printed for
             that shape and benchmark.py exits VOID_RC (4) with a "SPEED RULER VOID" line on stdout and stderr.
  ordering   rounds palindromic over arms, an even number of them, --gb-block (5) timed calls per arm per
             round, no lead calls (the burst, not the previous call, sets the card state). Timed calls per arm
             = ceil(gb_iters / gb_block) rounded up to an even round count, times gb_block.
  statistic  median of the per-call CUDA-event times (the bwd job keeps the MIN for `fast`, h83).
  witness    the card's hwmon freq1_input, sampled every 1 ms by a thread while (and only while) the gb loop
             runs: per call the median inside the call's window [t_end - ms, t_end] (gb_sclk) and in the 3 ms
             before it (gb_sclk_pre, the end of the burst). Expected on A0: call ~1.26-1.39 GHz, burst ~21-23 ms.
             A call-window median above --gb-max-sclk is reported (gb_clock_ok=False), not voided.
  A/A        --aa LABEL measures a byte copy of that arm (fresh temp dir) as LABEL_aa in the same rounds; the
             copy's row carries aa_of / aa_ratio (time ratio copy/original) = this session's noise floor.

Nothing here imports torch at module level: preimport() has to run before `import torch`.
"""
from __future__ import annotations

import argparse
import atexit
import bisect
import glob
import os
import shutil
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path

IMAGE_BLAS = "/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/gfx1250"
GB_M, GB_K, GB_N = 32768, 4096, 14336          # one bf16 GEMM of the burst: [M,K] x [N,K]^T
VOID_RC = 4
RULERS = ("auto", "blk", "gb", "both")


# ------------------------------------------------------------------------------------------------ arguments
def rulers_for(shape: str, ruler: str) -> tuple:
    """Which rulers run for a shape. The LAST one is the scored one.

    auto (default): `fast` keeps the blocked ruler only (launch-bound; a sentinel with weight 0); `proxy` and
    `prod` run blocked (reported as blk_*) and then gb, and are scored on gb.
    """
    if ruler == "blk":
        return ("blk",)
    if ruler == "gb":
        return ("gb",)
    if ruler == "both":
        return ("blk", "gb")
    return ("blk",) if shape == "fast" else ("blk", "gb")


def add_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("gb ruler (training operating point; gbruler.py, PT/output/1002__oe/RULER.md)")
    g.add_argument("--ruler", choices=RULERS, default="auto",
                   help="auto: fast blk; proxy/prod blk (reported) + gb (scored) | blk: blocked only (pre-10-02 "
                        "behaviour) | gb: gb only | both: blk + gb on every shape, gb scored")
    g.add_argument("--gb-iters", type=int, default=None,
                   help="timed calls per arm under gb (default: --iters; rounded up to whole even rounds)")
    g.add_argument("--gb-block", type=int, default=5, help="timed calls per arm per palindromic gb round")
    g.add_argument("--gb-ng", type=int, default=10, help="bf16 GEMMs per burst")
    g.add_argument("--gb-max-burst-ms", type=float, default=80.0,
                   help="burst median above this = image hipBLASLt not in use = ruler void (exit 4)")
    g.add_argument("--gb-max-sclk", type=float, default=1700.0,
                   help="call-window sclk median (MHz) above this is reported as gb_clock_ok=False")
    g.add_argument("--aa", action="append", default=[],
                   help="LABEL: also time a byte copy of that arm as LABEL_aa (A/A noise floor); repeatable")


def assign_image_blas() -> None:
    """ASSIGNED, never setdefault (the container's sitecustomize/profile already set these)."""
    os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "1"
    os.environ["HIPBLASLT_TENSILE_LIBPATH"] = IMAGE_BLAS


def preimport(argv, default_shapes: str) -> bool:
    """Call BEFORE `import torch`. Assigns the image hipBLASLt env iff a gb ruler will run in this process.

    `--ruler blk` (or auto with only `fast`) leaves the environment exactly as before this patch.
    """
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument("--ruler", default="auto")
    p.add_argument("--shapes", default=default_shapes)
    a, _ = p.parse_known_args(list(argv))
    if a.ruler not in RULERS:
        return False                                   # main()'s parser reports the error
    if any("gb" in rulers_for(s, a.ruler) for s in a.shapes.split(",") if s):
        assign_image_blas()
        return True
    return False


# ------------------------------------------------------------------------------------------------ clock witness
def find_sclk_file():
    """hwmon freq1_input of the card under test (A0: card1), else its pp_dpm_sclk; None if unreadable."""
    p = os.environ.get("OE_SCLK_FILE")
    if p:
        return p
    roots = []
    phys = os.environ.get("OE_PHYS_GPU")               # B0 convention (one container per card)
    if phys is not None:
        try:
            roots = [f"/sys/class/drm/renderD{128 + 8 * int(phys)}/device"]
        except ValueError:
            roots = []
    if not roots:
        roots = sorted(glob.glob("/sys/class/drm/card*/device"))
    for r in roots:
        if not os.path.exists(r + "/pp_dpm_sclk"):
            continue
        for c in sorted(glob.glob(r + "/hwmon/hwmon*/freq1_input")):
            return c
        return r + "/pp_dpm_sclk"
    return None


class Sclk:
    """Shader clock in MHz. A daemon thread samples every `period` s, but only while `on` is True."""

    def __init__(self, path, period=0.001):
        self.path, self.period = path, period
        self.T, self.V = [], []          # V appended before T: a reader that snapshots len(T) is safe
        self._go = threading.Event()
        self.fd = None
        if path:
            try:
                self.fd = os.open(path, os.O_RDONLY)
            except OSError:
                self.fd = None
        if self.fd is not None:
            threading.Thread(target=self._loop, daemon=True).start()

    @property
    def on(self) -> bool:
        return self._go.is_set()

    @on.setter
    def on(self, value: bool) -> None:
        if value:
            self._go.set()
        else:
            self._go.clear()

    @staticmethod
    def _parse(raw: bytes) -> int:
        s = raw.decode(errors="replace")
        if "Mhz" in s:                                  # pp_dpm_sclk: "1: 2355Mhz *"
            for line in s.splitlines():
                if line.rstrip().endswith("*"):
                    return int(line.split(":")[1].strip().split("Mhz")[0])
            return -1
        v = int(s.split()[0])                           # hwmon freq1_input: Hz
        return v // 1000000 if v > 100000 else v

    def read(self) -> int:
        if self.fd is None:
            return -1
        try:
            return self._parse(os.pread(self.fd, 256, 0))
        except (OSError, ValueError, IndexError):
            return -1

    def _loop(self) -> None:
        while True:
            self._go.wait()
            v = self.read()
            if v > 0:
                self.V.append(v)
                self.T.append(time.perf_counter())
            time.sleep(self.period)

    def window(self, a: float, b: float) -> list:
        n = len(self.T)
        i = bisect.bisect_left(self.T, a, 0, n)
        j = bisect.bisect_right(self.T, b, 0, n)
        return self.V[i:j]


# ------------------------------------------------------------------------------------------------ the loop
def gb_rounds(iters: int, block: int) -> int:
    rounds = -(-max(1, iters) // max(1, block))
    return rounds + rounds % 2                          # even: every arm the same mean position


def run_gb(labels, call, burst, new_event, sclk, iters, block):
    """Palindromic rounds; before every timed call burst.sync() + burst.launch() (GEMMs queued, no sync after
    them), then the call. `burst` is any object with sync() and launch(); `new_event()` returns an object with
    record(), synchronize() and elapsed_time(other) in ms (torch.cuda.Event semantics).

    Returns {label: {"ms": [...], "pre_ms": [...], "sclk": [...], "sclk_pre": [...]}}. Pure python around the
    callables, so it is tested on the CPU with fakes (test/test_gbruler.py).
    """
    rounds = gb_rounds(iters, block)
    evb, ev0, ev1 = new_event(), new_event(), new_event()
    rec = {lb: {"ms": [], "pre_ms": [], "sclk": [], "sclk_pre": []} for lb in labels}
    if sclk is not None:
        sclk.on = True
    try:
        for r in range(rounds):
            for lb in (labels if r % 2 == 0 else labels[::-1]):
                for _ in range(block):
                    burst.sync()                        # previous call (and any side stream) finished
                    evb.record()
                    burst.launch()                      # the GEMMs, queued; no sync before the call
                    ev0.record()
                    call(lb)
                    ev1.record()
                    ev1.synchronize()
                    t1 = time.perf_counter()
                    ms = ev0.elapsed_time(ev1)
                    rr = rec[lb]
                    rr["ms"].append(ms)
                    rr["pre_ms"].append(evb.elapsed_time(ev0))
                    if sclk is not None:
                        t0 = t1 - ms * 1e-3
                        win, prev = sclk.window(t0, t1), sclk.window(t0 - 0.003, t0)
                        rr["sclk"].append(statistics.median(win) if win else sclk.read())
                        rr["sclk_pre"].append(statistics.median(prev) if prev else -1)
                    else:
                        rr["sclk"].append(-1)
                        rr["sclk_pre"].append(-1)
    finally:
        if sclk is not None:
            sclk.on = False
    return rec


def _med_pos(xs):
    xs = [x for x in xs if x is not None and x > 0]
    return statistics.median(xs) if xs else -1


def gb_fields(rr: dict, max_sclk: float) -> dict:
    """The witness columns of one arm's gb record (added to its RESULT row)."""
    ms = sorted(rr["ms"])
    med = statistics.median(ms)
    q = statistics.quantiles(ms, n=4) if len(ms) >= 4 else [ms[0], med, ms[-1]]
    sclk = _med_pos(rr["sclk"])
    return {"gb_sclk": round(sclk), "gb_sclk_pre": round(_med_pos(rr["sclk_pre"])),
            "gb_burst_ms": round(_med_pos(rr["pre_ms"]), 3), "gb_iqr_pct": round((q[2] - q[0]) / med * 100, 3),
            "gb_clock_ok": bool(0 < sclk <= max_sclk) if sclk > 0 else None}


def blk_fields(ts, flop: float, use_min: bool = False, ndigits=None) -> dict:
    """The blocked ruler's figure for the same arm, reported beside the scored gb figure."""
    ts = sorted(ts)
    v = ts[0] if use_min else statistics.median(ts)
    tf = flop / (v / 1e3) / 1e12
    if ndigits:
        return {"blk_latency_ms": round(v, 5), "blk_min_ms": round(ts[0], 5), "blk_tflops": round(tf, 2),
                "blk_iters": len(ts)}
    return {"blk_latency_ms": v, "blk_min_ms": ts[0], "blk_tflops": tf, "blk_iters": len(ts)}


def void_reason(burst_ms, max_ms: float, where: str):
    xs = [x for x in burst_ms if x and x > 0]
    if not xs:
        return f"SPEED RULER VOID (gb, {where}): no burst time was recorded"
    m = statistics.median(xs)
    if m > max_ms:
        return (f"SPEED RULER VOID (gb, {where}): GEMM burst median {m:.1f} ms > {max_ms:.0f} ms -- the image "
                f"hipBLASLt library is not in use (HIPBLASLT_TENSILE_LIBPATH="
                f"{os.environ.get('HIPBLASLT_TENSILE_LIBPATH')}); no gb figure is reported")
    return None


def say_void(msg: str) -> None:
    """On stdout AND stderr: fwd validation.py shows the stderr tail when stderr is non-empty."""
    print(msg, flush=True)
    print(msg, file=sys.stderr, flush=True)


# ------------------------------------------------------------------------------------------------ torch side
class _Burst:
    def __init__(self, torch, ng: int, seed: int = 1234):
        import torch.nn.functional as F                 # noqa: PLC0415 -- torch is the caller's
        self.torch, self.F, self.ng = torch, F, ng
        for name in ("hipblaslt", "cublaslt"):          # the same backend under two names
            try:
                torch.backends.cuda.preferred_blas_library(name)
                break
            except Exception:                           # noqa: BLE001
                continue
        self.backend = str(torch.backends.cuda.preferred_blas_library())
        g = torch.Generator(device="cuda").manual_seed(seed)
        self.x = torch.randn(GB_M, GB_K, generator=g, device="cuda", dtype=torch.bfloat16)
        self.w = torch.randn(GB_N, GB_K, generator=g, device="cuda", dtype=torch.bfloat16) * 0.02
        for _ in range(2):                              # initialises hipBLASLt with the image library NOW
            self.F.linear(self.x, self.w)
        torch.cuda.synchronize()

    def sync(self) -> None:
        self.torch.cuda.synchronize()

    def launch(self) -> None:
        for _ in range(self.ng):
            self.F.linear(self.x, self.w)


class GbRuler:
    """Construct BEFORE loading any arm; call after_load() after loading them; run() per shape."""

    def __init__(self, torch, args, default_iters: int):
        self.torch = torch
        self.iters = args.gb_iters or default_iters
        self.block, self.ng = max(1, args.gb_block), max(1, args.gb_ng)
        self.max_ms, self.max_sclk = args.gb_max_burst_ms, args.gb_max_sclk
        self.order = f"gb{self.ng}x{GB_M}x{GB_K}x{GB_N}bf16+palindromic{self.block}"
        self.burst, self.sclk, self.check_ms, self.env_before = None, None, [], None
        self.void, self.note = None, ""
        if os.environ.get("HIPBLASLT_TENSILE_LIBPATH") != IMAGE_BLAS:
            # preimport() did not assign it (e.g. an abbreviated --ruler). hipBLASLt reads the path at its first
            # use, which is the warm-up below, and the backend is forced there too: assign now and say so.
            self.note = f"env assigned late (was {os.environ.get('HIPBLASLT_TENSILE_LIBPATH')}); "
            assign_image_blas()
        if not os.path.isdir(IMAGE_BLAS):
            self.void = f"SPEED RULER VOID (gb, init): image hipBLASLt library {IMAGE_BLAS} not found"
            return
        try:
            self.burst = _Burst(torch, self.ng)
        except Exception as exc:                        # noqa: BLE001 -- a clean void, never a traceback
            self.void = f"SPEED RULER VOID (gb, init): the GEMM burst did not run: {exc!r}"

    def after_load(self):
        """Re-assert the image env (an arm's _env.py may have re-pointed it), time 3 bursts, start the witness."""
        if self.void:
            return self.void
        self.env_before = os.environ.get("HIPBLASLT_TENSILE_LIBPATH")
        assign_image_blas()
        t = self.torch
        e0, e1 = t.cuda.Event(enable_timing=True), t.cuda.Event(enable_timing=True)
        try:
            for _ in range(3):
                self.burst.sync()
                e0.record()
                self.burst.launch()
                e1.record()
                e1.synchronize()
                self.check_ms.append(e0.elapsed_time(e1))
        except Exception as exc:                        # noqa: BLE001
            self.void = f"SPEED RULER VOID (gb, after load): the GEMM burst did not run: {exc!r}"
            return self.void
        self.sclk = Sclk(find_sclk_file())
        self.void = void_reason(self.check_ms, self.max_ms, "after load")
        return self.void

    def header(self) -> str:
        return (f"gb ruler: {self.note}{self.order}, {gb_rounds(self.iters, self.block) * self.block} timed calls "
                f"per arm; "
                f"blas {self.burst.backend if self.burst else '?'}; HIPBLASLT_TENSILE_LIBPATH re-assigned "
                f"{self.env_before} -> {IMAGE_BLAS}; burst {' '.join(f'{x:.1f}' for x in self.check_ms)} ms; "
                f"sclk witness {self.sclk.path if self.sclk else None} now "
                f"{self.sclk.read() if self.sclk else -1} MHz")

    def run(self, labels, call):
        return run_gb(labels, call, self.burst, lambda: self.torch.cuda.Event(enable_timing=True), self.sclk,
                      self.iters, self.block)

    def check(self, rec, shape: str):
        return void_reason([x for rr in rec.values() for x in rr["pre_ms"]], self.max_ms, shape)


# ------------------------------------------------------------------------------------------------ A/A
_AA_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "core", "core.*", ".runs", ".build", "build")


def add_aa_arms(arms, aa_labels):
    """arms: [(label, Path)]. Inserts (LABEL_aa, byte copy in a fresh temp dir) right after each named arm."""
    if not aa_labels:
        return list(arms)
    known = [lb for lb, _ in arms]
    missing = [a for a in aa_labels if a not in known]
    if missing:
        raise SystemExit(f"--aa {missing}: no such arm (arms: {known})")
    out = []
    for label, path in arms:
        out.append((label, path))
        if label in aa_labels:
            tmp = Path(tempfile.mkdtemp(prefix=f"bench_aa_{label}_"))
            dst = tmp / Path(path).name
            shutil.copytree(path, dst, symlinks=True, ignore=_AA_IGNORE)
            atexit.register(shutil.rmtree, str(tmp), True)
            out.append((f"{label}_aa", dst.resolve()))
    return out


def add_aa_ratios(rows) -> None:
    """In place: the copy's row gets aa_of / aa_ratio (time ratio copy/original, scored stat) and blk_aa_ratio."""
    by = {r["arm"]: r for r in rows}
    for r in rows:
        a = r["arm"]
        if a.endswith("_aa") and a[:-3] in by:
            o = by[a[:-3]]
            r["aa_of"] = a[:-3]
            r["aa_ratio"] = round(r["latency_ms"] / o["latency_ms"], 5)
            if "blk_latency_ms" in r and "blk_latency_ms" in o:
                r["blk_aa_ratio"] = round(r["blk_latency_ms"] / o["blk_latency_ms"], 5)
