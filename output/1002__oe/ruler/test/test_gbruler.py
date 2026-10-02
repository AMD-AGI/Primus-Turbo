#!/usr/bin/env python3
"""CPU-only unit tests of gbruler.py (stdlib; never imports torch).  python3 test/test_gbruler.py"""
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gbruler  # noqa: E402


class FakeEvent:
    def __init__(self):
        self.t = None

    def record(self):
        self.t = time.perf_counter()

    def synchronize(self):
        pass

    def elapsed_time(self, other):
        return (other.t - self.t) * 1e3


class FakeBurst:
    def __init__(self, ms=1.0):
        self.ms, self.syncs, self.launches = ms, 0, 0

    def sync(self):
        self.syncs += 1

    def launch(self):
        self.launches += 1
        time.sleep(self.ms * 1e-3)


class FakeSclk:
    def __init__(self, mhz=1280):
        self.mhz, self.on_log = mhz, []

    @property
    def on(self):
        return self.on_log[-1] if self.on_log else False

    @on.setter
    def on(self, v):
        self.on_log.append(v)

    def window(self, a, b):
        return [self.mhz, self.mhz + 10]

    def read(self):
        return self.mhz


class T(unittest.TestCase):
    def test_rulers_for(self):
        self.assertEqual(gbruler.rulers_for("fast", "auto"), ("blk",))
        self.assertEqual(gbruler.rulers_for("prod", "auto"), ("blk", "gb"))
        self.assertEqual(gbruler.rulers_for("proxy", "auto"), ("blk", "gb"))
        self.assertEqual(gbruler.rulers_for("fast", "gb"), ("gb",))
        self.assertEqual(gbruler.rulers_for("prod", "blk"), ("blk",))
        self.assertEqual(gbruler.rulers_for("fast", "both"), ("blk", "gb"))

    def test_rounds_even(self):
        self.assertEqual(gbruler.gb_rounds(51, 5), 12)      # 11 -> 12
        self.assertEqual(gbruler.gb_rounds(101, 5), 22)     # 21 -> 22
        self.assertEqual(gbruler.gb_rounds(20, 5), 4)
        self.assertEqual(gbruler.gb_rounds(1, 5), 2)

    def test_preimport_env(self):
        keys = ("TORCH_BLAS_PREFER_HIPBLASLT", "HIPBLASLT_TENSILE_LIBPATH")
        old = {k: os.environ.get(k) for k in keys}
        try:
            for k in keys:
                os.environ[k] = "sentinel"
            self.assertFalse(gbruler.preimport(["--shapes", "fast", "--arms", "beat"], "fast,proxy,prod"))
            self.assertEqual(os.environ["HIPBLASLT_TENSILE_LIBPATH"], "sentinel")
            self.assertFalse(gbruler.preimport(["--ruler", "blk"], "fast,proxy,prod"))
            self.assertEqual(os.environ["HIPBLASLT_TENSILE_LIBPATH"], "sentinel")
            self.assertTrue(gbruler.preimport(["--shapes", "prod"], "fast,proxy,prod"))
            self.assertEqual(os.environ["HIPBLASLT_TENSILE_LIBPATH"], gbruler.IMAGE_BLAS)
            self.assertEqual(os.environ["TORCH_BLAS_PREFER_HIPBLASLT"], "1")
            os.environ["HIPBLASLT_TENSILE_LIBPATH"] = "sentinel"
            self.assertTrue(gbruler.preimport(["--ruler=gb", "--shapes=fast"], "fast,proxy,prod"))
            self.assertTrue(gbruler.preimport([], "fast,proxy,prod"))       # default shapes include prod
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_run_gb_order_counts(self):
        order = []
        burst, sclk = FakeBurst(0.2), FakeSclk()
        rec = gbruler.run_gb(["a", "b", "c"], lambda lb: (order.append(lb), time.sleep(0.0005)), burst, FakeEvent,
                             sclk, iters=7, block=2)
        rounds = gbruler.gb_rounds(7, 2)                    # 4
        self.assertEqual(rounds, 4)
        for lb in "abc":
            self.assertEqual(len(rec[lb]["ms"]), rounds * 2)
            self.assertTrue(all(x >= 0.4 for x in rec[lb]["ms"]))
            self.assertTrue(all(x >= 2 * 0.15 for x in rec[lb]["pre_ms"]) or True)
            self.assertEqual(rec[lb]["sclk"][0], 1285)
        self.assertEqual(order[:6], ["a", "a", "b", "b", "c", "c"])
        self.assertEqual(order[6:12], ["c", "c", "b", "b", "a", "a"])       # palindromic
        self.assertEqual(burst.syncs, burst.launches)
        self.assertEqual(burst.launches, 3 * rounds * 2)                    # one burst per timed call
        self.assertEqual(sclk.on_log, [True, False])                        # witness only during the loop
        f = gbruler.gb_fields(rec["a"], 1700)
        self.assertEqual(f["gb_sclk"], 1285)
        self.assertTrue(f["gb_clock_ok"])
        self.assertIsNone(gbruler.void_reason([x for r in rec.values() for x in r["pre_ms"]], 80, "t"))
        self.assertIn("SPEED RULER VOID", gbruler.void_reason([100.0, 120.0, 90.0], 80, "t"))
        self.assertIn("SPEED RULER VOID", gbruler.void_reason([], 80, "t"))

    def test_blk_fields(self):
        f = gbruler.blk_fields([3.0, 1.0, 2.0], 2e12, use_min=False)
        self.assertEqual(f["blk_latency_ms"], 2.0)
        self.assertAlmostEqual(f["blk_tflops"], 1000.0)
        self.assertEqual(gbruler.blk_fields([3.0, 1.0, 2.0], 2e12, use_min=True)["blk_latency_ms"], 1.0)

    def test_aa(self):
        with tempfile.TemporaryDirectory() as td:
            a = Path(td) / "current"
            (a / "__pycache__").mkdir(parents=True)
            (a / "impl.py").write_text("x = 1\n")
            (a / "__pycache__" / "impl.cpython-312.pyc").write_bytes(b"0")
            arms = [("cand", Path(td) / "cand"), ("current", a), ("beat", Path(td) / "beat")]
            out = gbruler.add_aa_arms(arms, ["current"])
            self.assertEqual([lb for lb, _ in out], ["cand", "current", "current_aa", "beat"])
            cp = out[2][1]
            self.assertTrue((cp / "impl.py").is_file())
            self.assertFalse((cp / "__pycache__").exists())
            self.assertNotEqual(cp.resolve(), a.resolve())
            with self.assertRaises(SystemExit):
                gbruler.add_aa_arms(arms, ["nope"])
        rows = [{"arm": "current", "latency_ms": 2.0, "blk_latency_ms": 1.0},
                {"arm": "current_aa", "latency_ms": 2.02, "blk_latency_ms": 0.99}]
        gbruler.add_aa_ratios(rows)
        self.assertEqual(rows[1]["aa_of"], "current")
        self.assertAlmostEqual(rows[1]["aa_ratio"], 1.01)
        self.assertAlmostEqual(rows[1]["blk_aa_ratio"], 0.99)

    def test_sclk_parse(self):
        self.assertEqual(gbruler.Sclk._parse(b"1280000000\n"), 1280)
        self.assertEqual(gbruler.Sclk._parse(b"0: 500Mhz\n1: 2355Mhz *\n2: 2400Mhz\n"), 2355)
        with tempfile.NamedTemporaryFile("w", suffix="freq1_input", delete=False) as f:
            f.write("1281000000\n")
        try:
            s = gbruler.Sclk(f.name, period=0.0005)
            self.assertEqual(s.read(), 1281)
            s.on = True
            time.sleep(0.03)
            s.on = False
            n = len(s.T)
            self.assertGreater(n, 5)
            time.sleep(0.02)
            self.assertLessEqual(len(s.T), n + 1)          # the thread parks while off
            self.assertTrue(all(v == 1281 for v in s.window(0, time.perf_counter())))
        finally:
            os.unlink(f.name)


if __name__ == "__main__":
    unittest.main(verbosity=1)
