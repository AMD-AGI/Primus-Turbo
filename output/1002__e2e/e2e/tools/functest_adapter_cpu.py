"""CPU-only functional test of the 1002 e2e adapter's plumbing (NO GPU, NO HIP runtime: run with
HIP_VISIBLE_DEVICES=-1 inside fa-repro; CPU tensors; two fake FlyDSL trees written to a temp dir).

The autograd ENGINE is deliberately not used: its first backward() starts per-device worker threads,
which asks the HIP runtime for a device count and registers the process with KFD even with
HIP_VISIBLE_DEVICES=-1 (seen 2026-10-02 08:3x with an earlier version of this test: a short-lived
KFD holder, no GPU work). E2EAttnFunc.forward/backward are called directly with a fake ctx instead.

Checks: the 7-input forward / 7-output backward signature; per-step arm selection follows the
schedule for fwd AND bwd (bwd arm and (layer, step) travel in ctx); the BLAS guard restores a
re-pointed env and logs it; when event creation fails the attention timer disables itself instead
of raising into training; torch.cuda is never initialised. Prints FUNCTEST_OK.
"""
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

tmp = Path(tempfile.mkdtemp(prefix="e2e_functest_"))
for name, repoint in (("A", False), ("B", True)):
    d = tmp / f"tree{name}"
    d.mkdir()
    (d / "impl.py").write_text(f'''
import os, torch
{"os.environ['HIPBLASLT_TENSILE_LIBPATH'] = '/bogus/host/lib'" if repoint else ""}
LOG = []
def attn_fwd(q, k, v, scale, causal):
    LOG.append("fwd{name}")
    return q * 1.0, torch.zeros(q.shape[0], q.shape[2], q.shape[1])
def attn_bwd(do, q, k, v, o, lse, scale, causal):
    LOG.append("bwd{name}")
    return do * 1.0, torch.zeros_like(k), torch.zeros_like(v)
''')
os.environ["E2E_FLY_TREES"] = ('{"fa": {"fwd": "%s", "bwd": "%s"}, "fb": {"fwd": "%s", "bwd": "%s"}}'
                               % (tmp / "treeA", tmp / "treeA", tmp / "treeB", tmp / "treeB"))
os.environ["E2E_ATTN"] = "fa;fa,fb"
os.environ["E2E_EXPECT_BLAS_LIB"] = "/image/lib"
os.environ["HIPBLASLT_TENSILE_LIBPATH"] = "/image/lib"
os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "1"
os.environ["E2E_ATTN_EVENTS"] = str(tmp / "ev.jsonl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "attn_backends"))
import torch  # noqa: E402
import e2e_attn  # noqa: E402
from e2e_attn import arms  # noqa: E402

arms.check_versions = lambda: "flydsl (not imported in the CPU test)"


def _no_device():
    raise RuntimeError("no device in the CPU test")   # stands in for torch.cuda.Event without touching HIP


e2e_attn.TIMER._ev = _no_device
e2e_attn._log_blas_once = lambda: None


class Ctx(SimpleNamespace):
    def save_for_backward(self, *t):
        self.saved_tensors = t


# the module's own step counter + schedule, then the Function called by hand (no autograd engine)
mods = [e2e_attn.E2EAttention(causal=True) for _ in range(2)]
F = e2e_attn.E2EAttnFunc
logs = {}
for step in range(4):
    q = torch.randn(1, 8, 4, 16)
    k = torch.randn(1, 8, 2, 16)
    v = torch.randn(1, 8, 2, 16)
    ctxs, x = [], q
    for m in mods:
        s = m.calls
        m.calls += 1
        if m.layer == 0 and s > 0:
            e2e_attn.TIMER.drain(s)
        a = e2e_attn.arm_for_step(m.sched, s)
        c = Ctx()
        x = F.forward(c, x, k, v, 0.25, a[0], a[1], (m.layer, s))
        ctxs.append((c, a))
    g = torch.ones_like(x)
    for c, a in reversed(ctxs):
        out = F.backward(c, g)
        assert len(out) == 7 and out[3:] == (None, None, None, None), out
        assert c.bwd_arm == a[1] and c.meta[1] == step, (c.bwd_arm, c.meta)
        g = out[0]
    assert torch.allclose(g, torch.ones_like(q)), "dq did not flow through 2 layers"
    logs[step] = a[0]
want = ["fa", "fa", "fb", "fa"]                       # warm [fa], cycle [fa, fb]
assert [logs[s] for s in range(4)] == want, logs
fb = [sys.modules[n].LOG for n in sys.modules if n.startswith("e2e_fb_fwd_")]
assert fb and fb[0][:2] == ["fwdB", "fwdB"], fb
assert os.environ["HIPBLASLT_TENSILE_LIBPATH"] == "/image/lib", "BLAS guard did not restore"
assert arms.BLAS_EVENTS and "fb" in arms.BLAS_EVENTS[0][0], arms.BLAS_EVENTS
assert e2e_attn.TIMER is not None and not e2e_attn.TIMER.ok, "timer should have disabled itself"
assert not torch.cuda.is_initialized()
print("FUNCTEST_OK arms per step", want, "blas events", len(arms.BLAS_EVENTS))
