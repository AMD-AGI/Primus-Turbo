"""CPU-only check of bwd_s6_0341's stream fork/join and its env switches, through the REAL e2e loader.

NO GPU: run with HIP_VISIBLE_DEVICES=-1. torch.cuda.{current_stream,Stream} are replaced by fakes that only
log, every torch.cuda call that could initialise HIP raises, tensors are on the meta device, and the arm's
_launch (flyc.compile + launch) is replaced by a logger. What is exercised for real: the e2e loader
(attn_backends/e2e_attn/arms.py: _ensure_flydsl0341 + load-by-path of impl.py, which loads _env.py and
kernels.py under flydsl 0.3.4.1), impl.py's env parsing at import, its shape/grid decisions at the prod
shape (b4 s8192 hq32 hkv8 d128), and the exact order of launches / wait_stream / record_stream calls.

usage: sidestream_mock.py <mode>    mode in {default, norecord, serial}; one mode per process because the
switches are read at import (exactly as in the e2e process).
"""
import json
import os
import sys
import types

assert os.environ.get("HIP_VISIBLE_DEVICES") == "-1", "CPU-only test: HIP_VISIBLE_DEVICES=-1 required"
MODE = sys.argv[1]
ENV = {"default": {}, "norecord": {"FLY_BWD_RECORD_STREAM": "0"}, "serial": {"FLY_BWD_SIDE_STREAM": "0"}}[MODE]
for k in ("FLY_BWD_SIDE_STREAM", "FLY_BWD_RECORD_STREAM"):
    os.environ.pop(k, None)
os.environ.update(ENV)

E2E = "/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/e2e"
ARM = E2E + "/arms/bwd_s6_0341"
AITER = "/home/lihuzhan/code/aiter-src"
os.environ["E2E_FLY_TREES"] = json.dumps({"s6": {"fwd": E2E + "/arms/fwd_r16", "bwd": ARM}})

# aiter/__init__ may touch the device: stub the package chain (kernels.py needs only two leaf modules).
for pkg in ("aiter", "aiter.ops", "aiter.ops.flydsl", "aiter.ops.flydsl.kernels"):
    m = types.ModuleType(pkg)
    m.__path__ = [os.path.join(AITER, *pkg.split("."))]
    sys.modules[pkg] = m

import torch  # noqa: E402

LOG = []


class FakeStream:
    def __init__(self, name):
        self.name = name
        self.cuda_stream = id(self)

    def wait_stream(self, other):
        LOG.append(("wait_stream", self.name, other.name))

    def __repr__(self):
        return f"<{self.name}>"


MAIN = FakeStream("main")
N_SIDE = [0]


def _fake_stream_ctor(device=None, priority=0, **kw):
    N_SIDE[0] += 1
    return FakeStream(f"side{N_SIDE[0]}")


def _no_gpu(*a, **k):
    raise RuntimeError("GPU call reached in a CPU-only test")


torch.cuda.current_stream = lambda device=None: MAIN
torch.cuda.Stream = _fake_stream_ctor
for _n in ("init", "_lazy_init", "synchronize", "get_device_properties", "set_device", "device_count"):
    setattr(torch.cuda, _n, _no_gpu)
torch.Tensor.record_stream = lambda self, s: LOG.append(("record_stream", s.name, tuple(self.shape), str(self.dtype)))

sys.path.insert(0, E2E + "/attn_backends")
from e2e_attn import arms  # noqa: E402

bwd = arms.fly_bwd("s6")                      # the e2e shim's own loader and wrapper
mods = [m for n, m in sys.modules.items() if n.startswith("e2e_s6_bwd_")]
assert len(mods) == 1, list(sys.modules)
impl = mods[0]
import flydsl  # noqa: E402

assert flydsl.__version__ == "0.3.4.1" and "flydsl0341" in flydsl.__file__, flydsl.__file__
assert impl._k.__file__.startswith(ARM), impl._k.__file__


def _fake_assert_env():   # the real one minus the device-arch query
    assert flydsl.__version__.startswith("0.3.4") and "flydsl0341" in flydsl.__file__
    return flydsl.__version__, flydsl.__file__, "gfx1250 (not queried)"


impl._env.assert_environment = _fake_assert_env
impl._launch = lambda name, launcher, args: LOG.append(
    ("launch", name, [a.name for a in args if isinstance(a, FakeStream)][0]))

b, s, hq, hkv, d = 4, 8192, 32, 8, 128
M = "meta"
q = torch.empty((b, s, hq, d), dtype=torch.bfloat16, device=M)
k = torch.empty((b, s, hkv, d), dtype=torch.bfloat16, device=M)
v = torch.empty_like(k)
o = torch.empty_like(q)
do = torch.empty_like(q)
lse = torch.empty((b, hq, s), dtype=torch.float32, device=M)
dq, dk, dv = bwd(do, q, k, v, o, lse, d ** -0.5)
dq2, dk2, dv2 = bwd(do, q, k, v, o, lse, d ** -0.5)     # second call: side stream must be reused
assert (dq.shape, dk.shape, dv.shape) == (q.shape, k.shape, v.shape)
assert dq.dtype == dk.dtype == dv.dtype == torch.bfloat16
print(json.dumps({"mode": MODE, "env": ENV, "DQ_SIDE_STREAM": impl.DQ_SIDE_STREAM,
                  "DQ_SIDE_RECORD": impl.DQ_SIDE_RECORD, "side_streams_created": N_SIDE[0],
                  "flydsl": flydsl.__file__}))
half = len(LOG) // 2
assert LOG[:half] == LOG[half:], "two identical calls produced different event sequences"
for e in LOG[:half]:
    print("  ", e)

L = LOG[:half]
if MODE == "serial":
    assert L == [("launch", "delta", "main"), ("launch", "dkdv", "main"), ("launch", "dqg", "main")], L
    assert N_SIDE[0] == 0
else:
    assert N_SIDE[0] == 1
    i_delta = L.index(("launch", "delta", "main"))
    i_fork = L.index(("wait_stream", "side1", "main"))
    i_dkdv = L.index(("launch", "dkdv", "main"))
    i_dqg = L.index(("launch", "dqg", "side1"))
    i_join = L.index(("wait_stream", "main", "side1"))
    assert i_delta < i_fork < i_dkdv < i_dqg < i_join == len(L) - 1, L
    rec = [e for e in L if e[0] == "record_stream"]
    if MODE == "default":
        assert len(rec) == 8 and all(e[1] == "side1" for e in rec), rec      # q k v do o lse delta dq
        assert all(i_dqg < L.index(e) < i_join for e in rec)
    else:
        assert rec == [], rec
print(f"MOCK_OK {MODE}")
