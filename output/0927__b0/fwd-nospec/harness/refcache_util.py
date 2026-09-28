#!/usr/bin/env python3
"""Cached fp32 reference (o, lse) for the three op.shape entries, causal only.

    python3 refcache_util.py fast proxy prod     # build, one shape at a time

Why a cache (op.refcache in the spec): the reference itself ran clean at prod with the
hipBLASLt library path set, but this machine emits intermittent GPU page faults and one
of them occasionally kills its process; caching removes that exposure from every round's gate.

Provenance (shape, dims, seed, sha256 of eager/impl.py and ut/common.py) is checked on
load; a mismatch RECOMPUTES from op/eager and says so -- a stale reference must never be
used silently. Lives outside the two hashed files so editing it invalidates nothing.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "1"
os.environ["HIPBLASLT_TENSILE_LIBPATH"] = "/home/lihuzhan/.local/hipblaslt-gfx1250/gfx1250"

HERE = Path(__file__).resolve().parent
REFCACHE = HERE / "refcache"
sys.path.insert(0, str(HERE / "ut"))


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def provenance(shape):
    from common import SHAPES
    return {"shape": shape, "dims": tuple(SHAPES[shape]), "seed": 0,
            "eager_sha": _sha(HERE / "eager" / "impl.py"),
            "common_sha": _sha(HERE / "ut" / "common.py")}


def _eager():
    from common import load_impl
    return load_impl(HERE / "eager")


def reference(shape, q, k, v, causal=True):
    """(o, lse, source). source is 'refcache' or 'eager (<why>)'."""
    import torch
    path = REFCACHE / f"{shape}.pt"
    why = "non-causal" if not causal else "no cache"
    if causal and path.exists():
        try:
            blob = torch.load(path, map_location="cpu")
            want, got = provenance(shape), blob.get("provenance", {})
            bad = [k_ for k_ in want if (tuple(got.get(k_, ())) if k_ == "dims"
                                         else got.get(k_)) != want[k_]]
            if not bad:
                return blob["o"].to(q.device), blob["lse"].to(q.device), "refcache"
            why = "provenance differs: " + ",".join(bad)
        except Exception as exc:  # a broken cache must never be worse than no cache
            why = f"unloadable: {exc}"
    o, lse = _eager()(q, k, v, causal=causal)
    return o, lse, f"eager ({why})"


def build(shapes):
    import torch
    from common import make_inputs
    REFCACHE.mkdir(exist_ok=True)
    fn = _eager()
    for shape in shapes:
        q, k, v = make_inputs(shape, seed=0)
        o, lse = fn(q, k, v, causal=True)
        torch.cuda.synchronize()
        for tag, t in (("o", o), ("lse", lse)):
            if not bool(torch.isfinite(t).all()):
                raise SystemExit(f"{shape}: {tag} not all finite -- refusing to cache")
        prov = dict(provenance(shape), torch=str(torch.__version__),
                    device=torch.cuda.get_device_properties(0).gcnArchName)
        torch.save({"provenance": prov, "o": o.cpu(), "lse": lse.cpu()},
                   REFCACHE / f"{shape}.pt")
        print(f"  {shape:6s} cached o{tuple(o.shape)} lse{tuple(lse.shape)}  {prov}", flush=True)
        del q, k, v, o, lse
        torch.cuda.empty_cache()


if __name__ == "__main__":
    build(sys.argv[1:] or ["fast", "proxy", "prod"])
