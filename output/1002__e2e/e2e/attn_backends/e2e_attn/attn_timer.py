"""Per-training-step attention GPU time from CUDA event pairs (A0 e2e, 2026-10-02).

Why: on A0 the kineto trace of a profiled step held ~4 GPU kernels (a0_p3a, 09-28), so the
fwd/bwd attention time per step could not be read from traces. An event pair around every
attention call (the same scope as the `e2e::attn_fwd[..]` / `e2e::attn_bwd[..]` profiler
ranges: adapter copies, the arm's kernels, the ASM GQA sum) gives it for EVERY step and arm.

One JSON line per completed training step is appended to the file named by E2E_ATTN_EVENTS:
  {"step": 1-based, "arm_fwd": .., "arm_bwd": .., "fwd_ms": sum over layers, "bwd_ms": ..,
   "n_fwd": #layers seen, "n_bwd": .., "fwd": [ms per layer, model order], "bwd": [..]}

Cost: two event records per call (128 per step at 32 layers) and one non-blocking drain per
step (event.query(), elapsed_time() of completed pairs). Nothing here synchronises the device.
A step is written only when every pair of it has completed; the training loop's per-step
loss.item() (log_freq 1) normally guarantees that by the next step's first attention call.

Never raises into training: any exception disables the timer and is logged once.
This module does not import torch at import time (the CPU self-test injects fake events).
"""
from __future__ import annotations

import json
import sys
import threading


class AttnTimer:
    def __init__(self, path, event_factory=None, log=None):
        self.path = path
        self.lock = threading.Lock()
        self.pending = []        # (step0, layer, kind, arm, e0, e1)
        self.acc = {}            # step0 -> {"fwd": {layer: ms}, "bwd": {...}, "arm": {kind: arm}}
        self.ok = True
        self.written = 0
        self._ev = event_factory
        self._log = log or (lambda m: print(f"[e2e_attn] {m}", file=sys.stderr, flush=True))

    # ------------------------------------------------------------------ recording
    def _new_event(self):
        if self._ev is None:
            import torch
            self._ev = lambda: torch.cuda.Event(enable_timing=True)
        return self._ev()

    def begin(self):
        """Record the start event on the current stream; returns a token for end()."""
        if not self.ok:
            return None
        try:
            e = self._new_event()
            e.record()
            return e
        except Exception as exc:  # pragma: no cover - device specific
            self._fail("begin", exc)
            return None

    def end(self, e0, step0, layer, kind, arm):
        if e0 is None or not self.ok:
            return
        try:
            e1 = self._new_event()
            e1.record()
            with self.lock:
                self.pending.append((step0, layer, kind, arm, e0, e1))
        except Exception as exc:  # pragma: no cover
            self._fail("end", exc)

    # ------------------------------------------------------------------ folding
    def drain(self, before_step0=None):
        """Fold completed pairs of steps < before_step0 (all steps when None) and append every
        step whose pairs are all folded. Non-blocking: incomplete pairs stay pending."""
        if not self.ok:
            return
        try:
            with self.lock:
                keep = []
                for rec in self.pending:
                    step0, layer, kind, arm, e0, e1 = rec
                    if (before_step0 is None or step0 < before_step0) and e1.query():
                        a = self.acc.setdefault(step0, {"fwd": {}, "bwd": {}, "arm": {}})
                        a[kind][layer] = float(e0.elapsed_time(e1))
                        a["arm"][kind] = arm
                    else:
                        keep.append(rec)
                self.pending = keep
                busy = {r[0] for r in keep}
                ready = sorted(s for s in self.acc
                               if (before_step0 is None or s < before_step0) and s not in busy)
                lines = [self._line(s, self.acc.pop(s)) for s in ready]
            if lines:
                with open(self.path, "a") as f:
                    for line in lines:
                        f.write(line + "\n")
                self.written += len(lines)
        except Exception as exc:  # pragma: no cover
            self._fail("drain", exc)

    @staticmethod
    def _line(step0, a):
        def per_layer(d):
            if not d:
                return []
            n = max(d) + 1
            return [round(d[i], 4) if i in d else None for i in range(n)]
        rec = {"step": step0 + 1,
               "arm_fwd": a["arm"].get("fwd"), "arm_bwd": a["arm"].get("bwd"),
               "fwd_ms": round(sum(a["fwd"].values()), 4), "bwd_ms": round(sum(a["bwd"].values()), 4),
               "n_fwd": len(a["fwd"]), "n_bwd": len(a["bwd"]),
               "fwd": per_layer(a["fwd"]), "bwd": per_layer(a["bwd"])}
        return json.dumps(rec, separators=(",", ":"))

    def close(self):
        """atexit: fold whatever has completed; never blocks on the device."""
        self.drain(None)
        if self.ok:
            with self.lock:
                left = len(self.pending) + sum(len(a["fwd"]) + len(a["bwd"]) for a in self.acc.values())
            self._log(f"attn timer closed: {self.written} steps written, {left} pairs unwritten "
                      f"-> {self.path}")

    def _fail(self, where, exc):
        if self.ok:
            self.ok = False
            self._log(f"!! attn timer disabled ({where}: {exc!r}); training continues untimed")
