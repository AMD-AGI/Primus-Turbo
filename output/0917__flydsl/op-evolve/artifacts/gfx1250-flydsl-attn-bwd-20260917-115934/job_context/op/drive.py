#!/usr/bin/env python3
"""Run a command in the job's container, detached, and stream its output.

    python3 drive.py --key smoke1 --cmd '...'  [--timeout 1800] [--env K=V ...]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runner_util  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--key", required=True)
ap.add_argument("--cmd", required=True)
ap.add_argument("--timeout", type=float, default=1800)
ap.add_argument("--env", action="append", default=[])
a = ap.parse_args()
env = dict(e.split("=", 1) for e in a.env)
rc, out = runner_util.run_detached(a.cmd, a.key, timeout=a.timeout, env=env)
print(f"\n[runner] rc={rc}")
sys.exit(0 if rc == 0 else (124 if rc is None else rc))
