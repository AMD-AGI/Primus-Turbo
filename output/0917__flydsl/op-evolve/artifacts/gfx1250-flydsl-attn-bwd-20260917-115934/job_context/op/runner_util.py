"""Drive the job's runner so a run survives its own caller dying.

`runtime.runner` is `docker exec fa-repro bash -c ...`. The command inside the container
is a child of dockerd, not of the exec client, so a client timeout (or this process being
killed) leaves the remote process running and still holding the one GPU. Every helper here
therefore launches DETACHED, writes the exit code to a sentinel, and polls for it.

Run directories are keyed by the caller. On re-invocation with the same key:

  * sentinel absent, marker present -> the run is IN FLIGHT. Attach and keep polling.
    Starting a second one would put two processes on one card.
  * sentinel present               -> the run has FINISHED. Re-run it. Returning the
    cached result once handed a round the numbers from before its own fix.

Depends on nothing outside job_context except the shipped op-evolve `tools/` and the
framework's own runner classes, which is the point: the numbers in any report come from
the same path validation.py uses.
"""

from __future__ import annotations

import os
import shlex
import shutil
import sys
import time
from pathlib import Path

JOB_CONTEXT = Path(__file__).resolve().parent.parent
ARTIFACT = JOB_CONTEXT.parent
OP_EVOLVE = ARTIFACT.parent.parent
SPEC = JOB_CONTEXT / "gfx1250-flydsl-attn-bwd_final.yaml"
RUNS = JOB_CONTEXT / "op" / ".runs"

if str(OP_EVOLVE) not in sys.path:
    sys.path.insert(0, str(OP_EVOLVE))


def tools_path() -> Path:
    """The shipped tools/ directory -- op_flops.py is imported from here, never copied."""
    return OP_EVOLVE / "tools"


def build() -> object:
    from op_evolve.core.spec import load_spec
    from op_evolve.runners import build_runner

    spec = load_spec(SPEC)
    runner = build_runner(spec, name="gfx1250-flydsl-attn-bwd",
                          mounts=[Path("/home/lihuzhan")])
    runner.ensure()
    return runner


def run_detached(command: str, key: str, *, runner=None, timeout: float = 3600.0,
                 poll: float = 2.0, env: dict | None = None, quiet: bool = False):
    """Launch `command` through the runner, detached, and wait for its sentinel.

    Returns (returncode, stdout_text). A timeout here does NOT kill the remote process;
    it returns (None, partial_output) and leaves the sentinel to appear later, so a
    re-invocation with the same key attaches instead of starting a second run.
    """
    runner = runner or build()
    d = RUNS / key
    rc_file, out_file, marker = d / "rc", d / "out", d / "started"

    if rc_file.exists():
        # FINISHED. A finished run is re-run, never returned from cache.
        shutil.rmtree(d, ignore_errors=True)
    if not marker.exists():
        d.mkdir(parents=True, exist_ok=True)
        out_file.write_text("")
        marker.write_text(str(time.time()))
        exports = "".join(f"export {k}={shlex.quote(str(v))}; " for k, v in (env or {}).items())
        inner = f"{exports}{command}"
        launch = (
            f"cd {shlex.quote(str(d))} && "
            f"setsid bash -c {shlex.quote(f'{inner} > {d}/out 2>&1; echo $? > {d}/rc')} "
            f"</dev/null >/dev/null 2>&1 & disown; echo launched"
        )
        result = runner.run(launch, timeout=120)
        if not result.ok:
            raise RuntimeError(f"could not launch {key}: {result.summary(400)}")
        if not quiet:
            print(f"[runner] {key} launched", flush=True)

    deadline = time.time() + timeout
    shown = 0
    while time.time() < deadline:
        if rc_file.exists():
            rc = int(rc_file.read_text().strip() or 1)
            return rc, out_file.read_text()
        if not quiet:
            text = out_file.read_text() if out_file.exists() else ""
            if len(text) > shown:
                sys.stdout.write(text[shown:])
                sys.stdout.flush()
                shown = len(text)
        time.sleep(poll)
    return None, out_file.read_text() if out_file.exists() else ""


def python_in_container(script: Path, args: str = "", *, cwd: Path | None = None) -> str:
    """The command string that runs `script` under the container's python3."""
    cd = cwd or script.parent
    return f"cd {shlex.quote(str(cd))} && python3 {shlex.quote(str(script))} {args}".strip()
