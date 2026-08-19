#!/usr/bin/env python3
"""Phased, memory-backed, manager-driven mega-MoE kernel-optimization campaign.

Everything runs LOCALLY on this box: no ssh, no rsync. Every measurement goes
through report/bench_mega_moe.sh (which drives the rootless-runc ROCm container
itself), and every number is scored against a target JSON
(report/baseline/mi355x_19fe104.json by default) with report/extract_performance.py.

The score is the TOTAL fused latency of the five mega-MoE kernel stages
(dispatch fwd/bwd/wgrad + combine fwd/bwd) -- LOWER IS BETTER -- and the campaign
reports, at every round and at the end, the remaining gap to the target.

Phases (sizes scale with --rounds; the 30-round shape is the default:
3-5 analyze / ~23-25 optimize / 2 review):

  ANALYZE  (rounds 1..A, A in [analyze_min, analyze_max]):
      Deep analysis only. The agent profiles in-turn (`bash bench.sh`,
      `bash incontainer.sh '<rocprofv3 ...>'`), reads the knowledge base, and
      writes/refines a DETAILED goal file (goal.md). No kernel edit is kept in
      this phase. After analyze_min rounds the manager decides whether goal.md is
      detailed enough to start optimizing; if not, analysis continues up to
      analyze_max.

  OPTIMIZE (the middle rounds):
      Each round runs a real measure-driven session and leaves ONE coherent
      change on disk, which the orchestrator then re-measures authoritatively
      (--repeat runs, best = fastest total). Kept (committed as the new best) iff
      it is at least --keep-threshold faster. memory.md accumulates every scheme +
      result + pitfall + KB-accuracy note. If 5 consecutive optimize rounds yield
      < 2% cumulative gain, a REPLAN round rewrites goal.md.

  REVIEW   (last --review-rounds rounds):
      Code review, fixes applied, then ALL kept round-commits are squashed into
      ONE clean commit.

The manager runs EVERY round: it reviews the round, adjudicates any `NEED_RULING:`
the working agent raised, and hands the next round a concrete direction. Hard
rule: it may never emit a negative/ceiling verdict; such output is rejected and
re-asked.

This script NEVER pushes. Logs live under ~/flydsl_campaigns/<ts>/ (state.json,
goal.md, memory.md, profile_data.md, run.jsonl, run.log, runs/, rounds/round-NN/).

Run detached:
    IS_SANDBOX=1 nohup python3 flydsl_campaign_claude.py <args> > /tmp/campaign.log 2>&1 &
"""
import argparse
import glob
import importlib.util
import json
import os
import re
import shlex
import subprocess
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_BENCH_SH = REPO_ROOT / "report" / "bench_mega_moe.sh"
DEFAULT_TARGET_JSON = REPO_ROOT / "report" / "baseline" / "mi355x_19fe104.json"
DEFAULT_EXTRACTOR = REPO_ROOT / "report" / "extract_performance.py"
DEFAULT_KB_DIR = REPO_ROOT / "agent" / "skills" / "kernel-optimize"
DEFAULT_CAMPAIGN_ROOT = Path.home() / "flydsl_campaigns"
DEFAULT_RUNC_STATE = Path.home() / ".local" / "share" / "rootless-runc" / "state"
# $HOME is bind-mounted at /io by ~/start_container.sh
CONTAINER_HOME = "/io"

OPUS_1M = "claude-opus-5[1m]"

# The commit author for every commit the campaign makes (no AI coauthor).
COMMIT_NAME = "zhenhuang12"
COMMIT_EMAIL = "Zhen.Huang@amd.com"

# Substrings a manager ruling is forbidden to contain (no negative/ceiling verdicts).
NEGATIVE_MARKERS = [
    "不行", "不可能", "已经到上限", "已经到上线", "到顶", "到达上限", "达到上限",
    "无法优化", "没有优化空间", "没有空间", "收益低", "不值得", "没救", "极限了",
]


def sh(cmd, cwd=None, check=True, capture=True, timeout=None, input_text=None, env=None):
    p = subprocess.run(
        cmd, shell=True, cwd=cwd, timeout=timeout,
        input=input_text,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        text=True, env=env,
    )
    if check and p.returncode != 0:
        raise RuntimeError(
            f"command failed ({p.returncode}): {cmd}\n"
            f"--- stdout ---\n{p.stdout}\n--- stderr ---\n{p.stderr}"
        )
    return p.returncode, (p.stdout or ""), (p.stderr or "")


class Logger:
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = log_dir / "run.jsonl"
        self.text_path = log_dir / "run.log"

    def event(self, **kwargs):
        kwargs.setdefault("ts", time.time())
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(kwargs, ensure_ascii=False) + "\n")

    def line(self, msg):
        stamped = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(stamped, flush=True)
        with open(self.text_path, "a") as f:
            f.write(stamped + "\n")


# --------------------------------------------------------------------------- #
# Measurement                                                                  #
# --------------------------------------------------------------------------- #

def load_extractor(path: Path):
    """Import report/extract_performance.py as a module. It is stdlib-only, so the
    orchestrator can reuse its log parser / table renderer on the bare host instead
    of re-implementing (and drifting from) the scoring."""
    spec = importlib.util.spec_from_file_location("mega_moe_extract", str(path))
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import the extractor: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class BenchHarness:
    """Local measurement layer.

    - `bench()` runs report/bench_mega_moe.sh (it finds/starts the rootless-runc
      container itself) into a fresh OUT_DIR, then scores the produced logs.
    - `exec_in_container()` runs an arbitrary command inside the same container,
      for profiling (rocprofv3, ISA dumps, micro-benches) -- the host has no ROCm
      python environment.
    - `clear_caches()` drops flydsl's JIT and/or autotune disk caches. The autotune
      cache is keyed by shape/arch/toolchain, NOT by kernel source, so after
      changing a config list or a tile pin a stale tuned config would be reused
      and the new one never explored.
    """

    CACHE_PATHS = {"jit": "/root/.flydsl/cache", "autotune": "/root/.flydsl/autotune"}

    def __init__(self, repo: Path, campaign_dir: Path, extract, target_json: Path,
                 bench_sh: Path, modes: str, models: str, iters: int,
                 num_processes: int, case=None, runc_state=DEFAULT_RUNC_STATE,
                 container=""):
        self.repo = repo
        self.campaign_dir = campaign_dir
        self.extract = extract
        self.bench_sh = bench_sh
        self.modes = modes
        self.models = models
        self.iters = iters
        self.num_processes = num_processes
        self.case = case
        self.runc_state = Path(runc_state)
        self.container = container
        self.runs_dir = campaign_dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.target_json = Path(target_json)
        self.target = json.loads(self.target_json.read_text())
        self.target_stages = self.target["stages"]
        self.target_total = self._total(self.target_stages)

    # ---- container ------------------------------------------------------- #
    def container_name(self):
        """The running rootless-runc container (first one, like bench_mega_moe.sh).
        Re-discovered on every call unless one was pinned explicitly: the container
        can be restarted mid-campaign, and a cached id would then be dead."""
        if self.container:
            return self.container
        _, out, _ = sh(f"runc --root {shlex.quote(str(self.runc_state))} list --quiet",
                       check=False, timeout=60)
        for cid in (out or "").split():
            _, st, _ = sh(f"runc --root {shlex.quote(str(self.runc_state))} state {shlex.quote(cid)}",
                          check=False, timeout=60)
            try:
                if json.loads(st).get("status") == "running":
                    return cid
            except (ValueError, TypeError):
                continue
        return ""

    def to_container_path(self, host_path):
        host_path = str(Path(host_path).resolve())
        home = str(Path.home())
        if not host_path.startswith(home + "/"):
            raise RuntimeError(f"{host_path} is outside $HOME, so the container cannot see it")
        return f"{CONTAINER_HOME}/{host_path[len(home) + 1:]}"

    def exec_in_container(self, shell_cmd: str, timeout=1800, cwd=None):
        cid = self.container_name()
        if not cid:
            return 127, "", "no running container (start one with ~/start_container.sh)"
        cwd = self.to_container_path(cwd or self.repo)
        cmd = (f"runc --root {shlex.quote(str(self.runc_state))} exec --cwd {shlex.quote(cwd)} "
               f"{shlex.quote(cid)} bash -lc {shlex.quote(shell_cmd)}")
        return sh(cmd, check=False, timeout=timeout)

    def clear_caches(self, what):
        """what in {none, jit, autotune, all}."""
        if what in (None, "", "none"):
            return
        keys = list(self.CACHE_PATHS) if what == "all" else [what]
        paths = " ".join(self.CACHE_PATHS[k] for k in keys if k in self.CACHE_PATHS)
        if paths:
            self.exec_in_container(f"rm -rf {paths}", timeout=300)

    # ---- scoring --------------------------------------------------------- #
    def _total(self, stages, field="fused_ms"):
        vals = [stages.get(k, {}).get(field) for k, _ in self.extract.STAGES]
        return None if any(v is None for v in vals) else sum(vals)

    def score_run_dir(self, out_dir: Path):
        """Parse every *.log of a finished run dir into a score record.

        Returns a dict: ok / total_ms / gap_ms / gap_pct / stages / checks_failed /
        missing / summary (the human per-kernel table vs the target)."""
        logs = sorted(glob.glob(str(Path(out_dir) / "*.log")))
        merged, checks = {}, []
        for lg in logs:
            try:
                parsed = self.extract.parse_log(lg)
            except Exception:  # noqa: BLE001 - a truncated log must not kill the round
                continue
            for case, data in parsed.items():
                entry = merged.setdefault(case, {"meta": {}, "stages": {}, "checks": []})
                entry["meta"].update(data["meta"])
                entry["stages"].update(data["stages"])
                entry["checks"].extend(data.get("checks", []))
        if not merged:
            return {"ok": False, "total_ms": None, "gap_ms": None, "gap_pct": None,
                    "stages": {}, "checks_failed": [], "missing": [k for k, _ in self.extract.STAGES],
                    "summary": f"no parsable benchmark output in {out_dir}", "out_dir": str(out_dir)}
        case = self.case if self.case in merged else sorted(merged)[0]
        entry = merged[case]
        stages = entry["stages"]
        checks = entry.get("checks", [])
        missing = [k for k, _ in self.extract.STAGES if "fused_ms" not in stages.get(k, {})]
        failed = [what for what, verdict in checks if verdict != "PASS"]
        total = self._total(stages)
        gap = None if total is None else total - self.target_total
        meta = dict(entry["meta"])
        meta["checks"] = checks
        try:
            summary = self.extract.summarize(
                str(out_dir), case, stages, meta,
                self.target.get("name", str(self.target_json)),
                self.target.get("case", case), self.target_stages,
                dict(self.target.get("config", {}), gpu=self.target.get("gpu", "?")),
            )
        except Exception as e:  # noqa: BLE001 - the table is nice-to-have, the number is not
            summary = f"(summary render failed: {e})"
        return {
            "ok": not missing and not failed,
            "total_ms": total,
            "gap_ms": gap,
            "gap_pct": None if not total else (gap / self.target_total * 100),
            "stages": {k: stages.get(k, {}) for k, _ in self.extract.STAGES},
            "checks_failed": failed,
            "missing": missing,
            "summary": summary,
            "out_dir": str(out_dir),
        }

    # ---- bench ------------------------------------------------------------ #
    def bench(self, label, timeout=3600):
        """One full benchmark of the CURRENT working tree. Returns (score_record, rc)."""
        out_dir = self.runs_dir / f"{label}-{time.strftime('%Y%m%d-%H%M%S')}"
        env = dict(os.environ)
        env.update(OUT_DIR=str(out_dir), MODES=self.modes, MODELS=self.models,
                   ITERS=str(self.iters), NUM_PROCESSES=str(self.num_processes))
        if self.container:
            env["CONTAINER_NAME"] = self.container
        try:
            rc, out, err = sh(f"bash {shlex.quote(str(self.bench_sh))}", cwd=str(self.repo),
                              check=False, timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            rc, out, err = 124, "", f"bench timed out after {timeout}s"
        (out_dir).mkdir(parents=True, exist_ok=True)
        (out_dir / "orchestrator_stdout.txt").write_text((out or "") + "\n--- stderr ---\n" + (err or ""))
        rec = self.score_run_dir(out_dir)
        rec["rc"] = rc
        return rec, rc


def git(repo: Path, args, check=True):
    rc, out, err = sh(f"git {args}", cwd=str(repo), check=check)
    return out.strip()


def ensure_clean_tree(repo: Path):
    status = git(repo, "status --porcelain --untracked-files=no --ignore-submodules=all")
    if status:
        raise SystemExit(
            f"refusing to start: {repo} has uncommitted changes to tracked files:\n{status}\n"
            f"commit or stash them first."
        )


def untracked_files(repo: Path):
    out = git(repo, "ls-files --others --exclude-standard")
    return set(out.splitlines()) if out else set()


def fmt_ms(v):
    return "n/a" if v is None else f"{v:.3f}"


# --------------------------------------------------------------------------- #
# Agent invocation                                                            #
# --------------------------------------------------------------------------- #

_MAX_NUDGES = 20
_CONTINUE_NUDGE = ("先仔细阅读 agent/skills/kernel-optimize 知识库寻找可能的路，继续。"
                   "做完再输出最终 <<<ROUND_REPORT ... ROUND_REPORT。")


def _parse_stream(out):
    """Parse a claude stream-json stdout: return (session_id, result_text, cost, is_error)
    from the final type==result event (session_id also from any event carrying it)."""
    session_id, result_text, cost, is_error = None, "", 0.0, False
    for line in out.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        session_id = ev.get("session_id", session_id)
        if ev.get("type") == "result":
            result_text = ev.get("result", result_text)
            cost = ev.get("total_cost_usd", ev.get("cost_usd", cost)) or cost
            is_error = bool(ev.get("is_error", is_error))
    return session_id, result_text, cost, is_error


def call_agent(prompt: str, repo: Path, model: str, effort: str, budget: float,
               timeout: int, log: Logger, extra_dirs=(), label="", raw_path=None,
               min_seconds=0, continue_prompt=None):
    """One headless claude turn (deep turns keep a min-round floor). Returns
    (result_text, cost_usd, is_error).

    Uses stream-json so the FULL raw transcript (thinking + every tool call/result)
    is captured; raw_path gets the verbatim event stream (all turns concatenated).

    min_seconds>0 enforces a minimum round duration: if the agent returns early it
    is resumed (`claude -p --resume <session_id>`) with `continue_prompt` (default
    _CONTINUE_NUDGE) and told to keep working, looping until the floor is met, the
    deep-timeout is exhausted, it errors, or it no-ops _MAX_NUDGES times. Only deep
    rounds (analyze/optimize/replan) pass min_seconds; manager/review turns don't."""
    add_dirs = " ".join(f'--add-dir "{d}"' for d in extra_dirs)
    base = (f'--output-format stream-json --verbose --permission-mode bypassPermissions '
            f'--model {shlex.quote(model)} --effort {effort} --max-budget-usd {budget} {add_dirs}')

    def _turn(turn_prompt, resume_id, turn_timeout):
        resume = f'--resume {shlex.quote(resume_id)} ' if resume_id else ''
        # Prompt on stdin, NOT argv: goal+memory(300K)+kb grows and a >128KB prompt
        # overflows Linux MAX_ARG_STRLEN -> exec fails E2BIG. `claude -p` with no
        # prompt arg reads it from stdin.
        cmd = f'IS_SANDBOX=1 claude -p {resume}{base}'
        try:
            return sh(cmd, cwd=str(repo), check=False, timeout=turn_timeout, input_text=turn_prompt)
        except subprocess.TimeoutExpired:
            return None, "", "timeout"

    log.line(f"[{label}] claude model={model} effort={effort} budget=${budget} timeout={timeout}s"
             f"{f' min_round={min_seconds//60}min' if min_seconds else ''} ...")
    t0 = time.time()
    raw_chunks, reports, total_cost = [], [], 0.0
    session_id, is_error = None, False

    rc, out, err = _turn(prompt, None, timeout)
    if rc is None:
        log.line(f"[{label}] claude TIMED OUT after {timeout}s (turn kept whatever it wrote to disk)")
        return "", 0.0, True
    raw_chunks.append(out)
    if rc != 0:
        log.line(f"[{label}] claude failed rc={rc} in {time.time()-t0:.0f}s: {err[:1500]}")
        _write_raw(raw_path, raw_chunks)
        return "", 0.0, True
    sid, rtext, cost, is_error = _parse_stream(out)
    session_id = sid or session_id
    total_cost += cost
    if rtext:
        reports.append(rtext)

    nudges = 0
    cont = continue_prompt or _CONTINUE_NUDGE
    while (min_seconds and not is_error and session_id
           and (time.time() - t0) < min_seconds and nudges < _MAX_NUDGES):
        remaining = int(timeout - (time.time() - t0))
        if remaining < 120:
            break
        nudges += 1
        log.line(f"[{label}] {(time.time()-t0)/60:.1f}min < {min_seconds//60}min floor -> resume '继续' #{nudges}")
        rc, out, err = _turn(cont, session_id, remaining)
        if rc is None:
            log.line(f"[{label}] resume #{nudges} TIMED OUT; keeping work so far")
            break
        raw_chunks.append(out)
        if rc != 0:
            log.line(f"[{label}] resume #{nudges} failed rc={rc}: {err[:800]}; keeping work so far")
            break
        sid, rtext, cost, is_error = _parse_stream(out)
        session_id = sid or session_id
        total_cost += cost
        if rtext:
            reports.append(rtext)

    dt = time.time() - t0
    _write_raw(raw_path, raw_chunks)
    result_text = "\n\n".join(reports)  # concat so the REPORT block is never lost across nudges
    if not result_text:
        log.line(f"[{label}] WARNING: no result event parsed from stream (see raw)")
    log.line(f"[{label}] done in {dt/60:.1f}min ({nudges} nudge{'s' if nudges != 1 else ''}), cost=${total_cost:.2f}")
    return result_text, total_cost, is_error


def _write_raw(raw_path, chunks):
    if raw_path is None:
        return
    try:
        Path(raw_path).write_text("\n".join(chunks))
    except OSError:
        pass


def extract_markers(text: str, tag: str):
    """Pull every `TAG: ...` line's payload out of an agent message."""
    out = []
    for line in text.splitlines():
        m = re.match(rf"^\s*{tag}:\s*(.+?)\s*$", line)
        if m:
            out.append(m.group(1))
    return out


def extract_report(text: str):
    """Pull the LAST <<<ROUND_REPORT ... ROUND_REPORT block (or None). Last, because
    a min-round-sec nudged turn concatenates several turns' outputs and the final
    report is the most complete."""
    ms = re.findall(r"<<<ROUND_REPORT\s*(.+?)\s*ROUND_REPORT", text, re.DOTALL)
    return ms[-1].strip() if ms else None


# --------------------------------------------------------------------------- #
# Prompt blocks                                                               #
# --------------------------------------------------------------------------- #

KB_BLOCK = """MANDATORY, EVERY ROUND, BEFORE ANYTHING ELSE: read the knowledge base at {kb}
(the accumulated gfx950 / MI35x kernel-optimization memory of this repo). Its
navigation index (SKILL.md) is reproduced IN FULL below so you cannot skip it --
read it now, then open the specific cards relevant to this round.

- ALWAYS check the documented dead ends before proposing a change -- anything
  marked as already-tried-and-failed must not be re-walked; pick a different lever.
- As you work, JUDGE whether the KB is真实 (accurate). When the KB claims something
  and you observe otherwise, record it explicitly (skill 说 X, 实测 Y).
- You MUST report which card(s) you actually opened this round in the ROUND_REPORT
  `kb_read:` field. A round that consulted no card is not acceptable.

===== KB navigation index ({kb}/SKILL.md, verbatim) =====
{kb_index}
===== end KB index -- open the relevant cards from {kb} before proceeding =====
"""

NO_CEILING_RULE = """HARD RULE — no ceiling / no defeatism (even in the goal file, memory, and your
own reasoning): never write that the target is impossible, already at its
limit/ceiling, has no headroom, or that a regime "cannot" improve ("不可能 / 已经
到上限 / 到顶 / 无进一步可用杠杆 / 没有优化空间" and the like are FORBIDDEN). It is
always fine — and required — to say "lever X gave <measured result>, so the next
thing to try is Y". Report what a specific attempt measured and name the next
concrete lever; do not pronounce ceilings."""

DATA_TOOLS_BLOCK = """YOU HAVE THE LIVE 8-GPU BOX IN THIS TURN. Everything is local -- no ssh, no rsync.
Three ready-made commands (already wired to the right container / bench / target --
just run them with the Bash tool):

  bash {bench_sh} [label]
      THE RULER. Runs report/bench_mega_moe.sh on your CURRENT working tree
      (it finds/starts the rootless-runc ROCm container itself, sweeps both fused
      modes over 8 ranks), then prints the per-kernel table and the line
      "gap to baseline: ... ms" against the target {target_name}. This is the SAME
      ruler the orchestrator uses to keep or revert your round. One call ≈ {bench_min} min.
      Run it to measure EVERY edit.

  bash {incontainer_sh} '<shell command>'
      Runs an arbitrary command INSIDE the ROCm container (rocprofv3, ISA dump,
      rocm-smi, python, a targeted micro-bench). The host has no ROCm python, so
      all profiling must go through this.

  bash {cache_sh} <jit|autotune|all>
      Drops flydsl's disk caches in the container. The autotune cache is keyed by
      shape/arch/toolchain, NOT by kernel source: if you change an autotune config
      list, a tile pin, or anything that should change the chosen config, you MUST
      clear `autotune` before benching or you will re-measure the OLD config and
      the number will be a lie. Clear `jit` if you suspect a stale compiled kernel.

Because you can measure inside this turn, DO NOT guess-once-and-quit. Work like a
real optimization session: form a hypothesis from the profile + KB -> edit ->
`bash {bench_sh}` -> read the total / gap -> keep it only if faster, else revert your
edit and try the next hypothesis. Iterate through SEVERAL hypotheses. Keep the
best-performing version of the files on disk when you finish. A round that made one
blind edit and exited in a few minutes is a FAILED round.

(Optional deferred channel: a line `RUN_CMD: <cmd>` is executed in the container
AFTER the turn and fed back next round -- but prefer running commands yourself now,
in-turn, so you get the feedback immediately.)

If you hit a real fork in the road and want the manager to decide, write:
    NEED_RULING: <your specific question>
The manager answers before the next round. It always gives a concrete direction
-- there is no "give up" answer."""

SCORE_BLOCK = """SCORING (口径, identical for you and for the orchestrator):
- The score is the TOTAL fused latency of the 5 mega-MoE kernel stages
  (dispatch fwd nt / dispatch bwd dgrad nn / dispatch bwd wgrad tn /
   combine fwd nt / combine bwd dgrad nn), summed. LOWER IS BETTER.
- Target ({target_name}): {target_total:.3f} ms total. Per-stage target (fused ms):
{target_rows}
- Campaign start on this box: {base_total} ms. Current best: {best_total} ms.
  Remaining gap to target: {gap} ms ({gap_pct}).
- A measurement only counts if ALL 5 stages are present AND every `[check] ... PASS`
  accuracy check passes. A faster-but-wrong kernel is always rejected."""


ANALYZE_PROMPT = """You are the ANALYSIS phase of an autonomous mega-MoE kernel-optimization campaign in {repo}.

Campaign goal (what we must ultimately achieve): {goal}
Primary focus file(s): {focus}
This is analyze round {round_num} (analyze phase runs {analyze_min}-{analyze_max} rounds total).

必须非常详尽. Your job this round is NOT to edit the kernels for speed -- it is to
UNDERSTAND them deeply and PLAN.

{no_ceiling}

{score_block}

Specifically:
1. Read the focus file(s) and the surrounding flydsl mega-MoE machinery thoroughly.
2. Collect raw profiling data IN THIS TURN with `bash {incontainer_sh} '<rocprofv3 ...>'`
   (MFMA util, occupancy, LDS/bank conflicts, mem stalls, VGPR/AGPR pressure, L2 hit,
   XGMI traffic; ISA if useful) and `bash {bench_sh}` for the current per-stage numbers.
   Profile thoroughly -- multiple stages / multiple counters -- do not settle for one dump.
3. Analyze every metric against the KB and against the per-stage target above: WHERE
   is the gap (which stage, GEMM leg vs comm leg), what is each stage bound by, and
   which documented levers plausibly move that bound?
4. Write / refine the goal file at {goal_file}. It must end this phase as a
   DETAILED, ranked optimization plan: for each candidate optimization -- the
   hypothesis, the mechanism, the KB card it comes from (or why it's novel), which
   stage's ms it should buy back, the risk, and a rough priority. This file is what
   the optimize phase will execute against, so make it concrete and specific.

{kb_block}

{data_tools_block}

Latest raw data already collected (from earlier RUN_CMD calls):
{profile_data}

Per-run local context (READ THIS FIRST -- the plan so far and everything already tried):
{local_context}

Do NOT keep any speed edit this round (analysis only; the orchestrator reverts
kernel edits made during analysis). You MUST (re)write {goal_file} so it reflects
this round's conclusions before you finish.
End your message with:
<<<ROUND_REPORT
kb_read: <the KB card(s) you actually opened this round -- REQUIRED, non-empty>
findings: <the 2-3 most important things you learned this round>
kb_check: <any place the KB was inaccurate, or "consistent">
goal_status: <"ready to optimize" | "need another analyze round because ...">
ROUND_REPORT"""


OPTIMIZE_PROMPT = """You are ONE optimize round of an autonomous mega-MoE kernel-optimization campaign in {repo}.

必须非常详尽，必须进行深度思考. Use your full context budget; think hard before you
touch code. A shallow round is a wasted round.

{no_ceiling}

Campaign goal: {goal}
Primary focus file(s): {focus}
This is optimize round {opt_round} (round {round_num} overall).

{score_block}

READ FIRST, every round (given to you in full below; {goal_file} and {memory_file}
are also on disk if you need the untruncated versions): the goal plan and the
campaign memory of every scheme already tried + its measured result + pitfalls.
Then run a real, measure-driven optimization SESSION this round (not a single
blind edit): work down the goal plan's untried/not-yet-succeeded ideas, and for
EACH one -- edit -> `bash {bench_sh}` -> read the total/gap -> keep if faster, else
revert that edit and try the next. Iterate through several hypotheses; profile with
`bash {incontainer_sh}` whenever you need to know WHY a change did/didn't help. Do
not stop after one attempt. When you finish, leave the single best-performing
version of the files on disk (the orchestrator will re-measure it as the
authoritative keep/revert). This round should take real time and real thought -- a
few-minute one-shot round is a failure.

Per-run local context:
{local_context}

{kb_block}

{data_tools_block}

Latest raw data (profile_data.md):
{profile_data}

Manager's directive for this round:
{directive}

HARD CONSTRAINTS:
- Explore several hypotheses, but the change you LEAVE on disk must be one
  coherent, self-contained improvement -- not a grab-bag of half-tried edits.
- {focus} is your PRIMARY FOCUS / entry point, but you are NOT restricted to it:
  edit ANY production implementation file needed to land the win (the kernels,
  flydsl helpers, the EP intranode comm path, dispatch/wiring, etc.). Correctness
  (the [check] gate) and the measured benchmark are the real guardrails, not a
  file whitelist.
- FORBIDDEN to touch (would fake the result): the benchmark/verify harness
  (benchmark/ops/training/bench_mega_moe.py), report/ (bench_mega_moe.sh,
  extract_performance.py, baseline/*.json -- the ruler and the target), any test,
  and any config that defines the scored shapes. Never edit these to make the
  number look better.
- Do NOT run any git command. Leave your best version in the working tree; the
  orchestrator re-benchmarks it (repeated runs, all raw numbers logged) and
  commits or reverts it.
- The win must be a real, correctness-preserving speedup: every `[check] ... PASS`
  in the bench output must stay PASS.
- If your change should alter an autotune decision, run `bash {cache_sh} autotune`
  before benching, or you will re-measure the cached old config.
- Code conventions: comments in English only; naming consistent with the file;
  reuse the existing flydsl helpers instead of reimplementing.

End your message with:
<<<ROUND_REPORT
kb_read: <the KB card(s) you actually opened this round -- REQUIRED, non-empty>
scheme: <the change you LEFT on disk>
tried: <each hypothesis you tested in-turn and its bench result, e.g. "H1 wider combine store -> 17.51 ms (no); H2 bk unroll -> 17.28 ms (kept)"> -- REQUIRED, must show you measured
kb_cards: <card(s) that informed the change, or "none applied">
kb_check: <KB inaccuracy you found, or "consistent">
pitfalls: <anything future rounds should avoid>
expect: <which stage's ms this should buy back and why>
ROUND_REPORT"""


REPLAN_PROMPT = """You are a REPLAN round of an autonomous mega-MoE kernel-optimization campaign in {repo}.

The last {stall_window} optimize rounds produced < {stall_pct:.0f}% cumulative gain.
That does NOT mean the kernels are optimal -- it means the current plan's ideas are
exhausted or mis-prioritized. 必须非常详尽. Your job: produce a genuinely DIFFERENT
plan.

{no_ceiling}

Campaign goal: {goal}
Primary focus file(s): {focus}

{score_block}

READ FIRST (given in full below): the current goal plan and the full campaign
memory -- what was tried and what the measurements said. Then REWRITE {goal_file}
with a fresh, ranked plan that attacks the problem from angles not yet tried -- a
different stage, a different bound (GEMM leg vs comm leg vs overlap efficiency), a
coarser structural change, a combination. Use `bash {incontainer_sh} '<cmd>'` to
collect any new profiling data you need to justify the new plan.

Per-run local context:
{local_context}

{kb_block}

{data_tools_block}

Latest raw data:
{profile_data}

Do NOT keep kernel edits this round (planning only; the orchestrator reverts them).
Rewrite {goal_file}. End with:
<<<ROUND_REPORT
kb_read: <the KB card(s) you actually opened this round -- REQUIRED, non-empty>
new_angles: <the fresh directions the rewritten plan pursues>
ROUND_REPORT"""


MANAGER_PROMPT = """You are the INDEPENDENT MANAGER running this whole mega-MoE kernel-optimization
campaign. You are separate from the agents that edit code; you own the campaign.
Your job is not just to grade the last round -- it is to MANAGE the whole run:
steer the strategy, decide when each phase is done, decide when the plan must be
rewritten, allocate the remaining rounds, and adjudicate the working agents'
questions. The orchestrator will EXECUTE the phase action you choose below.

Campaign goal: {goal}
Primary focus file(s): {focus}

GLOBAL STATE
- Current phase: {phase}. This was round {round_num} of {rounds}; {remaining} rounds remain.
- Analyze budget: {analyze_min}-{analyze_max} rounds. Review reserved: last {review_rounds} rounds.
- Replans used: {replans_used}/{max_replans}.
- Score = total fused ms of the 5 mega-MoE stages, LOWER IS BETTER.
  Campaign start: {baseline} ms. Current best: {best} ms ({gain_pct:+.2f}% vs start).
  TARGET ({target_name}): {target_total:.3f} ms. Remaining gap: {gap} ms ({gap_pct}).
- Best-score trajectory / per-round outcomes (oldest→newest):
{progress}

Per-stage standing vs target (best measurement so far):
{stage_table}

You are FULLY AUTONOMOUS. Never stop and wait for a human -- there is always a
decision you can make yourself, so make it and keep the campaign moving. The human
owner may drop instructions into the inbox at ANY time; when present, treat them as
highest-priority and obey them, but never block waiting for them.

MESSAGES FROM THE HUMAN OWNER (obey; if one conflicts with your own plan, the human
wins; reflect them in your DIRECTIVE / PHASE_ACTION):
{user_inbox}

THIS ROUND
- Latest measurement: {ab_summary}
- Working agent's report:
{report}
- Question(s) it raised for you (may be empty):
{questions}

Current goal plan (goal.md):
{goal_excerpt}

ABSOLUTE RULE: you may NEVER give a negative or ceiling verdict -- never say a
thing is impossible, at its limit/ceiling, or that gains are used up ("不行 /
不可能 / 已经到上限 / 到顶 / 没有优化空间" and the like are forbidden). There is
always a next thing to try; name it concretely.

Decide, as the manager, in Chinese (keep it tight, <250 words):
1. 全局评估：进展是否健康？距离目标还差多少、差在哪个 stage / 哪条腿(GEMM vs comm)？是否在死磕低杠杆/死路？goal 计划是否还成立、还是该重写？剩余 {remaining} 轮该怎么分配？
2. 若有用户消息或 NEED_RULING，逐条给可执行回应/裁决。
3. 下一轮的 ONE 条明确、具体、可执行方向（引用 KB card 或具体机制/参数/结构/stage）。

Then end with EXACTLY these two lines:
PHASE_ACTION: <one of: CONTINUE_ANALYZE | START_OPTIMIZE | CONTINUE_OPTIMIZE | REPLAN | GO_REVIEW>
DIRECTIVE: <the single concrete instruction the next round must follow>

PHASE_ACTION guidance (the orchestrator obeys it within hard guardrails):
- CONTINUE_ANALYZE: analysis not deep enough yet / goal plan not concrete enough (only valid in ANALYZE, up to the analyze-max).
- START_OPTIMIZE: goal plan is concrete and ranked -> begin optimizing.
- CONTINUE_OPTIMIZE: keep executing the current plan next round.
- REPLAN: the current plan's ideas are exhausted or mis-prioritized -> spend one round rewriting goal.md from fresh angles.
- GO_REVIEW: optimization is in good shape (or nearly out of rounds) -> move to the code-review/finalize phase."""


REVIEW_PROMPT = """You are the CODE-REVIEW phase of an autonomous mega-MoE kernel-optimization campaign in {repo}.

The optimize phase committed a series of kept changes on top of base commit {base}
(primary focus was {focus}, but changes may span multiple implementation files).
Review the ENTIRE accumulated diff (base..HEAD) using the KB's quality methodology
(correctness first, then whether each change is a real, defensible optimization,
then code quality / conventions).

This is review round {review_idx} of {review_total}.

{no_ceiling}

{kb_block}

Do this:
1. `git diff {base}..HEAD` and read it ALL (every file touched).
2. Check: correctness/accuracy risk, any edit to the benchmark harness
   (benchmark/ops/training/bench_mega_moe.py) or the scoring path (report/*)
   (FORBIDDEN -- flag), comment language (English only), naming and helper-reuse
   conventions, dead code, leftover debug.
3. Fix real problems directly in the working tree (small, safe fixes only). Do NOT
   run git. Do NOT change kernel behavior in a way that would need re-benchmarking
   unless you are removing something clearly wrong.
End with:
<<<ROUND_REPORT
verdict: <"clean" | problems found and fixed: ...>
must_rebench: <"no" | "yes, because ...">
ROUND_REPORT"""


# --------------------------------------------------------------------------- #
# Campaign                                                                     #
# --------------------------------------------------------------------------- #

class Campaign:
    def __init__(self, args, log: Logger):
        self.args = args
        self.log = log
        self.repo = Path(args.repo).resolve()
        if not self.repo.is_dir():
            raise SystemExit(f"repo not found: {self.repo}")
        self.cdir = log.log_dir
        self.goal_file = self.cdir / "goal.md"
        self.memory_file = self.cdir / "memory.md"
        self.profile_file = self.cdir / "profile_data.md"
        self.state_file = self.cdir / "state.json"
        # human <-> manager interaction channel (file-based; the campaign is a
        # detached process, so you talk to the manager through these files).
        self.inbox_file = self.cdir / "inbox.md"            # you -> manager (append)
        self.inbox_archive = self.cdir / "inbox_archive.md"  # consumed messages
        self.status_file = self.cdir / "manager_status.md"   # manager -> you (overwritten each round)
        self.rounds_dir = self.cdir / "rounds"
        self.rounds_dir.mkdir(exist_ok=True)
        self.extra_dirs = [str(self.repo), str(self.cdir)]
        if args.kb_dir and Path(args.kb_dir).is_dir():
            self.extra_dirs.append(str(args.kb_dir))
        else:
            self.log.line(f"WARNING: KB dir not found: {args.kb_dir}")

        self.harness = BenchHarness(
            repo=self.repo, campaign_dir=self.cdir,
            extract=load_extractor(Path(args.extractor)),
            target_json=Path(args.target_json), bench_sh=Path(args.bench_sh),
            modes=args.modes, models=args.models, iters=args.iters,
            num_processes=args.num_processes, case=args.case,
            runc_state=args.runc_state, container=args.container,
        )
        self.total_cost = 0.0
        self.rounds_log = []
        self.kept_commits = []
        self.opt_window = []  # per-optimize-round kept ratios (1.0 = no change)
        self.no_improve_streak = 0        # consecutive optimize rounds without a new best
        self._replans_used = 0
        # progress cursor (persisted for --resume)
        self.phase = "ANALYZE"
        self.rnum = 0
        self.opt_round = 0
        self.directive = ""
        self.action = None
        self.resumed = False
        # set for real in run(); defaulted so revert/save never AttributeError
        self.baseline_untracked = set()
        self.base_commit = None
        self.base_score = None      # total ms at campaign start (higher = slower)
        self.best_score = None      # best (lowest) total ms so far
        self.best_record = None     # full score record of the best measurement

    # ---- file helpers ---------------------------------------------------- #
    def _read(self, p, default=""):
        return p.read_text() if p.exists() else default

    def _kb_index(self):
        """Read the KB navigation index (SKILL.md) fresh every round so it is
        physically present in each round's prompt -- 'must read KB every round'
        is enforced by injection, not left to the agent's discretion."""
        idx = Path(self.args.kb_dir) / "SKILL.md"
        return self._read(idx, "(KB SKILL.md not found -- knowledge base unavailable)")

    def _kb_block(self):
        return KB_BLOCK.format(kb=self.args.kb_dir, kb_index=self._kb_index())

    def _data_tools_block(self):
        return DATA_TOOLS_BLOCK.format(
            bench_sh=self.cdir / "bench.sh", incontainer_sh=self.cdir / "incontainer.sh",
            cache_sh=self.cdir / "cache.sh", bench_min=self.args.bench_minutes,
            target_name=self.harness.target.get("name", str(self.harness.target_json)))

    def _score_block(self):
        h = self.harness
        rows = "\n".join(
            f"    {label:28s} {h.target_stages.get(key, {}).get('fused_ms', float('nan')):.3f} ms"
            for key, label in h.extract.STAGES)
        gap = None if self.best_score is None else self.best_score - h.target_total
        return SCORE_BLOCK.format(
            target_name=h.target.get("name", str(h.target_json)),
            target_total=h.target_total, target_rows=rows,
            base_total=fmt_ms(self.base_score), best_total=fmt_ms(self.best_score),
            gap=fmt_ms(gap),
            gap_pct="n/a" if gap is None else f"{gap / h.target_total * 100:+.1f}%")

    def _stage_table(self):
        """Per-stage best-vs-target table, for the manager prompt and the final report."""
        h = self.harness
        rec = self.best_record or {}
        rows = []
        for key, label in h.extract.STAGES:
            now = rec.get("stages", {}).get(key, {}).get("fused_ms")
            tgt = h.target_stages.get(key, {}).get("fused_ms")
            d = None if (now is None or tgt is None) else now - tgt
            rows.append(f"  {label:28s} target {fmt_ms(tgt):>7s} | now {fmt_ms(now):>7s} | "
                        f"Δ {('n/a' if d is None else f'{d:+.3f}'):>7s} ms")
        return "\n".join(rows) or "  (no measurement yet)"

    def _write_helpers(self):
        """(Re)write the in-turn helper scripts the deep agents call via Bash:
        bench.sh (the ruler: report/bench_mega_moe.sh + the gap-vs-target table),
        incontainer.sh (arbitrary command inside the ROCm container), and cache.sh
        (drop flydsl's jit/autotune disk caches). This lets a round iterate
        edit->bench->edit in-turn, on exactly the ruler the orchestrator uses."""
        h = self.harness
        state = shlex.quote(str(h.runc_state))
        container_repo = h.to_container_path(self.repo)

        (self.cdir / "incontainer.sh").write_text(
            '#!/usr/bin/env bash\n'
            '# Run a command inside the running rootless-runc ROCm container.\n'
            '#   bash incontainer.sh \'rocprofv3 --stats -- python ...\'\n'
            'set -Eeuo pipefail\n'
            f'STATE={state}\n'
            f'CONTAINER=${{CONTAINER_NAME:-{shlex.quote(h.container)}}}\n'
            'if [[ -z "$CONTAINER" ]]; then\n'
            '  for id in $(runc --root "$STATE" list --quiet); do\n'
            '    if [[ "$(runc --root "$STATE" state "$id" | jq -r .status)" == running ]]; then\n'
            '      CONTAINER=$id; break\n'
            '    fi\n'
            '  done\n'
            'fi\n'
            '[[ -n "$CONTAINER" ]] || { echo "no running container; run ~/start_container.sh" >&2; exit 1; }\n'
            f'exec runc --root "$STATE" exec --cwd {shlex.quote(container_repo)} "$CONTAINER" bash -lc "$1"\n')

        (self.cdir / "cache.sh").write_text(
            '#!/usr/bin/env bash\n'
            '# Drop flydsl disk caches in the container: cache.sh <jit|autotune|all>\n'
            'set -Eeuo pipefail\n'
            'case "${1:-all}" in\n'
            f'  jit)      P={shlex.quote(BenchHarness.CACHE_PATHS["jit"])} ;;\n'
            f'  autotune) P={shlex.quote(BenchHarness.CACHE_PATHS["autotune"])} ;;\n'
            f'  all)      P={shlex.quote(" ".join(BenchHarness.CACHE_PATHS.values()))} ;;\n'
            '  *) echo "usage: cache.sh <jit|autotune|all>" >&2; exit 2 ;;\n'
            'esac\n'
            f'bash {shlex.quote(str(self.cdir / "incontainer.sh"))} "rm -rf $P"\n'
            'echo "cleared: $P"\n')

        (self.cdir / "bench.sh").write_text(
            '#!/usr/bin/env bash\n'
            '# THE RULER: bench the current working tree and print the gap to target.\n'
            '#   bash bench.sh [label]\n'
            'set -Eeuo pipefail\n'
            f'OUT="{self.cdir}/runs/${{1:-agent}}-$(date +%Y%m%d-%H%M%S)"\n'
            f'export OUT_DIR="$OUT" MODES={shlex.quote(self.args.modes)} '
            f'MODELS={shlex.quote(self.args.models)} ITERS={self.args.iters} '
            f'NUM_PROCESSES={self.args.num_processes}\n'
            + (f'export CONTAINER_NAME={shlex.quote(h.container)}\n' if h.container else '')
            + f'bash {shlex.quote(str(h.bench_sh))} || echo "(bench_mega_moe.sh exited non-zero -- '
              'read the logs below before trusting anything)"\n'
              'echo\n'
              'echo "===== vs TARGET ====="\n'
            f'python3 {shlex.quote(str(self.args.extractor))} "$OUT" '
            f'--baseline {shlex.quote(str(self.args.target_json))}\n'
            'echo\n'
            'echo "run dir: $OUT"\n')

        for f in ("bench.sh", "incontainer.sh", "cache.sh"):
            (self.cdir / f).chmod(0o755)

    def _local_context(self, include_goal=True):
        """The per-run local context, injected in full EVERY round so 'read the
        goal + memory at the start of each round' is guaranteed, not left to the
        agent. memory.md is tailed (it only grows); goal.md is small, sent whole."""
        parts = []
        if include_goal:
            parts.append("===== 当前 GOAL 计划 (goal.md 全文, 每轮必读) =====\n"
                         + self._read(self.goal_file, "(empty)") + "\n===== end goal =====")
        parts.append("===== campaign 本地 MEMORY (memory.md: 已试方案+测量结果+踩过的坑, tail) =====\n"
                     + self._read(self.memory_file, "(empty)")[-300000:] + "\n===== end memory =====")
        return "\n\n".join(parts)

    def _kb_gate(self, report, rnum, phase):
        """Enforce 'read the KB every round': verify the agent reported opening at
        least one card. Returns a manager note string ("" if the gate passed)."""
        read = " ".join(extract_markers(report or "", "kb_read")).strip()
        ok = bool(read) and read.lower() not in ("none", "n/a", "-", "无", "无。")
        self.log.event(type="kb_gate", round=rnum, phase=phase, kb_read=read, ok=ok)
        if ok:
            return ""
        self.log.line(f"round {rnum} [{phase}] WARNING: agent reported NO KB card read this round")
        return ("NOTE: 该轮 agent 未汇报读取任何 KB 卡片。请在裁决中明确要求下一轮"
                "必须先读知识库相关卡片再动手。")

    def _append_memory(self, text):
        with open(self.memory_file, "a") as f:
            f.write(text + "\n")

    def revert_worktree(self):
        git(self.repo, "checkout -- .", check=False)
        for f in untracked_files(self.repo) - self.baseline_untracked:
            (self.repo / f).unlink(missing_ok=True)

    def _handle_round_crash(self, rnum, exc):
        """A single round throwing must NOT kill the whole campaign. Save the crashed
        round's uncommitted worktree (INCLUDING new files) to a patch so a real win is
        never lost, log the full traceback, then revert to the clean HEAD (kept commits
        are already safe in git) and let the loop continue."""
        tb = traceback.format_exc()
        self.log.line(f"round {rnum} CRASHED: {type(exc).__name__}: {exc} -- campaign continues")
        self.log.event(type="round_crash", round=rnum, error=str(exc), traceback=tb[-4000:])
        try:
            git(self.repo, "add -A", check=False)  # stage incl. new files so the diff captures them
            patch = git(self.repo, "diff --cached HEAD", check=False)
            git(self.repo, "reset -q", check=False)  # unstage
            if patch.strip():
                p = self._round_dir(rnum) / "crash_worktree.patch"
                p.write_text(patch)
                self.log.line(f"  saved uncommitted worktree (incl. new files) -> {p} "
                              f"(recover with: git apply {p})")
        except Exception as e2:  # noqa: BLE001
            self.log.line(f"  WARNING: could not save crash patch: {e2}")
        self.revert_worktree()
        try:
            self._append_memory(f"\n## Round {rnum} — CRASHED ({type(exc).__name__})\n"
                                f"{str(exc)[:300]}\n(worktree saved to rounds/round-{rnum:02d}/crash_worktree.patch)\n")
        except Exception:  # noqa: BLE001
            pass

    def _commit(self, msg):
        """Commit staged changes with a fixed author, consistently for every commit
        the campaign makes (round, review, and final squash). --no-verify: the repo's
        pre-commit hooks (ruff-format/-check) would reformat files and fail the
        commit -- that would crash the campaign AND could silently rewrite the kernel
        we just measured; our commits are internal WIP, so bypass repo hooks."""
        author = f"{COMMIT_NAME} <{COMMIT_EMAIL}>"
        git(self.repo, f'-c user.name={shlex.quote(COMMIT_NAME)} -c user.email={shlex.quote(COMMIT_EMAIL)} '
                       f'commit --no-verify --author={shlex.quote(author)} -m {shlex.quote(msg)}')

    # ---- deferred profiling requests ------------------------------------- #
    def run_cmd_requests(self, result_text, rnum):
        cmds = extract_markers(result_text, "RUN_CMD")
        if not cmds:
            return
        self.log.line(f"round {rnum}: executing {len(cmds)} RUN_CMD profiling command(s) in the container")
        block = [f"\n## round {rnum} raw data ({time.strftime('%Y-%m-%d %H:%M:%S')})"]
        for c in cmds:
            rc, out, err = self.harness.exec_in_container(c, timeout=self.args.profile_timeout)
            tail = (out or "")[-6000:]
            errtail = (err or "")[-1500:]
            block.append(f"\n### $ {c}\n(rc={rc})\n```\n{tail}\n{errtail}\n```")
            self.log.event(type="run_cmd", round=rnum, cmd=c, rc=rc,
                           stdout_tail=tail, stderr_tail=errtail)
        with open(self.profile_file, "a") as f:
            f.write("\n".join(block) + "\n")

    # ---- measurement ----------------------------------------------------- #
    def measure(self, label, rnum):
        """Bench the current working tree --repeat times and take the BEST sample =
        MIN total fused ms (the least-throttled run is the fastest one). Every raw
        sample is logged to raw_scores.jsonl so drift is auditable.
        Returns (best_total_ms_or_None, ok, [raw_totals], best_record)."""
        self.harness.clear_caches(self.args.clear_cache)
        totals, best_rec, fails = [], None, 0
        for i in range(self.args.repeat):
            rec, rc = self.harness.bench(f"{label}-r{i}", timeout=self.args.bench_timeout)
            good = bool(rec["ok"] and rec["total_ms"])
            with open(self.cdir / "raw_scores.jsonl", "a") as f:
                f.write(json.dumps({"ts": time.time(), "round": rnum, "label": label, "rep": i,
                                    "rc": rc, "ok": good, "total_ms": rec["total_ms"],
                                    "gap_ms": rec["gap_ms"], "out_dir": rec["out_dir"],
                                    "checks_failed": rec["checks_failed"], "missing": rec["missing"]},
                                   ensure_ascii=False) + "\n")
            self.log.event(type="bench", round=rnum, label=label, rep=i, rc=rc,
                           total_ms=rec["total_ms"], gap_ms=rec["gap_ms"], ok=good,
                           checks_failed=rec["checks_failed"], missing=rec["missing"],
                           out_dir=rec["out_dir"])
            if good:
                totals.append(rec["total_ms"])
                if best_rec is None or rec["total_ms"] < best_rec["total_ms"]:
                    best_rec = rec
                self.log.line(f"  [{label}] rep {i+1}/{self.args.repeat}: total={rec['total_ms']:.3f} ms "
                              f"(gap to target {rec['gap_ms']:+.3f} ms)")
            else:
                fails += 1
                if rec["missing"]:
                    why = "missing stages " + ",".join(rec["missing"])
                elif rec["checks_failed"]:
                    why = "FAILED accuracy checks " + ",".join(rec["checks_failed"])
                else:
                    why = f"rc={rc}"
                self.log.line(f"  [{label}] rep {i+1}/{self.args.repeat}: INVALID ({why})")
        # Tolerate transient rep failures (a flaky rank / container blip): ok if AT
        # LEAST ONE rep produced a valid, all-checks-PASS number; use the fastest of
        # the good samples. A candidate that fails EVERY rep is rejected.
        if fails and totals:
            self.log.line(f"  [{label}] {fails}/{self.args.repeat} reps invalid (transient?); "
                          f"using {len(totals)} good sample(s)")
        return (min(totals) if totals else None), bool(totals), totals, best_rec

    def baseline(self):
        self.log.line("running baseline bench...")
        score, ok, totals, rec = self.measure("baseline", 0)
        if not ok or score is None:
            raise SystemExit(f"baseline bench produced no valid measurement; see {self.log.jsonl_path}")
        self.base_score = score
        self.best_score = score
        self.best_record = rec
        self.log.line(f"baseline total = {self.base_score:.3f} ms "
                      f"(best of {[round(s, 3) for s in totals]}); "
                      f"target {self.harness.target_total:.3f} ms -> gap "
                      f"{self.base_score - self.harness.target_total:+.3f} ms "
                      f"({(self.base_score / self.harness.target_total - 1) * 100:+.1f}%)")
        (self.cdir / "baseline_report.txt").write_text(rec["summary"])

    # ---- human <-> manager interaction ----------------------------------- #
    def _consume_inbox(self):
        """Atomically drain any messages the human appended to inbox.md since last
        round (rename-then-read avoids losing a concurrent append). Returns the
        text (or a placeholder) and archives what was consumed."""
        if not self.inbox_file.exists() or not self.inbox_file.read_text().strip():
            return "(no new messages)"
        tmp = self.cdir / "inbox.consuming"
        try:
            os.replace(self.inbox_file, tmp)     # atomic; a new inbox.md is recreated on next append
        except OSError:
            return "(no new messages)"
        msg = tmp.read_text().strip()
        with open(self.inbox_archive, "a") as f:
            f.write(f"\n## consumed {time.strftime('%Y-%m-%d %H:%M:%S')}\n{msg}\n")
        tmp.unlink(missing_ok=True)
        self.log.line(f"MANAGER inbox: consumed {len(msg)} chars of human input")
        return msg

    def _post_status(self, rnum, phase, action, directive, ruling):
        """Overwrite manager_status.md so the human can see, at any time, what the
        manager just decided and why."""
        gap = self.best_score - self.harness.target_total
        self.status_file.write_text(
            f"# MANAGER STATUS — campaign {self.cdir.name}\n"
            f"更新于 {time.strftime('%Y-%m-%d %H:%M:%S')} | round {rnum} | phase {phase}\n"
            f"best={self.best_score:.3f} ms (start {self.base_score:.3f} ms, "
            f"{(self.best_score/self.base_score-1)*100:+.2f}%)\n"
            f"目标 {self.harness.target.get('name','target')} = {self.harness.target_total:.3f} ms | "
            f"差距 {gap:+.3f} ms ({gap/self.harness.target_total*100:+.1f}%)\n\n"
            f"## 每个 kernel 与目标的差距\n{self._stage_table()}\n\n"
            f"PHASE_ACTION: {action}\nDIRECTIVE: {directive}\n\n"
            f"## 管理者本轮完整裁决\n{ruling}\n\n"
            f"---\n给管理者发指令：`bash {self.cdir / 'say.sh'} \"...\"`\n")

    # ---- manager (overall management + adjudication) --------------------- #
    VALID_ACTIONS = {"CONTINUE_ANALYZE", "START_OPTIMIZE", "CONTINUE_OPTIMIZE", "REPLAN", "GO_REVIEW"}

    def manage(self, phase, rnum, report, questions, ab_summary):
        """The independent MANAGER: every round it reviews the whole campaign and
        returns (ruling_text, directive, phase_action). The main loop executes the
        phase_action within hard guardrails, so the manager -- not hardcoded python
        -- steers phase transitions, replans, and finalization."""
        progress = "\n".join(
            f"  r{r['round']} [{r['phase']}] {r.get('decision','')} "
            f"best={r.get('best',0):.3f} ms :: {r.get('summary','')[:100]}"
            for r in self.rounds_log
        ) or "  (none yet)"
        gain_pct = (self.base_score / self.best_score - 1) * 100  # lower is better
        gap = self.best_score - self.harness.target_total
        prompt = MANAGER_PROMPT.format(
            goal=self.args.goal, focus=self.args.focus, phase=phase,
            round_num=rnum, rounds=self.args.rounds,
            remaining=max(0, self.args.rounds - rnum),
            analyze_min=self.args.analyze_min, analyze_max=self.args.analyze_max,
            review_rounds=self.args.review_rounds,
            replans_used=self._replans_used, max_replans=self.args.max_replans,
            baseline=f"{self.base_score:.3f}", best=f"{self.best_score:.3f}", gain_pct=gain_pct,
            target_name=self.harness.target.get("name", "target"),
            target_total=self.harness.target_total,
            gap=f"{gap:+.3f}", gap_pct=f"{gap / self.harness.target_total * 100:+.1f}%",
            stage_table=self._stage_table(),
            progress=progress, ab_summary=ab_summary,
            report=report or "(no structured report)",
            questions=questions or "(none)",
            user_inbox=self._consume_inbox(),
            goal_excerpt=self._read(self.goal_file, "(empty)")[:6000],
        )
        text = ""
        for attempt in range(2):
            text, cost, err = call_agent(
                prompt, self.repo, self.args.manager_model, self.args.manager_effort,
                self.args.manager_budget, self.args.manager_timeout, self.log,
                extra_dirs=self.extra_dirs, label=f"manager r{rnum}",
                raw_path=self._round_dir(rnum) / "manager_raw.jsonl",
            )
            self.total_cost += cost
            hits = [m for m in NEGATIVE_MARKERS if m in text]
            if not hits:
                break
            self.log.line(f"manager gave a forbidden negative verdict {hits}; re-asking")
            prompt += (
                f"\n\nYour previous answer contained forbidden negative wording {hits}. "
                f"That is not allowed. Re-answer with a concrete positive next step only."
            )
        dm = re.search(r"^DIRECTIVE:\s*(.+?)\s*$", text, re.MULTILINE)
        directive = dm.group(1) if dm else text.strip()[-400:]
        am = re.search(r"^PHASE_ACTION:\s*([A-Z_]+)\s*$", text, re.MULTILINE)
        action = am.group(1) if am and am.group(1) in self.VALID_ACTIONS else None
        self.log.line(f"manager r{rnum}: action={action or 'unspecified'}")
        self.log.event(type="manage", round=rnum, phase=phase, ruling=text,
                       directive=directive, action=action)
        self._post_status(rnum, phase, action, directive, text)
        return text, directive, action

    # ---- phase runners --------------------------------------------------- #
    def _round_dir(self, rnum):
        d = self.rounds_dir / f"round-{rnum:02d}"
        d.mkdir(exist_ok=True)
        return d

    def run_analyze(self, rnum):
        self._write_helpers()
        prompt = ANALYZE_PROMPT.format(
            repo=self.repo, goal=self.args.goal, focus=self.args.focus,
            round_num=rnum, analyze_min=self.args.analyze_min, analyze_max=self.args.analyze_max,
            goal_file=self.goal_file, local_context=self._local_context(),
            no_ceiling=NO_CEILING_RULE, score_block=self._score_block(),
            kb_block=self._kb_block(),
            data_tools_block=self._data_tools_block(),
            bench_sh=self.cdir / "bench.sh", incontainer_sh=self.cdir / "incontainer.sh",
            profile_data=self._read(self.profile_file, "(none yet)")[-20000:],
        )
        text, cost, err = call_agent(
            prompt, self.repo, self.args.deep_model, self.args.deep_effort,
            self.args.deep_budget, self.args.deep_timeout, self.log,
            extra_dirs=self.extra_dirs, label=f"analyze r{rnum}",
            raw_path=self._round_dir(rnum) / "deep_raw.jsonl", min_seconds=self.args.min_round_sec)
        self.total_cost += cost
        (self._round_dir(rnum) / "agent.txt").write_text(text)
        self.run_cmd_requests(text, rnum)
        self.revert_worktree()  # analysis keeps no kernel edit
        report = extract_report(text) or ""
        questions = "\n".join(extract_markers(text, "NEED_RULING") + [self._kb_gate(report, rnum, "ANALYZE")]).strip()
        ruling, directive, action = self.manage("ANALYZE", rnum, report, questions, "n/a (analysis round)")
        self._record(rnum, "ANALYZE", "analysis", report or text[:300], directive, kept_ratio=1.0)
        self._append_memory(
            f"\n## Round {rnum} (ANALYZE)\n{report}\n**管理者**: {directive}\n")
        return directive, action

    def run_optimize(self, rnum, opt_round, directive):
        self._write_helpers()
        prompt = OPTIMIZE_PROMPT.format(
            repo=self.repo, goal=self.args.goal, focus=self.args.focus,
            opt_round=opt_round, round_num=rnum,
            goal_file=self.goal_file, memory_file=self.memory_file,
            local_context=self._local_context(),
            no_ceiling=NO_CEILING_RULE, score_block=self._score_block(),
            kb_block=self._kb_block(),
            data_tools_block=self._data_tools_block(),
            bench_sh=self.cdir / "bench.sh", incontainer_sh=self.cdir / "incontainer.sh",
            cache_sh=self.cdir / "cache.sh",
            profile_data=self._read(self.profile_file, "(none yet)")[-20000:],
            directive=directive or "(first optimize round -- follow the goal plan's top priority)",
        )
        text, cost, err = call_agent(
            prompt, self.repo, self.args.optimize_model, self.args.optimize_effort,
            self.args.optimize_budget, self.args.deep_timeout, self.log,
            extra_dirs=self.extra_dirs, label=f"optimize r{rnum}",
            raw_path=self._round_dir(rnum) / "deep_raw.jsonl", min_seconds=self.args.min_round_sec)
        self.total_cost += cost
        rd = self._round_dir(rnum)
        (rd / "agent.txt").write_text(text)
        self.run_cmd_requests(text, rnum)
        report = extract_report(text) or ""
        questions = "\n".join(extract_markers(text, "NEED_RULING") + [self._kb_gate(report, rnum, "OPTIMIZE")]).strip()

        diff = git(self.repo, "diff --stat")
        (rd / "diff.txt").write_text(git(self.repo, "diff", check=False))
        kept, ratio, meas_summary, outcome = False, 1.0, "no change produced", "REVERTED"
        if err or not diff.strip():
            self.log.line(f"round {rnum}: no change produced; reverting")
            self.revert_worktree()
        else:
            self.log.line(f"round {rnum}: change produced, benchmarking candidate "
                          f"({self.args.repeat} rep(s), fastest wins)...")
            cand, cand_ok, totals, cand_rec = self.measure(f"r{rnum}-cand", rnum)
            # lower is better: gain > 0 means the candidate is faster than the best
            gain = (self.best_score / cand - 1) if cand else None
            if cand:
                meas_summary = (
                    f"cand_total={cand:.3f} ms vs best={self.best_score:.3f} ms "
                    f"gain={gain*100:+.2f}% | gap to target {cand - self.harness.target_total:+.3f} ms "
                    f"| raw={[round(s, 3) for s in totals]}")
            else:
                meas_summary = f"cand INVALID (missing stages or failed accuracy checks) raw={totals}"
            self.log.event(type="measure_result", round=rnum, cand_total_ms=cand,
                           best=self.best_score, gain=gain, ok=cand_ok, totals=totals)
            if cand_rec:
                (rd / "bench_report.txt").write_text(cand_rec["summary"])
            if not cand_ok or not cand:
                self.log.line(f"round {rnum}: not correct / no valid measurement ({meas_summary}); reverting")
                self.revert_worktree()
            elif gain is not None and gain >= self.args.keep_threshold:
                msg = f"mega-moe-campaign r{rnum}: {meas_summary}\n\n{report[:800]}"
                git(self.repo, "add -u")
                for f in untracked_files(self.repo) - self.baseline_untracked:
                    git(self.repo, f"add -- {shlex.quote(f)}")
                rc, _, _ = sh("git diff --cached --quiet", cwd=str(self.repo), check=False)
                if rc != 0:
                    kept, ratio, outcome = True, 1 + gain, "KEPT"
                    self._commit(msg)
                    self.kept_commits.append(git(self.repo, "rev-parse HEAD"))
                    self.best_score = cand
                    self.best_record = cand_rec
                    self.no_improve_streak = 0
                    (self.cdir / "best_report.txt").write_text(cand_rec["summary"])
                    self.log.line(f"round {rnum}: KEPT gain={gain*100:+.2f}%, best -> {self.best_score:.3f} ms "
                                  f"(gap to target {self.best_score - self.harness.target_total:+.3f} ms)")
                else:
                    self.log.line(f"round {rnum}: win measured but nothing staged; reverting")
                    self.revert_worktree()
            else:
                # correct but no new best -> keep the working copy for up to
                # revert_patience rounds so a multi-round scheme can climb out.
                self.no_improve_streak += 1
                ratio = (self.best_score / cand) if cand else 1.0
                if self.no_improve_streak < self.args.revert_patience:
                    outcome = f"KEPT_WIP({self.no_improve_streak}/{self.args.revert_patience})"
                    self.log.line(f"round {rnum}: no new best ({meas_summary}); KEEPING working copy to iterate "
                                  f"(streak {self.no_improve_streak}/{self.args.revert_patience})")
                else:
                    self.log.line(f"round {rnum}: {self.no_improve_streak} rounds without new best; reverting to best")
                    self.revert_worktree()
                    self.no_improve_streak = 0

        ruling, next_directive, action = self.manage("OPTIMIZE", rnum, report, questions, meas_summary)
        self.opt_window.append(ratio)
        self._record(rnum, "OPTIMIZE", outcome,
                     report or text[:300], next_directive, kept_ratio=ratio, ab=meas_summary)
        self._append_memory(
            f"\n## Round {rnum} (OPTIMIZE) — {outcome}\n"
            f"{report}\n**测量**: {meas_summary}\n**管理者下一步**: {next_directive}\n")
        return next_directive, action

    def run_replan(self, rnum):
        self._write_helpers()
        prompt = REPLAN_PROMPT.format(
            repo=self.repo, goal=self.args.goal, focus=self.args.focus,
            stall_window=self.args.stall_window, stall_pct=self.args.stall_threshold * 100,
            goal_file=self.goal_file, memory_file=self.memory_file,
            local_context=self._local_context(),
            no_ceiling=NO_CEILING_RULE, score_block=self._score_block(),
            kb_block=self._kb_block(),
            data_tools_block=self._data_tools_block(),
            incontainer_sh=self.cdir / "incontainer.sh",
            profile_data=self._read(self.profile_file, "(none yet)")[-20000:],
        )
        text, cost, err = call_agent(
            prompt, self.repo, self.args.deep_model, self.args.deep_effort,
            self.args.deep_budget, self.args.deep_timeout, self.log,
            extra_dirs=self.extra_dirs, label=f"replan r{rnum}",
            raw_path=self._round_dir(rnum) / "deep_raw.jsonl", min_seconds=self.args.min_round_sec)
        self.total_cost += cost
        (self._round_dir(rnum) / "agent.txt").write_text(text)
        self.run_cmd_requests(text, rnum)
        self.revert_worktree()
        report = extract_report(text) or ""
        questions = "\n".join(extract_markers(text, "NEED_RULING") + [self._kb_gate(report, rnum, "REPLAN")]).strip()
        ruling, directive, action = self.manage("REPLAN", rnum, report, questions, "n/a (replan round)")
        self.opt_window = []  # reset stagnation window after a replan
        self._record(rnum, "REPLAN", "replan", report or text[:300], directive, kept_ratio=1.0)
        self._append_memory(f"\n## Round {rnum} (REPLAN)\n{report}\n**管理者**: {directive}\n")
        return directive, action

    def run_review(self, rnum, review_idx):
        prompt = REVIEW_PROMPT.format(
            repo=self.repo, base=self.base_commit, focus=self.args.focus,
            review_idx=review_idx, review_total=self.args.review_rounds,
            no_ceiling=NO_CEILING_RULE,
            kb_block=self._kb_block(),
        )
        text, cost, err = call_agent(
            prompt, self.repo, self.args.review_model, self.args.review_effort,
            self.args.review_budget, self.args.review_timeout, self.log,
            extra_dirs=self.extra_dirs, label=f"review r{rnum}",
            raw_path=self._round_dir(rnum) / "review_raw.jsonl")
        self.total_cost += cost
        (self._round_dir(rnum) / "agent.txt").write_text(text)
        report = extract_report(text) or text[:400]
        # A review fix lands in the working tree on top of HEAD; commit it so the
        # tree stays clean for the final squash. Detect a real change by whether
        # anything is STAGED after `add -u` (NOT by `git diff --stat`, which counts
        # the perpetually-dirty 3rdparty/composable_kernel submodule and would make
        # us try to commit nothing -> git aborts rc=1 -> crash).
        git(self.repo, "add -u")
        rc, _, _ = sh("git diff --cached --quiet", cwd=str(self.repo), check=False)
        if rc != 0:
            self._commit(f"review r{rnum} fixes")
            self.kept_commits.append(git(self.repo, "rev-parse HEAD"))
            self.log.line(f"round {rnum}: review applied fixes, committed")
        else:
            self.log.line(f"round {rnum}: review made no tracked change; nothing to commit")
        self._record(rnum, "REVIEW", "review", report, "", kept_ratio=1.0)
        self._append_memory(f"\n## Round {rnum} (REVIEW)\n{report}\n")

    # ---- bookkeeping ----------------------------------------------------- #
    def _record(self, rnum, phase, decision, summary, directive, kept_ratio, ab=""):
        self.rounds_log.append({
            "round": rnum, "phase": phase, "decision": decision,
            "summary": summary[:600], "directive": directive[:400],
            "kept_ratio": kept_ratio, "ab": ab, "best": self.best_score,
        })
        self.save_state()

    def save_state(self):
        self.state_file.write_text(json.dumps({
            "base_commit": self.base_commit,
            "base_score": self.base_score,
            "best_score": self.best_score,
            "best_record": self.best_record,
            "target_total": self.harness.target_total,
            "kept_commits": self.kept_commits,
            "total_cost": self.total_cost,
            "rounds": self.rounds_log,
            "opt_window": self.opt_window,
            # progress cursor, for --resume:
            "phase": self.phase,
            "rnum": self.rnum,
            "opt_round": self.opt_round,
            "directive": self.directive,
            "action": self.action,
            "replans_used": self._replans_used,
            "no_improve_streak": self.no_improve_streak,
        }, ensure_ascii=False, indent=2))

    def load_state(self):
        """Restore a prior campaign's progress from state.json so --resume can
        continue instead of re-running baseline + analysis."""
        st = json.loads(self.state_file.read_text())
        self.base_commit = st["base_commit"]
        self.base_score = st["base_score"]
        self.best_score = st["best_score"]
        self.best_record = st.get("best_record")
        self.kept_commits = st.get("kept_commits", [])
        self.total_cost = st.get("total_cost", 0.0)
        self.rounds_log = st.get("rounds", [])
        self.opt_window = st.get("opt_window", [])
        self._replans_used = st.get("replans_used", 0)
        self.no_improve_streak = st.get("no_improve_streak", 0)
        # progress cursor; fall back to inferring from rounds_log for old states
        self.rnum = st.get("rnum", len(self.rounds_log))
        self.opt_round = st.get("opt_round", sum(1 for r in self.rounds_log if r["phase"] == "OPTIMIZE"))
        self.directive = st.get("directive", "")
        self.action = st.get("action")
        self.phase = st.get("phase") or (
            "OPTIMIZE" if any(r["phase"] in ("OPTIMIZE", "REPLAN") for r in self.rounds_log) else "ANALYZE")

    def stagnated(self):
        w = self.opt_window[-self.args.stall_window:]
        if len(w) < self.args.stall_window:
            return False
        cum = 1.0
        for r in w:
            cum *= r
        return (cum - 1.0) < self.args.stall_threshold

    def _setup_interaction(self):
        """Make it easy for the human to reach THIS campaign's manager, in a way
        that is safe when several campaigns run concurrently:
        - say.sh lives INSIDE the campaign dir (baked to this campaign's inbox), so
          it never collides with another concurrent run.
        - a per-tag stable pointer CURRENT_<tag> next to the campaign dirs points
          here, so `bash ~/flydsl_campaigns/CURRENT_<tag>/say.sh` always reaches the
          right campaign even without knowing the timestamp."""
        tag = self.args.tag or "mega_moe"
        self.tag = tag
        say = self.cdir / "say.sh"
        say.write_text(
            '#!/usr/bin/env bash\n'
            f'# Send a message to campaign "{tag}"\'s MANAGER (read next round).\n'
            '# Usage: bash say.sh "focus on combine bwd dgrad; raise per-round budget"\n'
            f'echo "[$(date +%H:%M:%S)] $*" >> {shlex.quote(str(self.inbox_file))}\n'
            f'echo "delivered to {tag} manager inbox"\n')
        say.chmod(0o755)
        ptr = self.cdir.parent / f"CURRENT_{tag}"
        try:
            if ptr.is_symlink() or ptr.exists():
                ptr.unlink()
            ptr.symlink_to(self.cdir)
        except OSError:
            (self.cdir.parent / f"CURRENT_{tag}.txt").write_text(str(self.cdir) + "\n")
        for f, hdr in ((self.inbox_file, f"# INBOX ({tag}) — 你给管理者的消息，管理者每轮读取\n"
                                          f"# 追加，或 `bash {say} \"...\"`\n"),
                       (self.status_file, f"# MANAGER STATUS ({tag}) — 管理者每轮覆写\n")):
            if not f.exists():
                f.write_text(hdr)
        self.log.line(f"interaction[{tag}]: `bash {say} \"...\"` (或 bash {ptr}/say.sh \"...\"); "
                      f"状态见 {self.status_file}")

    # ---- driver ---------------------------------------------------------- #
    def run(self):
        ensure_clean_tree(self.repo)
        # Show real paths (not git-quoted \xxx) so untracked add/unlink use the
        # actual filename even for non-ASCII / spaced names.
        git(self.repo, "config core.quotePath false", check=False)
        self._setup_interaction()
        self._write_helpers()
        self.baseline_untracked = untracked_files(self.repo)
        if self.resumed:
            if not self.base_commit:
                raise SystemExit("resume: state.json missing base_commit")
            self.log.line(f"RESUMED: {len(self.rounds_log)} rounds done, next round {self.rnum + 1}, "
                          f"phase={self.phase}, best={self.best_score:.3f} ms, base={self.base_commit[:10]}, "
                          f"kept={len(self.kept_commits)} (skipping baseline + completed rounds)")
        else:
            self.base_commit = git(self.repo, "rev-parse HEAD")
            self.log.line(f"base commit: {self.base_commit}")
            self.log.line(f"target: {self.harness.target.get('name', '?')} = "
                          f"{self.harness.target_total:.3f} ms total over 5 kernels")
            # Seed the per-run local context files with clear skeletons (fresh run only).
            if not self.goal_file.exists():
                self.goal_file.write_text(
                    f"# GOAL — campaign {self.cdir.name}\n\n"
                    f"最终目标: {self.args.goal}\n"
                    f"目标数字: {self.harness.target.get('name', 'target')} = "
                    f"{self.harness.target_total:.3f} ms (5 kernel fused 总和)\n\n"
                    f"> 由分析阶段(前 {self.args.analyze_min}-{self.args.analyze_max} 轮)填成"
                    f"「排序的优化方案表」：每条含 假设/机制/KB出处/预期买回多少 ms/风险/优先级。\n"
                    f"> 优化阶段逐条执行；停滞时 REPLAN 轮重写本文件。\n\n"
                    f"## 排序优化方案 (待分析阶段填写)\n")
            if not self.memory_file.exists():
                self.memory_file.write_text(
                    f"# MEMORY — campaign {self.cdir.name}\n\n"
                    f"目标: {self.args.goal}\n\n"
                    f"> 每轮结束由 orchestrator 追加：方案 / 测量结果 / 踩的坑 / KB 校验(skill 说X 实测Y) / 管理者裁决。\n"
                    f"> 每轮开始注入本文件全文，判断哪些已试、哪些是死路、KB 是否真实。\n")
            if not self.profile_file.exists():
                self.profile_file.write_text(
                    f"# PROFILE RAW DATA — campaign {self.cdir.name}\n\n"
                    f"> 每次 RUN_CMD(rocprofv3/ISA/micro-bench) 的原始输出按轮追加于此。\n")
            self.baseline()

        # The MANAGER drives phase transitions via the PHASE_ACTION it returns each
        # round; python only enforces hard guardrails: analyze_min/max bounds, the
        # 5-round-stall -> forced REPLAN, max_replans, and reserving the last
        # review_rounds rounds for REVIEW no matter what. Progress cursor lives in
        # self.rnum/phase/opt_round/action so --resume can pick up mid-campaign.
        R = self.args.rounds
        opt_budget_end = R - self.args.review_rounds  # last round that may be analyze/optimize

        consec_crashes = 0
        while self.rnum < opt_budget_end and self.phase != "REVIEW":
            self.rnum += 1
            self.log.line(f"===== round {self.rnum}/{R} (phase={self.phase}) =====")

            try:
                if self.phase == "ANALYZE":
                    self.directive, self.action = self.run_analyze(self.rnum)
                    start_ok = self.rnum >= self.args.analyze_min and self.action in (
                        "START_OPTIMIZE", "CONTINUE_OPTIMIZE", "GO_REVIEW")
                    if self.rnum >= self.args.analyze_max or start_ok:
                        self.phase = "OPTIMIZE"
                        self.log.line(f"phase ANALYZE -> OPTIMIZE "
                                      f"({'analyze_max' if self.rnum >= self.args.analyze_max else 'manager:'+str(self.action)})")
                elif self.action == "GO_REVIEW":
                    self.log.line("manager -> GO_REVIEW: ending optimize early")
                    self.phase = "REVIEW"
                else:  # OPTIMIZE
                    force_replan = self.stagnated() and self._replans_used < self.args.max_replans
                    want_replan = self.action == "REPLAN" and self._replans_used < self.args.max_replans
                    if force_replan or want_replan:
                        self._replans_used += 1
                        self.log.line(f"REPLAN #{self._replans_used} "
                                      f"({'stall guardrail' if force_replan else 'manager'})")
                        self.directive, self.action = self.run_replan(self.rnum)
                    else:
                        self.opt_round += 1
                        self.directive, self.action = self.run_optimize(self.rnum, self.opt_round, self.directive)
            except Exception as e:  # noqa: BLE001 - one bad round must not end the campaign
                consec_crashes += 1
                self._handle_round_crash(self.rnum, e)
                self.action = "CONTINUE_OPTIMIZE"  # keep moving
                if consec_crashes >= self.args.max_round_crashes:
                    self.log.line(f"{consec_crashes} consecutive round crashes -> stop optimizing, go finalize")
                    break
            else:
                consec_crashes = 0
            self.save_state()

        # REVIEW phase: exactly review_rounds rounds at the end (skip any already done on resume).
        self.phase = "REVIEW"
        done_review = sum(1 for r in self.rounds_log if r["phase"] == "REVIEW")
        for i in range(done_review + 1, self.args.review_rounds + 1):
            self.rnum += 1
            self.log.line(f"===== round {self.rnum}/{R} (phase=REVIEW {i}/{self.args.review_rounds}) =====")
            try:
                self.run_review(self.rnum, i)
            except Exception as e:  # noqa: BLE001
                self._handle_round_crash(self.rnum, e)
            self.save_state()

        try:
            self.finish()
        except Exception as e:  # noqa: BLE001
            self._handle_round_crash(self.rnum, e)
            self.log.line("finish() crashed; kept commits are safe in git -- squash/review manually.")

    def final_report(self):
        """The bottom line: where we started, where we are, and the remaining gap
        to the target, per kernel and in total."""
        h = self.harness
        gain = (self.base_score / self.best_score - 1) * 100
        gap = self.best_score - h.target_total
        lines = [
            f"# 结果 — campaign {self.cdir.name}",
            f"生成于 {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"目标 (target)      : {h.target.get('name', h.target_json)} = {h.target_total:.3f} ms "
            f"(5 kernel fused 总和)",
            f"起点 (campaign 开始): {self.base_score:.3f} ms "
            f"({self.base_score - h.target_total:+.3f} ms vs 目标)",
            f"最好 (campaign 结束): {self.best_score:.3f} ms ({gain:+.2f}% vs 起点)",
            "",
            f"**距离目标还差 {gap:+.3f} ms ({gap / h.target_total * 100:+.1f}%)**"
            + ("  ← 已达成/超过目标" if gap <= 0 else ""),
            "",
            "## 每个 kernel 与目标的差距 (fused ms)",
            self._stage_table(),
            "",
            f"kept commits: {len(self.kept_commits)} | rounds: {len(self.rounds_log)} | "
            f"spend: ${self.total_cost:.2f}",
        ]
        if self.best_record and self.best_record.get("summary"):
            lines += ["", "## 最好一次测量的完整对比表", "```", self.best_record["summary"], "```"]
        return "\n".join(lines)

    def finish(self):
        report = self.final_report()
        (self.cdir / "FINAL_REPORT.md").write_text(report + "\n")
        print("\n" + report + "\n", flush=True)
        with open(self.log.text_path, "a") as f:
            f.write("\n" + report + "\n")
        gap = self.best_score - self.harness.target_total
        self.log.event(type="done", base_ms=self.base_score, best_ms=self.best_score,
                       target_ms=self.harness.target_total, gap_ms=gap, cost=self.total_cost)
        if not self.kept_commits:
            self.log.line("nothing kept -- repo left at base commit.")
            return
        if self.args.no_squash:
            self.log.line("--no-squash: leaving per-round commits in place.")
            return
        self.log.line("squashing all kept commits into ONE clean commit...")
        summary = [
            f"perf(mega-moe): {self.args.goal}", "",
            f"total fused latency (5 kernels): {self.base_score:.3f} -> {self.best_score:.3f} ms "
            f"({(self.base_score / self.best_score - 1) * 100:+.2f}%)",
            f"target {self.harness.target.get('name', 'target')} = {self.harness.target_total:.3f} ms; "
            f"remaining gap {gap:+.3f} ms.",
            f"{len([r for r in self.rounds_log if r['decision'] == 'KEPT'])} optimize rounds kept.",
        ]
        for r in self.rounds_log:
            if r["decision"] == "KEPT":
                summary.append(f"- round {r['round']}: {r['summary'][:160]}")
        git(self.repo, f"reset --soft {self.base_commit}")
        # If the net diff is empty (e.g. review undid the only kept change, or kept
        # commits cancel out), `git commit` would abort with check=True and crash
        # at the finish line. Detect and restore base cleanly instead.
        rc, _, _ = sh("git diff --cached --quiet", cwd=str(self.repo), check=False)
        if rc == 0:
            self.log.line("net diff is empty -- nothing to squash; restoring base commit.")
            git(self.repo, f"reset --hard {self.base_commit}", check=False)
            return
        self._commit(chr(10).join(summary))
        self.log.line(f"final commit: {git(self.repo, 'rev-parse HEAD')} "
                      f"(nothing pushed; review with git log/diff)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=str(REPO_ROOT), help="local repo dir (default: this script's dir)")
    ap.add_argument("--focus", default="primus_turbo/flydsl/mega/",
                    help="PRIMARY focus file(s)/dir(s), repo-relative, space-separated -- the optimization "
                         "entry point. NOT a whitelist: agents may edit any implementation file needed; "
                         "only the bench harness + report/ scoring path are off-limits.")
    ap.add_argument("--goal", default=None,
                    help="what to optimize, plain language (default: close the gap to the target JSON)")
    ap.add_argument("--tag", default="mega_moe",
                    help="short name for this campaign. Used for the stable pointer CURRENT_<tag> so "
                         "concurrent campaigns don't collide.")
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--analyze-min", type=int, default=3)
    ap.add_argument("--analyze-max", type=int, default=5)
    ap.add_argument("--review-rounds", type=int, default=2)
    ap.add_argument("--max-replans", type=int, default=3)
    ap.add_argument("--min-round-sec", type=int, default=1800,
                    help="min duration of a deep round (analyze/optimize/replan); if the agent returns "
                         "early it is resumed with a 继续 nudge until this floor. 0 disables.")
    ap.add_argument("--max-round-crashes", type=int, default=3,
                    help="stop optimizing after N consecutive round exceptions (progress + kept commits kept)")
    # ---- measurement (local; report/bench_mega_moe.sh is the only ruler) ---- #
    ap.add_argument("--bench-sh", default=str(DEFAULT_BENCH_SH),
                    help="the bench driver; run once per measurement with OUT_DIR/MODES/... in the env")
    ap.add_argument("--extractor", default=str(DEFAULT_EXTRACTOR),
                    help="report/extract_performance.py -- parses the logs and diffs vs the target")
    ap.add_argument("--target-json", default=str(DEFAULT_TARGET_JSON),
                    help="the TARGET we measure the gap against (report/baseline/*.json)")
    ap.add_argument("--modes", default="dispatch_grouped_gemm grouped_gemm_combine",
                    help="MODES passed to bench_mega_moe.sh (all 5 scored stages need both)")
    ap.add_argument("--models", default="DeepSeek-V3")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--num-processes", type=int, default=8)
    ap.add_argument("--case", default=None, help="MoE case name in the logs (default: the first one found)")
    ap.add_argument("--repeat", type=int, default=2,
                    help="bench repeats per measurement; the FASTEST total is taken (least throttled)")
    ap.add_argument("--clear-cache", default="none", choices=["none", "jit", "autotune", "all"],
                    help="flydsl disk caches to drop in the container before each orchestrator measurement. "
                         "The autotune cache is keyed by shape/arch, NOT by source: use 'autotune' (or 'all') "
                         "when rounds are expected to change tuned configs, or a stale config gets re-measured.")
    ap.add_argument("--bench-timeout", type=int, default=3600, help="per bench_mega_moe.sh invocation")
    ap.add_argument("--bench-minutes", type=int, default=4, help="rough bench duration, only shown to the agent")
    ap.add_argument("--profile-timeout", type=int, default=1800)
    ap.add_argument("--runc-state", default=str(DEFAULT_RUNC_STATE),
                    help="rootless-runc state dir (used for in-container profiling / cache clears)")
    ap.add_argument("--container", default="", help="runc container id (default: the first running one)")
    # ---- keep / revert policy (LOWER total ms is better) ------------------- #
    ap.add_argument("--keep-threshold", type=float, default=0.005,
                    help="min relative speedup vs the current best to keep a round (0.5%%)")
    ap.add_argument("--revert-patience", type=int, default=5,
                    help="consecutive optimize rounds without a new best before reverting the working copy")
    ap.add_argument("--stall-window", type=int, default=5)
    ap.add_argument("--stall-threshold", type=float, default=0.02,
                    help="cumulative gain over the window below which we replan")
    ap.add_argument("--resume", default=None, metavar="CAMPAIGN_DIR",
                    help="resume a prior campaign dir (reuse its goal.md/memory.md/state.json, "
                         "skip baseline + completed rounds).")
    # ---- agents ------------------------------------------------------------ #
    # deep phase (analyze/replan): opus, max thinking, ~1M ctx, long turns
    ap.add_argument("--deep-model", default=OPUS_1M)
    ap.add_argument("--deep-effort", default="max", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--deep-budget", type=float, default=30.0)
    ap.add_argument("--deep-timeout", type=int, default=5400, help="per deep turn (default 90min ceiling)")
    # optimize (implement) phase
    ap.add_argument("--optimize-model", default="claude-sonnet-5")
    ap.add_argument("--optimize-effort", default="max", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--optimize-budget", type=float, default=30.0)
    # manager (every round)
    ap.add_argument("--manager-model", default="opus")
    ap.add_argument("--manager-effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--manager-budget", type=float, default=5.0)
    ap.add_argument("--manager-timeout", type=int, default=1200)
    # review phase
    ap.add_argument("--review-model", default="sonnet")
    ap.add_argument("--review-effort", default="xhigh", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--review-budget", type=float, default=10.0)
    ap.add_argument("--review-timeout", type=int, default=2400)
    ap.add_argument("--kb-dir", default=str(DEFAULT_KB_DIR))
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--no-squash", action="store_true")
    args = ap.parse_args()

    for path, what in ((args.bench_sh, "--bench-sh"), (args.extractor, "--extractor"),
                       (args.target_json, "--target-json")):
        if not Path(path).is_file():
            raise SystemExit(f"{what} not found: {path}")
    if not args.goal:
        target = json.loads(Path(args.target_json).read_text())
        total = sum(s["fused_ms"] for s in target["stages"].values())
        args.goal = (f"把本机 mega-MoE 5 个 kernel 的 fused 总延迟压到目标 "
                     f"{target.get('name', 'target')} = {total:.3f} ms 以内（越低越好）")

    if args.resume:
        log_dir = Path(args.resume).resolve()
        if not (log_dir / "state.json").exists():
            raise SystemExit(f"--resume: no state.json in {log_dir}")
    elif args.log_dir:
        log_dir = Path(args.log_dir)
    else:
        log_dir = DEFAULT_CAMPAIGN_ROOT / time.strftime("%Y%m%d_%H%M%S")
    # bench_mega_moe.sh writes its logs into OUT_DIR *from inside the container*,
    # which only sees $HOME (bind-mounted at /io). A campaign dir outside $HOME
    # would fail every bench, so refuse up front instead of at the first measurement.
    log_dir = log_dir.resolve()
    if not str(log_dir).startswith(str(Path.home()) + os.sep):
        raise SystemExit(f"campaign dir must live under {Path.home()} (the container only "
                         f"sees $HOME); got {log_dir}")
    log = Logger(log_dir)
    log.event(type="start", args=vars(args), resume=bool(args.resume))
    log.line(f"{'RESUME ' if args.resume else ''}campaign dir: {log_dir}")
    camp = Campaign(args, log)
    if args.resume:
        camp.resumed = True
        camp.load_state()
    camp.run()


if __name__ == "__main__":
    main()
