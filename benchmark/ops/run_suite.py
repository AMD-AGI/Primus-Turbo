#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""GPU pool benchmark runner.

Schedules benchmark tasks across available GPUs with automatic GPU assignment,
log isolation, and exit code tracking.

Usage:
    python3 run_suite.py -d /path/to/output                          # run all tasks
    python3 run_suite.py -d /path/to/output -g attention             # filter by group
    python3 run_suite.py -d /path/to/output -l gemm_mxfp8_turbo      # run one task by label
    python3 run_suite.py -d /path/to/output -g gemm_bf16 -n 4        # 4 GPUs
    python3 run_suite.py -d /path/to/output -l gemm_fp8_blockwise_triton -s 8
        # split a single shardable task into 8 shards across 8 GPUs
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import NamedTuple

import yaml

SEPARATOR = "=" * 60
REQUIRED_TASK_FIELDS = {"label", "script", "output"}
SHARD_LABEL_SEP = "#shard"


def log(msg=""):
    """Print with immediate flush for CI real-time visibility."""
    print(msg, flush=True)


class RunningTask(NamedTuple):
    task_id: int
    gpu_id: int
    label: str
    proc: subprocess.Popen
    log_path: str
    log_file: object
    start_time: float


def load_config(config_path, groups=None, labels=None):
    """Load and validate tasks from config, optionally filtered by group/label.

    Returns (single_gpu_tasks, multi_gpu_tasks, num_gpus).
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    num_gpus = config.get("num_gpus", 8)
    tasks = config["tasks"]

    for i, task in enumerate(tasks):
        missing = REQUIRED_TASK_FIELDS - task.keys()
        if missing:
            raise ValueError(f"Task #{i} ({task.get('label', '?')}) " f"missing required fields: {missing}")

    if groups:
        group_set = set(groups)
        tasks = [t for t in tasks if t.get("group") in group_set]

    if labels:
        label_set = set(labels)
        tasks = [t for t in tasks if t["label"] in label_set]

    single = [t for t in tasks if t.get("gpus", 1) == 1]
    multi = [t for t in tasks if t.get("gpus", 1) > 1]
    return single, multi, num_gpus


def expand_shardable_tasks(tasks, num_shards):
    """Expand each shardable single-GPU task into ``num_shards`` shard tasks.

    A task is sharded only if it sets ``shardable: true`` in YAML and uses a
    single GPU. Each shard appends ``--num-shards N --shard-id i`` to ``args``
    and writes to a per-shard CSV named ``<output>.part-<i><ext>``. Returns
    ``(expanded_tasks, shard_groups)`` where ``shard_groups`` maps the original
    label to merge metadata used by :func:`merge_shard_outputs`.
    """
    if num_shards <= 1:
        return list(tasks), {}

    expanded = []
    shard_groups = {}
    for task in tasks:
        if not task.get("shardable", False) or task.get("gpus", 1) > 1:
            expanded.append(task)
            continue

        original_label = task["label"]
        original_output = task["output"]
        base, ext = os.path.splitext(original_output)
        partials = []
        for i in range(num_shards):
            shard_task = dict(task)
            shard_task["label"] = f"{original_label}{SHARD_LABEL_SEP}{i}"
            shard_task["args"] = list(task.get("args", [])) + [
                "--num-shards",
                str(num_shards),
                "--shard-id",
                str(i),
            ]
            partial_output = f"{base}.part-{i}{ext}"
            shard_task["output"] = partial_output
            partials.append(partial_output)
            expanded.append(shard_task)

        shard_groups[original_label] = {
            "final_output": original_output,
            "partials": partials,
        }
        log(f"[shard] {original_label}: split into {num_shards} shards -> {original_output}")

    return expanded, shard_groups


def merge_shard_outputs(shard_groups, output_dir):
    """Merge per-shard CSVs into the final output CSV for each shard group.

    Uses a header-aware CSV merge and sorts rows by the first column (TestID).
    Missing partials (e.g. a shard crashed before writing) are skipped with a
    warning. Partial files are removed after a successful merge.
    """
    if not shard_groups:
        return

    log()
    for label, info in shard_groups.items():
        final_path = os.path.join(output_dir, info["final_output"])
        partial_paths = [os.path.join(output_dir, p) for p in info["partials"]]
        present = [p for p in partial_paths if os.path.exists(p)]

        if not present:
            log(f"[merge] {label}: no partial CSVs found, nothing to merge")
            continue

        header = None
        rows = []
        for p in present:
            with open(p, newline="") as f:
                reader = csv.reader(f)
                file_header = next(reader, None)
                if header is None:
                    header = file_header
                rows.extend(reader)

        def _sort_key(row):
            try:
                return (0, int(row[0]))
            except (ValueError, IndexError):
                return (1, 0)

        rows.sort(key=_sort_key)

        with open(final_path, "w", newline="") as f:
            writer = csv.writer(f)
            if header is not None:
                writer.writerow(header)
            writer.writerows(rows)

        for p in present:
            try:
                os.remove(p)
            except OSError:
                pass

        log(
            f"[merge] {label}: merged {len(present)}/{len(partial_paths)} parts "
            f"({len(rows)} rows) -> {info['final_output']}"
        )


def launch_task(task, gpu_id, output_dir, script_dir, log_dir):
    """Launch a benchmark task as a subprocess with isolated log file."""
    env = os.environ.copy()
    if gpu_id is not None:
        env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    for k, v in task.get("env", {}).items():
        env[k] = str(v)

    cmd = [sys.executable, os.path.join(script_dir, task["script"])]
    cmd += [str(a) for a in task.get("args", [])]
    cmd += ["-o", os.path.join(output_dir, task["output"])]

    log_path = os.path.join(log_dir, f"{task['label']}.log")
    log_file = open(log_path, "w")
    try:
        proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=log_file)
    except Exception:
        log_file.close()
        raise
    return proc, log_path, log_file


def print_task_log(label, returncode, log_path, elapsed):
    """Print a completed task's captured log with a clear header."""
    status = "OK" if returncode == 0 else f"FAILED (exit={returncode})"
    log(f"\n{SEPARATOR}")
    log(f"  {label}  [{status}, {elapsed:.0f}s]")
    log(SEPARATOR)
    with open(log_path) as f:
        content = f.read()
    if content:
        print(content, end="" if content.endswith("\n") else "\n", flush=True)


def wait_any(running):
    """Poll until any running process completes."""
    while True:
        for entry in running:
            if entry.proc.poll() is not None:
                return entry
        time.sleep(0.5)


def run_single_gpu_tasks(tasks, num_gpus, output_dir, script_dir, log_dir, total):
    """Run single-GPU tasks using a GPU pool scheduler.

    Returns (num_failed, num_started).
    """
    free_gpus = list(range(num_gpus))
    running: list[RunningTask] = []
    queue = list(tasks)
    failed = 0
    started = 0

    while queue or running:
        while queue and free_gpus:
            task = queue.pop(0)
            gpu_id = free_gpus.pop(0)
            proc, log_path, log_file = launch_task(task, gpu_id, output_dir, script_dir, log_dir)
            started += 1
            running.append(
                RunningTask(
                    task_id=started,
                    gpu_id=gpu_id,
                    label=task["label"],
                    proc=proc,
                    log_path=log_path,
                    log_file=log_file,
                    start_time=time.time(),
                )
            )
            log(f"[{started}/{total}] Start: {task['label']} (GPU {gpu_id})")

        entry = wait_any(running)
        running.remove(entry)
        entry.log_file.close()
        elapsed = time.time() - entry.start_time

        rc = entry.proc.returncode
        tag = "OK" if rc == 0 else f"FAIL(exit={rc})"
        log(f"[{entry.task_id}/{total}] Done:  {entry.label} " f"(GPU {entry.gpu_id}, {elapsed:.0f}s, {tag})")
        print_task_log(entry.label, rc, entry.log_path, elapsed)

        free_gpus.append(entry.gpu_id)
        if rc != 0:
            failed += 1

    return failed, started


def run_multi_gpu_tasks(tasks, output_dir, script_dir, log_dir, total, task_id_offset):
    """Run multi-GPU tasks sequentially (no HIP_VISIBLE_DEVICES restriction).

    Returns num_failed.
    """
    failed = 0
    task_id = task_id_offset

    for task in tasks:
        task_id += 1
        gpus = task.get("gpus", 1)
        log(f"[{task_id}/{total}] Start: {task['label']} ({gpus} GPUs)")

        t0 = time.time()
        proc, log_path, log_file = launch_task(task, None, output_dir, script_dir, log_dir)
        proc.wait()
        log_file.close()
        elapsed = time.time() - t0

        rc = proc.returncode
        tag = "OK" if rc == 0 else f"FAIL(exit={rc})"
        log(f"[{task_id}/{total}] Done:  {task['label']} ({elapsed:.0f}s, {tag})")
        print_task_log(task["label"], rc, log_path, elapsed)

        if rc != 0:
            failed += 1

    return failed


def main():
    parser = argparse.ArgumentParser(
        description="GPU pool benchmark runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="Path to benchmark_suite.yaml (default: next to this script)",
    )
    parser.add_argument(
        "-d",
        "--output-dir",
        required=True,
        help="Output directory for benchmark CSV files",
    )
    parser.add_argument(
        "-g",
        "--group",
        nargs="+",
        default=None,
        help="Only run tasks matching these group(s)",
    )
    parser.add_argument(
        "-l",
        "--label",
        nargs="+",
        default=None,
        help="Only run tasks matching these exact label(s)",
    )
    parser.add_argument(
        "-n",
        "--num-gpus",
        type=int,
        default=None,
        help="Override the number of GPUs",
    )
    parser.add_argument(
        "-s",
        "--shards",
        type=int,
        default=1,
        help=(
            "Split each shardable single-GPU task into this many shards across "
            "the GPU pool (default: 1, no sharding). Tasks must declare "
            "`shardable: true` in YAML and the underlying script must accept "
            "`--num-shards` and `--shard-id`."
        ),
    )
    args = parser.parse_args()

    if args.shards < 1:
        raise SystemExit(f"--shards must be >= 1, got {args.shards}")

    config_path = args.config or os.path.join(os.path.dirname(__file__), "benchmark_suite.yaml")
    script_dir = os.path.dirname(os.path.abspath(config_path))

    single_tasks, multi_tasks, num_gpus = load_config(config_path, args.group, args.label)
    if args.num_gpus is not None:
        num_gpus = args.num_gpus

    single_tasks, shard_groups = expand_shardable_tasks(single_tasks, args.shards)
    if args.shards > 1 and not shard_groups:
        log(
            f"[shard] --shards={args.shards} requested but no matched task "
            f"declares `shardable: true`; running without sharding."
        )

    total = len(single_tasks) + len(multi_tasks)
    if total == 0:
        log("No tasks to run.")
        return

    log(
        f"Benchmark suite: {total} tasks "
        f"({len(single_tasks)} single-GPU, {len(multi_tasks)} multi-GPU), "
        f"{num_gpus} GPUs"
    )
    if args.group:
        log(f"Filtered by group(s): {', '.join(args.group)}")
    if args.label:
        log(f"Filtered by label(s): {', '.join(args.label)}")
    if shard_groups:
        log(f"Sharded labels: {', '.join(shard_groups.keys())} ({args.shards} shards each)")
    log()

    os.makedirs(args.output_dir, exist_ok=True)

    log_dir = tempfile.mkdtemp(prefix="bench_logs_")
    failed = 0

    try:
        started = 0
        if single_tasks:
            f, started = run_single_gpu_tasks(
                single_tasks,
                num_gpus,
                args.output_dir,
                script_dir,
                log_dir,
                total,
            )
            failed += f

        if multi_tasks:
            failed += run_multi_gpu_tasks(
                multi_tasks,
                args.output_dir,
                script_dir,
                log_dir,
                total,
                started,
            )

        merge_shard_outputs(shard_groups, args.output_dir)

        passed = total - failed
        log(f"\n{SEPARATOR}")
        log(f"  SUMMARY: {passed}/{total} passed, {failed} failed")
        log(SEPARATOR)

    finally:
        shutil.rmtree(log_dir, ignore_errors=True)

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
