"""Terminal dashboard for monitoring optimizer state (standalone process)."""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
except ImportError:
    print("Dashboard requires 'rich'. Install: pip install rich")
    sys.exit(1)


# --- Formatting helpers (tested independently) ---

_STATUS_ICONS = {
    "running": "▶ RUN",
    "bottleneck": "⚠ BTNK",
    "completed": "✓ DONE",
    "failed": "✗ FAIL",
    "pending": "◻ WAIT",
    "stuck": "✗ STUCK",
    "paused": "⏸ PAUSE",
}


def format_status_icon(status: str) -> str:
    return _STATUS_ICONS.get(status, status)


def format_gain(pct: float | None) -> str:
    if pct is None:
        return "-"
    return f"+{pct:.1f}%" if pct >= 0 else f"{pct:.1f}%"


def format_worker_table_data(
    workers: dict, max_rounds: int = 10
) -> list[dict[str, str]]:
    rows = []
    for task_id, w in workers.items():
        status_str = format_status_icon(w["status"])
        if w.get("bottleneck"):
            status_str += f" {w['bottleneck']}"

        gpu = str(w["gpu_id"]) if w.get("gpu_id") is not None else "-"

        if w["current_round"] > 0 or w["status"] == "running":
            mr = w.get("max_rounds", max_rounds)
            round_str = f"{w['current_round']}/{mr}"
        else:
            round_str = "-"

        if w["rounds"]:
            first_baseline = w["rounds"][0].get("baseline_tflops", 0)
            last_result = w["rounds"][-1].get("result_tflops", 0)
            tflops = f"{first_baseline:.0f}→{last_result:.0f}"
            total_gain = ((last_result - first_baseline) / first_baseline * 100) if first_baseline > 0 else 0
            gain = format_gain(total_gain)
        else:
            tflops = "-"
            gain = "-"

        rows.append({
            "task": task_id,
            "status": status_str,
            "gpu": gpu,
            "round": round_str,
            "tflops": tflops,
            "gain": gain,
        })
    return rows


# --- Dashboard rendering ---

def build_worker_table(rows: list[dict[str, str]]) -> Table:
    table = Table(title="Worker Status", expand=True)
    table.add_column("Task", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("GPU", justify="center")
    table.add_column("Round", justify="center")
    table.add_column("TFLOPS", justify="right")
    table.add_column("Gain", justify="right", style="green")

    for r in rows:
        style = None
        if "BTNK" in r["status"]:
            style = "yellow"
        elif "FAIL" in r["status"] or "STUCK" in r["status"]:
            style = "red"
        elif "DONE" in r["status"]:
            style = "green"
        table.add_row(r["task"], r["status"], r["gpu"], r["round"], r["tflops"], r["gain"], style=style)
    return table


def build_activity_panel(activity_entries: list[dict], max_lines: int = 8) -> Panel:
    lines = []
    for e in activity_entries[-max_lines:]:
        ts = e.get("ts", "")[:19].replace("T", " ")
        worker = e.get("worker", "?")
        msg = e.get("msg", "")
        lines.append(f"[dim]{ts}[/dim] [{worker:<16}] {msg}")
    content = "\n".join(lines) if lines else "[dim]No activity yet[/dim]"
    return Panel(content, title="Live Activity", border_style="blue")


def load_activities(pattern: str, n: int = 20) -> list[dict]:
    entries = []
    for path in glob.glob(pattern):
        try:
            with open(path) as f:
                for line in f:
                    if line.strip():
                        entries.append(json.loads(line))
        except (json.JSONDecodeError, OSError):
            pass
    entries.sort(key=lambda e: e.get("ts", ""))
    return entries[-n:]


def run_dashboard(state_path: str, activity_pattern: str | None = None, interval: int = 5):
    console = Console()
    state_path = Path(state_path)

    if activity_pattern is None:
        activity_pattern = str(state_path.parent / "*" / "activity.jsonl")

    console.print(f"[bold]Primus Optimizer Dashboard[/bold]")
    console.print(f"Watching: {state_path}")
    console.print(f"Activity: {activity_pattern}")
    console.print("Press Ctrl+C to exit\n")

    try:
        with Live(console=console, refresh_per_second=1) as live:
            while True:
                if state_path.exists():
                    with open(state_path) as f:
                        state = json.load(f)

                    # Header
                    sid = state.get("session_id", "?")
                    hw = state.get("config", {}).get("hw", "?")
                    started = state.get("started_at", "")
                    elapsed = ""
                    if started:
                        try:
                            start_dt = datetime.fromisoformat(started)
                            delta = datetime.now(timezone.utc) - start_dt
                            h, rem = divmod(int(delta.total_seconds()), 3600)
                            m, s = divmod(rem, 60)
                            elapsed = f"{h:02d}:{m:02d}:{s:02d}"
                        except ValueError:
                            elapsed = "?"

                    workers = state.get("workers", {})
                    gpu_total = len(state.get("gpu_pool", {}).get("total", []))
                    gpu_busy = len(state.get("gpu_pool", {}).get("allocated", {}))

                    header = Text(
                        f" Session: {sid} | HW: {hw} | Elapsed: {elapsed} | GPUs: {gpu_busy}/{gpu_total} busy",
                        style="bold white on blue",
                    )

                    max_rounds = state.get("config", {}).get("defaults", {}).get("max_rounds", 10)
                    rows = format_worker_table_data(workers, max_rounds)
                    table = build_worker_table(rows)

                    activities = load_activities(activity_pattern)
                    activity_panel = build_activity_panel(activities)

                    layout = Layout()
                    layout.split_column(
                        Layout(header, size=1),
                        Layout(table, name="table"),
                        Layout(activity_panel, name="activity", size=12),
                    )
                    live.update(layout)
                else:
                    live.update(Text(f"Waiting for {state_path} ...", style="yellow"))

                time.sleep(interval)
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard closed.[/yellow]")


def main():
    parser = argparse.ArgumentParser(description="Primus Optimizer Dashboard")
    parser.add_argument("--state", required=True, help="Path to optimizer-state.json")
    parser.add_argument("--activity", default=None, help="Glob pattern for activity.jsonl files")
    parser.add_argument("--interval", type=int, default=5, help="Refresh interval in seconds")
    parser.add_argument("--keep-alive", action="store_true", help="Stay open after all workers finish")
    args = parser.parse_args()
    run_dashboard(args.state, args.activity, args.interval)


if __name__ == "__main__":
    main()
