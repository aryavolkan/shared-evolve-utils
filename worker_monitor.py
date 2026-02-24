#!/usr/bin/env python3
"""
Generic worker monitor and auto-spawn utilities for Godot training projects.

Used by both Evolve and Chess-Evolve.  Each project provides a small config
dict; the functions here do the generic heavy lifting.

Typical usage::

    from worker_monitor import WorkerConfig, monitor_once, spawn_worker

    cfg = WorkerConfig(
        godot_data_dir=godot_user_dir("Chess Evolve"),
        worker_script=Path("overnight-agent/chess_sweep_worker.py"),
        worker_script_names=["chess_sweep_worker.py"],
        sweep_flag="--sweep-id",     # flag used by this project's worker
        project_dir=Path("."),
        wandb_project="chess-evolve",
    )
    monitor_once(cfg, auto_spawn=True, notify=True)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable


# ---------------------------------------------------------------------------
# Nanoclaw IPC — common across all projects
# ---------------------------------------------------------------------------

NANOCLAW_IPC_CANDIDATES = [
    Path.home() / "projects/nanoclaw/data/ipc/main/messages",
    Path.home() / "nanoclaw/data/ipc/main/messages",
]
WHATSAPP_JID = "12066088083@s.whatsapp.net"

# CPU threshold below which we consider spawning new workers
DEFAULT_CPU_THRESHOLD = 50.0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class WorkerConfig:
    """Per-project configuration for the monitor/spawn system.

    Attributes:
        godot_data_dir: Godot user data directory (e.g. ``godot_user_dir("Chess Evolve")``).
        worker_script: Absolute path to the worker Python script to spawn.
        worker_script_names: List of script filenames to look for in ``ps aux``
            (e.g. ``["chess_sweep_worker.py", "overnight_sweep.py"]``).
        project_dir: Root directory of the Godot project.
        wandb_project: Default W&B project name for spawned workers.
        sweep_flag: CLI flag used to pass a sweep ID to the worker
            (e.g. ``"--sweep-id"`` or ``"--join"``).
        display_name: Human-readable name for display/reports.
        extra_spawn_args: Extra CLI args to append when spawning a worker.
    """

    godot_data_dir: Path
    worker_script: Path
    worker_script_names: list[str]
    project_dir: Path
    wandb_project: str = "evolve"
    sweep_flag: str = "--sweep-id"
    display_name: str = "Worker"
    extra_spawn_args: list[str] = field(default_factory=list)
    python_bin: str = sys.executable  # Python interpreter used to spawn workers


# ---------------------------------------------------------------------------
# System resource helpers
# ---------------------------------------------------------------------------


def auto_max_workers(mem_per_worker_gb: float = 0.75, headroom_gb: float = 2.5) -> int:
    """Compute a sensible max-worker count from available CPU cores and free memory."""
    cpu_max = max(1, (os.cpu_count() or 4) - 2)

    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    avail_gb = int(line.split()[1]) / 1024 / 1024
                    break
            else:
                return cpu_max
    except OSError:
        return cpu_max

    try:
        ps = subprocess.run(["ps", "aux"], capture_output=True, text=True)
        current = sum(
            1
            for ln in ps.stdout.splitlines()
            if any(s in ln for s in ["overnight", "sweep_worker"])
            and "grep" not in ln
        )
    except Exception:
        current = 0

    extra_slots = max(0, int((avail_gb - headroom_gb) / mem_per_worker_gb))
    mem_max = current + extra_slots
    return max(1, min(cpu_max, mem_max))


# ---------------------------------------------------------------------------
# Process detection
# ---------------------------------------------------------------------------


def _parse_flag_value(cmd: str, flag: str) -> str | None:
    """Extract value after *flag* from a command string (``--flag VALUE`` style)."""
    match = re.search(re.escape(flag) + r"\s+(\S+)", cmd)
    if match:
        return match.group(1)
    # Also handle --flag=VALUE style
    match = re.search(re.escape(flag) + r"=(\S+)", cmd)
    return match.group(1) if match else None


def get_running_workers(script_names: list[str]) -> list[dict]:
    """Return a list of running Python worker processes matching *script_names*.

    Each entry is a dict with keys: pid, cpu, mem, start_time, command, sweep_id.
    """
    try:
        ps = subprocess.run(["ps", "aux"], capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError:
        return []

    workers = []
    for line in ps.stdout.splitlines():
        if any(name in line for name in script_names) and "grep" not in line:
            parts = line.split()
            if len(parts) < 11:
                continue
            try:
                cmd = " ".join(parts[10:])
                # Try common sweep-id flags
                sweep_id = (
                    _parse_flag_value(cmd, "--sweep-id")
                    or _parse_flag_value(cmd, "--join")
                    or _parse_flag_value(cmd, "--sweep_id")
                )
                workers.append({
                    "pid": int(parts[1]),
                    "cpu": float(parts[2]),
                    "mem": float(parts[3]),
                    "start_time": parts[8],
                    "command": cmd,
                    "sweep_id": sweep_id,
                })
            except (ValueError, IndexError):
                continue

    return workers


def get_godot_instances() -> list[dict]:
    """Return a list of running headless Godot processes."""
    try:
        ps = subprocess.run(["ps", "aux"], capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError:
        return []

    instances = []
    for line in ps.stdout.splitlines():
        if "godot" in line.lower() and "--headless" in line and "grep" not in line:
            parts = line.split()
            if len(parts) < 11:
                continue
            try:
                worker_id = "unknown"
                for part in parts:
                    if "--worker-id=" in part:
                        worker_id = part.split("=")[1]
                        break
                instances.append({
                    "pid": int(parts[1]),
                    "cpu": float(parts[2]),
                    "mem": float(parts[3]),
                    "start_time": parts[8],
                    "worker_id": worker_id,
                })
            except (ValueError, IndexError):
                continue

    return instances


def get_active_metrics(godot_data_dir: Path, stale_seconds: int = 300) -> list[dict]:
    """Return recently-updated metrics files from *godot_data_dir*."""
    if not godot_data_dir.exists():
        return []

    cutoff = time.time() - stale_seconds
    results = []
    for mf in godot_data_dir.glob("metrics*.json"):
        mtime = mf.stat().st_mtime
        if mtime > cutoff:
            try:
                data = json.loads(mf.read_text())
                results.append({
                    "file": mf.name,
                    "age_seconds": int(time.time() - mtime),
                    "generation": data.get("generation", "?"),
                    "best_fitness": data.get("best_fitness", "?"),
                    "avg_fitness": data.get("avg_fitness", "?"),
                    "training_complete": data.get("training_complete", False),
                })
            except Exception:
                pass

    return results


def get_active_sweeps(workers: list[dict]) -> set[str]:
    """Return set of unique sweep IDs from running workers."""
    return {w["sweep_id"] for w in workers if w.get("sweep_id")}


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

_WIDTH = 110


def print_status_table(
    workers: list[dict],
    godot_instances: list[dict],
    metrics: list[dict],
    display_name: str = "Worker",
) -> float:
    """Print a human-readable status table. Returns average CPU of workers."""
    print(f"\n{'=' * _WIDTH}")
    print(f"{'WORKER STATUS':^{_WIDTH}}")
    print("=" * _WIDTH)
    print(
        f"{'PID':<8} {'CPU %':<8} {'MEM %':<8} {'START':<10} "
        f"{'TYPE':<15} {'SWEEP':<12} {'COMMAND/INFO':<39}"
    )
    print("-" * _WIDTH)

    for w in workers:
        cmd = w["command"]
        sweep = w.get("sweep_id") or "-"
        cmd_short = next((name for name in [cmd] if len(name) <= 39), cmd[:39])
        print(
            f"{w['pid']:<8} {w['cpu']:<8.1f} {w['mem']:<8.1f} "
            f"{w['start_time']:<10} {display_name + ' Worker':<15} {sweep:<12} {cmd_short:<39}"
        )

    for inst in godot_instances:
        info = f"Worker ID: {inst['worker_id']}"
        print(
            f"{inst['pid']:<8} {inst['cpu']:<8.1f} {inst['mem']:<8.1f} "
            f"{inst['start_time']:<10} {'Godot Instance':<15} {'-':<12} {info:<39}"
        )

    if not workers and not godot_instances:
        print(f"{'No workers or Godot instances running':<{_WIDTH}}")

    print("-" * _WIDTH)

    # Active training sessions
    if metrics:
        print(f"\n{'=' * _WIDTH}")
        print(f"{'ACTIVE TRAINING SESSIONS':^{_WIDTH}}")
        print("=" * _WIDTH)
        for m in metrics:
            status = "COMPLETE" if m["training_complete"] else "TRAINING"
            age_str = f"{m['age_seconds']}s ago"
            print(
                f"  {m['file']:<25} [{status:<10}] Updated: {age_str:<12} "
                f"Gen: {m['generation']:<5} Best: {m['best_fitness']:<10} Avg: {m['avg_fitness']:<10}"
            )

    # Summary
    avg_cpu = sum(w["cpu"] for w in workers) / len(workers) if workers else 0
    total_godot_cpu = sum(g["cpu"] for g in godot_instances)
    sweeps = get_active_sweeps(workers)

    print(f"\n{'=' * _WIDTH}")
    print(f"{'SUMMARY':^{_WIDTH}}")
    print("=" * _WIDTH)
    print(f"  Python Workers:       {len(workers)}")
    print(f"  Godot Instances:      {len(godot_instances)}")
    active_count = len([m for m in metrics if not m["training_complete"]])
    done_count = len([m for m in metrics if m["training_complete"]])
    print(f"  Active Training:      {active_count}")
    print(f"  Completed Training:   {done_count}")
    if sweeps:
        print(f"  Active Sweeps:        {', '.join(sorted(sweeps))}")
    else:
        print("  Active Sweeps:        none")
    if workers:
        print(f"  Avg Worker CPU:       {avg_cpu:.1f}%")
    if godot_instances:
        print(f"  Total Godot CPU:      {total_godot_cpu:.1f}%")

    return avg_cpu


# ---------------------------------------------------------------------------
# WhatsApp reporting
# ---------------------------------------------------------------------------


def find_nanoclaw_ipc() -> Path | None:
    """Return the NanoClaw IPC messages directory, or None if not found."""
    for candidate in NANOCLAW_IPC_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def build_whatsapp_report(
    workers: list[dict],
    godot_instances: list[dict],
    metrics: list[dict],
    display_name: str = "Workers",
    sweep_id: str | None = None,
    spawned: bool = False,
) -> str:
    """Build a concise WhatsApp status message."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    hostname = os.uname().nodename

    active = [m for m in metrics if not m["training_complete"]]
    total_cpu = sum(g["cpu"] for g in godot_instances)
    sweeps = get_active_sweeps(workers)
    if sweep_id:
        sweeps.add(sweep_id)

    lines = [f"*{display_name}* [{hostname}] {now}"]

    if not workers and not godot_instances:
        lines.append("No workers running.")
    else:
        lines.append(f"Workers: {len(workers)} | Godot: {len(godot_instances)} | CPU: {total_cpu:.0f}%")

    if sweeps:
        lines.append(f"Sweeps: {', '.join(sorted(sweeps))}")

    if active:
        lines.append("")
        lines.append("*Training progress:*")
        for m in sorted(active, key=lambda x: x.get("generation", 0), reverse=True)[:5]:
            gen = m.get("generation", "?")
            best = m.get("best_fitness", 0)
            avg = m.get("avg_fitness", 0)
            try:
                lines.append(f"  Gen {gen:<3} | best {float(best):,.0f} | avg {float(avg):,.0f}")
            except (TypeError, ValueError):
                lines.append(f"  Gen {gen} | best {best} | avg {avg}")

    if spawned and sweep_id:
        lines.append(f"\nSpawned new worker → sweep {sweep_id}")

    return "\n".join(lines)


def send_whatsapp_report(message: str, notify_host: str | None = None) -> bool:
    """Write IPC message for NanoClaw. Falls back to SSH if IPC is not local."""
    ipc_dir = find_nanoclaw_ipc()
    if ipc_dir:
        ts = int(time.time() * 1000)
        msg_file = ipc_dir / f"monitor-{ts}.json"
        msg_file.write_text(json.dumps({
            "type": "message",
            "chatJid": WHATSAPP_JID,
            "text": message,
        }))
        print(f"  WhatsApp report queued → {msg_file.name}")
        return True

    host = notify_host or os.environ.get("NANOCLAW_HOST")
    if not host:
        print("  (NanoClaw IPC not found; set NANOCLAW_HOST or use --notify-host)")
        return False

    ts = int(time.time() * 1000)
    payload = json.dumps({"type": "message", "chatJid": WHATSAPP_JID, "text": message})
    remote_path = None
    for candidate in NANOCLAW_IPC_CANDIDATES:
        check = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
             f"aryasen@{host}", f"test -d {candidate} && echo ok"],
            capture_output=True, text=True,
        )
        if check.stdout.strip() == "ok":
            remote_path = str(candidate / f"monitor-{ts}.json")
            break

    if not remote_path:
        print(f"  (NanoClaw IPC not found on {host})")
        return False

    result = subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
         f"aryasen@{host}", f"cat > {remote_path}"],
        input=payload, text=True, capture_output=True,
    )
    if result.returncode == 0:
        print(f"  WhatsApp report queued on {host} → {Path(remote_path).name}")
        return True
    print(f"  SSH notify failed: {result.stderr.strip()}")
    return False


# ---------------------------------------------------------------------------
# Worker spawning
# ---------------------------------------------------------------------------


def spawn_worker(
    cfg: WorkerConfig,
    sweep_id: str | None = None,
    count: int = 5,
    log_dir: Path | None = None,
) -> bool:
    """Spawn a new sweep worker in the background.

    Args:
        cfg: Project configuration.
        sweep_id: W&B sweep ID to join. If None, worker creates its own.
        count: Number of runs this worker will execute.
        log_dir: Directory for the worker log file. Defaults to
            ``cfg.project_dir / "overnight-agent"``.
    """
    if not cfg.worker_script.exists():
        print(f"Error: worker script not found: {cfg.worker_script}", file=sys.stderr)
        return False

    if log_dir is None:
        log_dir = cfg.project_dir / "overnight-agent"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"worker_{timestamp}.log"

    cmd = [
        "nohup",
        cfg.python_bin,
        str(cfg.worker_script),
        "--project", cfg.wandb_project,
        "--count", str(count),
        *cfg.extra_spawn_args,
    ]
    if sweep_id:
        cmd.extend([cfg.sweep_flag, sweep_id])

    try:
        with open(log_file, "w") as f:
            proc = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                cwd=cfg.project_dir,
            )
        print(f"\n✓ Spawned new worker (PID: {proc.pid})")
        print(f"  Log: {log_file}")
        return True
    except Exception as e:
        print(f"Error spawning worker: {e}", file=sys.stderr)
        return False


def ensure_sweep_id(
    cfg: WorkerConfig,
    sweep_id: str | None,
    create_sweep_fn: Callable[[], str] | None = None,
) -> str:
    """Return *sweep_id* as-is, or create a new one via *create_sweep_fn*."""
    if sweep_id:
        return sweep_id
    if create_sweep_fn:
        return create_sweep_fn()
    from godot_wandb import create_or_join_sweep
    return create_or_join_sweep({}, cfg.wandb_project)


# ---------------------------------------------------------------------------
# High-level: monitor once
# ---------------------------------------------------------------------------


def monitor_once(
    cfg: WorkerConfig,
    *,
    auto_spawn: bool = False,
    fill: bool = False,
    max_workers: int | None = None,
    sweep_id: str | None = None,
    cpu_threshold: float = DEFAULT_CPU_THRESHOLD,
    notify: bool = False,
    notify_host: str | None = None,
    as_json: bool = False,
    count_per_worker: int = 5,
    create_sweep_fn: Callable[[], str] | None = None,
) -> dict:
    """Run one monitoring pass.

    Args:
        cfg: Project-specific worker configuration.
        auto_spawn: Spawn a worker if utilization is low or none running.
        fill: Keep spawning until ``max_workers`` is reached.
        max_workers: Maximum concurrent workers (auto-detected if None).
        sweep_id: W&B sweep ID to pass to spawned workers.
        cpu_threshold: Spawn threshold (default 50%).
        notify: Send WhatsApp report via NanoClaw IPC.
        notify_host: SSH host for remote NanoClaw.
        as_json: Return JSON dict instead of printing human-readable output.
        count_per_worker: ``--count`` passed to each spawned worker.
        create_sweep_fn: Callable that creates/returns a sweep ID when needed.

    Returns:
        Summary dict with worker_count, godot_count, active_training, etc.
    """
    if max_workers is None:
        max_workers = auto_max_workers()

    workers = get_running_workers(cfg.worker_script_names)
    godot_instances = get_godot_instances()
    metrics = get_active_metrics(cfg.godot_data_dir)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "workers": workers,
        "godot_instances": godot_instances,
        "metrics": metrics,
        "summary": {
            "worker_count": len(workers),
            "godot_count": len(godot_instances),
            "active_training": len([m for m in metrics if not m["training_complete"]]),
            "avg_cpu": sum(w["cpu"] for w in workers) / len(workers) if workers else 0.0,
            "sweeps": list(get_active_sweeps(workers)),
        },
    }

    if as_json:
        import json as _json
        print(_json.dumps(summary, indent=2))
        return summary

    print(f"\n{cfg.display_name} Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    avg_cpu = print_status_table(workers, godot_instances, metrics, cfg.display_name)

    # Spawn logic
    spawned = False
    if auto_spawn or fill:
        if len(workers) < max_workers:
            if not workers:
                print("\n→ No workers running. Spawning first worker...")
                sweep_id = ensure_sweep_id(cfg, sweep_id, create_sweep_fn)
                spawned = spawn_worker(cfg, sweep_id=sweep_id, count=count_per_worker)
            elif avg_cpu < cpu_threshold:
                print(f"\n→ Low CPU ({avg_cpu:.1f}% < {cpu_threshold}%). Spawning worker...")
                sweep_id = ensure_sweep_id(cfg, sweep_id, create_sweep_fn)
                spawned = spawn_worker(cfg, sweep_id=sweep_id, count=count_per_worker)
            else:
                print(f"\n→ Workers active (avg CPU: {avg_cpu:.1f}%). No spawn needed.")
        else:
            print(f"\n→ Already at max workers ({max_workers}). No spawn needed.")

        if spawned:
            print(f"\n✓ Worker spawned (sweep: {sweep_id})")
            print(f"  Add more: --sweep-id {sweep_id}")

    if fill and sweep_id:
        current = get_running_workers(cfg.worker_script_names)
        while len(current) < max_workers:
            slots = max_workers - len(current)
            print(f"\n  Filling {slots} slot(s) (have {len(current)}/{max_workers})...")
            for _ in range(slots):
                spawn_worker(cfg, sweep_id=sweep_id, count=count_per_worker)
            time.sleep(5)  # Give workers time to start before recounting
            current = get_running_workers(cfg.worker_script_names)

    print("\n" + "=" * _WIDTH + "\n")

    if notify:
        report = build_whatsapp_report(
            workers, godot_instances, metrics, cfg.display_name, sweep_id, spawned
        )
        send_whatsapp_report(report, notify_host=notify_host)

    return summary


# ---------------------------------------------------------------------------
# CLI helper (for quick standalone testing)
# ---------------------------------------------------------------------------


def add_monitor_args(parser) -> None:  # type: ignore[type-arg]
    """Add standard monitor CLI args to an argparse.ArgumentParser."""
    parser.add_argument("--auto-spawn", action="store_true",
                        help="Spawn a worker if utilization is low")
    parser.add_argument("--fill", action="store_true",
                        help="Spawn until max-workers is reached")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Max concurrent workers (auto-detected if omitted)")
    parser.add_argument("--sweep-id", type=str, default=None,
                        help="W&B sweep ID for spawned workers")
    parser.add_argument("--cpu-threshold", type=float, default=DEFAULT_CPU_THRESHOLD,
                        help=f"CPU %% threshold for spawning (default: {DEFAULT_CPU_THRESHOLD})")
    parser.add_argument("--notify", action="store_true",
                        help="Send status report via WhatsApp/NanoClaw")
    parser.add_argument("--notify-host", type=str, default=None,
                        help="SSH host running NanoClaw (if IPC not local)")
    parser.add_argument("--json", action="store_true", dest="as_json",
                        help="Output JSON instead of human-readable table")
