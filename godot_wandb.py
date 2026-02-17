#!/usr/bin/env python3
"""
Shared Godot + W&B training utilities.

Common helpers for launching Godot training, polling metrics, and logging to
Weights & Biases. Used by both Evolve and Chess-Evolve projects.
"""

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# Force unbuffered output for nohup/sweep compatibility
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Default Godot binary
DEFAULT_GODOT_PATH = os.environ.get(
    "GODOT_PATH", "/opt/homebrew/bin/godot"
)


def godot_user_dir(app_name: str) -> Path:
    """Return the Godot user data directory for a project on the current platform.

    Args:
        app_name: The Godot project name (e.g. "evolve", "Chess Evolve", "chess-evolve").
    """
    override = os.environ.get("GODOT_USER_DIR")
    if override:
        return Path(override)

    system = platform.system()
    if system == "Darwin":
        return Path.home() / "Library/Application Support/Godot/app_userdata" / app_name
    elif system == "Windows":
        appdata = os.environ.get("APPDATA", "")
        return Path(appdata) / "Godot/app_userdata" / app_name
    else:  # Linux
        return Path.home() / ".local/share/godot/app_userdata" / app_name


def write_config(config: dict, config_path: Path) -> None:
    """Write a JSON config file for Godot to read at startup."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"✓ Config written to {config_path}")


def read_metrics(metrics_path: Path) -> Optional[dict]:
    """Read current metrics from Godot's JSON file. Returns None if unavailable."""
    try:
        if not metrics_path.exists():
            return None
        with open(metrics_path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def launch_godot(
    project_path: str,
    godot_path: str = DEFAULT_GODOT_PATH,
    visible: bool = False,
    extra_args: list[str] = None,
    metrics_path: Optional[Path] = None,
) -> subprocess.Popen:
    """Launch a Godot training process.

    Args:
        project_path: Path to the Godot project directory.
        godot_path: Path to the Godot binary.
        visible: If False, run headless.
        extra_args: Extra Godot user args (after --).
        metrics_path: If provided, clear old metrics before launch.

    Returns:
        The subprocess.Popen instance.
    """
    if metrics_path and metrics_path.exists():
        metrics_path.unlink()
        print("✓ Cleared old metrics")

    cmd = [godot_path, "--path", project_path]
    if not visible:
        cmd.extend(["--headless", "--rendering-driver", "dummy"])
    if extra_args:
        cmd.extend(["--"] + extra_args)
    else:
        cmd.extend(["--", "--auto-train"])

    print(f"🚀 Launching: {' '.join(cmd)}")
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


def wait_for_metrics(
    metrics_path: Path,
    timeout: float = 30.0,
) -> bool:
    """Wait for metrics.json to appear (training started).

    Returns True if training started within timeout.
    """
    print(f"⏳ Waiting for training to start (timeout: {timeout}s)...")
    start = time.time()
    while time.time() - start < timeout:
        if metrics_path.exists():
            print("✓ Training started!")
            return True
        time.sleep(1)
    print("❌ Timeout waiting for training start")
    return False


def poll_metrics(
    wandb_run,
    metrics_path: Path,
    max_generations: int = 100,
    poll_interval: float = 5.0,
    max_stale: int = 60,
    log_keys: Optional[list[str]] = None,
) -> Optional[dict]:
    """Poll metrics.json and log each new generation to W&B.

    Args:
        wandb_run: An active wandb.Run instance.
        metrics_path: Path to the metrics.json file.
        max_generations: Stop after this many generations.
        poll_interval: Seconds between checks.
        max_stale: Max consecutive stale polls before aborting.
        log_keys: If provided, only log these keys. Otherwise log all.

    Returns:
        The final metrics dict, or None.
    """
    last_gen = -1
    stale_count = 0
    final_metrics = None
    print(f"📊 Polling metrics every {poll_interval}s...")

    while True:
        metrics = read_metrics(metrics_path)
        if metrics and "generation" in metrics:
            gen = metrics["generation"]
            if gen > last_gen:
                last_gen = gen
                stale_count = 0
                final_metrics = metrics

                if log_keys:
                    log_data = {k: metrics.get(k, 0) for k in log_keys}
                else:
                    log_data = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}

                wandb_run.log(log_data)

                # Print summary line
                parts = [f"Gen {gen}"]
                for k in sorted(log_data):
                    if k != "generation":
                        v = log_data[k]
                        parts.append(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}")
                print(", ".join(parts[:6]))

                if gen >= max_generations - 1:
                    print(f"\n✓ Training complete! Reached generation {gen}")
                    break
            else:
                stale_count += 1
                if stale_count >= max_stale:
                    print("❌ Training appears stuck")
                    return final_metrics

        time.sleep(poll_interval)

    return final_metrics


def run_training(
    config: dict,
    project_path: str,
    app_name: str,
    wandb_project: str,
    wandb_tags: Optional[list[str]] = None,
    visible: bool = False,
    godot_path: str = DEFAULT_GODOT_PATH,
    log_keys: Optional[list[str]] = None,
) -> None:
    """Run a complete training session: config → launch → poll → log → cleanup.

    Args:
        config: Training hyperparameter config dict.
        project_path: Path to the Godot project.
        app_name: Godot app name for user data dir.
        wandb_project: W&B project name.
        wandb_tags: Tags for the W&B run.
        visible: Show Godot window.
        godot_path: Path to Godot binary.
        log_keys: Specific metric keys to log (or None for all).
    """
    import wandb

    user_dir = godot_user_dir(app_name)
    metrics_path = user_dir / "metrics.json"
    config_path = user_dir / "sweep_config.json"

    run = wandb.init(
        project=wandb_project,
        config=config,
        tags=wandb_tags or [],
    )

    max_gens = config.get("max_generations", 100)
    print(f"\n🎮 Starting training (pop={config.get('population_size', '?')}, "
          f"gens={max_gens})\n")

    write_config(config, config_path)
    process = launch_godot(
        project_path, godot_path, visible,
        metrics_path=metrics_path,
    )

    try:
        if not wait_for_metrics(metrics_path):
            process.kill()
            run.finish(exit_code=1)
            return

        final = poll_metrics(run, metrics_path, max_gens, log_keys=log_keys)

        time.sleep(5)
        process.terminate()
        process.wait(timeout=10)

        if final:
            for k, v in final.items():
                if isinstance(v, (int, float)):
                    run.summary[f"final_{k}"] = v

        print("\n✅ Training complete!")
        run.finish(exit_code=0)

    except KeyboardInterrupt:
        print("\n🛑 Interrupted")
        process.kill()
        run.finish(exit_code=130)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        process.kill()
        run.finish(exit_code=1)
