# Shared Evolve Utilities

Common Python utilities shared between [Evolve](~/Projects/evolve) and [Chess-Evolve](~/Projects/chess-evolve).

## `godot_wandb.py`

Reusable helpers for running Godot training with W&B logging:

- `write_config(config, path)` — Write JSON config for Godot to read
- `read_metrics(path)` — Read metrics.json from Godot
- `launch_godot(godot_path, project_path, visible, extra_args)` — Launch Godot subprocess
- `wait_for_metrics(path, timeout)` — Wait for training to start
- `poll_metrics(wandb_run, metrics_path, max_gens, poll_interval)` — Poll & log loop
- `run_training(config, ...)` — Full training session with W&B

## Usage

```python
# In your project's train script:
import sys
sys.path.insert(0, os.path.expanduser("~/Projects/shared-evolve-utils"))
from godot_wandb import launch_godot, write_config, wait_for_metrics, poll_metrics
```

Or symlink/pip install as needed.
