# ruff: noqa: I001
"""Tests for shared godot_wandb utilities."""

import json
import os
import platform
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

# Allow importing the module from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import godot_wandb

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


class TestGodotUserDir:
    def test_returns_path_object(self):
        result = godot_wandb.godot_user_dir("test-app")
        assert isinstance(result, Path)

    def test_contains_app_name(self):
        result = godot_wandb.godot_user_dir("my-project")
        assert "my-project" in str(result)

    def test_override_via_env(self):
        with patch.dict(os.environ, {"GODOT_USER_DIR": "/tmp/custom"}):
            result = godot_wandb.godot_user_dir("ignored")
            assert result == Path("/tmp/custom")

    def test_platform_specific_path(self):
        result = godot_wandb.godot_user_dir("evolve")
        system = platform.system()
        if system == "Darwin":
            assert "Library/Application Support/Godot" in str(result)
        elif system == "Linux":
            assert ".local/share/godot" in str(result)


# ---------------------------------------------------------------------------
# Config / metrics I/O
# ---------------------------------------------------------------------------


class TestWriteConfig:
    def test_writes_valid_json(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "config.json"
            godot_wandb.write_config({"pop_size": 100, "lr": 0.01}, path)
            with open(path) as f:
                data = json.load(f)
            assert data["pop_size"] == 100
            assert data["lr"] == 0.01

    def test_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "sub" / "deep" / "config.json"
            godot_wandb.write_config({"x": 1}, path)
            assert path.exists()


class TestReadMetrics:
    def test_returns_none_for_missing_file(self):
        result = godot_wandb.read_metrics(Path("/nonexistent/path/metrics.json"))
        assert result is None

    def test_reads_valid_json(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "metrics.json"
            path.write_text(json.dumps({"generation": 5, "best_fitness": 42.0}))
            result = godot_wandb.read_metrics(path)
            assert result["generation"] == 5
            assert result["best_fitness"] == 42.0

    def test_returns_none_for_corrupt_json(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "metrics.json"
            path.write_text("not json {{{")
            result = godot_wandb.read_metrics(path)
            assert result is None


# ---------------------------------------------------------------------------
# SweepWorker
# ---------------------------------------------------------------------------


class TestSweepWorker:
    def test_unique_ids(self):
        with tempfile.TemporaryDirectory() as d:
            w1 = godot_wandb.SweepWorker(Path(d))
            w2 = godot_wandb.SweepWorker(Path(d))
            assert w1.worker_id != w2.worker_id

    def test_custom_id(self):
        with tempfile.TemporaryDirectory() as d:
            w = godot_wandb.SweepWorker(Path(d), worker_id="abc123")
            assert w.worker_id == "abc123"
            assert "abc123" in str(w.config_path)
            assert "abc123" in str(w.metrics_path)

    def test_write_and_clear(self):
        with tempfile.TemporaryDirectory() as d:
            w = godot_wandb.SweepWorker(Path(d))
            w.write_config({"test": True})
            assert w.config_path.exists()

            # Write fake metrics
            w.metrics_path.write_text(json.dumps({"gen": 1}))
            assert w.metrics_path.exists()

            w.clear_metrics()
            assert not w.metrics_path.exists()

    def test_cleanup(self):
        with tempfile.TemporaryDirectory() as d:
            w = godot_wandb.SweepWorker(Path(d))
            w.write_config({"x": 1})
            w.metrics_path.write_text("{}")
            w.cleanup()
            assert not w.config_path.exists()
            assert not w.metrics_path.exists()
