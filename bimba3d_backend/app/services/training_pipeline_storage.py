"""Training pipeline storage service using file-based JSON storage."""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from bimba3d_backend.app.config import DATA_DIR

PIPELINES_DIR = DATA_DIR / "training_pipelines"
PIPELINES_DIR.mkdir(parents=True, exist_ok=True)


def _pipeline_path(pipeline_id: str) -> Path:
    """Get path to pipeline JSON file."""
    return PIPELINES_DIR / f"{pipeline_id}.json"


def _timestamp_now() -> str:
    """Get current timestamp in ISO format."""
    return datetime.utcnow().isoformat() + "Z"


def create_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    """Create a new training pipeline.

    Args:
        config: Pipeline configuration including projects, phases, thermal settings

    Returns:
        Complete pipeline state with metadata
    """
    pipeline_id = f"pipeline_{uuid.uuid4().hex[:12]}"

    # Calculate total runs
    total_runs = 0
    for phase in config.get("phases", []):
        runs_per_project = phase.get("runs_per_project", 1)
        passes = phase.get("passes", 1)
        project_count = len(config.get("projects", []))
        total_runs += runs_per_project * passes * project_count

    pipeline = {
        "id": pipeline_id,
        "name": config.get("name", f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
        "status": "pending",
        "created_at": _timestamp_now(),
        "started_at": None,
        "completed_at": None,

        # Configuration
        "config": config,

        # Progress tracking
        "current_phase": 1,
        "current_pass": 1,
        "current_project_index": 0,
        "total_runs": total_runs,
        "completed_runs": 0,
        "failed_runs": 0,

        # Statistics
        "mean_reward": None,
        "success_rate": None,
        "best_reward": None,

        # Thermal management
        "last_run_ended_at": None,
        "next_run_scheduled_at": None,
        "cooldown_active": False,

        # Error handling
        "last_error": None,
        "retry_count": 0,

        # Run history
        "runs": [],
    }

    # Save to disk
    with open(_pipeline_path(pipeline_id), "w") as f:
        json.dump(pipeline, f, indent=2)

    return pipeline


def get_pipeline(pipeline_id: str) -> Optional[dict[str, Any]]:
    """Load pipeline by ID."""
    path = _pipeline_path(pipeline_id)
    if not path.exists():
        return None

    with open(path, "r") as f:
        return json.load(f)


def update_pipeline(pipeline_id: str, updates: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Update pipeline with new data."""
    pipeline = get_pipeline(pipeline_id)
    if not pipeline:
        return None

    pipeline.update(updates)

    # Save to disk
    with open(_pipeline_path(pipeline_id), "w") as f:
        json.dump(pipeline, f, indent=2)

    return pipeline


def list_pipelines(limit: int = 50) -> list[dict[str, Any]]:
    """List all pipelines, most recent first."""
    pipelines = []

    for path in PIPELINES_DIR.glob("pipeline_*.json"):
        try:
            with open(path, "r") as f:
                pipeline = json.load(f)
                pipelines.append(pipeline)
        except Exception:
            continue

    # Sort by created_at descending
    pipelines.sort(key=lambda p: p.get("created_at", ""), reverse=True)

    return pipelines[:limit]


def add_run_result(pipeline_id: str, run_result: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Add a run result to pipeline history and update statistics."""
    pipeline = get_pipeline(pipeline_id)
    if not pipeline:
        return None

    # Add to history
    pipeline["runs"].append(run_result)
    pipeline["completed_runs"] += 1

    # Update statistics
    rewards = [r["reward"] for r in pipeline["runs"] if r.get("reward") is not None]
    if rewards:
        pipeline["mean_reward"] = sum(rewards) / len(rewards)
        pipeline["success_rate"] = len([r for r in rewards if r > 0]) / len(rewards)
        pipeline["best_reward"] = max(rewards)

    # Save
    with open(_pipeline_path(pipeline_id), "w") as f:
        json.dump(pipeline, f, indent=2)

    return pipeline


def delete_pipeline(pipeline_id: str) -> bool:
    """Delete a pipeline."""
    path = _pipeline_path(pipeline_id)
    if not path.exists():
        return False

    path.unlink()
    return True
