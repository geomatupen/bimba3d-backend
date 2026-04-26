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

    # Determine where to create pipeline folder
    pipeline_directory = config.get("pipeline_directory")
    if pipeline_directory:
        # User specified custom location
        pipeline_root = Path(pipeline_directory)
    else:
        # Default: same location as DATA_DIR
        pipeline_root = DATA_DIR

    # Create pipeline folder with sanitized name (replace spaces with underscores)
    pipeline_name = config.get("name", f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    sanitized_name = pipeline_name.replace(" ", "_")
    pipeline_folder = pipeline_root / sanitized_name
    pipeline_folder.mkdir(parents=True, exist_ok=True)

    # Store pipeline folder path in config for orchestrator
    config["pipeline_folder"] = str(pipeline_folder)

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

    # Save to centralized pipelines registry
    with open(_pipeline_path(pipeline_id), "w") as f:
        json.dump(pipeline, f, indent=2)

    # Also save pipeline.json marker in the pipeline folder so it's not listed as a project
    pipeline_marker = pipeline_folder / "pipeline.json"
    with open(pipeline_marker, "w") as f:
        json.dump({
            "pipeline_id": pipeline_id,
            "pipeline_name": pipeline["name"],
            "created_at": pipeline["created_at"],
        }, f, indent=2)

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

    # Update counters based on actual status
    if run_result.get("status") == "success":
        pipeline["completed_runs"] += 1
    elif run_result.get("status") == "failed":
        pipeline["failed_runs"] += 1

    # Update statistics
    # Only calculate reward stats for runs that have rewards (AI learning phases, not baseline)
    rewards = [r["reward"] for r in pipeline["runs"] if r.get("reward") is not None]
    if rewards:
        pipeline["mean_reward"] = sum(rewards) / len(rewards)
        pipeline["best_reward"] = max(rewards)

    # Success rate is based on run status, not rewards
    successful_runs = [r for r in pipeline["runs"] if r.get("status") == "success"]
    total_runs = len(pipeline["runs"])
    if total_runs > 0:
        pipeline["success_rate"] = (len(successful_runs) / total_runs) * 100

    # Save
    with open(_pipeline_path(pipeline_id), "w") as f:
        json.dump(pipeline, f, indent=2)

    return pipeline


def delete_pipeline(pipeline_id: str) -> bool:
    """Delete a pipeline and its folder."""
    import shutil
    import logging

    logger = logging.getLogger(__name__)
    path = _pipeline_path(pipeline_id)

    if not path.exists():
        return False

    # Read pipeline config to get folder path
    try:
        with open(path, "r") as f:
            pipeline = json.load(f)

        pipeline_folder = pipeline.get("config", {}).get("pipeline_folder")

        # Delete the pipeline folder if it exists
        if pipeline_folder:
            folder_path = Path(pipeline_folder)
            if folder_path.exists():
                try:
                    shutil.rmtree(folder_path)
                    logger.info(f"Deleted pipeline folder: {folder_path}")
                except Exception as e:
                    logger.error(f"Failed to delete pipeline folder {folder_path}: {e}")

        # Also clean up any symlinks in DATA_DIR for pipeline projects
        if "projects" in pipeline.get("config", {}):
            for project in pipeline["config"]["projects"]:
                project_name = project.get("name")
                if project_name:
                    project_dir = Path(pipeline_folder) / project_name if pipeline_folder else None
                    if project_dir and project_dir.exists():
                        # Read project config to get UUID
                        config_file = project_dir / "config.json"
                        if config_file.exists():
                            try:
                                with open(config_file, "r") as f:
                                    proj_config = json.load(f)
                                project_id = proj_config.get("id")
                                if project_id:
                                    # Remove symlink in DATA_DIR
                                    symlink = DATA_DIR / project_id
                                    if symlink.exists():
                                        try:
                                            symlink.unlink()
                                            logger.info(f"Deleted project symlink: {symlink}")
                                        except Exception as e:
                                            logger.warning(f"Failed to delete symlink {symlink}: {e}")
                            except Exception as e:
                                logger.warning(f"Failed to clean up symlinks for project {project_name}: {e}")

    except Exception as e:
        logger.error(f"Failed to read pipeline config during deletion: {e}")

    # Delete the pipeline metadata JSON
    path.unlink()
    return True
