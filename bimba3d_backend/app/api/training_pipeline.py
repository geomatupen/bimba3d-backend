"""Training pipeline API endpoints for automated cross-project training."""
from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field

from bimba3d_backend.app.config import DATA_DIR
from bimba3d_backend.app.services import training_pipeline_storage
from bimba3d_backend.app.services import training_pipeline_orchestrator
from PIL import Image

logger = logging.getLogger(__name__)
router = APIRouter()


# ========== Request/Response Models ==========

class ScanDirectoryRequest(BaseModel):
    base_directory: str = Field(..., description="Path to directory containing dataset folders")


class DatasetInfo(BaseModel):
    name: str
    path: str
    image_count: int
    size_mb: float
    has_images: bool


class ScanDirectoryResponse(BaseModel):
    datasets: list[DatasetInfo]
    total: int


class ThermalConfig(BaseModel):
    enabled: bool = True
    strategy: str = "fixed_interval"  # or "temperature_based", "time_scheduled"
    cooldown_minutes: int = 10
    gpu_temp_threshold: int = 70
    check_interval_seconds: int = 30
    max_wait_minutes: int = 30


class PhaseConfig(BaseModel):
    phase_number: int
    name: str
    runs_per_project: int = 1
    passes: int = 1
    strategy_override: Optional[str] = None  # Override ai_selector_strategy for this phase
    preset_override: Optional[str] = None  # For baseline phase
    update_model: bool = True
    context_jitter: bool = False
    context_jitter_mode: str = "uniform"  # "uniform" (sample bounds), "mild" (±10%), "gaussian" (±15%)
    shuffle_order: bool = True
    session_execution_mode: str = "train"


class ProjectConfig(BaseModel):
    project_id: Optional[str] = None
    name: str
    dataset_path: str
    baseline_run_id: Optional[str] = None
    image_count: int
    created: bool = False


class CreatePipelineRequest(BaseModel):
    name: str
    base_directory: str
    pipeline_directory: Optional[str] = None  # Where to create pipeline folder (default: DATA_DIR)
    projects: list[ProjectConfig]
    shared_config: dict[str, Any]  # Training parameters shared across all runs
    phases: list[PhaseConfig]
    thermal_management: ThermalConfig
    failure_handling: dict[str, Any] = {
        "continue_on_failure": True,
        "max_retries_per_run": 1,
        "skip_project_after_failures": 3
    }


class PipelineStatusResponse(BaseModel):
    id: str
    name: str
    status: str
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    current_phase: int
    current_pass: int
    current_project_index: int
    total_runs: int
    completed_runs: int
    failed_runs: int
    mean_reward: Optional[float]
    success_rate: Optional[float]
    best_reward: Optional[float]
    last_error: Optional[str]
    cooldown_active: bool
    next_run_scheduled_at: Optional[str]


class BatchCreateProjectsRequest(BaseModel):
    datasets: list[DatasetInfo]
    shared_config: dict[str, Any]


class BatchCreateProjectsResponse(BaseModel):
    created: list[ProjectConfig]
    existing: list[ProjectConfig]
    failed: list[dict[str, str]]


# ========== Utility Functions ==========

def _scan_dataset_folder(folder_path: Path) -> Optional[DatasetInfo]:
    """Scan a folder to extract dataset information."""
    if not folder_path.is_dir():
        return None

    # Look for images
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    image_files = [f for f in folder_path.glob("*") if f.suffix.lower() in image_exts]

    if not image_files:
        return None

    # Calculate size
    total_size = sum(f.stat().st_size for f in image_files)
    size_mb = total_size / (1024 * 1024)

    return DatasetInfo(
        name=folder_path.name,
        path=str(folder_path.absolute()),
        image_count=len(image_files),
        size_mb=round(size_mb, 2),
        has_images=True
    )


# ========== API Endpoints ==========

@router.post("/scan-directory", response_model=ScanDirectoryResponse)
async def scan_directory(request: ScanDirectoryRequest):
    """Scan a directory for dataset folders."""
    base_path = Path(request.base_directory)

    if not base_path.exists():
        raise HTTPException(status_code=404, detail="Directory not found")

    if not base_path.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")

    datasets = []
    for item in base_path.iterdir():
        if item.is_dir():
            dataset_info = _scan_dataset_folder(item)
            if dataset_info:
                datasets.append(dataset_info)

    # Sort by name
    datasets.sort(key=lambda d: d.name)

    return ScanDirectoryResponse(
        datasets=datasets,
        total=len(datasets)
    )


@router.post("/batch-create-projects", response_model=BatchCreateProjectsResponse)
async def batch_create_projects(request: BatchCreateProjectsRequest):
    """Create projects for multiple datasets in batch."""
    created = []
    existing = []
    failed = []

    for dataset in request.datasets:
        try:
            # Check if project already exists
            project_name = dataset.name
            project_dir = DATA_DIR / project_name

            if project_dir.exists():
                # Load existing project
                config_path = project_dir / "config.json"
                if config_path.exists():
                    with open(config_path, "r") as f:
                        config = json.load(f)
                        existing.append(ProjectConfig(
                            project_id=config.get("id"),
                            name=project_name,
                            dataset_path=dataset.path,
                            image_count=dataset.image_count,
                            created=False
                        ))
                continue

            # Create new project
            project_id = str(uuid.uuid4())
            project_dir.mkdir(parents=True, exist_ok=True)

            # Create config
            config = {
                "id": project_id,
                "name": project_name,
                "source_dir": dataset.path,
                "created_at": datetime.utcnow().isoformat() + "Z",
                **request.shared_config
            }

            config_path = project_dir / "config.json"
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

            created.append(ProjectConfig(
                project_id=project_id,
                name=project_name,
                dataset_path=dataset.path,
                image_count=dataset.image_count,
                created=True
            ))

        except Exception as e:
            logger.error(f"Failed to create project for {dataset.name}: {e}")
            failed.append({
                "dataset_name": dataset.name,
                "error": str(e)
            })

    return BatchCreateProjectsResponse(
        created=created,
        existing=existing,
        failed=failed
    )


@router.post("/create", response_model=PipelineStatusResponse)
async def create_pipeline(request: CreatePipelineRequest):
    """Create a new training pipeline."""
    try:
        config = request.dict()
        pipeline = training_pipeline_storage.create_pipeline(config)

        return PipelineStatusResponse(**pipeline)

    except Exception as e:
        logger.error(f"Failed to create pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{pipeline_id}/start")
async def start_pipeline(pipeline_id: str):
    """Start pipeline execution."""
    pipeline = training_pipeline_storage.get_pipeline(pipeline_id)

    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    if pipeline["status"] in ["running"]:
        raise HTTPException(status_code=400, detail="Pipeline is already running")

    # Update status
    updates = {
        "status": "running",
        "started_at": datetime.utcnow().isoformat() + "Z",
        "current_phase": 1,
        "current_pass": 1,
        "current_project_index": 0,
    }

    pipeline = training_pipeline_storage.update_pipeline(pipeline_id, updates)

    # Start orchestrator in background thread
    training_pipeline_orchestrator.start_pipeline_orchestrator(pipeline_id)

    return {"status": "running", "message": "Pipeline started"}


@router.post("/{pipeline_id}/pause")
async def pause_pipeline(pipeline_id: str):
    """Pause pipeline execution."""
    pipeline = training_pipeline_storage.get_pipeline(pipeline_id)

    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    if pipeline["status"] != "running":
        raise HTTPException(status_code=400, detail="Pipeline is not running")

    pipeline = training_pipeline_storage.update_pipeline(pipeline_id, {"status": "paused"})

    # Signal orchestrator to pause
    training_pipeline_orchestrator.stop_pipeline_orchestrator(pipeline_id)

    return {"status": "paused", "message": "Pipeline paused"}


@router.post("/{pipeline_id}/resume")
async def resume_pipeline(pipeline_id: str):
    """Resume paused pipeline."""
    pipeline = training_pipeline_storage.get_pipeline(pipeline_id)

    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    if pipeline["status"] != "paused":
        raise HTTPException(status_code=400, detail="Pipeline is not paused")

    pipeline = training_pipeline_storage.update_pipeline(pipeline_id, {"status": "running"})

    # Resume orchestrator
    training_pipeline_orchestrator.start_pipeline_orchestrator(pipeline_id)

    return {"status": "running", "message": "Pipeline resumed"}


@router.post("/{pipeline_id}/stop")
async def stop_pipeline(pipeline_id: str):
    """Stop pipeline execution."""
    pipeline = training_pipeline_storage.get_pipeline(pipeline_id)

    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    pipeline = training_pipeline_storage.update_pipeline(pipeline_id, {
        "status": "stopped",
        "completed_at": datetime.utcnow().isoformat() + "Z"
    })

    # Signal orchestrator to stop
    training_pipeline_orchestrator.stop_pipeline_orchestrator(pipeline_id)

    return {"status": "stopped", "message": "Pipeline stopped"}


@router.get("/{pipeline_id}", response_model=PipelineStatusResponse)
async def get_pipeline(pipeline_id: str):
    """Get pipeline status and details."""
    pipeline = training_pipeline_storage.get_pipeline(pipeline_id)

    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    return PipelineStatusResponse(**pipeline)


@router.get("/{pipeline_id}/runs")
async def get_pipeline_runs(pipeline_id: str):
    """Get all runs for a pipeline."""
    pipeline = training_pipeline_storage.get_pipeline(pipeline_id)

    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    return {"runs": pipeline.get("runs", [])}


@router.get("/list")
async def list_pipelines(limit: int = 50):
    """List all pipelines."""
    pipelines = training_pipeline_storage.list_pipelines(limit=limit)

    return {"pipelines": [PipelineStatusResponse(**p) for p in pipelines]}


@router.delete("/{pipeline_id}")
async def delete_pipeline(pipeline_id: str):
    """Delete a pipeline."""
    success = training_pipeline_storage.delete_pipeline(pipeline_id)

    if not success:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    return {"message": "Pipeline deleted successfully"}
