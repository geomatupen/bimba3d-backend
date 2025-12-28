import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.projects import router as projects_router
from app.config import ALLOWED_ORIGINS
from app.config import DATA_DIR
import json
from pathlib import Path
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

app = FastAPI(title="Gaussian Splat Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(projects_router, prefix="/projects")


@app.on_event("startup")
def mark_interrupted_projects():
    """On backend start, mark any projects that were 'processing' as stopped/resumable.

    This ensures the frontend doesn't continue to show 'processing' for jobs
    that were interrupted by a backend restart or crash.
    """
    note = "Backend restarted — processing interrupted. Please resume when ready."
    for proj_dir in DATA_DIR.iterdir():
        try:
            if not proj_dir.is_dir():
                continue
            status_file = proj_dir / "status.json"
            if not status_file.exists():
                continue
            try:
                with open(status_file, 'r') as f:
                    data = json.load(f)
            except Exception:
                data = {}
            if data.get("status") == "processing":
                data["status"] = "stopped"
                data["progress"] = data.get("progress", 0)
                data["error"] = note
                data["stop_requested"] = True
                data["stopped_stage"] = data.get("stage", "unknown")
                data["resumable"] = True
                data["percentage"] = data.get("percentage", 0.0)
                # write atomically
                tmp = status_file.with_suffix('.tmp')
                with open(tmp, 'w') as f:
                    json.dump(data, f)
                tmp.replace(status_file)
        except Exception:
            logging.exception(f"Failed to mark interrupted project: {proj_dir}")


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/health/gpu")
def gpu_health():
    """Report GPU availability and basic CUDA/device info."""
    try:
        import torch
        available = torch.cuda.is_available()
        count = torch.cuda.device_count() if available else 0
        devices = []
        for i in range(count):
            try:
                devices.append(torch.cuda.get_device_name(i))
            except Exception:
                devices.append(f"cuda:{i}")
        return {
            "gpu_available": available,
            "device_count": count,
            "devices": devices,
            "cuda_version": getattr(torch.version, "cuda", None),
        }
    except Exception:
        return {
            "gpu_available": False,
            "device_count": 0,
            "devices": [],
            "cuda_version": None,
        }
