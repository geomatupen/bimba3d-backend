import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.projects import router as projects_router
from app.config import ALLOWED_ORIGINS

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
