# Bimba3d Backend

FastAPI backend for project management, processing pipeline (COLMAP + training), and status/preview endpoints.

## Quick Start
- Install deps: `pip install -r requirements.txt`
- Run dev server: `uvicorn app.main:app --reload --port 8005`

## Key Endpoints
- `POST /projects` — create a project
- `POST /projects/{id}/images` — upload images
- `POST /projects/{id}/process` — start pipeline
  - Body params include:
    - `stage`: `full` | `colmap_only` | `train_only`
    - `max_steps`, `batch_size`
    - `splat_export_interval`, `png_export_interval`, `auto_early_stop`
- `POST /projects/{id}/stop` — request manual stop
- `GET /projects/{id}/status` — status with `stage`, `message`, `device`
- `GET /projects/{id}/preview` — latest preview PNG
- `GET /health/gpu` — GPU availability and device info

## Pipeline Stages
- `full`: COLMAP sparse + training
- `colmap_only`: Only run COLMAP sparse reconstruction
- `train_only`: Only run training (requires existing sparse outputs)



## Notes
- Default `max_steps` is 300; previews and splat exports occur at configured intervals.
- Manual stop triggers a final export and sets status to `stopped`.
- GPU detection uses PyTorch; when CUDA is unavailable, training runs on CPU (slower).
