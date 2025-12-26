"""Resume detection utilities for interrupted pipelines."""
from pathlib import Path
from typing import Optional, Dict
from app.config import DATA_DIR


def can_resume_project(project_id: str) -> Dict[str, any]:
    """
    Check if a project has resumable state.
    Returns dict with:
      - can_resume: bool
      - has_sparse: bool (COLMAP completed)
      - has_checkpoints: bool (training checkpoints exist)
      - last_checkpoint_step: Optional[int]
    """
    project_dir = DATA_DIR / project_id
    output_dir = project_dir / "outputs"
    
    # Check for COLMAP sparse output
    sparse_dir = output_dir / "sparse" / "0"
    has_sparse = sparse_dir.exists() and any(sparse_dir.iterdir())
    
    # Check for training checkpoints
    ckpt_dir = output_dir / "checkpoints"
    has_checkpoints = False
    last_checkpoint_step = None
    
    if ckpt_dir.exists():
        checkpoints = sorted(ckpt_dir.glob("ckpt_*.pt"))
        if checkpoints:
            has_checkpoints = True
            # Extract step from filename like ckpt_001000.pt
            latest_ckpt = checkpoints[-1]
            try:
                step_str = latest_ckpt.stem.split("_")[-1]
                last_checkpoint_step = int(step_str)
            except (IndexError, ValueError):
                pass
    
    # Check for full completion: status.json status=="completed" and all outputs present
    status_file = project_dir / "status.json"
    fully_completed = False
    if status_file.exists():
        try:
            import json
            with open(status_file) as f:
                s = json.load(f)
            if s.get("status") == "completed":
                # Check for main outputs: splats.splat and splats.ply
                splat = output_dir / "splats.splat"
                ply = output_dir / "splats.ply"
                if splat.exists() and ply.exists():
                    fully_completed = True
        except Exception:
            pass
    can_resume = (has_sparse or has_checkpoints) and not fully_completed
    return {
        "can_resume": can_resume,
        "has_sparse": has_sparse,
        "has_checkpoints": has_checkpoints,
        "last_checkpoint_step": last_checkpoint_step,
    }
