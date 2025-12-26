import subprocess
import time
import logging
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Check if running in Docker mode
USE_DOCKER = os.getenv("USE_DOCKER_WORKER", "true").lower() == "true"


def run_colmap_docker(project_id: str, params: dict = None) -> None:
    """
    Run COLMAP via Docker worker.
    """
    from app.config import DATA_DIR
    
    params_json = json.dumps(params or {})
    # DATA_DIR is now /path/to/websplat-backend/data/projects
    # Mount parent: /path/to/websplat-backend/data -> /data
    data_dir = DATA_DIR.parent  
    
    cmd = [
        "docker", "run", "--rm",
        "--gpus", "all",  # Enable GPU access
        "-v", f"{data_dir}:/data",
        "bimba3d-worker:latest",
        project_id,
        "--data-dir", "/data/projects",
        "--params", params_json
    ]
    
    logger.info(f"Running Docker worker: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Docker worker failed: {e.stderr}")
        raise


def _run_cmd_with_retry(cmd: list[str], retries: int = 3, delay_sec: float = 2.0):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            res = subprocess.run(cmd, check=True, capture_output=True, text=True)
            if res.stdout:
                logger.info(res.stdout.strip())
            if res.stderr:
                logger.debug(res.stderr.strip())
            return
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or "").lower()
            last_err = e
            if "database is locked" in stderr or "busy" in stderr:
                logger.warning(f"SQLite busy/locked (attempt {attempt}/{retries}). Retrying after {delay_sec}s...")
                time.sleep(delay_sec)
                continue
            logger.error(f"Command failed: {cmd}\nSTDERR: {e.stderr}")
            raise
    logger.error(f"Command failed after retries: {cmd}\nERR: {last_err}")
    raise last_err


def _cleanup_sqlite_sidecars(db_path: Path):
    for suffix in ("-wal", "-shm"):
        sidecar = db_path.with_name(db_path.name + suffix)
        if sidecar.exists():
            try:
                sidecar.unlink()
                logger.info(f"Removed stale SQLite sidecar: {sidecar}")
            except Exception as e:
                logger.warning(f"Failed to remove sidecar {sidecar}: {e}")


def run_colmap(image_dir: Path, output_dir: Path):
    """
    Run COLMAP feature extraction, matching, and sparse reconstruction.
    
    Args:
        image_dir: Directory containing input images
        output_dir: Directory for COLMAP outputs
        
    Returns:
        Path to sparse reconstruction directory
        
    Raises:
        subprocess.CalledProcessError: If COLMAP commands fail
        FileNotFoundError: If COLMAP is not installed
    """
    db_path = output_dir / "database.db"
    sparse_dir = output_dir / "sparse"
    
    sparse_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove existing database and sidecars to prevent locking issues
    if db_path.exists():
        logger.info(f"Removing existing database: {db_path}")
        try:
            db_path.unlink()
        except Exception:
            db_path.write_bytes(b"")
    _cleanup_sqlite_sidecars(db_path)
    
    try:
        # 1️⃣ Feature extraction
        logger.info("Running COLMAP feature extraction...")
        _run_cmd_with_retry([
            "colmap", "feature_extractor",
            "--database_path", str(db_path),
            "--image_path", str(image_dir),
            "--ImageReader.camera_model", "OPENCV",
            "--ImageReader.single_camera", "1",
            "--SiftExtraction.max_image_size", "3200",
            "--SiftExtraction.peak_threshold", "0.01",
        ])
        logger.info("✓ Feature extraction completed")
        
        # 2️⃣ Feature matching
        logger.info("Running COLMAP feature matching...")
        _run_cmd_with_retry([
            "colmap", "exhaustive_matcher",
            "--database_path", str(db_path),
            "--SiftMatching.guided_matching", "1",
        ])
        logger.info("✓ Feature matching completed")
        
        # 3️⃣ Sparse reconstruction
        logger.info("Running COLMAP sparse reconstruction (mapper)...")
        _run_cmd_with_retry([
            "colmap", "mapper",
            "--database_path", str(db_path),
            "--image_path", str(image_dir),
            "--output_path", str(sparse_dir),
            "--Mapper.ba_refine_principal_point", "1",
            "--Mapper.ba_refine_focal_length", "1",
            "--Mapper.ba_refine_extra_params", "1",
        ])
        logger.info("✓ Sparse reconstruction completed")
        
        # Verify outputs
        if not (sparse_dir / "0").exists():
            raise FileNotFoundError(f"COLMAP reconstruction failed - no output in {sparse_dir}")
        
        logger.info(f"COLMAP outputs saved to: {sparse_dir}")
        return sparse_dir
        
    except FileNotFoundError as e:
        logger.error(f"COLMAP not found. Install with: apt-get install colmap")
        raise
    except subprocess.CalledProcessError as e:
        logger.error(f"COLMAP command failed: {e.stderr}")
        raise
