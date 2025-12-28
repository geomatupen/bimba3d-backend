import logging
from pathlib import Path
from app.config import DATA_DIR

logger = logging.getLogger(__name__)


def get_output_files(project_id: str) -> dict:
    """
    Get list of output files for a project.
    """
    project_dir = DATA_DIR / project_id
    output_dir = project_dir / "outputs"
    
    if not output_dir.exists():
        return {}
    
    files = {}
    
    # Check for standard outputs
    # Support multiple splats formats: .splat (optimized), .ply, .bin
    splats_splat = output_dir / "splats.splat"
    splats_ply = output_dir / "splats.ply"
    splats_bin = output_dir / "splats.bin"
    if splats_splat.exists():
        files["splats"] = {"format": "splat", "path": str(splats_splat), "size": splats_splat.stat().st_size}
    elif splats_ply.exists():
        files["splats"] = {"format": "ply", "path": str(splats_ply), "size": splats_ply.stat().st_size}
    elif splats_bin.exists():
        files["splats"] = {"format": "bin", "path": str(splats_bin), "size": splats_bin.stat().st_size}
    
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        files["metadata"] = {
            "path": str(metadata_path),
            "size": metadata_path.stat().st_size,
            "type": "json"
        }

    # Previews (latest PNGs during training)
    previews_dir = output_dir / "previews"
    if previews_dir.exists():
        previews = []
        for preview in sorted(previews_dir.glob("preview_*.png")):
            previews.append({
                "name": preview.name,
                "path": str(preview),
                "size": preview.stat().st_size,
            })
        latest_preview = previews_dir / "preview_latest.png"
        files["previews"] = {
            "items": previews,
            "latest": str(latest_preview) if latest_preview.exists() else None,
        }
    
    # Check for checkpoints
    ckpt_dir = output_dir / "checkpoints"
    if ckpt_dir.exists():
        checkpoints = []
        for ckpt in sorted(ckpt_dir.glob("ckpt_*.pt")):
            checkpoints.append({
                "name": ckpt.name,
                "path": str(ckpt),
                "size": ckpt.stat().st_size,
            })
        if checkpoints:
            files["checkpoints"] = checkpoints
    
    # Check for images
    images_dir = project_dir / "images"
    if images_dir.exists():
        images = []
        for img in sorted(images_dir.glob("*")):
            if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"} and img.is_file():
                images.append({
                    "name": img.name,
                    "path": str(img),
                    "size": img.stat().st_size,
                })
        if images:
            files["images"] = images

    # Check for COLMAP sparse reconstructions under outputs/sparse/*
    sparse_root = output_dir / "sparse"
    if sparse_root.exists() and sparse_root.is_dir():
        reconstructions = []
        for recon in sorted([p for p in sparse_root.iterdir() if p.is_dir()]):
            recon_info = {"name": recon.name, "path": str(recon), "files": []}
            # Look for common COLMAP outputs
            points = recon / "points3D.bin"
            cams = recon / "cameras.bin"
            imgs = recon / "images.bin"
            proj = recon / "project.ini"
            if points.exists():
                recon_info["complete"] = True
            else:
                recon_info["complete"] = False
            for f in (points, cams, imgs, proj):
                if f.exists():
                    recon_info["files"].append({"name": f.name, "size": f.stat().st_size, "path": str(f)})
            reconstructions.append(recon_info)
        if reconstructions:
            files["sparse"] = reconstructions
    
    return files


def get_file_path(project_id: str, file_type: str, filename: str = None) -> Path:
    """
    Get path to a specific output file.
    """
    project_dir = DATA_DIR / project_id
    output_dir = project_dir / "outputs"
    
    if file_type == "splats":
        return output_dir / "splats.bin"
    elif file_type == "metadata":
        return output_dir / "metadata.json"
    elif file_type == "image" and filename:
        return project_dir / "images" / filename
    else:
        raise ValueError(f"Unknown file type: {file_type}")
