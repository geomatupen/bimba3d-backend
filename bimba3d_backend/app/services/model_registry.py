import hashlib
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from bimba3d_backend.app.config import DATA_DIR

SHARED_CONFIG_FILE = "shared_config.json"
MODELS_DIR = DATA_DIR.parent / "models"
MODELS_INDEX_FILE = MODELS_DIR / "models_index.json"
ALLOWED_SNAPSHOT_FILENAMES = {"run_config.json", "shared_config.json", "metadata.json"}
INDEX_ALLOWED_KEYS = {
    "model_id",
    "model_name",
    "engine",
    "created_at",
    "source",
    "paths",
    "artifact_format",
    "provenance_summary",
}


def ensure_models_store() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_model_token(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip().lower())
    cleaned = re.sub(r"-+", "-", cleaned).strip("-_")
    return cleaned or "model"


def utc_now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def build_model_id(model_name: str) -> str:
    return f"model_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{sanitize_model_token(model_name)}"


def read_json_if_exists(path: Path | None):
    if path is None or not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def write_json_atomic(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    tmp_path.replace(path)


def load_models_index() -> list[dict]:
    ensure_models_store()
    raw = read_json_if_exists(MODELS_INDEX_FILE)
    if isinstance(raw, list):
        filtered = [item for item in raw if isinstance(item, dict)]
        compacted = [compact_index_record(item) for item in filtered]
        if compacted != filtered:
            # One-way migration: keep the global index lightweight.
            write_json_atomic(MODELS_INDEX_FILE, compacted)
        return compacted
    return []


def save_models_index(items: list[dict]) -> None:
    ensure_models_store()
    compacted = [compact_index_record(item) for item in items if isinstance(item, dict)]
    write_json_atomic(MODELS_INDEX_FILE, compacted)


def compact_index_record(model_record: dict) -> dict:
    compact = {key: model_record.get(key) for key in INDEX_ALLOWED_KEYS if key in model_record}
    source = compact.get("source")
    if isinstance(source, dict):
        compact["source"] = {
            "project_id": source.get("project_id"),
            "project_name": source.get("project_name"),
            "run_id": source.get("run_id"),
        }
    paths = compact.get("paths")
    if isinstance(paths, dict):
        compact["paths"] = {
            "checkpoint": paths.get("checkpoint"),
            "artifact": paths.get("artifact"),
            "model_dir": paths.get("model_dir"),
            "configs_dir": paths.get("configs_dir"),
            "lineage": paths.get("lineage"),
        }
    summary = compact.get("provenance_summary")
    if isinstance(summary, dict):
        compact["provenance_summary"] = {
            "contributor_count": summary.get("contributor_count"),
            "unique_project_count": summary.get("unique_project_count"),
            "project_names": summary.get("project_names"),
        }
    return compact


def find_latest_gsplat_checkpoint(run_dir: Path) -> Path | None:
    ckpt_dir = run_dir / "outputs" / "engines" / "gsplat" / "ckpts"
    if not ckpt_dir.exists():
        return None

    candidates = sorted(ckpt_dir.glob("ckpt_*_rank0.pt"))
    if not candidates:
        candidates = sorted(ckpt_dir.glob("ckpt_*.pt"))
    if not candidates:
        return None
    return candidates[-1]


def resolve_reusable_model(model_id: str) -> dict | None:
    model_id = str(model_id or "").strip()
    if not model_id:
        return None
    for item in load_models_index():
        if str(item.get("model_id") or "") == model_id:
            return item
    return None


def get_model_record(model_id: str) -> dict | None:
    return resolve_reusable_model(model_id)


def rename_model(model_id: str, model_name: str) -> dict | None:
    model_id = str(model_id or "").strip()
    normalized_name = str(model_name or "").strip()
    if not model_id or not normalized_name:
        return None

    records = load_models_index()
    target_record = None
    for item in records:
        if str(item.get("model_id") or "") == model_id:
            item["model_name"] = normalized_name
            target_record = item
            break

    if not isinstance(target_record, dict):
        return None

    save_models_index(records)

    model_dir = _model_dir_from_record(model_id, target_record)
    model_json = read_json_if_exists(model_dir / "model.json")
    if isinstance(model_json, dict):
        model_json["model_name"] = normalized_name
        write_json_atomic(model_dir / "model.json", model_json)
    else:
        write_json_atomic(model_dir / "model.json", target_record)

    return target_record


def remove_model(model_id: str) -> dict | None:
    model_id = str(model_id or "").strip()
    if not model_id:
        return None

    records = load_models_index()
    removed = None
    kept: list[dict] = []
    for item in records:
        if str(item.get("model_id") or "") == model_id and removed is None:
            removed = item
            continue
        kept.append(item)

    if removed is None:
        return None

    save_models_index(kept)
    return removed


def _model_dir_from_record(model_id: str, model_record: dict) -> Path:
    paths = model_record.get("paths") if isinstance(model_record.get("paths"), dict) else {}
    model_dir_raw = paths.get("model_dir") if isinstance(paths, dict) else None
    if isinstance(model_dir_raw, str) and model_dir_raw.strip():
        return Path(model_dir_raw).expanduser()
    return MODELS_DIR / model_id


def _build_configs_tree(configs_dir: Path) -> dict:
    projects: list[dict] = []
    if not configs_dir.exists() or not configs_dir.is_dir():
        return {"projects": projects}

    for project_path in sorted([p for p in configs_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
        runs: list[dict] = []
        for run_path in sorted([p for p in project_path.iterdir() if p.is_dir()], key=lambda p: p.name):
            files = []
            for candidate in sorted([p for p in run_path.iterdir() if p.is_file()], key=lambda p: p.name):
                files.append(
                    {
                        "name": candidate.name,
                        "path": candidate.as_posix(),
                        "size": candidate.stat().st_size,
                    }
                )
            runs.append(
                {
                    "run_id": run_path.name,
                    "path": run_path.as_posix(),
                    "files": files,
                }
            )
        projects.append(
            {
                "project_id": project_path.name,
                "path": project_path.as_posix(),
                "runs": runs,
            }
        )
    return {"projects": projects}


def get_model_lineage_summary(model_id: str) -> dict | None:
    model_record = get_model_record(model_id)
    if not isinstance(model_record, dict):
        return None

    model_dir = _model_dir_from_record(model_id, model_record)
    lineage = read_json_if_exists(model_dir / "lineage.json")
    lineage_doc = lineage if isinstance(lineage, dict) else {"contributors": []}
    contributors = lineage_doc.get("contributors") if isinstance(lineage_doc.get("contributors"), list) else []

    summary = model_record.get("provenance_summary") if isinstance(model_record.get("provenance_summary"), dict) else summarize_lineage(contributors)
    configs_dir = model_dir / "configs"

    return {
        "model": model_record,
        "lineage": lineage_doc,
        "provenance_summary": summary,
        "configs": _build_configs_tree(configs_dir),
    }


def resolve_config_snapshot_file(model_id: str, project_id: str, run_id: str, filename: str) -> Path | None:
    model_record = get_model_record(model_id)
    if not isinstance(model_record, dict):
        return None

    filename = str(filename or "").strip()
    if filename not in ALLOWED_SNAPSHOT_FILENAMES:
        return None

    model_dir = _model_dir_from_record(model_id, model_record)
    target = (model_dir / "configs" / str(project_id) / str(run_id) / filename).resolve()
    root = model_dir.resolve()

    try:
        target.relative_to(root)
    except ValueError:
        return None

    if not target.exists() or not target.is_file():
        return None
    return target


def sha256_for_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def copy_if_exists(src: Path, dst: Path) -> Optional[dict]:
    if not src.exists() or not src.is_file():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {
        "path": dst.as_posix(),
        "sha256": sha256_for_file(dst),
        "size": dst.stat().st_size,
    }


def import_parent_configs_into_model(parent_model_record: dict | None, model_dir: Path) -> None:
    if not isinstance(parent_model_record, dict):
        return
    parent_model_dir_raw = ((parent_model_record.get("paths") or {}).get("model_dir") if isinstance(parent_model_record.get("paths"), dict) else None)
    if not parent_model_dir_raw:
        return
    parent_model_dir = Path(str(parent_model_dir_raw)).expanduser()
    parent_configs_dir = parent_model_dir / "configs"
    if not parent_configs_dir.exists() or not parent_configs_dir.is_dir():
        return

    target_configs_dir = model_dir / "configs"
    shutil.copytree(parent_configs_dir, target_configs_dir, dirs_exist_ok=True)


def load_parent_lineage_contributors(parent_model_record: dict | None) -> list[dict]:
    if not isinstance(parent_model_record, dict):
        return []
    parent_model_dir_raw = ((parent_model_record.get("paths") or {}).get("model_dir") if isinstance(parent_model_record.get("paths"), dict) else None)
    if not parent_model_dir_raw:
        return []
    parent_model_dir = Path(str(parent_model_dir_raw)).expanduser()
    lineage = read_json_if_exists(parent_model_dir / "lineage.json")
    if isinstance(lineage, dict) and isinstance(lineage.get("contributors"), list):
        return [item for item in lineage["contributors"] if isinstance(item, dict)]
    return []


def snapshot_contributor_configs(
    model_dir: Path,
    project_dir: Path,
    run_dir: Path,
    project_id: str,
    project_name: Optional[str],
    run_id: str,
    captured_at: str,
) -> dict:
    contributor_dir = model_dir / "configs" / project_id / run_id
    contributor_id = f"{project_id}:{run_id}"

    run_cfg = copy_if_exists(run_dir / "run_config.json", contributor_dir / "run_config.json")
    shared_cfg = copy_if_exists(project_dir / SHARED_CONFIG_FILE, contributor_dir / SHARED_CONFIG_FILE)
    metadata_cfg = copy_if_exists(
        run_dir / "outputs" / "engines" / "gsplat" / "metadata.json",
        contributor_dir / "metadata.json",
    )

    files = {
        "run_config": run_cfg,
        "shared_config": shared_cfg,
        "metadata": metadata_cfg,
    }
    compact_files = {key: value for key, value in files.items() if value is not None}

    return {
        "contributor_id": contributor_id,
        "project_id": project_id,
        "project_name": project_name,
        "run_id": run_id,
        "captured_at": captured_at,
        "files": compact_files,
    }


def dedupe_contributors(contributors: list[dict]) -> list[dict]:
    by_id: dict[str, dict] = {}
    ordered_ids: list[str] = []
    for item in contributors:
        if not isinstance(item, dict):
            continue
        contributor_id = str(item.get("contributor_id") or "").strip()
        if not contributor_id:
            project_id = str(item.get("project_id") or "").strip()
            run_id = str(item.get("run_id") or "").strip()
            if project_id and run_id:
                contributor_id = f"{project_id}:{run_id}"
                item = {**item, "contributor_id": contributor_id}
            else:
                continue
        if contributor_id not in by_id:
            ordered_ids.append(contributor_id)
        by_id[contributor_id] = item
    return [by_id[item_id] for item_id in ordered_ids]


def summarize_lineage(contributors: list[dict]) -> dict:
    project_names = []
    for item in contributors:
        name = str(item.get("project_name") or "").strip()
        if name and name not in project_names:
            project_names.append(name)
    unique_project_ids = {
        str(item.get("project_id") or "").strip()
        for item in contributors
        if str(item.get("project_id") or "").strip()
    }
    return {
        "contributor_count": len(contributors),
        "unique_project_count": len(unique_project_ids),
        "project_names": project_names,
    }


def write_lineage(
    model_dir: Path,
    model_id: str,
    source_model_id: Optional[str],
    contributors: list[dict],
    captured_at: str,
) -> dict:
    doc = {
        "version": 1,
        "model_id": model_id,
        "captured_at": captured_at,
        "source_model_id": source_model_id,
        "contributors": dedupe_contributors(contributors),
    }
    write_json_atomic(model_dir / "lineage.json", doc)
    return doc
