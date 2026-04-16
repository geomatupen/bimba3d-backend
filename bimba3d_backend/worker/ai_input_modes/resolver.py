from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .common import ModeContext, apply_preset_updates
from .exif_only import build_preset as build_exif_only_preset
from .exif_plus_flight_plan import build_preset as build_exif_plus_flight_plan_preset
from .exif_plus_flight_plan_plus_external import (
    build_preset as build_exif_plus_flight_plan_plus_external_preset,
)
from .learner import select_preset

VALID_AI_INPUT_MODES = {
    "exif_only",
    "exif_plus_flight_plan",
    "exif_plus_flight_plan_plus_external",
}

CACHE_VERSION = 1
VALID_PRESET_OVERRIDES = {"conservative", "balanced", "geometry_fast", "appearance_fast"}


def _normalize_preset_override(value: Any) -> str:
    token = str(value or "").strip().lower()
    return token if token in VALID_PRESET_OVERRIDES else ""


def _feature_cache_dir(project_dir: Path) -> Path:
    return project_dir / "outputs" / "ai_input_modes"


def _feature_cache_path(project_dir: Path, mode: str) -> Path:
    return _feature_cache_dir(project_dir) / f"{mode}.json"


def _image_fingerprint(image_dir: Path) -> str:
    digest = hashlib.sha256()
    files = [p for p in Path(image_dir).glob("*") if p.is_file()]
    files.sort()
    for path in files:
        try:
            stat = path.stat()
        except Exception:
            continue
        rel_name = path.name.lower().encode("utf-8", errors="ignore")
        digest.update(rel_name)
        digest.update(str(int(stat.st_size)).encode("utf-8"))
        digest.update(str(int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000)))).encode("utf-8"))
    return digest.hexdigest()


def _load_feature_cache(project_dir: Path, mode: str, fingerprint: str) -> dict[str, Any] | None:
    cache_path = _feature_cache_path(project_dir, mode)
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if int(payload.get("version", 0) or 0) != CACHE_VERSION:
        return None
    if str(payload.get("fingerprint") or "") != fingerprint:
        return None
    return payload


def _save_feature_cache(project_dir: Path, mode: str, fingerprint: str, payload: dict[str, Any]) -> Path:
    cache_dir = _feature_cache_dir(project_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _feature_cache_path(project_dir, mode)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    tmp_payload = {
        "version": CACHE_VERSION,
        "mode": mode,
        "fingerprint": fingerprint,
        "features": dict(payload.get("features") or {}),
        "notes": list(payload.get("notes") or []),
        "heuristic_preset": str(payload.get("heuristic_preset") or "balanced"),
    }
    tmp_path.write_text(json.dumps(tmp_payload, indent=2), encoding="utf-8")
    tmp_path.replace(cache_path)
    return cache_path


def normalize_ai_input_mode(value: Any) -> str:
    mode = str(value or "").strip().lower()
    if mode in VALID_AI_INPUT_MODES:
        return mode
    return ""


def _count_supported_images(image_dir: Path) -> int:
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
    try:
        return sum(1 for p in Path(image_dir).glob("*") if p.is_file() and p.suffix.lower() in exts)
    except Exception:
        return 0


def _build_feature_log_details(mode: str, features: dict[str, Any], image_count: int) -> dict[str, Any]:
    details: dict[str, Any] = {
        "image_count": int(image_count),
    }

    exif_fields = [
        "camera_make",
        "camera_model",
        "focal_length_mm",
        "focal_missing",
        "aperture_f",
        "aperture_missing",
        "iso",
        "iso_missing",
        "gps_missing",
        "timestamp_mode",
        "timestamp_missing",
        "img_width_median",
        "img_height_median",
        "img_size_missing",
    ]
    for key in exif_fields:
        if key in features:
            details[key] = features.get(key)

    if mode in {"exif_plus_flight_plan", "exif_plus_flight_plan_plus_external"}:
        flight_fields = [
            "flight_type",
            "flight_type_missing",
            "heading_consistency",
            "heading_missing",
            "coverage_spread",
            "coverage_missing",
            "overlap_proxy",
            "overlap_missing",
            "camera_angle_profile",
            "angle_profile_missing",
        ]
        for key in flight_fields:
            if key in features:
                details[key] = features.get(key)

    if mode == "exif_plus_flight_plan_plus_external":
        external_fields = [
            "vegetation_cover_percentage",
            "green_area_missing",
            "terrain_roughness_proxy",
            "roughness_missing",
            "texture_density",
            "texture_missing",
            "blur_motion_risk",
            "blur_missing",
        ]
        for key in external_fields:
            if key in features:
                details[key] = features.get(key)

    return details


def _build_initial_params_log(params: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "tune_start_step",
        "tune_end_step",
        "tune_interval",
        "tune_min_improvement",
        "trend_scope",
        "feature_lr",
        "opacity_lr",
        "scaling_lr",
        "rotation_lr",
        "position_lr_init",
        "position_lr_final",
        "densification_interval",
        "densify_grad_threshold",
        "opacity_threshold",
        "lambda_dssim",
    ]
    details: dict[str, Any] = {}
    for key in keys:
        value = params.get(key)
        if value is not None:
            details[key] = value
    return details


def apply_initial_preset(
    params: dict[str, Any],
    *,
    image_dir: Path,
    colmap_dir: Path,
    logger,
) -> dict[str, Any]:
    """Apply optional initial parameter preset from selected AI input mode.

    This function is intentionally additive: it keeps legacy behavior when no
    mode is selected and only updates a small bounded set of initial params.
    """
    mode = normalize_ai_input_mode(params.get("ai_input_mode"))
    if not mode:
        return {
            "mode": "legacy",
            "applied": False,
            "updates": {},
            "features": {},
            "notes": ["No ai_input_mode selected; using existing behavior."],
            "cache_used": False,
        }

    ctx = ModeContext(image_dir=Path(image_dir), colmap_dir=Path(colmap_dir), params=params)
    project_dir = Path(image_dir).resolve().parent
    fingerprint = _image_fingerprint(Path(image_dir))
    cached = _load_feature_cache(project_dir, mode, fingerprint)
    cache_used = cached is not None

    if cached is not None:
        result_features = dict(cached.get("features") or {})
        result_notes = list(cached.get("notes") or [])
        heuristic_preset = str(cached.get("heuristic_preset") or "balanced")
    else:
        if mode == "exif_only":
            result = build_exif_only_preset(ctx)
        elif mode == "exif_plus_flight_plan":
            result = build_exif_plus_flight_plan_preset(ctx)
        else:
            result = build_exif_plus_flight_plan_plus_external_preset(ctx)

        result_features = dict(result.features)
        result_notes = list(result.notes)
        heuristic_preset = str(result.updates.get("preset_name") or "balanced")
        _save_feature_cache(
            project_dir,
            mode,
            fingerprint,
            {
                "features": result_features,
                "notes": result_notes,
                "heuristic_preset": heuristic_preset,
            },
        )

    selection = select_preset(
        project_dir=project_dir,
        mode=mode,
        heuristic_preset=heuristic_preset,
        params=params,
    )
    selected_updates = dict(selection.get("updates") or {})
    selected_preset = str(selection.get("selected_preset") or heuristic_preset)
    forced_preset = _normalize_preset_override(params.get("ai_preset_override"))
    preset_forced = bool(forced_preset)
    if preset_forced:
        # Optional override for controlled exploration experiments; default path remains adaptive.
        selected_preset = forced_preset
        selected_updates = apply_preset_updates(params, selected_preset)

    for key, value in selected_updates.items():
        if key == "preset_name":
            continue
        params[key] = value

    feature_details = _build_feature_log_details(mode, result_features, _count_supported_images(Path(image_dir)))
    logger.info(
        "AI_INPUT_MODE_FEATURES mode=%s details=%s",
        mode,
        json.dumps(feature_details, sort_keys=True),
    )
    logger.info(
        "AI_INPUT_MODE_PRESET mode=%s heuristic=%s selected=%s cache_used=%s forced=%s",
        mode,
        heuristic_preset,
        selected_preset,
        str(bool(cache_used)).lower(),
        str(preset_forced).lower(),
    )
    logger.info(
        "AI_INPUT_MODE_INITIAL_PARAMS mode=%s params=%s",
        mode,
        json.dumps(_build_initial_params_log(params), sort_keys=True),
    )

    logger.info(
        "AI input preset applied mode=%s selected_preset=%s cache_used=%s updates=%s features=%s",
        mode,
        selected_preset,
        cache_used,
        selected_updates,
        result_features,
    )

    return {
        "mode": mode,
        "applied": True,
        "updates": selected_updates,
        "features": result_features,
        "notes": result_notes,
        "heuristic_preset": heuristic_preset,
        "selected_preset": selected_preset,
        "preset_forced": preset_forced,
        "yhat_scores": dict(selection.get("yhat_scores") or {}),
        "project_dir": str(project_dir),
        "cache_used": cache_used,
        "cache_path": str(_feature_cache_path(project_dir, mode)),
    }
