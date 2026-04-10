from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    tmp_path.replace(path)


def _read_text_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            return [line.rstrip("\n") for line in handle]
    except Exception:
        return []


def _parse_log_timestamp(line: str) -> datetime | None:
    prefix = line.split(" - ", 1)[0] if " - " in line else ""
    try:
        return datetime.strptime(prefix, "%Y-%m-%d %H:%M:%S,%f")
    except Exception:
        return None


def _parse_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    token = str(raw).strip().lower()
    if token in {"", "n/a", "na", "none", "null"}:
        return None
    try:
        return float(token)
    except Exception:
        return None


def _build_fallback_summary(project_id: str, run_id: str, run_dir: Path, engine: str = "gsplat") -> dict[str, Any]:
    run_config_path = run_dir / "run_config.json"
    run_config = _read_json(run_config_path) if run_config_path.exists() else {}
    if not isinstance(run_config, dict):
        run_config = {}

    resolved = run_config.get("resolved_params") if isinstance(run_config.get("resolved_params"), dict) else {}
    requested = run_config.get("requested_params") if isinstance(run_config.get("requested_params"), dict) else {}

    log_path = run_dir / "processing.log"
    lines = _read_text_lines(log_path)

    loss_points: list[dict[str, float | int]] = []
    time_points: list[dict[str, float | int]] = []
    preview_points: list[tuple[int, datetime]] = []
    first_ts: datetime | None = None
    last_ts: datetime | None = None
    best_splat_step: int | None = None
    best_splat_loss: float | None = None
    early_stop: dict[str, Any] = {
        "enabled": bool(requested.get("auto_early_stop", False)),
        "candidate": False,
        "candidate_since_step": None,
        "triggered": False,
        "trigger_step": None,
        "reason": None,
        "ema_loss": None,
        "monitor_points": [],
    }

    decision_re = re.compile(
        r"Core-AI adaptive decision step=(?P<step>\d+).*?loss=(?P<loss>[0-9.eE+-]+).*?prev_loss=(?P<prev_loss>[^\s]+).*?rel_improve=(?P<rel>[0-9.eE+-]+|n/a).*?"
    )
    best_re = re.compile(r"Updated best\.splat at step (?P<step>\d+) with loss (?P<loss>[0-9.eE+-]+)")
    preview_re = re.compile(r"Updated preview_latest\.png from eval step (?P<step>\d+)")
    early_stop_re = re.compile(r"Early stop triggered at step (?P<step>\d+)")

    for line in lines:
        ts = _parse_log_timestamp(line)
        if ts is not None:
            first_ts = ts if first_ts is None else min(first_ts, ts)
            last_ts = ts if last_ts is None else max(last_ts, ts)

        m = decision_re.search(line)
        if m:
            step = int(m.group("step"))
            loss = float(m.group("loss"))
            prev_loss = _parse_float(m.group("prev_loss"))
            rel = _parse_float(m.group("rel"))
            loss_points.append({"step": step, "loss": loss})
            if ts is not None:
                time_points.append({"step": step, "elapsed_seconds": 0.0})
            if prev_loss is not None and rel is None and prev_loss != 0:
                rel = (prev_loss - loss) / prev_loss
            continue

        m = best_re.search(line)
        if m:
            best_splat_step = int(m.group("step"))
            best_splat_loss = float(m.group("loss"))
            continue

        m = preview_re.search(line)
        if m and ts is not None:
            preview_points.append((int(m.group("step")), ts))
            continue

        m = early_stop_re.search(line)
        if m:
            early_stop["triggered"] = True
            early_stop["trigger_step"] = int(m.group("step"))
            early_stop["reason"] = "triggered_in_log"

    if preview_points:
        base_ts = preview_points[0][1]
        time_points = [
            {"step": step, "elapsed_seconds": round((ts - base_ts).total_seconds(), 6)}
            for step, ts in preview_points
        ]
    elif first_ts is not None and last_ts is not None and loss_points:
        base_ts = first_ts
        time_points = [{"step": point["step"], "elapsed_seconds": 0.0} for point in loss_points]

    stats_dir = run_dir / "outputs" / "engines" / engine / "stats"
    eval_psnr_series: list[dict[str, float | int]] = []
    eval_ssim_series: list[dict[str, float | int]] = []
    eval_lpips_series: list[dict[str, float | int]] = []
    num_gaussians = None
    last_eval_payload: dict[str, Any] = {}

    if stats_dir.exists():
        for stats_file in sorted(stats_dir.glob("val_step*.json")):
            match = re.match(r"val_step(?P<step>\d+)\.json$", stats_file.name)
            if not match:
                continue
            step = int(match.group("step")) + 1
            payload = _read_json(stats_file)
            if not isinstance(payload, dict):
                continue
            last_eval_payload = payload
            if isinstance(payload.get("psnr"), (int, float)):
                eval_psnr_series.append({"step": step, "value": float(payload["psnr"])})
            if isinstance(payload.get("ssim"), (int, float)):
                eval_ssim_series.append({"step": step, "value": float(payload["ssim"])})
            if isinstance(payload.get("lpips"), (int, float)):
                eval_lpips_series.append({"step": step, "value": float(payload["lpips"])})
            if isinstance(payload.get("num_GS"), (int, float)):
                num_gaussians = int(payload["num_GS"])

    final_loss = loss_points[-1]["loss"] if loss_points else None
    total_time_seconds = round((last_ts - first_ts).total_seconds(), 6) if first_ts is not None and last_ts is not None else None

    summary = {
        "project_id": project_id,
        "run_id": run_id,
        "run_name": run_config.get("run_name") or run_id,
        "name": project_id,
        "status": "completed",
        "mode": resolved.get("mode") or requested.get("mode"),
        "engine": engine,
        "metrics": {
            "convergence_speed": None,
            "final_loss": final_loss,
            "lpips_mean": last_eval_payload.get("lpips") if isinstance(last_eval_payload.get("lpips"), (int, float)) else None,
            "sharpness_mean": None,
            "num_gaussians": num_gaussians,
            "total_time_seconds": total_time_seconds,
            "best_splat_step": best_splat_step,
            "best_splat_loss": best_splat_loss,
            "stopped_early": bool(early_stop.get("triggered")),
            "early_stop_step": early_stop.get("trigger_step"),
        },
        "tuning": {
            "initial": {},
            "final": {},
            "end_params": {},
            "end_step": resolved.get("tune_end_step"),
            "runs": None,
            "history_count": len(loss_points),
            "history": [],
            "tune_interval": resolved.get("tune_interval"),
            "log_interval": resolved.get("log_interval"),
            "runtime_series": [],
        },
        "major_params": {
            "max_steps": resolved.get("max_steps"),
            "total_steps_completed": loss_points[-1]["step"] if loss_points else None,
            "densify_from_iter": resolved.get("densify_from_iter"),
            "densify_until_iter": resolved.get("densify_until_iter"),
            "densification_interval": resolved.get("densification_interval"),
            "eval_interval": resolved.get("eval_interval"),
            "save_interval": resolved.get("save_interval"),
            "splat_export_interval": resolved.get("splat_export_interval"),
            "best_splat_interval": resolved.get("best_splat_interval"),
            "auto_early_stop": resolved.get("auto_early_stop"),
            "early_stop_monitor_interval": resolved.get("early_stop_monitor_interval"),
            "early_stop_decision_points": resolved.get("early_stop_decision_points"),
            "early_stop_min_eval_points": resolved.get("early_stop_min_eval_points"),
            "early_stop_min_step_ratio": resolved.get("early_stop_min_step_ratio"),
            "early_stop_monitor_min_relative_improvement": resolved.get("early_stop_monitor_min_relative_improvement"),
            "early_stop_eval_min_relative_improvement": resolved.get("early_stop_eval_min_relative_improvement"),
            "early_stop_max_volatility_ratio": resolved.get("early_stop_max_volatility_ratio"),
            "early_stop_ema_alpha": resolved.get("early_stop_ema_alpha"),
            "batch_size": resolved.get("batch_size"),
        },
        "loss_milestones": {},
        "log_loss_series": loss_points,
        "log_time_series": time_points,
        "eval_series": loss_points,
        "eval_time_series": time_points,
        "eval_psnr_series": eval_psnr_series,
        "eval_ssim_series": eval_ssim_series,
        "eval_lpips_series": eval_lpips_series,
        "preview_url": None,
        "eval_points": len(eval_psnr_series) or len(loss_points),
        "early_stop": early_stop if early_stop else None,
    }

    metadata = {
        "evaluation_metrics": {
            "lpips_score": last_eval_payload.get("lpips") if isinstance(last_eval_payload.get("lpips"), (int, float)) else None,
            "sharpness": None,
            "convergence_speed": None,
            "final_loss": final_loss,
            "gaussian_count": num_gaussians,
        },
        "final_metrics": {
            "convergence_speed": None,
            "final_loss": final_loss,
            "lpips_mean": last_eval_payload.get("lpips") if isinstance(last_eval_payload.get("lpips"), (int, float)) else None,
            "sharpness_mean": None,
        },
        "num_gaussians": num_gaussians,
        "mode": summary["mode"],
        "tune_scope": resolved.get("tune_scope") or requested.get("tune_scope"),
        "best_splat": {
            "step": best_splat_step,
            "loss": best_splat_loss,
            "path": str(run_dir / "outputs" / "engines" / engine / "best.splat"),
        },
        "early_stop": {
            "enabled": bool(requested.get("auto_early_stop", False)),
            "candidate": False,
            "candidate_since_step": None,
            "triggered": bool(early_stop.get("triggered")),
            "trigger_step": early_stop.get("trigger_step"),
            "reason": early_stop.get("reason"),
            "ema_loss": None,
            "monitor_points": [],
        },
    }

    return {"summary": summary, "metadata": metadata}


def build_summary(project_id: str, run_id: str, run_dir: Path, engine: str = "gsplat") -> dict[str, Any]:
    engine_dir = run_dir / "outputs" / "engines" / engine
    eval_path = engine_dir / "eval_history.json"
    metadata_path = engine_dir / "metadata.json"
    tuning_path = engine_dir / "adaptive_tuning_results.json"
    run_config_path = run_dir / "run_config.json"
    processing_log_path = run_dir / "processing.log"

    eval_history_raw = _read_json(eval_path)
    if not isinstance(eval_history_raw, list):
        eval_history_raw = []
    eval_history = sorted(
        [item for item in eval_history_raw if isinstance(item, dict)],
        key=lambda item: item.get("step") if isinstance(item.get("step"), (int, float)) else float("inf"),
    )

    metadata = _read_json(metadata_path)
    if not isinstance(metadata, dict):
        metadata = {}

    run_config = _read_json(run_config_path)
    if not isinstance(run_config, dict):
        run_config = {}

    tuning_results: dict[str, Any] = {}
    if tuning_path.exists():
        parsed = _read_json(tuning_path)
        if isinstance(parsed, dict):
            tuning_results = parsed

    log_lines = _read_text_lines(processing_log_path)
    log_loss_series: list[dict[str, float | int]] = []
    log_time_series: list[dict[str, float | int]] = []
    first_log_ts: datetime | None = None
    log_loss_re = re.compile(r"step=(?P<step>\d+).*?loss=(?P<loss>[0-9.eE+-]+)")
    for line in log_lines:
        ts = None
        try:
            prefix = line.split(" - ", 1)[0] if " - " in line else ""
            ts = datetime.strptime(prefix, "%Y-%m-%d %H:%M:%S,%f")
        except Exception:
            ts = None
        if ts is not None and first_log_ts is None:
            first_log_ts = ts

        match = log_loss_re.search(line)
        if match:
            step = int(match.group("step"))
            loss = float(match.group("loss"))
            log_loss_series.append({"step": step, "loss": loss})
            if first_log_ts is not None and ts is not None:
                log_time_series.append({"step": step, "elapsed_seconds": round((ts - first_log_ts).total_seconds(), 6)})

    latest_eval = eval_history[-1] if eval_history else {}
    first_eval = eval_history[0] if eval_history else {}

    eval_series: list[dict[str, float | int]] = []
    eval_time_series: list[dict[str, float | int]] = []
    loss_milestones: dict[str, float] = {}

    for point in eval_history:
        step = point.get("step")
        if isinstance(step, (int, float)):
            loss_value = point.get("final_loss")
            if isinstance(loss_value, (int, float)):
                eval_series.append({"step": int(step), "loss": float(loss_value)})

            elapsed_seconds = point.get("elapsed_seconds")
            if isinstance(elapsed_seconds, (int, float)) and elapsed_seconds >= 0:
                eval_time_series.append({
                    "step": int(step),
                    "elapsed_seconds": float(elapsed_seconds),
                })

        for key, value in point.items():
            if isinstance(key, str) and key.startswith("loss_at_") and isinstance(value, (int, float)):
                loss_milestones[key] = float(value)

    total_time_seconds = None
    if eval_time_series:
        total_time_seconds = max(float(point["elapsed_seconds"]) for point in eval_time_series)

    resolved = run_config.get("resolved_params")
    if not isinstance(resolved, dict):
        resolved = {}

    tuning_history = tuning_results.get("tuning_history")
    if not isinstance(tuning_history, list):
        tuning_history = []

    runtime_series = [
        {"step": item.get("step"), "params": item.get("params")}
        for item in tuning_history
        if isinstance(item, dict) and isinstance(item.get("step"), (int, float)) and isinstance(item.get("params"), dict)
    ]

    final_loss = latest_eval.get("final_loss") if isinstance(latest_eval, dict) else None
    if not isinstance(final_loss, (int, float)):
        final_loss = None
    if final_loss is None and log_loss_series:
        final_loss = float(log_loss_series[-1]["loss"])

    best_splat = metadata.get("best_splat") if isinstance(metadata.get("best_splat"), dict) else {}
    best_splat_step = best_splat.get("step") if isinstance(best_splat.get("step"), (int, float)) else None
    best_splat_loss = best_splat.get("loss") if isinstance(best_splat.get("loss"), (int, float)) else None
    early_stop = metadata.get("early_stop") if isinstance(metadata.get("early_stop"), dict) else {}
    early_stop_triggered = bool(early_stop.get("triggered"))
    early_stop_step = early_stop.get("trigger_step") if isinstance(early_stop.get("trigger_step"), (int, float)) else None

    return {
        "project_id": project_id,
        "run_id": run_id,
        "run_name": run_config.get("run_name") or run_id,
        "name": project_id,
        "status": "completed",
        "mode": metadata.get("mode"),
        "engine": engine,
        "metrics": {
            "convergence_speed": latest_eval.get("convergence_speed") if isinstance(latest_eval, dict) else None,
            "final_loss": final_loss,
            "lpips_mean": latest_eval.get("lpips_mean") if isinstance(latest_eval, dict) else None,
            "sharpness_mean": latest_eval.get("sharpness_mean") if isinstance(latest_eval, dict) else None,
            "num_gaussians": latest_eval.get("num_gaussians") if isinstance(latest_eval, dict) else None,
            "total_time_seconds": total_time_seconds,
            "best_splat_step": int(best_splat_step) if isinstance(best_splat_step, (int, float)) else None,
            "best_splat_loss": float(best_splat_loss) if isinstance(best_splat_loss, (int, float)) else None,
            "stopped_early": early_stop_triggered,
            "early_stop_step": int(early_stop_step) if isinstance(early_stop_step, (int, float)) else None,
        },
        "tuning": {
            "initial": first_eval.get("tuning_params") if isinstance(first_eval, dict) and isinstance(first_eval.get("tuning_params"), dict) else {},
            "final": latest_eval.get("tuning_params") if isinstance(latest_eval, dict) and isinstance(latest_eval.get("tuning_params"), dict) else {},
            "end_params": tuning_results.get("final_params") if isinstance(tuning_results.get("final_params"), dict) else {},
            "end_step": resolved.get("tune_end_step") if resolved.get("tune_end_step") is not None else tuning_results.get("tune_end_step"),
            "runs": metadata.get("tuning_runs"),
            "history_count": len(tuning_history),
            "history": tuning_history,
            "tune_interval": resolved.get("tune_interval"),
            "log_interval": resolved.get("log_interval"),
            "runtime_series": runtime_series,
        },
        "major_params": {
            "max_steps": resolved.get("max_steps"),
            "total_steps_completed": latest_eval.get("step") if isinstance(latest_eval, dict) else None,
            "densify_from_iter": resolved.get("densify_from_iter"),
            "densify_until_iter": resolved.get("densify_until_iter"),
            "densification_interval": resolved.get("densification_interval"),
            "eval_interval": resolved.get("eval_interval"),
            "save_interval": resolved.get("save_interval"),
            "splat_export_interval": resolved.get("splat_export_interval"),
            "best_splat_interval": resolved.get("best_splat_interval"),
            "auto_early_stop": resolved.get("auto_early_stop"),
            "early_stop_monitor_interval": resolved.get("early_stop_monitor_interval"),
            "early_stop_decision_points": resolved.get("early_stop_decision_points"),
            "early_stop_min_eval_points": resolved.get("early_stop_min_eval_points"),
            "early_stop_min_step_ratio": resolved.get("early_stop_min_step_ratio"),
            "early_stop_monitor_min_relative_improvement": resolved.get("early_stop_monitor_min_relative_improvement"),
            "early_stop_eval_min_relative_improvement": resolved.get("early_stop_eval_min_relative_improvement"),
            "early_stop_max_volatility_ratio": resolved.get("early_stop_max_volatility_ratio"),
            "early_stop_ema_alpha": resolved.get("early_stop_ema_alpha"),
            "batch_size": resolved.get("batch_size"),
        },
        "loss_milestones": loss_milestones,
        "log_loss_series": log_loss_series or eval_series,
        "log_time_series": log_time_series or eval_time_series,
        "eval_series": eval_series,
        "eval_time_series": eval_time_series,
        "preview_url": None,
        "eval_points": len(eval_history),
        "early_stop": early_stop if isinstance(early_stop, dict) and early_stop else None,
    }


def backfill_runs(data_projects_dir: Path, project_filter: str | None = None, dry_run: bool = False) -> tuple[int, int]:
    updated = 0
    skipped = 0

    project_dirs = [p for p in data_projects_dir.iterdir() if p.is_dir()]
    for project_dir in sorted(project_dirs):
        project_id = project_dir.name
        if project_filter and project_filter != project_id:
            continue

        runs_dir = project_dir / "runs"
        if not runs_dir.exists():
            continue

        for run_dir in sorted([p for p in runs_dir.iterdir() if p.is_dir()]):
            run_id = run_dir.name
            engine_dir = run_dir / "outputs" / "engines" / "gsplat"
            comparison_dir = run_dir / "comparison"
            summary_path = comparison_dir / "experiment_summary.json"
            required = [
                run_dir / "run_config.json",
                engine_dir / "eval_history.json",
                engine_dir / "metadata.json",
            ]

            if not all(path.exists() for path in required):
                fallback = _build_fallback_summary(project_id, run_id, run_dir, engine="gsplat")
                if dry_run:
                    print(f"[DRY-RUN] would rebuild {summary_path} from logs/stats")
                    updated += 1
                    continue

                _write_json(summary_path, fallback["summary"])
                comparison_dir.mkdir(parents=True, exist_ok=True)
                _write_json(comparison_dir / "metadata.json", fallback["metadata"])
                _write_json(comparison_dir / "eval_history.json", {
                    "log_loss_series": fallback["summary"].get("log_loss_series", []),
                    "log_time_series": fallback["summary"].get("log_time_series", []),
                    "eval_series": fallback["summary"].get("eval_series", []),
                    "eval_time_series": fallback["summary"].get("eval_time_series", []),
                    "eval_psnr_series": fallback["summary"].get("eval_psnr_series", []),
                    "eval_ssim_series": fallback["summary"].get("eval_ssim_series", []),
                    "eval_lpips_series": fallback["summary"].get("eval_lpips_series", []),
                })
                run_config_path = run_dir / "run_config.json"
                if run_config_path.exists():
                    shutil.copy2(run_config_path, comparison_dir / "run_config.json")
                updated += 1
                continue

            summary_payload = build_summary(project_id, run_id, run_dir, engine="gsplat")
            if dry_run:
                print(f"[DRY-RUN] would write {summary_path}")
                updated += 1
                continue

            _write_json(summary_path, summary_payload)

            for name in ("eval_history.json", "adaptive_tuning_results.json", "metadata.json"):
                source = engine_dir / name
                if source.exists():
                    comparison_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, comparison_dir / name)

            run_cfg = run_dir / "run_config.json"
            if run_cfg.exists():
                comparison_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(run_cfg, comparison_dir / "run_config.json")

            updated += 1

    return updated, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill run comparison summaries for existing gsplat runs.")
    parser.add_argument("--project-id", help="Optional single project id to backfill.")
    parser.add_argument("--data-projects-dir", default=str(Path(__file__).resolve().parents[1] / "data" / "projects"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_projects_dir = Path(args.data_projects_dir)
    if not data_projects_dir.exists():
        raise SystemExit(f"Data projects dir not found: {data_projects_dir}")

    updated, skipped = backfill_runs(data_projects_dir, project_filter=args.project_id, dry_run=args.dry_run)
    print(f"Backfill complete. updated={updated} skipped={skipped} dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
