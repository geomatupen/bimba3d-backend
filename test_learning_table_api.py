#!/usr/bin/env python
"""Test the AI Learning Table API endpoint logic."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "bimba3d_backend"))

from bimba3d_backend.app.services import training_pipeline_storage
from bimba3d_backend.app.api.projects import _read_json_if_exists

# Test with learning_test_20260425_134003
pipeline_id = "pipeline_d3b870b17d0a"
pipeline = training_pipeline_storage.get_pipeline(pipeline_id)

print(f"Pipeline: {pipeline['name']}")
print(f"Status: {pipeline['status']}")
print(f"Runs: {pipeline['completed_runs']}/{pipeline['total_runs']}")
print()

config = pipeline.get("config", {})
pipeline_folder = Path(config.get("pipeline_folder"))

print(f"Pipeline folder: {pipeline_folder}")
print(f"Exists: {pipeline_folder.exists()}")
print()

learning_rows = []

# Iterate through project directories
for project_dir in pipeline_folder.iterdir():
    if not project_dir.is_dir():
        continue

    if project_dir.name in ["shared_models", "training_pipelines"]:
        continue

    runs_dir = project_dir / "runs"
    if not runs_dir.exists():
        continue

    print(f"Project: {project_dir.name}")

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        # Read run analytics (always available)
        analytics_file = run_dir / "analytics" / "run_analytics_v1.json"
        if not analytics_file.exists():
            print(f"  {run_dir.name}: NO ANALYTICS")
            continue

        analytics_data = _read_json_if_exists(analytics_file)
        if not analytics_data:
            print(f"  {run_dir.name}: INVALID ANALYTICS")
            continue

        # Extract summary data
        summary = analytics_data.get("summary", {})
        ai_insights = analytics_data.get("ai", {}).get("input_mode_insights", {})

        # Read learning results if available
        learning_file = run_dir / "outputs" / "engines" / "gsplat" / "input_mode_learning_results.json"
        learning_data = _read_json_if_exists(learning_file) if learning_file.exists() else {}

        is_baseline = summary.get("mode") == "baseline"

        row = {
            "run_name": summary.get("run_name") or run_dir.name,
            "mode": summary.get("mode"),
            "is_baseline": is_baseline,
            "baseline_session_id": learning_data.get("baseline_run_id") or ai_insights.get("baseline_session_id"),
            "selected_preset": ai_insights.get("selected_preset"),
            "final_psnr": summary.get("metrics", {}).get("convergence_speed"),
            "final_loss": summary.get("metrics", {}).get("final_loss"),
            "reward": learning_data.get("reward") or ai_insights.get("reward"),
            "has_learning_file": learning_file.exists(),
        }

        learning_rows.append(row)

        print(f"  {run_dir.name}:")
        print(f"    Mode: {row['mode']}")
        print(f"    Baseline: {row['is_baseline']}")
        print(f"    Baseline Session ID: {row['baseline_session_id'] or 'None'}")
        print(f"    Preset: {row['selected_preset']}")
        print(f"    PSNR: {row['final_psnr']:.2f}" if row['final_psnr'] else "    PSNR: None")
        print(f"    Loss: {row['final_loss']:.4f}" if row['final_loss'] else "    Loss: None")
        print(f"    Reward: {row['reward']}" if row['reward'] is not None else "    Reward: None")
        print(f"    Learning file: {'YES' if row['has_learning_file'] else 'NO'}")
        print()

print(f"\nTotal rows: {len(learning_rows)}")
print(f"\n{'='*80}")
print("AI LEARNING TABLE SHOULD SHOW:")
print(f"{'='*80}")
for i, row in enumerate(learning_rows, 1):
    psnr_str = f"{row['final_psnr']:.2f}" if row['final_psnr'] is not None else 'N/A'
    reward_str = str(row['reward']) if row['reward'] is not None else 'N/A'
    print(f"{i}. {row['run_name'][:50]}")
    print(f"   Mode: {row['mode']}, Preset: {row['selected_preset']}, PSNR: {psnr_str}, Reward: {reward_str}")
