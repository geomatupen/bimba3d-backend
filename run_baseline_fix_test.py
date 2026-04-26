#!/usr/bin/env python
"""Test that baseline_run_id is correctly passed to phase 2."""
import json
import sys
import time
from pathlib import Path
from datetime import datetime
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent / "bimba3d_backend"))

from bimba3d_backend.app.services import training_pipeline_storage
from bimba3d_backend.app.services.training_pipeline_orchestrator import PipelineOrchestrator

print("=" * 80)
print("BASELINE FIX TEST")
print("=" * 80)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
pipeline_name = f"baseline_fix_{timestamp}"

config = {
    "name": pipeline_name,
    "pipeline_directory": "E:\\Thesis\\PipelineProjects",
    "projects": [
        {
            "name": "podoli",
            "dataset_path": "E:\\Thesis\\Training Data\\podoli",
            "image_count": 138,
        }
    ],
    "shared_config": {
        "engine": "gsplat",
        "max_steps": 100,  # Very few steps
        "ai_input_mode": "exif_plus_flight_plan",
        "colmap": {
            "max_image_size": 600,
            "mapper_num_threads": 1,
            "camera_model": "OPENCV",
            "single_camera": True,
            "matching_type": "sequential"
        }
    },
    "phases": [
        {
            "phase_number": 1,
            "name": "Baseline",
            "runs_per_project": 1,
            "passes": 1,
            "preset_override": "baseline",
            "update_model": False,
        },
        {
            "phase_number": 2,
            "name": "AI",
            "runs_per_project": 1,
            "passes": 1,
            "update_model": True,
        }
    ],
    "thermal_management": {"enabled": False}
}

print(f"\nCreating: {pipeline_name}")
pipeline_data = training_pipeline_storage.create_pipeline(config)
pipeline_id = pipeline_data['id']
print(f"✓ ID: {pipeline_id}")

print(f"\nStarting...")
orchestrator = PipelineOrchestrator(pipeline_id)

import threading
thread = threading.Thread(target=orchestrator.start, daemon=True)
thread.start()
time.sleep(2)

print(f"✓ Running")
print(f"\nMonitoring...")

start = time.time()
last_completed = 0
while time.time() - start < 900:  # 15 min max
    pipeline = training_pipeline_storage.get_pipeline(pipeline_id)
    if not pipeline:
        break

    status = pipeline['status']
    completed = pipeline['completed_runs']

    if completed > last_completed:
        print(f"[{int(time.time()-start)}s] {completed}/2 completed")

        # Check config after phase 1
        if completed == 1:
            project = pipeline['config']['projects'][0]
            baseline_id = project.get('baseline_run_id')
            if baseline_id:
                print(f"  ✓ baseline_run_id stored: {baseline_id}")
            else:
                print(f"  ❌ baseline_run_id NOT stored")

        last_completed = completed

    if status in ['completed', 'failed', 'stopped']:
        break

    time.sleep(10)

# Final check
pipeline = training_pipeline_storage.get_pipeline(pipeline_id)
print(f"\n{'='*80}")
print(f"Status: {pipeline['status']}")
print(f"Runs: {pipeline['completed_runs']}/2")

# Check learning results
folder = Path(f"E:/Thesis/PipelineProjects/{pipeline_name}/podoli/runs")
phase2_dirs = list(folder.glob("*phase2*"))

if phase2_dirs:
    phase2_run = phase2_dirs[0]

    # Check analytics
    analytics_file = phase2_run / "analytics" / "run_analytics_v1.json"
    if analytics_file.exists():
        with open(analytics_file) as f:
            analytics = json.load(f)
        baseline_id = analytics['ai']['input_mode_insights'].get('baseline_session_id')
        print(f"\nPhase 2 baseline_session_id: {baseline_id or 'NULL'}")

        if baseline_id:
            print(f"✓✓✓ SUCCESS: Baseline ID passed to phase 2!")
        else:
            print(f"❌ FAILED: Baseline ID still null in phase 2")

    # Check learning results file
    learning_file = phase2_run / "outputs" / "engines" / "gsplat" / "input_mode_learning_results.json"
    if learning_file.exists():
        with open(learning_file) as f:
            results = json.load(f)
        reward = results.get('reward')
        print(f"✓✓✓ Learning results file exists!")
        print(f"Reward: {reward}")
    else:
        print(f"❌ Learning results file not found")
else:
    print(f"❌ No phase 2 run found")

print(f"{'='*80}")
