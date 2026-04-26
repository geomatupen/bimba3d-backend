#!/usr/bin/env python
"""Quick test to verify AI learning results are generated."""
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
print("QUICK AI LEARNING TEST")
print("=" * 80)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
pipeline_name = f"learning_test_{timestamp}"

config = {
    "name": pipeline_name,
    "pipeline_directory": "E:\\Thesis\\PipelineProjects",
    "projects": [
        {
            "name": "podoli",
            "dataset_path": "E:\\Thesis\\Training Data\\podoli",
            "image_count": 138,
            "colmap_source_project_id": None
        }
    ],
    "shared_config": {
        "engine": "gsplat",
        "max_steps": 150,  # Even fewer steps for faster test
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
            "context_jitter": False,
            "shuffle_order": False
        },
        {
            "phase_number": 2,
            "name": "AI Learning",
            "runs_per_project": 1,
            "passes": 1,
            "preset_override": None,  # Use AI
            "update_model": True,  # Update model
            "context_jitter": True,
            "context_jitter_mode": "temporal",
            "shuffle_order": False
        }
    ],
    "thermal_management": {
        "enabled": False
    }
}

print(f"\nCreating pipeline: {pipeline_name}")
pipeline_data = training_pipeline_storage.create_pipeline(config)
pipeline_id = pipeline_data['id']
print(f"✓ Pipeline ID: {pipeline_id}")

print(f"\nStarting pipeline...")
orchestrator = PipelineOrchestrator(pipeline_id)

import threading
def run_async():
    try:
        orchestrator.start()
    except Exception as e:
        print(f"\n❌ Error: {e}")

thread = threading.Thread(target=run_async, daemon=True)
thread.start()
time.sleep(2)

print(f"✓ Running in background")
print(f"\nMonitoring (checking every 10s)...")

start = time.time()
last_completed = 0
while time.time() - start < 1200:  # 20 minutes max
    pipeline = training_pipeline_storage.get_pipeline(pipeline_id)
    if not pipeline:
        break

    status = pipeline['status']
    completed = pipeline['completed_runs']

    if completed > last_completed:
        print(f"[{int(time.time()-start)}s] Progress: {completed}/2, Status: {status}")
        last_completed = completed

    if status in ['completed', 'failed', 'stopped']:
        print(f"\n✓ Pipeline {status}: {completed}/2 runs completed")
        break

    time.sleep(10)

# Check for learning results
pipeline_folder = Path(f"E:/Thesis/PipelineProjects/{pipeline_name}")
run_dirs = list((pipeline_folder / "podoli" / "runs").glob("*phase2*"))

if run_dirs:
    phase2_run = run_dirs[0]
    learning_results = phase2_run / "outputs" / "engines" / "gsplat" / "input_mode_learning_results.json"

    if learning_results.exists():
        print(f"\n✓✓✓ SUCCESS: Learning results file exists!")
        with open(learning_results) as f:
            data = json.load(f)
        reward = data.get('reward')
        print(f"Reward: {reward}")
        print(f"\nFile: {learning_results}")
    else:
        print(f"\n❌ FAILED: Learning results file not found")
        print(f"Expected: {learning_results}")
else:
    print(f"\n❌ No phase 2 run directories found")

print(f"\nPipeline folder: {pipeline_folder}")
