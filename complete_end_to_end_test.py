#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Complete end-to-end test with backend server, API monitoring, and pipeline execution.
"""
import json
import sys
import time
import subprocess
import requests
from pathlib import Path
from datetime import datetime
import io
import signal

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent / "bimba3d_backend"))

BASE_URL = "http://localhost:8000"
backend_process = None

def start_backend():
    """Start backend server."""
    global backend_process
    print("\n" + "=" * 80)
    print("STARTING BACKEND SERVER")
    print("=" * 80)

    # Start uvicorn in background
    backend_process = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "bimba3d_backend.app.main:app",
            "--host", "0.0.0.0",
            "--port", "8000"
        ],
        cwd=str(Path(__file__).parent),
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parent), "WORKER_MODE": "local"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # Wait for server to start
    print("Waiting for backend to start...")
    for i in range(30):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                print(f"✓ Backend started successfully on {BASE_URL}")
                return True
        except:
            time.sleep(1)
            print(f"  Attempt {i+1}/30...")

    print("❌ Backend failed to start")
    return False

def stop_backend():
    """Stop backend server."""
    global backend_process
    if backend_process:
        print("\n✓ Stopping backend server...")
        backend_process.terminate()
        try:
            backend_process.wait(timeout=5)
        except:
            backend_process.kill()

def create_test_pipeline():
    """Create a minimal test pipeline via API."""
    print("\n" + "=" * 80)
    print("CREATING TEST PIPELINE VIA API")
    print("=" * 80)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    pipeline_name = f"e2e_test_{timestamp}"

    config = {
        "name": pipeline_name,
        "pipeline_directory": "E:\\Thesis\\PipelineProjects",  # No spaces!
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
            "max_steps": 50,  # Ultra minimal for quick test
            "ai_input_mode": "exif_plus_flight_plan",
            "colmap": {
                "max_image_size": 400,  # Very small for speed
                "mapper_num_threads": 1,
                "camera_model": "OPENCV",
                "single_camera": True,
                "matching_type": "sequential"
            }
        },
        "phases": [
            {
                "phase_number": 1,
                "name": "Quick Test",
                "runs_per_project": 1,
                "passes": 1,
                "preset_override": "baseline",
                "update_model": False,
                "context_jitter": False,
                "shuffle_order": False,
                "strategy_override": None
            }
        ],
        "thermal_management": {
            "enabled": False
        }
    }

    print(f"Pipeline configuration:")
    print(f"  Name: {pipeline_name}")
    print(f"  Max steps: 50 (ultra minimal)")
    print(f"  COLMAP resolution: 400px (very small)")
    print(f"  Total runs: 1")

    try:
        response = requests.post(f"{BASE_URL}/api/training-pipeline", json=config, timeout=30)
        if response.status_code == 200:
            pipeline = response.json()
            print(f"\n✓ Pipeline created successfully!")
            print(f"  ID: {pipeline['id']}")
            print(f"  Name: {pipeline['name']}")
            print(f"  Status: {pipeline['status']}")
            return pipeline['id']
        else:
            print(f"\n❌ Failed to create pipeline: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
    except Exception as e:
        print(f"\n❌ Error creating pipeline: {e}")
        return None

def start_pipeline(pipeline_id):
    """Start pipeline execution."""
    print("\n" + "=" * 80)
    print("STARTING PIPELINE EXECUTION")
    print("=" * 80)

    try:
        response = requests.post(f"{BASE_URL}/api/training-pipeline/{pipeline_id}/start", timeout=10)
        if response.status_code == 200:
            print(f"✓ Pipeline started successfully")
            return True
        else:
            print(f"❌ Failed to start pipeline: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error starting pipeline: {e}")
        return False

def monitor_pipeline(pipeline_id, max_duration=600):
    """Monitor pipeline execution with detailed progress."""
    print("\n" + "=" * 80)
    print("MONITORING PIPELINE EXECUTION")
    print("=" * 80)
    print(f"Max monitoring time: {max_duration} seconds")
    print(f"Check interval: 5 seconds\n")

    start_time = time.time()
    last_status = None
    last_progress = None
    last_message = None

    while time.time() - start_time < max_duration:
        try:
            # Get pipeline status
            response = requests.get(f"{BASE_URL}/api/training-pipeline/{pipeline_id}", timeout=5)
            if response.status_code != 200:
                print(f"⚠ API error: {response.status_code}")
                time.sleep(5)
                continue

            pipeline = response.json()
            status = pipeline['status']
            completed = pipeline['completed_runs']
            total = pipeline['total_runs']
            failed = pipeline['failed_runs']
            elapsed = int(time.time() - start_time)

            # Print updates when something changes
            progress_str = f"{completed}/{total}"
            if status != last_status or progress_str != last_progress:
                print(f"\n[{elapsed}s] Status: {status}, Progress: {progress_str}, Failed: {failed}")
                last_status = status
                last_progress = progress_str

                # Show last error if any
                if pipeline.get('last_error'):
                    print(f"       Error: {pipeline['last_error'][:100]}...")

            # Check if completed or failed
            if status in ['completed', 'failed', 'stopped']:
                print(f"\n{'='*80}")
                print(f"PIPELINE {status.upper()}")
                print(f"{'='*80}")
                print(f"  Total time: {elapsed}s")
                print(f"  Completed runs: {completed}/{total}")
                print(f"  Failed runs: {failed}")

                if pipeline.get('mean_reward') is not None:
                    print(f"  Mean reward: {pipeline['mean_reward']:.4f}")
                if pipeline.get('best_reward') is not None:
                    print(f"  Best reward: {pipeline['best_reward']:.4f}")

                return status == 'completed'

            time.sleep(5)

        except KeyboardInterrupt:
            print("\n\n⚠ Monitoring interrupted by user")
            return False
        except Exception as e:
            print(f"\n⚠ Monitoring error: {e}")
            time.sleep(5)

    print(f"\n⚠ Max monitoring duration ({max_duration}s) reached")
    print("   Pipeline may still be running in the background")
    return False

def check_filesystem(pipeline_id):
    """Check filesystem to verify files were created."""
    print("\n" + "=" * 80)
    print("CHECKING FILESYSTEM")
    print("=" * 80)

    # Find pipeline folder
    pipeline_root = Path("E:/Thesis/PipelineProjects")
    pipeline_folders = list(pipeline_root.glob("e2e_test_*"))

    if not pipeline_folders:
        print("❌ No pipeline folders found")
        return False

    pipeline_folder = pipeline_folders[-1]  # Latest
    print(f"✓ Pipeline folder: {pipeline_folder}")

    # Check project folder
    project_folder = pipeline_folder / "podoli"
    if not project_folder.exists():
        print(f"❌ Project folder not found: {project_folder}")
        return False
    print(f"✓ Project folder: {project_folder}")

    # Check key directories and files
    checks = {
        "Images copied": project_folder / "images",
        "Config file": project_folder / "config.json",
        "Processing log": project_folder / "processing.log",
        "COLMAP database": project_folder / "outputs" / "database.db",
        "COLMAP sparse": project_folder / "outputs" / "colmap" / "sparse",
        "Training outputs": project_folder / "outputs" / "gsplat",
        "Runs folder": project_folder / "runs",
    }

    all_good = True
    for name, path in checks.items():
        if path.exists():
            if path.is_dir():
                count = len(list(path.iterdir()))
                print(f"  ✓ {name}: {path} ({count} items)")
            else:
                size = path.stat().st_size
                print(f"  ✓ {name}: {path} ({size} bytes)")
        else:
            print(f"  ❌ {name}: NOT FOUND")
            all_good = False

    return all_good

def check_ai_learning_table(pipeline_id):
    """Check if AI learning table has data."""
    print("\n" + "=" * 80)
    print("CHECKING AI LEARNING TABLE")
    print("=" * 80)

    try:
        response = requests.get(f"{BASE_URL}/api/training-pipeline/{pipeline_id}/learning-table", timeout=10)
        if response.status_code == 200:
            data = response.json()
            rows = data.get('rows', [])
            print(f"✓ Learning table endpoint works")
            print(f"  Rows returned: {len(rows)}")

            if rows:
                print(f"\n  Sample row:")
                row = rows[0]
                print(f"    Project: {row.get('project_name')}")
                print(f"    Run: {row.get('run_name')}")
                print(f"    Reward: {row.get('reward')}")
                print(f"    PSNR: {row.get('final_psnr')}")
                return True
            else:
                print(f"  ⚠ No rows in learning table")
                return False
        else:
            print(f"❌ Learning table API error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error checking learning table: {e}")
        return False

import os

def run_complete_test():
    """Run complete end-to-end test."""
    print("\n" + "🚀" * 40)
    print("COMPLETE END-TO-END PIPELINE TEST")
    print("🚀" * 40)

    try:
        # Step 1: Start backend
        if not start_backend():
            return False

        # Step 2: Create pipeline
        pipeline_id = create_test_pipeline()
        if not pipeline_id:
            return False

        # Step 3: Start pipeline
        if not start_pipeline(pipeline_id):
            return False

        # Step 4: Monitor execution
        success = monitor_pipeline(pipeline_id, max_duration=600)  # 10 minutes max

        # Step 5: Check filesystem
        fs_ok = check_filesystem(pipeline_id)

        # Step 6: Check AI learning table
        table_ok = check_ai_learning_table(pipeline_id)

        # Final summary
        print("\n" + "=" * 80)
        print("FINAL TEST SUMMARY")
        print("=" * 80)
        print(f"Pipeline ID: {pipeline_id}")
        print(f"Pipeline execution: {'✓ SUCCESS' if success else '❌ FAILED'}")
        print(f"Filesystem check: {'✓ PASS' if fs_ok else '❌ FAIL'}")
        print(f"AI learning table: {'✓ PASS' if table_ok else '❌ FAIL'}")
        print(f"\nOverall: {'✓✓✓ ALL TESTS PASSED ✓✓✓' if (success and fs_ok) else '⚠ SOME TESTS FAILED'}")

        print(f"\nTo view in frontend:")
        print(f"  1. Open browser: http://localhost:8000/pipelines")
        print(f"  2. Find pipeline: {pipeline_id}")
        print(f"  3. Check live progress in Overview tab")

        return success and fs_ok

    except KeyboardInterrupt:
        print("\n\n⚠ Test interrupted by user")
        return False
    finally:
        stop_backend()

if __name__ == "__main__":
    try:
        success = run_complete_test()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ TEST CRASHED: {e}")
        import traceback
        traceback.print_exc()
        stop_backend()
        sys.exit(1)
