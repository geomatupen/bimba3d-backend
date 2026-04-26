#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create and run a minimal pipeline for testing with small resolution and few steps.
"""
import json
import sys
import time
from pathlib import Path
from datetime import datetime
import io
import threading

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent / "bimba3d_backend"))

from bimba3d_backend.app.services import training_pipeline_storage
from bimba3d_backend.app.services.training_pipeline_orchestrator import PipelineOrchestrator

def create_minimal_pipeline():
    """Create a minimal test pipeline with small settings."""
    print("=" * 80)
    print("CREATING MINIMAL TEST PIPELINE")
    print("=" * 80)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    pipeline_name = f"minimal_test_{timestamp}"

    # Minimal configuration
    config = {
        "name": pipeline_name,
        "pipeline_directory": "E:\\Thesis\\PipelineProjects",  # No spaces in path!
        "projects": [
            {
                "name": "podoli",
                "dataset_path": "E:\\Thesis\\Training Data\\podoli",
                "image_count": 138,
                "colmap_source_project_id": None  # Will run COLMAP
            }
        ],
        "shared_config": {
            "engine": "gsplat",
            "max_steps": 100,  # Very few steps for quick test
            "ai_input_mode": "exif_plus_flight_plan",
            "colmap": {
                "max_image_size": 800,  # Small resolution
                "mapper_num_threads": 1,
                "camera_model": "OPENCV",
                "single_camera": True,
                "matching_type": "sequential"  # Faster than exhaustive
            }
        },
        "phases": [
            {
                "phase_number": 1,
                "name": "Quick Test",
                "runs_per_project": 1,  # Just 1 run
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

    print(f"\nConfiguration:")
    print(f"  Name: {pipeline_name}")
    print(f"  Dataset: podoli (138 images)")
    print(f"  Max steps: 100 (minimal)")
    print(f"  COLMAP resolution: 800px (small)")
    print(f"  Total runs: 1")
    print(f"  Storage: E:\\Thesis\\Pipeline Projects\\{pipeline_name}")

    # Create pipeline
    print(f"\nCreating pipeline...")
    pipeline_data = training_pipeline_storage.create_pipeline(config)
    pipeline_id = pipeline_data['id']

    print(f"✓ Pipeline created successfully!")
    print(f"  ID: {pipeline_id}")
    print(f"  Folder: {pipeline_data['config']['pipeline_folder']}")

    return pipeline_id, pipeline_name


def monitor_pipeline(pipeline_id: str, check_interval: int = 5, max_duration: int = 300):
    """Monitor pipeline execution with live updates."""
    print(f"\n" + "=" * 80)
    print("MONITORING PIPELINE EXECUTION")
    print("=" * 80)

    start_time = time.time()
    last_status = None
    last_progress = None

    while time.time() - start_time < max_duration:
        try:
            pipeline = training_pipeline_storage.get_pipeline(pipeline_id)

            status = pipeline['status']
            progress = f"{pipeline['completed_runs']}/{pipeline['total_runs']}"
            phase = pipeline['current_phase']
            elapsed = int(time.time() - start_time)

            # Only print if something changed
            if status != last_status or progress != last_progress:
                print(f"\n[{elapsed}s] Status: {status}, Progress: {progress}, Phase: {phase}")

                if pipeline.get('last_error'):
                    print(f"      Error: {pipeline['last_error']}")

                last_status = status
                last_progress = progress

            # Check if completed or failed
            if status in ['completed', 'failed', 'stopped']:
                print(f"\n✓ Pipeline {status}")
                print(f"  Completed runs: {pipeline['completed_runs']}")
                print(f"  Failed runs: {pipeline['failed_runs']}")
                print(f"  Total time: {elapsed}s")
                return True

            time.sleep(check_interval)

        except KeyboardInterrupt:
            print(f"\n\n⚠ Monitoring interrupted by user")
            return False
        except Exception as e:
            print(f"\n⚠ Monitoring error: {e}")
            time.sleep(check_interval)

    print(f"\n⚠ Max monitoring duration ({max_duration}s) reached")
    return False


def run_pipeline_test():
    """Complete pipeline test: create, start, monitor."""

    # Step 1: Create pipeline
    try:
        pipeline_id, pipeline_name = create_minimal_pipeline()
    except Exception as e:
        print(f"\n❌ Failed to create pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 2: Start pipeline
    print(f"\n" + "=" * 80)
    print("STARTING PIPELINE")
    print("=" * 80)

    try:
        orchestrator = PipelineOrchestrator(pipeline_id)

        # Run in background thread
        def run_async():
            try:
                print(f"\nStarting pipeline execution...")
                orchestrator.start()
            except Exception as e:
                print(f"\n❌ Pipeline execution error: {e}")
                import traceback
                traceback.print_exc()

        pipeline_thread = threading.Thread(target=run_async, daemon=True)
        pipeline_thread.start()

        print(f"✓ Pipeline started in background")

        # Wait a moment for it to actually start
        time.sleep(3)

    except Exception as e:
        print(f"\n❌ Failed to start pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 3: Monitor execution
    success = monitor_pipeline(pipeline_id, check_interval=5, max_duration=600)  # 10 min max

    # Step 4: Final summary
    print(f"\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    final_pipeline = training_pipeline_storage.get_pipeline(pipeline_id)

    print(f"\nPipeline: {final_pipeline['name']}")
    print(f"Status: {final_pipeline['status']}")
    print(f"Progress: {final_pipeline['completed_runs']}/{final_pipeline['total_runs']}")

    if final_pipeline.get('runs'):
        print(f"\nRuns completed:")
        for run in final_pipeline['runs']:
            print(f"  - {run.get('run_name', run.get('run_id'))}: {run.get('status')}")
            if run.get('reward') is not None:
                print(f"    Reward: {run['reward']:.4f}")

    # Check pipeline folder
    pipeline_folder = Path(final_pipeline['config']['pipeline_folder'])
    if pipeline_folder.exists():
        print(f"\n✓ Pipeline folder exists: {pipeline_folder}")

        # Check project folder
        project_name = final_pipeline['config']['projects'][0]['name']
        project_folder = pipeline_folder / project_name
        if project_folder.exists():
            print(f"✓ Project folder exists: {project_folder}")

            # Check key directories
            dirs_to_check = [
                "images",
                "outputs/colmap/sparse",
                "outputs/gsplat"
            ]
            for dir_path in dirs_to_check:
                full_path = project_folder / dir_path
                if full_path.exists():
                    print(f"  ✓ {dir_path} exists")
                else:
                    print(f"  ⚠ {dir_path} not found")

    print(f"\nTest {'PASSED' if success else 'INCOMPLETE'}")

    return success


if __name__ == "__main__":
    try:
        print("\n🚀 Starting minimal pipeline test...")
        print("   This will create and run a small test pipeline:")
        print("   - 1 project (podoli, 138 images)")
        print("   - COLMAP with 800px resolution")
        print("   - Training with only 100 steps")
        print("   - 1 run total")
        print("\n   Press Ctrl+C to stop at any time\n")

        success = run_pipeline_test()
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n\n⚠ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
