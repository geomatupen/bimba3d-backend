#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create and run a REAL training pipeline with AI learning, model updates, and full monitoring.
This will populate the AI Learning Table and create models.
"""
import json
import sys
import time
from pathlib import Path
from datetime import datetime
import io
import threading

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent / "bimba3d_backend"))

from bimba3d_backend.app.services import training_pipeline_storage
from bimba3d_backend.app.services.training_pipeline_orchestrator import PipelineOrchestrator

def create_real_training_pipeline():
    """Create a real training pipeline with proper AI learning phases."""
    print("=" * 80)
    print("CREATING REAL TRAINING PIPELINE WITH AI LEARNING")
    print("=" * 80)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    pipeline_name = f"real_training_{timestamp}"

    # Real training configuration with AI learning
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
            "max_steps": 200,  # Minimal but enough for meaningful training
            "ai_input_mode": "exif_plus_flight_plan",  # AI-based selection
            "colmap": {
                "max_image_size": 600,  # Low resolution for speed
                "mapper_num_threads": 1,
                "camera_model": "OPENCV",
                "single_camera": True,
                "matching_type": "sequential"
            }
        },
        "phases": [
            {
                "phase_number": 1,
                "name": "Baseline Phase",
                "runs_per_project": 1,  # 1 baseline run
                "passes": 1,
                "preset_override": "baseline",  # This is the baseline
                "update_model": False,  # Don't update on baseline
                "context_jitter": False,
                "shuffle_order": False,
                "strategy_override": None
            },
            {
                "phase_number": 2,
                "name": "AI Learning Phase",
                "runs_per_project": 1,  # 1 AI run to test learning
                "passes": 1,
                "preset_override": None,  # Use AI selection
                "update_model": True,  # UPDATE MODEL - this creates learner models!
                "context_jitter": True,
                "context_jitter_mode": "temporal",
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
    print(f"  Max steps: 200 (minimal but meaningful)")
    print(f"  COLMAP resolution: 600px (low)")
    print(f"  Total phases: 2 (Baseline + AI Learning)")
    print(f"  Total runs: 2 (1 per phase)")
    print(f"  Storage: E:\\Thesis\\PipelineProjects\\{pipeline_name}")
    print(f"\n  Phase 1: Baseline - establishes reference performance")
    print(f"  Phase 2: AI Learning - uses AI selection, updates model, calculates reward")

    # Create pipeline
    print(f"\nCreating pipeline...")
    pipeline_data = training_pipeline_storage.create_pipeline(config)
    pipeline_id = pipeline_data['id']

    print(f"✓ Pipeline created successfully!")
    print(f"  ID: {pipeline_id}")
    print(f"  Folder: {pipeline_data['config']['pipeline_folder']}")

    return pipeline_id, pipeline_name


def monitor_pipeline_detailed(pipeline_id: str, check_interval: int = 5, max_duration: int = 1800):
    """Monitor pipeline with detailed progress reporting."""
    print(f"\n" + "=" * 80)
    print("MONITORING PIPELINE EXECUTION")
    print("=" * 80)
    print(f"Check interval: {check_interval}s")
    print(f"Max duration: {max_duration}s ({max_duration//60} minutes)")

    start_time = time.time()
    last_status = None
    last_progress = None
    last_phase = None
    last_completed = None

    while time.time() - start_time < max_duration:
        try:
            pipeline = training_pipeline_storage.get_pipeline(pipeline_id)
            if not pipeline:
                print(f"\n⚠ Pipeline not found!")
                time.sleep(check_interval)
                continue

            status = pipeline['status']
            completed = pipeline['completed_runs']
            total = pipeline['total_runs']
            failed = pipeline['failed_runs']
            phase = pipeline['current_phase']
            elapsed = int(time.time() - start_time)

            # Check for changes
            progress_changed = (completed != last_completed)
            phase_changed = (phase != last_phase)
            status_changed = (status != last_status)

            if progress_changed or phase_changed or status_changed:
                print(f"\n[{elapsed}s] Phase {phase}/2, Progress: {completed}/{total}, Status: {status}, Failed: {failed}")

                # Show phase details
                if phase_changed and pipeline.get('config', {}).get('phases'):
                    phases = pipeline['config']['phases']
                    if phase <= len(phases):
                        phase_info = phases[phase - 1]
                        print(f"       Current Phase: {phase_info.get('name', f'Phase {phase}')}")

                # Show completed run details
                if progress_changed and pipeline.get('runs'):
                    latest_run = pipeline['runs'][-1]
                    print(f"       Latest Run: {latest_run.get('run_name', 'N/A')}")
                    print(f"       Run Status: {latest_run.get('status', 'N/A')}")
                    if latest_run.get('reward') is not None:
                        print(f"       Reward: {latest_run['reward']:.4f}")
                    if latest_run.get('completed_at'):
                        print(f"       Completed: {latest_run['completed_at']}")

                # Show statistics
                if pipeline.get('mean_reward') is not None:
                    print(f"       Mean Reward: {pipeline['mean_reward']:.4f}")
                if pipeline.get('best_reward') is not None:
                    print(f"       Best Reward: {pipeline['best_reward']:.4f}")

                # Show errors
                if pipeline.get('last_error'):
                    print(f"       ⚠ Last Error: {pipeline['last_error'][:100]}...")

                last_status = status
                last_progress = f"{completed}/{total}"
                last_phase = phase
                last_completed = completed

            # Check if done
            if status in ['completed', 'failed', 'stopped']:
                print(f"\n{'='*80}")
                print(f"PIPELINE {status.upper()}")
                print(f"{'='*80}")
                print(f"Total time: {elapsed}s ({elapsed//60} minutes)")
                print(f"Completed runs: {completed}/{total}")
                print(f"Failed runs: {failed}")

                if pipeline.get('runs'):
                    print(f"\nAll Runs:")
                    for i, run in enumerate(pipeline['runs'], 1):
                        reward_str = f"reward={run['reward']:.4f}" if run.get('reward') is not None else "no reward"
                        print(f"  {i}. {run.get('run_name', 'N/A')}: {run['status']} ({reward_str})")

                if pipeline.get('mean_reward') is not None:
                    print(f"\nStatistics:")
                    print(f"  Mean Reward: {pipeline['mean_reward']:.4f}")
                    print(f"  Best Reward: {pipeline.get('best_reward', 'N/A')}")
                    print(f"  Success Rate: {pipeline.get('success_rate', 'N/A')}")

                return status == 'completed', pipeline

            time.sleep(check_interval)

        except KeyboardInterrupt:
            print(f"\n\n⚠ Monitoring interrupted by user")
            return False, None
        except Exception as e:
            print(f"\n⚠ Monitoring error: {e}")
            time.sleep(check_interval)

    print(f"\n⚠ Max monitoring duration reached")
    return False, training_pipeline_storage.get_pipeline(pipeline_id)


def check_outputs(pipeline_id: str, pipeline_name: str):
    """Check all outputs were created."""
    print(f"\n" + "=" * 80)
    print("CHECKING OUTPUTS AND FILES")
    print("=" * 80)

    pipeline_folder = Path(f"E:/Thesis/PipelineProjects/{pipeline_name}")

    checks = []

    # Check pipeline folder
    checks.append(("Pipeline folder", pipeline_folder, pipeline_folder.exists()))

    # Check shared models (created by update_model=True)
    shared_models = pipeline_folder / "shared_models"
    checks.append(("Shared models folder", shared_models, shared_models.exists()))

    if shared_models.exists():
        model_files = list(shared_models.glob("*.pth")) + list(shared_models.glob("*.pt"))
        checks.append(("Model files", shared_models, len(model_files) > 0))
        if model_files:
            print(f"  Found {len(model_files)} model files:")
            for mf in model_files[:3]:  # Show first 3
                print(f"    - {mf.name}")

    # Check project folder
    project_folder = pipeline_folder / "podoli"
    checks.append(("Project folder", project_folder, project_folder.exists()))

    if project_folder.exists():
        # Check key files
        checks.append(("Images folder", project_folder / "images", (project_folder / "images").exists()))
        checks.append(("Config file", project_folder / "config.json", (project_folder / "config.json").exists()))
        checks.append(("Processing log", project_folder / "processing.log", (project_folder / "processing.log").exists()))

        # Check COLMAP outputs
        colmap_db = project_folder / "outputs" / "database.db"
        checks.append(("COLMAP database", colmap_db, colmap_db.exists()))

        colmap_sparse = project_folder / "outputs" / "colmap" / "sparse" / "0"
        checks.append(("COLMAP sparse/0", colmap_sparse, colmap_sparse.exists()))

        # Check training outputs
        gsplat_dir = project_folder / "outputs" / "gsplat"
        checks.append(("Training outputs", gsplat_dir, gsplat_dir.exists()))

        # Check runs folder
        runs_folder = project_folder / "runs"
        checks.append(("Runs folder", runs_folder, runs_folder.exists()))

        if runs_folder.exists():
            run_dirs = [d for d in runs_folder.iterdir() if d.is_dir()]
            checks.append(("Run directories", runs_folder, len(run_dirs) >= 2))
            print(f"  Found {len(run_dirs)} run directories:")
            for rd in run_dirs:
                print(f"    - {rd.name}")

                # Check for learning results
                learning_results = rd / "outputs" / "engines" / "gsplat" / "input_mode_learning_results.json"
                if learning_results.exists():
                    print(f"      ✓ Learning results found")
                    try:
                        with open(learning_results) as f:
                            results = json.load(f)
                            if results.get('reward') is not None:
                                print(f"      Reward: {results['reward']:.4f}")
                    except:
                        pass

    print(f"\nSummary:")
    for name, path, exists in checks:
        status = "✓" if exists else "❌"
        print(f"  {status} {name}")

    return all(exists for _, _, exists in checks)


def check_ai_learning_table(pipeline_id: str):
    """Check AI learning table data."""
    print(f"\n" + "=" * 80)
    print("CHECKING AI LEARNING TABLE DATA")
    print("=" * 80)

    # This would normally be via API, but we'll check the files directly
    pipeline = training_pipeline_storage.get_pipeline(pipeline_id)

    if not pipeline or not pipeline.get('runs'):
        print("❌ No runs found in pipeline")
        return False

    print(f"Pipeline has {len(pipeline['runs'])} runs:")

    has_rewards = False
    for i, run in enumerate(pipeline['runs'], 1):
        print(f"\n  Run {i}: {run.get('run_name', 'N/A')}")
        print(f"    Status: {run['status']}")
        print(f"    Phase: {run.get('phase', 'N/A')}")
        print(f"    Pass: {run.get('pass', 'N/A')}")

        if run.get('reward') is not None:
            print(f"    Reward: {run['reward']:.4f} ✓")
            has_rewards = True
        else:
            print(f"    Reward: Not calculated (normal for baseline)")

        if run.get('completed_at'):
            print(f"    Completed: {run['completed_at']}")

    if has_rewards:
        print(f"\n✓ AI Learning Table will have reward data")
    else:
        print(f"\n⚠ No rewards calculated (may be all baseline runs)")

    return True


def run_real_training_test():
    """Run complete real training pipeline test."""
    print("\n" + "🎯" * 40)
    print("REAL TRAINING PIPELINE WITH AI LEARNING")
    print("🎯" * 40)

    try:
        # Step 1: Create pipeline
        pipeline_id, pipeline_name = create_real_training_pipeline()

        # Step 2: Start pipeline
        print(f"\n" + "=" * 80)
        print("STARTING PIPELINE")
        print("=" * 80)

        orchestrator = PipelineOrchestrator(pipeline_id)

        # Run in background thread
        def run_async():
            try:
                print(f"\nStarting pipeline execution in background...")
                orchestrator.start()
            except Exception as e:
                print(f"\n❌ Pipeline execution error: {e}")
                import traceback
                traceback.print_exc()

        pipeline_thread = threading.Thread(target=run_async, daemon=True)
        pipeline_thread.start()

        print(f"✓ Pipeline started")

        # Wait a moment for it to actually start
        time.sleep(3)

        # Step 3: Monitor execution
        success, final_pipeline = monitor_pipeline_detailed(pipeline_id, check_interval=10, max_duration=1800)

        # Step 4: Check outputs
        outputs_ok = check_outputs(pipeline_id, pipeline_name)

        # Step 5: Check AI learning table
        table_ok = check_ai_learning_table(pipeline_id)

        # Final summary
        print(f"\n" + "=" * 80)
        print("FINAL TEST SUMMARY")
        print("=" * 80)
        print(f"Pipeline ID: {pipeline_id}")
        print(f"Pipeline Name: {pipeline_name}")
        print(f"Pipeline Status: {final_pipeline['status'] if final_pipeline else 'Unknown'}")
        print(f"Execution: {'✓ SUCCESS' if success else '❌ FAILED'}")
        print(f"Outputs: {'✓ COMPLETE' if outputs_ok else '❌ INCOMPLETE'}")
        print(f"AI Learning Data: {'✓ PRESENT' if table_ok else '❌ MISSING'}")

        print(f"\n{'='*80}")
        if success and outputs_ok:
            print("✓✓✓ TRAINING PIPELINE COMPLETED SUCCESSFULLY ✓✓✓")
            print(f"\nYou can now:")
            print(f"1. Start backend: cd bimba3d_backend && uvicorn app.main:app")
            print(f"2. Open browser: http://localhost:8000/pipelines")
            print(f"3. View pipeline: {pipeline_name}")
            print(f"4. Check 'Logs' tab → 'AI Learning Table' for rewards")
            print(f"5. Check 'Models' tab for trained models")
        else:
            print("⚠ PIPELINE DID NOT COMPLETE SUCCESSFULLY")
            print("Check the logs above for errors")
        print(f"{'='*80}")

        return success

    except KeyboardInterrupt:
        print("\n\n⚠ Test interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    try:
        success = run_real_training_test()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ CRASHED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
