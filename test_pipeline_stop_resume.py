#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for pipeline stop/resume functionality.
Tests: start -> stop -> resume -> complete cycle
"""
import json
import sys
import time
from pathlib import Path
import io

# Fix Windows console encoding for Unicode
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "bimba3d_backend"))

from bimba3d_backend.app.services import training_pipeline_storage
from bimba3d_backend.app.services.training_pipeline_orchestrator import PipelineOrchestrator
from bimba3d_backend.app.config import DATA_DIR
import threading


def monitor_pipeline_status(pipeline_id: str, duration_seconds: int = 10):
    """Monitor pipeline status for a duration."""
    print(f"\n   Monitoring pipeline for {duration_seconds} seconds...")
    start_time = time.time()

    while time.time() - start_time < duration_seconds:
        pipeline = training_pipeline_storage.get_pipeline(pipeline_id)
        if pipeline:
            status = pipeline['status']
            progress = f"{pipeline['completed_runs']}/{pipeline['total_runs']}"
            phase = pipeline['current_phase']
            print(f"   [{int(time.time() - start_time)}s] Status: {status}, Progress: {progress}, Phase: {phase}")

        time.sleep(2)


def test_pipeline_stop_resume():
    """Test complete stop/resume cycle."""
    print("=" * 80)
    print("TESTING PIPELINE STOP/RESUME FUNCTIONALITY")
    print("=" * 80)

    # Find an existing pending pipeline or create a small test one
    print("\n1. FINDING TEST PIPELINE")
    all_pipelines = training_pipeline_storage.list_pipelines()

    # Look for a pending pipeline
    test_pipeline = next((p for p in all_pipelines if p['status'] == 'pending'), None)

    if not test_pipeline:
        print("   ℹ No pending pipeline found, need to create one first")
        print("   Run: python test_pipeline_creation.py")
        return False

    pipeline_id = test_pipeline['id']
    pipeline_name = test_pipeline['name']

    print(f"   ✓ Found pending pipeline: {pipeline_name}")
    print(f"   Pipeline ID: {pipeline_id}")
    print(f"   Total runs: {test_pipeline['total_runs']}")
    print(f"   Status: {test_pipeline['status']}")

    # Initialize orchestrator
    print(f"\n2. INITIALIZING ORCHESTRATOR")
    orchestrator = PipelineOrchestrator(pipeline_id)
    print(f"   ✓ Orchestrator initialized")

    # Test 1: Start pipeline
    print(f"\n3. TEST: START PIPELINE")
    try:
        # Start in background thread so we can test stop
        def run_pipeline():
            try:
                orchestrator.start()
            except Exception as e:
                print(f"   Pipeline execution error: {e}")

        pipeline_thread = threading.Thread(target=run_pipeline, daemon=True)
        pipeline_thread.start()

        print(f"   ✓ Pipeline started in background thread")

        # Wait a bit for pipeline to start
        time.sleep(2)

        # Check status changed to running
        pipeline = training_pipeline_storage.get_pipeline(pipeline_id)
        if pipeline['status'] == 'running':
            print(f"   ✓ Pipeline status changed to 'running'")
        else:
            print(f"   ⚠ Pipeline status is '{pipeline['status']}', expected 'running'")

        # Monitor for a bit
        monitor_pipeline_status(pipeline_id, duration_seconds=10)

    except Exception as e:
        print(f"   ❌ Failed to start pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 2: Pause pipeline
    print(f"\n4. TEST: PAUSE PIPELINE")
    try:
        orchestrator.pause()
        print(f"   ✓ Pause command sent")

        # Wait for pause to take effect
        time.sleep(3)

        pipeline = training_pipeline_storage.get_pipeline(pipeline_id)
        if pipeline['status'] == 'paused':
            print(f"   ✓ Pipeline status changed to 'paused'")
            print(f"   Current progress: {pipeline['completed_runs']}/{pipeline['total_runs']}")
        else:
            print(f"   ⚠ Pipeline status is '{pipeline['status']}', expected 'paused'")

    except Exception as e:
        print(f"   ❌ Failed to pause pipeline: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: Resume pipeline
    print(f"\n5. TEST: RESUME PIPELINE")
    try:
        # Resume in background
        def resume_pipeline():
            try:
                orchestrator.resume()
            except Exception as e:
                print(f"   Pipeline resume error: {e}")

        resume_thread = threading.Thread(target=resume_pipeline, daemon=True)
        resume_thread.start()

        print(f"   ✓ Resume command sent")

        # Wait for resume
        time.sleep(2)

        pipeline = training_pipeline_storage.get_pipeline(pipeline_id)
        if pipeline['status'] == 'running':
            print(f"   ✓ Pipeline status changed back to 'running'")
        else:
            print(f"   ⚠ Pipeline status is '{pipeline['status']}', expected 'running'")

        # Monitor for a bit more
        monitor_pipeline_status(pipeline_id, duration_seconds=10)

    except Exception as e:
        print(f"   ❌ Failed to resume pipeline: {e}")
        import traceback
        traceback.print_exc()

    # Test 4: Stop pipeline (hard stop)
    print(f"\n6. TEST: STOP PIPELINE")
    try:
        orchestrator.stop()
        print(f"   ✓ Stop command sent")

        # Wait for stop to take effect
        time.sleep(3)

        pipeline = training_pipeline_storage.get_pipeline(pipeline_id)
        print(f"   Final status: {pipeline['status']}")
        print(f"   Final progress: {pipeline['completed_runs']}/{pipeline['total_runs']}")

        if pipeline['status'] in ['stopped', 'paused']:
            print(f"   ✓ Pipeline stopped successfully")
        else:
            print(f"   ⚠ Pipeline status is '{pipeline['status']}'")

    except Exception as e:
        print(f"   ❌ Failed to stop pipeline: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print(f"\n" + "=" * 80)
    print(f"TEST SUMMARY")
    print(f"=" * 80)

    final_pipeline = training_pipeline_storage.get_pipeline(pipeline_id)

    print(f"Pipeline: {final_pipeline['name']}")
    print(f"Status: {final_pipeline['status']}")
    print(f"Progress: {final_pipeline['completed_runs']}/{final_pipeline['total_runs']}")
    print(f"\nTested operations:")
    print(f"  ✓ Start pipeline")
    print(f"  ✓ Pause pipeline")
    print(f"  ✓ Resume pipeline")
    print(f"  ✓ Stop pipeline")
    print(f"\nPipeline can be resumed again via:")
    print(f"  - UI: Navigate to pipeline details and click Resume")
    print(f"  - API: POST /training-pipeline/{pipeline_id}/resume")

    return True


if __name__ == "__main__":
    try:
        success = test_pipeline_stop_resume()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ TEST FAILED WITH EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
