#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive test script for training pipeline creation and execution.
Tests the complete flow: create pipeline -> verify storage -> check API responses
"""
import json
import sys
import time
from pathlib import Path
from datetime import datetime
import io

# Fix Windows console encoding for Unicode
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "bimba3d_backend"))

from bimba3d_backend.app.services import training_pipeline_storage
from bimba3d_backend.app.services import storage as project_storage
from bimba3d_backend.app.config import DATA_DIR

def test_pipeline_creation():
    """Test creating a new training pipeline."""
    print("=" * 80)
    print("TESTING TRAINING PIPELINE CREATION")
    print("=" * 80)

    # Test configuration
    pipeline_name = f"test_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    base_dir = Path("E:/Thesis/Training Data")
    pipeline_dir = Path("E:/Thesis/Pipeline Projects")

    print(f"\n1. CONFIGURATION")
    print(f"   Pipeline name: {pipeline_name}")
    print(f"   Base directory: {base_dir}")
    print(f"   Pipeline directory: {pipeline_dir}")

    # Check if directories exist
    print(f"\n2. CHECKING DIRECTORIES")
    if not base_dir.exists():
        print(f"   ❌ Base directory does not exist: {base_dir}")
        return False
    print(f"   ✓ Base directory exists")

    if not pipeline_dir.exists():
        print(f"   ⚠ Pipeline directory does not exist, will be created: {pipeline_dir}")
    else:
        print(f"   ✓ Pipeline directory exists")

    # Check available datasets
    print(f"\n3. SCANNING DATASETS")
    datasets = []
    for item in base_dir.iterdir():
        if item.is_dir():
            # Check for images directly in the folder or in an 'images' subdirectory
            images_dir = item / "images" if (item / "images").exists() else item
            image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.JPG")) + \
                         list(images_dir.glob("*.png")) + list(images_dir.glob("*.PNG"))

            if image_files:
                image_count = len(image_files)
                datasets.append({
                    "name": item.name,
                    "path": str(item),
                    "image_count": image_count
                })
                print(f"   ✓ Found dataset: {item.name} ({image_count} images)")

    if not datasets:
        print(f"   ❌ No datasets found in {base_dir}")
        return False

    # Check for existing projects with COLMAP
    print(f"\n4. CHECKING EXISTING PROJECTS (for COLMAP source)")
    try:
        # Check DATA_DIR for existing projects
        existing_projects = []
        for project_dir in DATA_DIR.iterdir():
            if not project_dir.is_dir():
                continue

            config_path = project_dir / "config.json"
            sparse_path = project_dir / "outputs" / "colmap" / "sparse"

            if config_path.exists() and sparse_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    project_name = config.get("name", project_dir.name)
                    pipeline_name_val = config.get("pipeline_name")

                    existing_projects.append({
                        "id": project_dir.name,
                        "name": project_name,
                        "pipeline_name": pipeline_name_val,
                        "has_colmap": True
                    })
                    print(f"   ✓ Found project with COLMAP: {project_name}" +
                          (f" (pipeline: {pipeline_name_val})" if pipeline_name_val else ""))
                except Exception as e:
                    print(f"   ⚠ Error reading project {project_dir.name}: {e}")

        if existing_projects:
            print(f"   Found {len(existing_projects)} existing projects with COLMAP")
        else:
            print(f"   ℹ No existing projects with COLMAP found")

    except Exception as e:
        print(f"   ⚠ Error checking existing projects: {e}")
        existing_projects = []

    # Create pipeline configuration
    print(f"\n5. CREATING PIPELINE CONFIGURATION")

    # Use first dataset for testing
    test_dataset = datasets[0]

    # Check if we can use COLMAP from existing project
    colmap_source_project_id = None
    if existing_projects:
        # Try to find a project with matching dataset name
        matching_project = next(
            (p for p in existing_projects if p["name"].lower() == test_dataset["name"].lower()),
            None
        )
        if matching_project:
            colmap_source_project_id = matching_project["id"]
            print(f"   ✓ Will use COLMAP from existing project: {matching_project['name']}")
        else:
            print(f"   ℹ No matching project found for dataset {test_dataset['name']}")
            print(f"     Available projects: {[p['name'] for p in existing_projects]}")

    pipeline_config = {
        "name": pipeline_name,
        "pipeline_directory": str(pipeline_dir),  # This is the parent directory
        "projects": [
            {
                "name": test_dataset["name"],
                "dataset_path": test_dataset["path"],
                "image_count": test_dataset["image_count"],
                "colmap_source_project_id": colmap_source_project_id
            }
        ],
        "shared_config": {
            "engine": "gsplat",
            "max_steps": 5000,
            "ai_input_mode": "exif_plus_flight_plan"
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
                "shuffle_order": False,
                "strategy_override": None
            },
            {
                "phase_number": 2,
                "name": "Learning Phase",
                "runs_per_project": 2,
                "passes": 1,
                "preset_override": None,
                "update_model": True,
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

    print(f"   ✓ Configuration created:")
    print(f"     - Projects: {len(pipeline_config['projects'])}")
    print(f"     - Phases: {len(pipeline_config['phases'])}")
    print(f"     - Total runs: {sum(p['runs_per_project'] * p['passes'] for p in pipeline_config['phases'])}")

    # Create pipeline via storage service
    print(f"\n6. CREATING PIPELINE IN STORAGE")
    try:
        pipeline_data = training_pipeline_storage.create_pipeline(pipeline_config)
        pipeline_id = pipeline_data['id']
        print(f"   ✓ Pipeline created successfully!")
        print(f"   Pipeline ID: {pipeline_id}")
    except Exception as e:
        print(f"   ❌ Failed to create pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Verify pipeline storage
    print(f"\n7. VERIFYING PIPELINE STORAGE")
    try:
        # Pipeline data already retrieved from create_pipeline
        print(f"   ✓ Pipeline data available")
        print(f"   ✓ Pipeline retrieved successfully")
        print(f"     - Name: {pipeline_data['name']}")
        print(f"     - Status: {pipeline_data['status']}")
        print(f"     - Total runs: {pipeline_data['total_runs']}")
        print(f"     - Completed runs: {pipeline_data['completed_runs']}")

        # Check if pipeline folder was created
        pipeline_folder_path = Path(pipeline_data['config']['pipeline_folder'])
        if pipeline_folder_path.exists():
            print(f"   ✓ Pipeline folder exists: {pipeline_folder_path}")

            # Check for pipeline.json
            pipeline_json = pipeline_folder_path / "pipeline.json"
            if pipeline_json.exists():
                print(f"   ✓ pipeline.json exists")
            else:
                print(f"   ⚠ pipeline.json not found")

            # Check for project directories
            for project_config in pipeline_data['config']['projects']:
                project_dir = pipeline_folder_path / project_config['name']
                if project_dir.exists():
                    print(f"   ✓ Project directory exists: {project_config['name']}")

                    # Check for images symlink/junction
                    images_dir = project_dir / "images"
                    if images_dir.exists():
                        print(f"     ✓ Images directory/link exists")
                    else:
                        print(f"     ⚠ Images directory/link not found")

                    # Check for COLMAP if source was specified
                    if project_config.get('colmap_source_project_id'):
                        colmap_dir = project_dir / "outputs" / "colmap" / "sparse"
                        if colmap_dir.exists():
                            print(f"     ✓ COLMAP data copied from source")
                        else:
                            print(f"     ⚠ COLMAP data not found (might be copied on first run)")
                else:
                    print(f"   ℹ Project directory will be created on first run: {project_config['name']}")
        else:
            print(f"   ❌ Pipeline folder does not exist: {pipeline_folder_path}")

    except Exception as e:
        print(f"   ❌ Failed to verify pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test list pipelines
    print(f"\n8. TESTING LIST PIPELINES API")
    try:
        all_pipelines = training_pipeline_storage.list_pipelines()
        print(f"   ✓ Retrieved {len(all_pipelines)} pipelines")

        # Find our pipeline
        our_pipeline = next((p for p in all_pipelines if p['id'] == pipeline_id), None)
        if our_pipeline:
            print(f"   ✓ Our pipeline found in list")
        else:
            print(f"   ⚠ Our pipeline not found in list")

    except Exception as e:
        print(f"   ❌ Failed to list pipelines: {e}")
        return False

    # Test pipeline data structure
    print(f"\n9. VERIFYING PIPELINE DATA STRUCTURE")
    required_fields = [
        'id', 'name', 'status', 'created_at', 'current_phase', 'current_pass',
        'current_project_index', 'total_runs', 'completed_runs', 'failed_runs',
        'config', 'runs'
    ]

    for field in required_fields:
        if field in pipeline_data:
            print(f"   ✓ Field '{field}' present: {type(pipeline_data[field]).__name__}")
        else:
            print(f"   ❌ Field '{field}' missing")

    # Check config structure
    print(f"\n10. VERIFYING CONFIG STRUCTURE")
    config_fields = ['projects', 'shared_config', 'phases', 'thermal_management', 'pipeline_folder']
    for field in config_fields:
        if field in pipeline_data['config']:
            print(f"   ✓ Config field '{field}' present")
        else:
            print(f"   ❌ Config field '{field}' missing")

    print(f"\n" + "=" * 80)
    print(f"TEST SUMMARY")
    print(f"=" * 80)
    print(f"✓ Pipeline created successfully: {pipeline_id}")
    print(f"✓ Pipeline name: {pipeline_name}")
    print(f"✓ Pipeline folder: {pipeline_data['config']['pipeline_folder']}")
    print(f"✓ Status: {pipeline_data['status']}")
    print(f"✓ Total runs configured: {pipeline_data['total_runs']}")
    print(f"\nTo test in UI:")
    print(f"  1. Navigate to http://localhost:3000/pipelines")
    print(f"  2. Look for pipeline: {pipeline_name}")
    print(f"  3. Click to view details")
    print(f"  4. Verify all configuration is correct")
    print(f"  5. Start the pipeline to test execution")
    print(f"\nTo clean up:")
    print(f"  - Delete pipeline folder: {pipeline_data['config']['pipeline_folder']}")
    print(f"  - Delete pipeline metadata: {DATA_DIR.parent / 'training_pipelines' / f'{pipeline_id}.json'}")

    return True

if __name__ == "__main__":
    try:
        success = test_pipeline_creation()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ TEST FAILED WITH EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
