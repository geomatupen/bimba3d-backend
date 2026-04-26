#!/usr/bin/env python
"""Fix the 'Second Test Pipeline' by moving it to ID-based folder."""
import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "bimba3d_backend"))

from bimba3d_backend.app.services import training_pipeline_storage

pipeline_id = "pipeline_2c5819cc8493"
pipeline = training_pipeline_storage.get_pipeline(pipeline_id)

old_folder = Path("E:/Thesis/PipelineProjects/Second Test Pipeline")
new_folder = Path(f"E:/Thesis/PipelineProjects/{pipeline_id}")

print(f"Pipeline: {pipeline['name']}")
print(f"ID: {pipeline_id}")
print(f"Old folder: {old_folder}")
print(f"New folder: {new_folder}")
print(f"Old folder exists: {old_folder.exists()}")
print(f"New folder exists: {new_folder.exists()}")

if old_folder.exists() and not new_folder.exists():
    print(f"\nMoving folder...")
    try:
        shutil.move(str(old_folder), str(new_folder))
        print(f"✓ Moved successfully")

        # Update pipeline config
        pipeline['config']['pipeline_folder'] = str(new_folder)
        training_pipeline_storage.update_pipeline(pipeline_id, {'config': pipeline['config']})
        print(f"✓ Updated pipeline config")

        print(f"\nDone! Pipeline folder is now: {new_folder}")
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"\nPlease manually:")
        print(f"1. Close all programs that might have files open in the folder")
        print(f"2. Rename the folder in Windows Explorer from:")
        print(f"   'Second Test Pipeline' to '{pipeline_id}'")
        sys.exit(1)
elif new_folder.exists():
    print(f"\n✓ New folder already exists, updating config...")
    pipeline['config']['pipeline_folder'] = str(new_folder)
    training_pipeline_storage.update_pipeline(pipeline_id, {'config': pipeline['config']})
    print(f"✓ Updated pipeline config")
else:
    print(f"\n❌ Old folder doesn't exist. The pipeline may have been deleted or moved already.")
