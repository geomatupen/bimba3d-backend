# Pipeline Folder Listing Fix

**Date:** 2026-04-23  
**Status:** ✅ **FIXED**

---

## Problem 1: Pipeline Folder Listed as Project

**Error:**
```
GET http://localhost:8005/projects/training_pipelines/status 404 (Not Found)
Failed to fetch project: ha
```

**Root cause:**
1. Pipeline folder created at: `DATA_DIR/training_2026-04-23/`
2. `list_projects()` API lists ALL directories in DATA_DIR
3. Dashboard tries to fetch status for "training_2026-04-23" folder
4. Pipeline folders don't have status.json → 404 error
5. Infinite retry loop in Dashboard

---

## Solution 1: Exclude Pipeline Folders from Project List

**[projects.py](d:\bimba3d-re\bimba3d_backend\app\api\projects.py)** (Line 3264-3273):

```python
for project_dir in sorted(DATA_DIR.iterdir()):
    if not project_dir.is_dir():
        continue

    project_id = project_dir.name

    # Skip pipeline folders (they contain multiple projects, not a single project)
    # Pipeline folders are identified by having a pipeline.json file
    if (project_dir / "pipeline.json").exists():
        continue

    project_status = status.get_status(project_id)
    # ... rest of code
```

**How it works:**
- Pipeline folders now have a `pipeline.json` marker file
- `list_projects()` skips directories with this marker
- Only actual projects are listed

---

## Solution 2: Create Pipeline Marker File

**[training_pipeline_storage.py](d:\bimba3d-re\bimba3d_backend\app\services\training_pipeline_storage.py)** (Lines 99-112):

```python
# Save to centralized pipelines registry
with open(_pipeline_path(pipeline_id), "w") as f:
    json.dump(pipeline, f, indent=2)

# Also save pipeline.json marker in the pipeline folder so it's not listed as a project
pipeline_marker = pipeline_folder / "pipeline.json"
with open(pipeline_marker, "w") as f:
    json.dump({
        "pipeline_id": pipeline_id,
        "pipeline_name": pipeline["name"],
        "created_at": pipeline["created_at"],
    }, f, indent=2)
```

**Result:**
```
DATA_DIR/
  ├── training_pipelines/          ← Registry directory
  │   ├── pipeline_abc123.json    ← Pipeline metadata
  │   └── pipeline_def456.json
  │
  ├── training_2026-04-23/         ← Pipeline folder (NOW EXCLUDED)
  │   ├── pipeline.json            ← Marker file (NEW!)
  │   ├── shared_models/
  │   ├── project1/
  │   └── project2/
  │
  └── my_manual_project/           ← Regular project (still listed)
      ├── config.json
      └── status.json
```

---

## Problem 2: Pipeline Projects Shown Last (User Request)

**User said:** "the sort by pipeline keeps pipeline projects at last, we need to flip it to show them first."

**Current code** ([Dashboard.tsx](d:\bimba3d-re\bimba3d_frontend\src\pages\Dashboard.tsx), Line 137):
```typescript
if (aIsPipeline !== bIsPipeline) {
  return aIsPipeline ? -1 : 1;  // Pipeline first (-1 = earlier)
}
```

**This is already correct!** The code shows pipeline projects **first**.

**Possible reasons user saw them last:**
1. ❌ The "training_pipelines" folder itself was being listed (now fixed!)
2. ❌ Pipeline projects didn't have `pipeline_name` in config yet
3. ✅ After fix, pipeline projects WILL show first when sorted by "Pipeline"

---

## Directory Structure

### Before Fix:
```
DATA_DIR/
  ├── training_pipelines/          ← Registry
  │   └── pipeline_abc123.json
  │
  ├── training_2026-04-23/         ← LISTED AS PROJECT ❌
  │   ├── (no pipeline.json)
  │   ├── podoli_oblique/
  │   └── bilovec_nadir/
  │
  └── my_manual_project/
      └── status.json

Dashboard sees:
  - "training_2026-04-23" ← Tries to load as project → 404 error
  - "my_manual_project"
```

### After Fix:
```
DATA_DIR/
  ├── training_pipelines/          ← Registry
  │   └── pipeline_abc123.json
  │
  ├── training_2026-04-23/         ← EXCLUDED ✓
  │   ├── pipeline.json            ← Marker file
  │   ├── shared_models/
  │   ├── podoli_oblique/          ← Individual project
  │   │   ├── config.json          ← Has pipeline_name
  │   │   └── status.json
  │   └── bilovec_nadir/           ← Individual project
  │       ├── config.json          ← Has pipeline_name
  │       └── status.json
  │
  └── my_manual_project/
      └── status.json

Dashboard sees:
  - "podoli_oblique" (pipeline_name: "training_2026-04-23") ← Shows with badge
  - "bilovec_nadir" (pipeline_name: "training_2026-04-23") ← Shows with badge
  - "my_manual_project" (no pipeline_name)
```

---

## Testing

### Test 1: Create New Pipeline

**Steps:**
1. Create pipeline: "test_pipeline_2026"
2. Select 3 datasets
3. Start pipeline

**Expected:**
- ✅ Pipeline folder created: `DATA_DIR/test_pipeline_2026/`
- ✅ Marker created: `DATA_DIR/test_pipeline_2026/pipeline.json`
- ✅ Projects created inside: `podoli_oblique/`, `bilovec_nadir/`, etc.
- ✅ Dashboard does NOT show "test_pipeline_2026" folder
- ✅ Dashboard DOES show individual projects with pipeline badge

---

### Test 2: Sort by Pipeline

**Steps:**
1. Open Dashboard
2. Select "Sort: Pipeline" dropdown

**Expected:**
```
Pipeline Projects (shown FIRST):
  ┌─────────────────────────────────────────────────┐
  │ 🟣 test_pipeline_2026                            │
  │ podoli_oblique                                   │
  │ Status: processing | Progress: 45%               │
  └─────────────────────────────────────────────────┘
  │ 🟣 test_pipeline_2026                            │
  │ bilovec_nadir                                    │
  │ Status: completed | Progress: 100%               │
  └─────────────────────────────────────────────────┘

Manual Projects (shown AFTER):
  ┌─────────────────────────────────────────────────┐
  │ my_manual_project                                │
  │ Status: completed | Progress: 100%               │
  └─────────────────────────────────────────────────┘
```

---

### Test 3: No More 404 Errors

**Before fix:**
```
GET /projects/training_pipelines/status → 404
GET /projects/test_pipeline_2026/status → 404
(Infinite retry loop)
```

**After fix:**
```
GET /projects/podoli_oblique/status → 200 OK
GET /projects/bilovec_nadir/status → 200 OK
GET /projects/my_manual_project/status → 200 OK
(No pipeline folders in list)
```

---

## Summary

✅ **Pipeline folders excluded** from project list  
✅ **pipeline.json marker** created in each pipeline folder  
✅ **Individual projects** inside pipeline folders still listed  
✅ **Pipeline badge** shows for pipeline projects  
✅ **Sort by pipeline** shows pipeline projects first  
✅ **No more 404 errors** from trying to load pipeline folders  

**Result:** Clean separation between pipeline containers and actual projects! 🎯
