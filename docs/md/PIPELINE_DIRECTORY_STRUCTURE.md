# Pipeline Directory Structure - Implementation Summary

**Date:** 2026-04-22  
**Status:** ✅ **IMPLEMENTED**

---

## User's Requirement

> "i need to create separate folder per training pipeline. data will be separate which is reads from. we dont need to touch data folder which we create projects from based on the folder names in the data folder. but when it runs it should create a new folder with new pipeline and create projects inside it. independent from data folder. we can reuse data folder later with new pipeline."

---

## Implementation

### ✅ Clear Separation of Concerns

**1. Source Data Folder (Read-Only)**
```
E:/Thesis/exp_new_method/
├── podoli_oblique/
│   ├── IMG_0001.jpg
│   ├── IMG_0002.jpg
│   └── ...                  ← Original images, NEVER modified
├── bilovec_nadir/
│   ├── IMG_0001.jpg
│   └── ...
└── terrain_rough/
    └── ...
```

**Purpose:** Contains original dataset images  
**Access:** Read-only reference  
**Modified:** Never  
**Reusable:** Yes, by multiple pipelines

---

**2. Pipeline Working Directory**
```
D:/bimba3d-pipelines/                          ← User selects this location
├── contextual_learning_2026_04_22/            ← Pipeline 1
│   ├── podoli_oblique/                        ← Project folder
│   │   ├── config.json
│   │   │   {
│   │   │     "source_dir": "E:/Thesis/.../podoli_oblique",  ← References data
│   │   │     "created_by": "training_pipeline",
│   │   │     "pipeline_id": "pipeline_abc123",
│   │   │     ...
│   │   │   }
│   │   ├── images/                            ← Symlink or copy (if needed)
│   │   ├── sparse/                            ← COLMAP outputs
│   │   ├── runs/                              ← Training runs
│   │   │   ├── baseline_phase1/
│   │   │   ├── exploration_phase2/
│   │   │   └── refinement_phase3_pass2/
│   │   ├── models/                            ← Trained models
│   │   │   └── contextual_continuous_selector/
│   │   └── outputs/                           ← Previews, analytics
│   ├── bilovec_nadir/
│   └── terrain_rough/
│
└── new_experiment_2026_04_25/                 ← Pipeline 2 (REUSES same data!)
    ├── podoli_oblique/                        ← NEW project, SAME source
    │   └── config.json
    │       {
    │         "source_dir": "E:/Thesis/.../podoli_oblique",  ← Same data!
    │         "pipeline_id": "pipeline_xyz789",
    │         ...
    │       }
    ├── bilovec_nadir/
    └── terrain_rough/
```

**Purpose:** Working area for training, COLMAP, models  
**Access:** Read/write by pipeline  
**Modified:** Continuously during training  
**Reusable:** Each pipeline is independent

---

## How It Works

### User Workflow

**Step 1: Specify Directories in Wizard**
```
Training Pipeline Page
  ├─ Step 1: Dataset Selection
  │   ├─ Source Data Directory (Read-Only):
  │   │   E:/Thesis/exp_new_method          ← Scan for datasets
  │   │   [Scan Directory]
  │   │
  │   └─ Pipeline Output Directory:
  │       D:/bimba3d-pipelines              ← Where to create pipeline
  │       (Optional: leave empty for default)
```

**Step 2-5:** Configure training, phases, thermal, review

**Result:**
```
Pipeline creates folder structure:
  D:/bimba3d-pipelines/contextual_learning_2026_04_22/
    ├── podoli_oblique/     ← Project 1
    ├── bilovec_nadir/      ← Project 2
    └── terrain_rough/      ← Project 3

Each project's config.json references:
  "source_dir": "E:/Thesis/exp_new_method/{dataset_name}"
```

---

### Backend Implementation

**1. Pipeline Creation (storage.py)**
```python
def create_pipeline(config):
    # User specifies pipeline output directory
    pipeline_directory = config.get("pipeline_directory")
    
    if pipeline_directory:
        pipeline_root = Path(pipeline_directory)  # User choice
    else:
        pipeline_root = DATA_DIR  # Default (same as manual projects)
    
    # Create: {pipeline_root}/{pipeline_name}/
    pipeline_folder = pipeline_root / config["name"]
    pipeline_folder.mkdir(parents=True, exist_ok=True)
    
    # Store path for orchestrator
    config["pipeline_folder"] = str(pipeline_folder)
```

**2. Project Creation (orchestrator.py)**
```python
def _get_or_create_project_dir(pipeline, project):
    # Get pipeline folder
    pipeline_folder = Path(pipeline["config"]["pipeline_folder"])
    
    # Create: {pipeline_folder}/{project_name}/
    project_dir = pipeline_folder / project["name"]
    project_dir.mkdir(parents=True, exist_ok=True)
    
    # Write config with source reference
    config = {
        "id": uuid4(),
        "name": project["name"],
        "source_dir": project["dataset_path"],  # Points to read-only data!
        "created_by": "training_pipeline",
        "pipeline_id": pipeline["id"],
        ...
    }
    
    with open(project_dir / "config.json", "w") as f:
        json.dump(config, f)
    
    return project_dir
```

**3. Training Run Execution**
```python
def _execute_run(pipeline, project, phase, pass_num):
    # Get project directory in pipeline folder
    project_dir = self._get_or_create_project_dir(pipeline, project)
    
    # Build run config
    run_config = {
        "source_dir": project["dataset_path"],  # Read images from here
        "project_dir": project_dir,              # Write outputs here
        ...
    }
    
    # Execute training (reads data, writes to project_dir)
    gsplat.run_training(project_dir, run_config)
```

---

## Key Benefits

### ✅ 1. Data Reusability
```
Source data: E:/Thesis/exp_new_method/

Pipeline 1 (baseline experiment):
  D:/pipelines/baseline_2026_04_22/
    └── References: E:/Thesis/.../

Pipeline 2 (new algorithm):
  D:/pipelines/new_algo_2026_04_25/
    └── References: E:/Thesis/.../  ← SAME data, different work

Source data used TWICE, stored ONCE
```

### ✅ 2. Clear Organization
```
Before (all mixed):
  projects/
    ├── manual_project_1/
    ├── pipeline_proj_1/  ← Hard to tell which came from where
    ├── manual_project_2/
    ├── pipeline_proj_2/
    └── ...

After (organized):
  projects/
    ├── manual_project_1/
    └── manual_project_2/
    
  pipelines/
    ├── contextual_learning_2026_04_22/  ← Clear grouping
    │   ├── proj_1/
    │   ├── proj_2/
    │   └── proj_3/
    └── new_experiment_2026_04_25/
        └── ...
```

### ✅ 3. Safety
```
Source data folder:
  ✓ Never modified by pipeline
  ✓ Can't be accidentally deleted
  ✓ Preserved for future experiments

Pipeline folder:
  ✓ Can delete entire pipeline safely
  ✓ Won't affect source data
  ✓ Won't affect other pipelines
```

### ✅ 4. Flexibility
```
User scenarios:

1. Quick experiment (default location):
   Pipeline Output Directory: [leave empty]
   → Creates in default projects directory

2. Organized experiments (separate drive):
   Pipeline Output Directory: D:/experiments/
   → Creates pipeline folders there

3. Network storage (shared team folder):
   Pipeline Output Directory: \\server\team\pipelines\
   → Creates pipeline folders on network drive

4. Multiple experiments (same source data):
   Source: E:/data/
   Pipeline 1: D:/experiment_A/
   Pipeline 2: D:/experiment_B/
   Both read from E:/data/, write to different locations
```

---

## User Questions Answered

### Q1: "does it create a projects folder as well where all newly being created projects folder will be kept?"

**A:** Yes! But you control WHERE:

```
Option 1 (default - leave Pipeline Directory empty):
  Creates in default location (same as manual projects)
  
  projects/
    ├── contextual_learning_2026_04_22/
    │   ├── podoli_oblique/
    │   └── ...
    └── manual_projects/

Option 2 (specify custom location):
  Creates in your chosen location
  
  D:/bimba3d-pipelines/              ← You specify this
    ├── contextual_learning_2026_04_22/
    │   ├── podoli_oblique/
    │   └── ...
    └── ...
```

### Q2: "there is no difference [from manually created projects]. that global config is then set to individual configs or individual projects keep using the global configs till the end. which way is better?"

**A:** Pipeline-created projects ARE normal projects, just auto-created in batch:

```
Manual project creation:
  Dashboard → Create Project → Upload images → Configure

Pipeline project creation:
  Pipeline Wizard → Scan data folder → Configure once → Creates N projects

Both result in: Independent projects with their own config.json

Global config approach:
  - Pipeline wizard settings → COPIED to each project's config.json
  - Each project is independent after creation
  - Change Project A settings → Only affects Project A
  - Better: Projects can evolve independently
```

### Q3: "data will be separate which is reads from. we dont need to touch data folder"

**A:** Correct! Source data is NEVER modified:

```
Source folder (read-only):
  E:/Thesis/exp_new_method/
    └── podoli_oblique/
        ├── IMG_0001.jpg  ← Original, never touched
        └── ...

Project folder (work area):
  {pipeline_folder}/podoli_oblique/
    ├── config.json       → source_dir: "E:/Thesis/.../podoli_oblique"
    ├── sparse/           ← COLMAP output (generated)
    ├── runs/             ← Training runs (generated)
    └── models/           ← Learned models (generated)

Images read from source, outputs written to project folder
```

### Q4: "we can reuse data folder later with new pipeline"

**A:** Absolutely! Multiple pipelines can reference the same data:

```
1 data folder → N pipelines

E:/data/ (200 GB of images, stored ONCE)
  ↓ Referenced by
  
Pipeline 1: D:/pipelines/baseline/
Pipeline 2: D:/pipelines/experiment_A/
Pipeline 3: D:/pipelines/experiment_B/

All read from E:/data/, write to their own folders
No duplication, maximum reusability
```

---

## Implementation Details

### Frontend Changes

**TrainingPipelinePage.tsx:**
```typescript
// Added state
const [pipelineDirectory, setPipelineDirectory] = useState("");

// Added UI field in Step 1
<input
  value={pipelineDirectory}
  onChange={(e) => setPipelineDirectory(e.target.value)}
  placeholder="Leave empty to use default projects directory"
/>

// Preview
Pipeline will create: {pipelineDirectory || "[default]"}/{pipelineName}/

// Sent to backend
const config = {
  name: pipelineName,
  base_directory: baseDirectory,       // Source data (scan from)
  pipeline_directory: pipelineDirectory || null,  // Output (create in)
  ...
};
```

### Backend Changes

**API Model:**
```python
class CreatePipelineRequest(BaseModel):
    base_directory: str         # Source data folder
    pipeline_directory: Optional[str] = None  # Output folder (default: DATA_DIR)
    ...
```

**Storage Service:**
```python
# Determine location
pipeline_root = Path(config["pipeline_directory"]) if config.get("pipeline_directory") else DATA_DIR

# Create folder
pipeline_folder = pipeline_root / config["name"]
pipeline_folder.mkdir(parents=True, exist_ok=True)

# Store for orchestrator
config["pipeline_folder"] = str(pipeline_folder)
```

**Orchestrator:**
```python
# Create project in pipeline folder
def _get_or_create_project_dir(pipeline, project):
    pipeline_folder = Path(pipeline["config"]["pipeline_folder"])
    project_dir = pipeline_folder / project["name"]
    
    # Write config with source_dir reference
    config = {
        "source_dir": project["dataset_path"],  # Read-only reference!
        ...
    }
```

---

## Example: Complete Workflow

**Day 1: Create Pipeline**
```
User opens Training Pipeline wizard

Step 1:
  Source Data Directory: E:/Thesis/exp_new_method
  [Scan Directory] → Finds 15 datasets
  
  Pipeline Output Directory: D:/bimba3d-pipelines
  (Preview: D:/bimba3d-pipelines/contextual_learning_2026_04_22/)
  
Step 2-5: Configure, launch

Backend creates:
  D:/bimba3d-pipelines/contextual_learning_2026_04_22/
    ├── podoli_oblique/
    │   └── config.json: "source_dir": "E:/Thesis/.../podoli_oblique"
    ├── bilovec_nadir/
    └── ...
```

**Day 2-5: Pipeline Runs**
```
Orchestrator executes Phase 1, 2, 3

For each run:
  1. Read images from: E:/Thesis/.../podoli_oblique
  2. Run COLMAP, write to: D:/pipelines/.../podoli_oblique/sparse/
  3. Train model, write to: D:/pipelines/.../podoli_oblique/runs/
  4. Save model to: D:/pipelines/.../podoli_oblique/models/

Source folder: Untouched, still has original images
Pipeline folder: Accumulates COLMAP, runs, models
```

**Day 10: New Experiment (Reuse Data)**
```
User opens Training Pipeline wizard again

Step 1:
  Source Data Directory: E:/Thesis/exp_new_method  ← SAME data!
  [Scan Directory] → Finds same 15 datasets
  
  Pipeline Output Directory: D:/bimba3d-pipelines
  (Preview: D:/bimba3d-pipelines/new_experiment_2026_05_02/)

Creates NEW pipeline folder, references SAME source data:
  D:/bimba3d-pipelines/new_experiment_2026_05_02/
    ├── podoli_oblique/
    │   └── config.json: "source_dir": "E:/Thesis/.../podoli_oblique"  ← Same!
    └── ...

Result:
  - 2 pipelines, 1 source data folder
  - No data duplication
  - Independent experiments
```

---

## Summary

✅ **Implemented:** Separate pipeline directories with user-selectable path  
✅ **Source data:** Read-only, never modified, reusable  
✅ **Pipeline folders:** Independent working areas per batch  
✅ **Projects:** Normal projects, just auto-created in pipeline folder  
✅ **Backward compatible:** Default location = same as manual projects  

The architecture achieves exactly what you requested:
- Data folder separate (read-only source)
- Pipeline creates new folder for work area
- Independent from data folder
- Can reuse data folder with new pipeline

Perfect separation of concerns! 🎯
