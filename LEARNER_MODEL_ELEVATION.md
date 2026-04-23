# Learner Model Elevation

**Date:** 2026-04-23  
**Status:** ✅ **IMPLEMENTED**

---

## Overview

Pipelines train **contextual continuous learner models** that accumulate knowledge across multiple projects. These models learn the relationship between context features (focal length, GSD, vegetation, etc.) and optimal training parameters.

**Model Elevation** allows you to promote a pipeline's shared learner model to the **global model registry**, making it reusable across other projects and pipelines.

---

## Why Elevate Learner Models?

### Problem
After a pipeline completes 15 projects:
- Shared model has learned from 15 diverse datasets
- Model can predict good parameters for new, unseen contexts
- BUT: Model is buried in pipeline folder, not easily reusable

### Solution
Elevate the model → Global registry:
- ✅ Discoverable in model browser
- ✅ Reusable by manual projects
- ✅ Can be starting point for new pipelines
- ✅ Provenance tracked (which pipeline, how many runs)

---

## Directory Structure

### Before Elevation

```
Pipeline folder:
  D:/pipelines/contextual_learning_2026_04_22/
    ├── shared_models/
    │   └── contextual_continuous_selector/
    │       ├── exif_only.json  ← Learned from 15 projects
    │       │   {
    │       │     "runs": 15,
    │       │     "reward_mean": 0.08,
    │       │     "models": {...}  ← Ridge regression weights
    │       │   }
    │       ├── exif_plus_flight_plan.json
    │       └── exif_plus_flight_plan_plus_external.json
    │
    ├── podoli_oblique/
    ├── bilovec_nadir/
    └── ... (15 projects)

Global registry:
  models/
    └── (empty, no learner models yet)
```

### After Elevation

```
Pipeline folder:
  D:/pipelines/contextual_learning_2026_04_22/
    └── shared_models/
        └── contextual_continuous_selector/
            └── exif_only.json  ← Original remains

Global registry:
  models/
    └── model_20260423_143022_oblique_expert_exif_only/
        ├── learner_exif_only.json  ← Copy of shared model
        ├── metadata.json
        │   {
        │     "model_type": "contextual_continuous_learner",
        │     "ai_input_mode": "exif_only",
        │     "context_dim": 9,
        │     "runs": 15,
        │     "reward_mean": 0.08
        │   }
        └── provenance.json
            {
              "source_type": "training_pipeline",
              "pipeline_id": "pipeline_abc123",
              "pipeline_name": "contextual_learning_2026_04_22",
              "mode": "exif_only",
              "elevated_at": "2026-04-23T14:30:22Z"
            }
```

---

## API Usage

### Endpoint

```http
POST /api/training-pipeline/{pipeline_id}/elevate-learner-model
```

### Request Body

```json
{
  "model_name": "Oblique Expert - 15 Projects",
  "mode": "exif_only"
}
```

**Parameters:**
- `model_name`: User-friendly name for the elevated model
- `mode`: AI input mode
  - `"exif_only"` - Basic EXIF features (9D context)
  - `"exif_plus_flight_plan"` - + Flight plan features (19D context)
  - `"exif_plus_flight_plan_plus_external"` - + External features (29D context)

### Response

```json
{
  "success": true,
  "model_id": "model_20260423_143022_oblique_expert_exif_only",
  "model_name": "Oblique Expert - 15 Projects",
  "mode": "exif_only",
  "created_at": "2026-04-23T14:30:22Z",
  "paths": {
    "model_dir": "D:/models/model_20260423_143022_oblique_expert_exif_only",
    "learner_model": "D:/models/.../learner_exif_only.json"
  },
  "provenance": {
    "mode": "exif_only",
    "runs": 15,
    "reward_mean": 0.08
  }
}
```

### Error Cases

**Pipeline not found (404):**
```json
{
  "detail": "Pipeline not found"
}
```

**Learner model not found (404):**
```json
{
  "detail": "Learner model for mode 'exif_only' not found. Has the pipeline completed any training runs?"
}
```

**Invalid mode (400):**
```json
{
  "detail": "Invalid mode. Must be one of: exif_only, exif_plus_flight_plan, exif_plus_flight_plan_plus_external"
}
```

---

## Implementation Details

### Backend Function (`model_registry.py`)

```python
def elevate_learner_model(
    shared_model_dir: Path,
    mode: str,
    model_name: str,
    pipeline_id: str,
    pipeline_name: str,
) -> dict:
    """
    Copy shared learner model to global registry with metadata.
    
    Steps:
    1. Validate source model exists
    2. Read model to extract metadata
    3. Create model_id and directory
    4. Copy learner model file
    5. Create metadata.json
    6. Create provenance.json
    7. Register in models_index.json
    """
```

### API Endpoint (`training_pipeline.py`)

```python
@router.post("/{pipeline_id}/elevate-learner-model")
async def elevate_learner_model(pipeline_id: str, request: ElevateLearnerModelRequest):
    """
    1. Load pipeline from storage
    2. Locate shared_models directory
    3. Validate mode and model file exists
    4. Call model_registry.elevate_learner_model()
    5. Return model record
    """
```

---

## Use Cases

### Use Case 1: Reuse Across Pipelines

```
Pipeline 1 (Oblique datasets):
  - Trains on 15 oblique projects
  - Learns: High angles → specific parameters
  - Elevate model → "Oblique Expert"

Pipeline 2 (Mixed datasets):
  - Can load "Oblique Expert" as starting model
  - Continues learning from mixed data
  - Better initial predictions for oblique images
```

### Use Case 2: Manual Project with Trained Model

```
User creates manual project:
  1. Upload images (oblique, vegetation)
  2. Configure training
  3. Select pre-trained model: "Oblique Expert"
  4. Training starts with learned parameters
  5. Faster convergence, better results
```

### Use Case 3: Model Versioning

```
Experiment evolution:
  v1: "Baseline Model" (5 projects, reward=0.05)
  v2: "Improved Model" (15 projects, reward=0.08)
  v3: "Expert Model" (30 projects, reward=0.10)

Each version elevated → Can compare performance
```

---

## Model Types Comparison

### Gaussian Splat Models (Already Elevated)

**What:** Trained 3D reconstruction (point clouds, Gaussians)  
**File:** `.pt` checkpoint files (PyTorch)  
**Size:** Large (hundreds of MB)  
**Purpose:** Render 3D scenes  
**Elevation:** Per-run (elevate best checkpoint from run)

### Learner Models (NEW - This Feature)

**What:** Parameter selection models (ridge regression)  
**File:** `.json` files (model weights, statistics)  
**Size:** Small (< 1 MB)  
**Purpose:** Predict training parameters from context  
**Elevation:** Per-pipeline (elevate accumulated knowledge)

---

## Example Workflow

### Step 1: Create and Run Pipeline

```bash
# User creates pipeline with 15 datasets
POST /api/training-pipeline/create
{
  "name": "contextual_learning_2026_04_22",
  "projects": [...15 datasets...],
  "shared_config": {
    "ai_input_mode": "exif_only",
    "ai_selector_strategy": "contextual_continuous"
  }
}

# Pipeline runs for 5 days
# Shared model accumulates knowledge
```

### Step 2: Check Model Status

```bash
# View pipeline
GET /api/training-pipeline/{pipeline_id}

# Response shows:
{
  "completed_runs": 15,
  "mean_reward": 0.08,
  "success_rate": 0.87
}

# Check shared_models directory:
D:/pipelines/contextual_learning_2026_04_22/shared_models/
  └── contextual_continuous_selector/
      └── exif_only.json
          {"runs": 15, "reward_mean": 0.08, ...}
```

### Step 3: Elevate the Model

```bash
POST /api/training-pipeline/{pipeline_id}/elevate-learner-model
{
  "model_name": "Oblique Expert - EXIF Only",
  "mode": "exif_only"
}

# Response:
{
  "model_id": "model_20260423_143022_oblique_expert_exif_only",
  "provenance": {
    "runs": 15,
    "reward_mean": 0.08
  }
}
```

### Step 4: Verify in Registry

```bash
GET /api/models

# Response includes:
{
  "models": [
    {
      "model_id": "model_20260423_143022_oblique_expert_exif_only",
      "model_name": "Oblique Expert - EXIF Only",
      "engine": "contextual_continuous_learner",
      "created_at": "2026-04-23T14:30:22Z",
      "source": {
        "pipeline_id": "pipeline_abc123",
        "pipeline_name": "contextual_learning_2026_04_22"
      }
    }
  ]
}
```

### Step 5: Reuse in New Project

```bash
# Create new manual project
POST /api/projects
{
  "name": "new_oblique_project"
}

# Configure training with elevated model
POST /api/projects/{project_id}/process
{
  "ai_input_mode": "exif_only",
  "ai_selector_strategy": "contextual_continuous",
  "pretrained_learner_model_id": "model_20260423_143022_oblique_expert_exif_only"
}

# Training uses learned parameters from 15-project pipeline!
```

---

## Validation and Safety

### Validation Checks

1. **Pipeline exists:** 404 if pipeline_id invalid
2. **Shared model directory exists:** 404 if pipeline hasn't trained
3. **Mode is valid:** 400 if mode not in allowed list
4. **Learner model file exists:** 404 if mode not trained
5. **Model format valid:** 400 if JSON invalid or version mismatch

### Safety Features

1. **Non-destructive:** Original shared model remains in pipeline
2. **Copy operation:** Elevated model is independent copy
3. **Provenance tracked:** Know which pipeline, when, how many runs
4. **Version in model_id:** Timestamp prevents conflicts

---

## Limitations

1. **No automatic detection:** User must manually trigger elevation
2. **One mode at a time:** Must elevate each mode separately if trained multiple
3. **Static snapshot:** Elevated model doesn't update when pipeline continues
4. **No model merging:** Can't combine multiple elevated models (yet)

---

## Future Enhancements

### Possible Improvements

1. **Auto-elevation on completion:**
   ```python
   if pipeline["status"] == "completed" and pipeline["success_rate"] > 0.8:
       auto_elevate_best_models()
   ```

2. **Multi-mode elevation:**
   ```python
   POST /api/training-pipeline/{pipeline_id}/elevate-all-learner-models
   # Elevates all trained modes at once
   ```

3. **Model comparison UI:**
   ```
   Models Browser:
     - Compare reward_mean across elevated models
     - Visualize learning curves
     - Test on sample contexts
   ```

4. **Model inheritance:**
   ```python
   # New pipeline starts with elevated model
   POST /api/training-pipeline/create
   {
     "pretrained_learner_model_id": "model_...",
     "projects": [...new datasets...]
   }
   # Continues learning from elevated model
   ```

---

## Summary

✅ **Implemented learner model elevation**  
✅ **API endpoint**: `POST /api/training-pipeline/{id}/elevate-learner-model`  
✅ **Backend function**: `model_registry.elevate_learner_model()`  
✅ **Metadata and provenance tracking**  
✅ **Registered in global models index**  

**Key Benefits:**
- Share learned knowledge across projects
- Reuse pipeline training investment
- Track model evolution and provenance
- Enable model-based transfer learning

Pipelines now have **two elevation paths**:
1. **Gaussian splat models** (per-run, for rendering)
2. **Learner models** (per-pipeline, for parameter selection)

Both types contribute to the model registry! 🚀
