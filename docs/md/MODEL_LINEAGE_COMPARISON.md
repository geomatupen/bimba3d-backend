# Model Lineage Comparison - Gsplat vs Learner Models

**Date:** 2026-04-23  
**Status:** ✅ **COMPLETE PARITY**

---

## Overview

Both model types now have **complete lineage tracking** with equivalent metadata:

1. **Gaussian Splat Models** - Trained 3D reconstructions (per-run elevation)
2. **Learner Models** - Parameter selection models (per-pipeline elevation)

---

## Feature Parity Matrix

| Feature | Gsplat Models | Learner Models | Status |
|---------|--------------|----------------|--------|
| **Model ID** | ✅ Unique timestamp-based | ✅ Unique timestamp-based | ✅ Equal |
| **Model Name** | ✅ User-provided | ✅ User-provided | ✅ Equal |
| **Engine** | ✅ `"gsplat"` | ✅ `"gsplat"` | ✅ Equal |
| **Artifact Format** | ✅ `"pytorch_checkpoint"` | ✅ `"learner_json"` | ✅ Different (appropriate) |
| **AI Profile** | ✅ Yes (mode, strategy) | ✅ Yes (mode, strategy, context_dim) | ✅ Equal |
| **Source Tracking** | ✅ Project + Run | ✅ Pipeline | ✅ Equal (different source types) |
| **Lineage** | ✅ Contributors list | ✅ Contributors list | ✅ Equal |
| **Configs** | ✅ Per-run configs | ✅ Shared config | ✅ Equal (appropriate approach) |
| **Provenance Summary** | ✅ Yes | ✅ Yes (with project count, runs, reward) | ✅ Equal |
| **Created Timestamp** | ✅ ISO 8601 | ✅ ISO 8601 | ✅ Equal |
| **Model Directory** | ✅ `models/{model_id}/` | ✅ `models/{model_id}/` | ✅ Equal |

---

## Directory Structure Comparison

### Gaussian Splat Model (Elevated from Run)

```
models/model_20260423_143022_best_reconstruction/
  ├── source_checkpoint.pt              ← PyTorch checkpoint (100-500 MB)
  ├── metadata.json                     ← Gsplat training metadata
  ├── provenance.json                   ← Source tracking
  ├── lineage.json                      ← Contributors from batch/lineage
  │   {
  │     "contributors": [
  │       {
  │         "contributor_id": "proj_123:run_abc",
  │         "project_id": "proj_123",
  │         "project_name": "oblique_dataset",
  │         "run_id": "run_abc",
  │         "captured_at": "2026-04-23T14:30:22Z",
  │         "files": {
  │           "run_config": {...},      ← Run-specific config
  │           "shared_config": {...},   ← Project shared config
  │           "metadata": {...}
  │         }
  │       }
  │     ]
  │   }
  └── configs/                          ← Per-run configs
      └── proj_123/
          └── run_abc/
              ├── run_config.json       ← Run parameters
              ├── shared_config.json    ← Project defaults
              └── metadata.json         ← Gsplat metadata
```

### Learner Model (Elevated from Pipeline)

```
models/model_20260423_150030_oblique_expert_exif_only/
  ├── learner_exif_only.json            ← Ridge regression model (< 1 MB)
  │   {
  │     "version": 2,
  │     "mode": "exif_only",
  │     "context_dim": 9,
  │     "lambda_ridge": 2.0,
  │     "runs": 15,                      ← Total training runs
  │     "reward_mean": 0.08,             ← Average reward
  │     "models": {
  │       "feature_lr_mult": {
  │         "A": [[...]],                ← Ridge regression A matrix
  │         "b": [...],                  ← Ridge regression b vector
  │         "n": 15
  │       },
  │       ...8 total multipliers
  │     },
  │     "last": {
  │       "run_id": "proj_15_run_1",
  │       "s_best": 0.85,
  │       "s_run": 0.82,
  │       "reward_signal": 0.08
  │     }
  │   }
  │
  ├── metadata.json                     ← Model type metadata
  │   {
  │     "model_type": "contextual_continuous_learner",
  │     "ai_input_mode": "exif_only",
  │     "context_dim": 9,
  │     "runs": 15,
  │     "reward_mean": 0.08,
  │     "lambda_ridge": 2.0,
  │     "last_update": "proj_15_run_1"
  │   }
  │
  ├── provenance.json                   ← Pipeline source
  │   {
  │     "source_type": "training_pipeline",
  │     "pipeline_id": "pipeline_abc123",
  │     "pipeline_name": "contextual_learning_2026_04_22",
  │     "mode": "exif_only",
  │     "elevated_at": "2026-04-23T15:00:30Z"
  │   }
  │
  ├── lineage.json                      ← Contributors (all pipeline projects)
  │   {
  │     "contributors": [
  │       {
  │         "project_name": "podoli_oblique",
  │         "dataset_path": "E:/Thesis/.../podoli_oblique",
  │         "source": "training_pipeline",
  │         "image_count": 245
  │       },
  │       {
  │         "project_name": "bilovec_nadir",
  │         "dataset_path": "E:/Thesis/.../bilovec_nadir",
  │         "source": "training_pipeline",
  │         "image_count": 312
  │       },
  │       ...13 more projects
  │     ],
  │     "pipeline_id": "pipeline_abc123",
  │     "pipeline_name": "contextual_learning_2026_04_22",
  │     "total_runs": 15,
  │     "reward_mean": 0.08,
  │     "note": "Pipeline uses shared config across all projects"
  │   }
  │
  └── config/                           ← Shared config (ONE for all projects)
      └── shared_config.json            ← Pipeline training parameters
          {
            "ai_input_mode": "exif_only",
            "ai_selector_strategy": "contextual_continuous",
            "max_steps": 15000,
            "densify_grad_threshold": 0.0002,
            "opacity_threshold": 0.005,
            ...
          }
```

---

## Key Differences (By Design)

### 1. Config Storage Approach

**Gsplat Models:**
```
Multiple configs per model:
  configs/
    ├── project1/run1/
    │   ├── run_config.json    ← Run-specific params
    │   └── shared_config.json ← Project defaults
    ├── project1/run2/
    └── project2/run1/

Reason: Each run may have different parameters
Use case: Elevate best run from batch of experiments
```

**Learner Models:**
```
Single shared config per model:
  config/
    └── shared_config.json     ← ONE config for entire pipeline

Reason: Pipeline uses consistent parameters across all projects
Use case: All projects trained with same configuration
Note: Per-project variation comes from CONTEXT (features), not config
```

**Why different?**
- Gsplat models: Elevate specific successful run → preserve its exact config
- Learner models: Elevate accumulated knowledge → shared config represents pipeline approach

### 2. Contributor Granularity

**Gsplat Models:**
```
contributors: [
  {
    "contributor_id": "proj_123:run_abc",  ← Project + Run
    "project_id": "proj_123",
    "run_id": "run_abc",
    "files": {run_config, shared_config, metadata}
  }
]

Tracks: Individual runs that contributed to model
```

**Learner Models:**
```
contributors: [
  {
    "project_name": "podoli_oblique",    ← Project only
    "dataset_path": "E:/Thesis/.../podoli_oblique",
    "source": "training_pipeline",
    "image_count": 245
  }
]

Tracks: Projects (datasets) that contributed to learning
```

**Why different?**
- Gsplat models: Specific runs matter (different parameters, warmup stages)
- Learner models: Projects matter (diverse contexts), runs within project are similar

### 3. Artifact Size

**Gsplat Models:**
- File: `source_checkpoint.pt`
- Size: 100-500 MB (PyTorch checkpoint with millions of Gaussians)
- Type: Binary

**Learner Models:**
- File: `learner_exif_only.json`
- Size: < 1 MB (8 ridge regression models with 9-29D weights)
- Type: JSON

---

## Lineage Tracking Examples

### Example 1: Gsplat Model from Batch Run

```
User workflow:
  1. Create project "oblique_dataset"
  2. Configure: run_count=5, ai_input_mode=exif_only
  3. Run batch training
  4. Elevate best run (run_3)

Resulting lineage:
  models/model_20260423_best_oblique/
    └── configs/
        └── oblique_dataset/
            ├── run_1/ ← Contributor 1
            ├── run_2/ ← Contributor 2
            ├── run_3/ ← Contributor 3 (main)
            ├── run_4/ ← Contributor 4
            └── run_5/ ← Contributor 5

  lineage.json:
    {
      "contributors": [5 runs],
      "provenance_summary": {
        "contributor_count": 5,
        "unique_project_count": 1,
        "project_names": ["oblique_dataset"]
      }
    }
```

### Example 2: Learner Model from Pipeline

```
User workflow:
  1. Create pipeline "contextual_learning_2026_04_22"
  2. Add 15 datasets from E:/Thesis/exp_new_method/
  3. Configure shared params: ai_input_mode=exif_only
  4. Run pipeline (15 projects × 3 phases = 45 runs)
  5. Elevate learner model

Resulting lineage:
  models/model_20260423_oblique_expert/
    ├── config/
    │   └── shared_config.json ← ONE config for all 15 projects
    └── lineage.json
        {
          "contributors": [
            {"project_name": "podoli_oblique", ...},
            {"project_name": "bilovec_nadir", ...},
            ...15 projects total
          ],
          "total_runs": 45,
          "reward_mean": 0.08,
          "pipeline_id": "pipeline_abc123",
          "note": "Pipeline uses shared config across all projects"
        }
```

---

## Provenance Summary Comparison

### Gsplat Model

```json
{
  "provenance_summary": {
    "contributor_count": 5,
    "unique_project_count": 1,
    "project_names": ["oblique_dataset"]
  }
}
```

**Shows:** How many runs contributed, from how many projects

### Learner Model

```json
{
  "provenance_summary": {
    "mode": "exif_only",
    "runs": 45,
    "reward_mean": 0.08,
    "contributor_count": 15,
    "unique_project_count": 15,
    "project_names": [
      "podoli_oblique",
      "bilovec_nadir",
      "terrain_rough",
      ...10 more (first 10 shown)
    ]
  }
}
```

**Shows:** Training mode, total runs, performance metrics, contributor projects

---

## API Comparison

### Elevate Gsplat Model

```http
POST /api/projects/{project_id}/runs/{run_id}/elevate-model
{
  "model_name": "Best Oblique Reconstruction"
}

→ Copies checkpoint from run
→ Snapshots configs from run + batch contributors
→ Creates lineage from batch_lineage_latest.json
```

### Elevate Learner Model

```http
POST /api/training-pipeline/{pipeline_id}/elevate-learner-model
{
  "model_name": "Oblique Expert - 15 Projects",
  "mode": "exif_only"
}

→ Copies learner model from shared_models/
→ Snapshots shared_config from pipeline
→ Creates lineage from pipeline projects list
```

---

## Use Cases

### Gsplat Model Reuse

```
Scenario: User wants to continue training from best result

1. Elevate run_3 → "Best Baseline Model"
2. Create new project
3. Select "Best Baseline Model" as start_model
4. Training continues from checkpoint

Lineage preserved:
  - Original run's exact config
  - Can reproduce exact starting point
  - Config per run allows variation
```

### Learner Model Reuse

```
Scenario: User wants to apply learned parameters to new project

1. Elevate pipeline model → "Oblique Expert"
2. Create new manual project (oblique dataset)
3. Select ai_input_mode="exif_only"
4. Select "Oblique Expert" as pretrained model
5. Training uses learned parameter predictions

Lineage preserved:
  - Which 15 projects contributed
  - Average performance (reward_mean=0.08)
  - Shared config used across pipeline
  - Can understand model's training diversity
```

---

## Summary

✅ **Both models now have complete lineage tracking**  
✅ **Both store configs** (gsplat: per-run, learner: shared)  
✅ **Both track contributors** (gsplat: runs, learner: projects)  
✅ **Both have provenance summaries**  
✅ **Both compatible with UI filtering** (ai_profile)  
✅ **Both registered in global index**  

**Differences are intentional:**
- Config approach matches usage pattern (per-run vs shared)
- Contributor granularity matches what matters (runs vs projects)
- Artifact formats match content type (pytorch vs json)

**Result:** Complete parity with appropriate specialization! 🎯
