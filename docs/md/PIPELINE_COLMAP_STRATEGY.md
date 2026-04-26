# Pipeline COLMAP Strategy

**Date:** 2026-04-24  
**Status:** 📋 **DOCUMENTED**

---

## Critical Design Decision

**COLMAP runs ONCE per project** (during baseline phase), then **ALL subsequent runs reuse the same sparse reconstruction**.

---

## Why?

### What COLMAP Does:
1. Analyzes input images
2. Extracts features (SIFT/etc)
3. Matches features between images
4. Estimates camera poses (position + orientation)
5. Triangulates 3D points
6. Creates sparse 3D reconstruction

### COLMAP Output:
```
project_dir/
└── colmap/
    ├── database.db           # Feature matches, image pairs
    └── sparse/
        └── 0/                # Sparse reconstruction
            ├── cameras.bin   # Camera intrinsics
            ├── images.bin    # Camera poses/extrinsics
            └── points3D.bin  # Sparse 3D points
```

### Why COLMAP is Project-Specific, NOT Run-Specific:
- COLMAP output depends **only on the images**, not training parameters
- Camera poses are geometric facts about the capture setup
- Running COLMAP again on the same images gives the same result (deterministic)
- COLMAP is **slow** (5-30 minutes depending on image count)
- Gaussian splatting training uses COLMAP output as **initialization**, then optimizes Gaussians

**Analogy:** COLMAP is like measuring a room's dimensions. You measure once, then you can try different furniture arrangements (training parameters) without re-measuring the room each time.

---

## Project Directory Structure

```
pipeline_folder/
├── shared_models/              # Shared AI models across projects
│   └── contextual_continuous_selector/
│       └── exif_plus_flight_plan.json
│
├── podoli_oblique/             # Project 1
│   ├── config.json
│   ├── colmap/                 # ✅ CREATED ONCE (Phase 1)
│   │   ├── database.db
│   │   └── sparse/0/           # Camera poses, sparse points
│   └── runs/
│       ├── run_001_baseline/   # Phase 1: Creates COLMAP + training
│       │   ├── config.json
│       │   └── outputs/
│       │       └── engines/gsplat/
│       │           ├── input_mode_learning_results.json
│       │           ├── stats/
│       │           └── checkpoints/
│       ├── run_002_learning/   # Phase 2: Reuses COLMAP, different params
│       │   └── outputs/...
│       └── run_003_learning/   # Phase 3: Reuses COLMAP, different params
│           └── outputs/...
│
└── bilovec_nadir/              # Project 2 (same structure)
    ├── config.json
    ├── colmap/                 # ✅ CREATED ONCE (Phase 1)
    └── runs/
        ├── run_001_baseline/
        ├── run_002_learning/
        └── run_003_learning/
```

---

## Training Pipeline Flow

### Phase 1: Baseline (First Run)

**Purpose:** Create reference run with fixed parameters

**Steps:**
1. ✅ Check if `project_dir/colmap/sparse/0/` exists
   - If NO: **Run COLMAP** (images → sparse reconstruction)
   - If YES: Skip COLMAP (resume scenario)
2. ✅ Run gaussian splatting training
   - Use fixed preset (e.g., "balanced")
   - No AI parameter selection
   - Use colmap sparse as input
3. ✅ Save results as baseline
   - No reward calculation (baseline is reference)
   - Store metrics for future comparisons

**Key:** COLMAP is created here and **never run again** for this project.

---

### Phase 2+: AI Learning (Subsequent Runs)

**Purpose:** Learn better parameters by comparing to baseline

**Steps:**
1. ✅ Check if `project_dir/colmap/sparse/0/` exists
   - **MUST exist** (from Phase 1)
   - If missing: ERROR - cannot proceed without COLMAP
2. ⚠️ **SKIP COLMAP** - reuse existing reconstruction
3. ✅ AI selects training parameters
   - Learner model suggests densification params
   - Different from baseline preset
4. ✅ Run gaussian splatting training
   - **Same COLMAP data** as baseline
   - **Different training parameters**
   - Use colmap sparse as input (same path)
5. ✅ Calculate reward
   - Compare quality metrics to baseline
   - Reward = s_run - s_base
   - Update learner model

**Key:** Training parameters vary, COLMAP data stays constant.

---

## What Changes Between Runs?

### Same Across All Runs (Project-Specific):
- ✅ Input images
- ✅ COLMAP sparse reconstruction (cameras, poses, points)
- ✅ Image resolution
- ✅ Camera intrinsics/extrinsics

### Different Per Run (Training-Specific):
- ⚠️ Densification parameters:
  - `densify_grad_threshold`
  - `densify_from_iter`
  - `densify_until_iter`
  - `densification_interval`
- ⚠️ Optimization parameters:
  - Learning rates
  - Lambda SSIM
  - Opacity threshold
- ⚠️ Training schedule:
  - Max steps
  - Eval interval

---

## Resume Logic

### Scenario 1: Pipeline Stopped After Completing Phase 1

**State:**
```
podoli_oblique/
  ├── colmap/sparse/0/  ✅ EXISTS
  └── runs/
      └── run_001_baseline/  ✅ COMPLETED
```

**Resume behavior:**
- Skip Phase 1 (already completed)
- Start Phase 2 for this project
- Reuse existing COLMAP
- AI selects new parameters
- Run training with different params

---

### Scenario 2: Pipeline Stopped During Phase 2 Training

**State:**
```
podoli_oblique/
  ├── colmap/sparse/0/  ✅ EXISTS
  └── runs/
      ├── run_001_baseline/  ✅ COMPLETED
      └── run_002_learning/  ⚠️ PARTIAL (no learning_results.json)
```

**Resume behavior:**
- Phase 1: Skip (completed)
- Phase 2: Check if learning_results.json exists
  - If NO: **Restart this run** (partial run is discarded)
  - If YES: Skip (completed), move to next project or phase
- Reuse existing COLMAP

**Why restart?** Partial runs may have incomplete training, bad checkpoints, or corrupted state. Safer to restart from scratch.

---

### Scenario 3: Multiple Projects, One Failed

**State:**
```
Pipeline:
  - podoli_oblique/  ✅ All phases completed
  - bilovec_nadir/   ⚠️ Phase 2 failed
  - terrain_rough/   ⏸️ Not started
```

**Resume behavior:**
- podoli_oblique: Skip all (completed)
- bilovec_nadir: 
  - Phase 1: Skip (COLMAP + baseline exist)
  - Phase 2: Retry failed run
    - Reuse COLMAP
    - AI suggests new parameters (or same)
    - Retry training
- terrain_rough: Normal execution

---

## Implementation Checklist

### Backend (training_pipeline_orchestrator.py):

- [x] Check if COLMAP exists before running
  - Path: `project_dir/colmap/sparse/0/`
- [x] Run COLMAP only in Phase 1 if missing
  - Command: COLMAP feature_extractor → matcher → mapper
- [ ] Pass COLMAP path to training
  - Config: `colmap_sparse_path: str(project_dir / "colmap" / "sparse" / "0")`
- [x] Check if run already completed (learning_results.json exists)
  - If yes: Skip run, mark as completed
  - If no: Execute run
- [ ] Handle failed runs
  - Delete partial outputs
  - Allow retry on resume

### Run Configuration:

```python
run_config = {
    # SAME across all runs (from COLMAP)
    "colmap_sparse_path": str(project_dir / "colmap" / "sparse" / "0"),
    "images_path": str(project["dataset_path"]),
    
    # DIFFERENT per run (AI selected or preset)
    "densify_grad_threshold": 0.0002,  # Phase 1: preset, Phase 2+: AI
    "densify_from_iter": 500,
    "densify_until_iter": 4000,
    "densification_interval": 100,
    "max_steps": 5000,
    "eval_interval": 1000,
    
    # Phase metadata
    "phase_number": 2,
    "pass_number": 1,
    "baseline_run_id": "run_001_baseline",  # Phase 2+ only
}
```

---

## Error Handling

### Error: COLMAP missing in Phase 2+

```python
if phase_num > 1 and not (project_dir / "colmap" / "sparse" / "0").exists():
    raise RuntimeError(
        f"COLMAP reconstruction missing for {project_name}. "
        f"Baseline phase must complete successfully before AI learning phases. "
        f"Check if Phase 1 baseline run completed."
    )
```

### Error: Baseline results missing

```python
baseline_run_dir = project_dir / "runs" / baseline_run_id
if not (baseline_run_dir / "outputs" / "engines" / "gsplat" / "input_mode_learning_results.json").exists():
    raise RuntimeError(
        f"Baseline results missing for {project_name}. "
        f"Cannot calculate reward without baseline comparison. "
        f"Run baseline phase first."
    )
```

---

## Benefits of This Approach

### Performance:
- ⚡ **Saves 5-30 minutes per run** by not repeating COLMAP
- 🚀 Enables rapid parameter exploration
- 📊 More runs = better learning

### Correctness:
- ✅ **Fair comparison** - all runs use same initialization
- ✅ Reward differences come from parameters, not COLMAP variance
- ✅ Consistent camera poses across runs

### Storage:
- 💾 COLMAP output is ~50-200MB per project
- 💾 Reusing saves 50-200MB × (num_runs - 1) per project
- 💾 Example: 10 projects × 7 runs = save ~3-14GB

---

## Testing

### Test 1: New Project, Full Pipeline

1. Create pipeline with 2 projects, 3 phases
2. Start pipeline
3. **Phase 1 (Baseline):**
   - Verify COLMAP runs
   - Verify `colmap/sparse/0/cameras.bin` created
   - Verify baseline run completes
4. **Phase 2 (Learning):**
   - Verify COLMAP does NOT run again
   - Verify training uses existing COLMAP
   - Verify different parameters used
   - Verify reward calculated
5. **Phase 3 (Multi-pass):**
   - Verify COLMAP still not run
   - Verify multiple passes reuse same COLMAP

### Test 2: Resume After Stop

1. Start pipeline, let Phase 1 complete
2. Stop pipeline during Phase 2
3. Resume pipeline
4. **Expected:**
   - Phase 1 skipped (already completed)
   - Phase 2 continues from next project
   - COLMAP not run again
   - No duplicate runs

### Test 3: COLMAP Missing Error

1. Manually delete `colmap/sparse/0/` after baseline
2. Try to run Phase 2
3. **Expected:**
   - Error: "COLMAP reconstruction missing"
   - Pipeline fails gracefully
   - Clear error message

---

## Summary

**Golden Rule:** 🏗️ **COLMAP = Blueprint. Training = Building.**

- You measure the blueprint once (COLMAP)
- Then you try different construction methods (training parameters)
- You don't re-measure the blueprint for each construction attempt

**Key Takeaways:**
1. ✅ COLMAP runs once per project (Phase 1 baseline)
2. ✅ All subsequent runs reuse the same COLMAP reconstruction
3. ✅ Only training parameters change between runs
4. ✅ Resume checks for completed runs to avoid duplicates
5. ✅ Fair comparison: same initialization, different optimization

This design enables efficient parameter exploration while maintaining fair comparisons across runs.
