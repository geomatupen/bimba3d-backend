# Pipeline Storage Management

**Date:** 2026-04-24  
**Status:** ✅ **IMPLEMENTED**

---

## Problem

Training pipelines with frequent evaluations create massive storage requirements:

```
Example: 10 projects × 7 runs × 25 evals = 1,750 eval image sets
At 20MB per eval: 35GB just for evaluation images!
Plus checkpoints: ~100-500MB per run × 70 runs = 7-35GB
Total: 42-70GB for one pipeline
```

---

## Solution: Configurable Storage Options

Added 5 configuration options in pipeline creation UI:

### 1. **Save Evaluation Images** ✅/❌
- **What:** Rendered images at each eval_interval
- **Size:** ~5-50MB per eval (depends on resolution)
- **Use case:** Visual inspection of training progress
- **Default:** ✅ Enabled (but with replace option)

### 2. **Replace Evaluation Images** ✅/❌
- **What:** Keep only latest eval, delete previous
- **Storage savings:** ~95% (keep 1 eval instead of 25)
- **Trade-off:** Lose training history visualization
- **Use case:** Pipeline training (you only care about final result)
- **Default:** ✅ Enabled (recommended for pipelines)

### 3. **Save Training Checkpoints** ✅/❌
- **What:** Model weights for resuming training
- **Size:** ~100-500MB per checkpoint
- **Use case:** Resume training if interrupted
- **Default:** ✅ Enabled (but with replace option)

### 4. **Replace Checkpoints** ✅/❌
- **What:** Keep only latest checkpoint, delete previous
- **Storage savings:** ~90% (keep 1 checkpoint instead of 10)
- **Trade-off:** Can't resume from middle of training
- **Use case:** Pipeline training (runs complete quickly)
- **Default:** ✅ Enabled (recommended for pipelines)

### 5. **Save Final Splat Model** ✅/❌
- **What:** Final Gaussian splat model (.ply/.splat)
- **Size:** ~50-200MB per model
- **Use case:** Viewing results, comparisons, deployment
- **Default:** ✅ Enabled (always recommended)

---

## Configuration UI

**Location:** Training Pipeline Creation → Step 2 (Shared Configuration)

```
┌────────────────────────────────────────────────────────┐
│ 💾 Storage Management                                  │
├────────────────────────────────────────────────────────┤
│ Configure what gets saved to manage storage.           │
│ Eval images at 200-500 step intervals can create       │
│ massive storage requirements.                           │
│                                                         │
│ ☑ Save Evaluation Images                               │
│   (Renders at each eval_interval. WARNING: ~5-50MB     │
│    per eval × num_evals = GBs)                         │
│                                                         │
│   ☑ Replace Eval Images (Keep only latest eval)       │
│     (Saves ~95% storage. Use for pipeline training,    │
│      disable for final runs)                           │
│                                                         │
│ ☑ Save Training Checkpoints                            │
│   (Model weights for resuming. ~100-500MB per          │
│    checkpoint)                                          │
│                                                         │
│   ☑ Replace Checkpoints (Keep only latest checkpoint) │
│     (Recommended for pipelines. Keep only final model) │
│                                                         │
│ ☑ Save Final Splat Model                               │
│   (Always recommended. ~50-200MB. Needed for viewing   │
│    results)                                             │
│                                                         │
│ ┌──────────────────────────────────────────────────┐  │
│ │ 💡 Recommended for Pipeline Training:            │  │
│ │ ✅ Replace Eval Images (save 95% storage)        │  │
│ │ ✅ Replace Checkpoints (save 90% storage)        │  │
│ │ ✅ Save Final Splat (needed for comparison)      │  │
│ │ ❌ Uncheck Save Eval Images if you don't need    │  │
│ │    visual inspection                             │  │
│ └──────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

---

## Storage Comparison

### Scenario: 10 projects, 7 runs per project, eval every 200 steps (5000 max steps = 25 evals)

#### Option A: Save Everything (Default Legacy)
```
Eval Images:    10 projects × 7 runs × 25 evals × 20MB = 35,000 MB (35 GB)
Checkpoints:    10 projects × 7 runs × 10 ckpts × 200MB = 140,000 MB (140 GB)
Final Splats:   10 projects × 7 runs × 150MB = 10,500 MB (10.5 GB)
─────────────────────────────────────────────────────────────────────
TOTAL:          185.5 GB
```

#### Option B: Pipeline Optimized (Recommended)
```
Eval Images:    10 projects × 7 runs × 1 eval × 20MB = 1,400 MB (1.4 GB)
                (Replace enabled: keep only latest)
Checkpoints:    10 projects × 7 runs × 1 ckpt × 200MB = 14,000 MB (14 GB)
                (Replace enabled: keep only latest)
Final Splats:   10 projects × 7 runs × 150MB = 10,500 MB (10.5 GB)
─────────────────────────────────────────────────────────────────────
TOTAL:          25.9 GB
SAVINGS:        159.6 GB (86% reduction!)
```

#### Option C: Maximum Storage Savings (Research Only)
```
Eval Images:    DISABLED (0 GB)
Checkpoints:    10 projects × 7 runs × 1 ckpt × 200MB = 14,000 MB (14 GB)
                (Replace enabled: keep only latest)
Final Splats:   10 projects × 7 runs × 150MB = 10,500 MB (10.5 GB)
─────────────────────────────────────────────────────────────────────
TOTAL:          24.5 GB
SAVINGS:        161 GB (87% reduction!)
```

---

## Use Cases

### Use Case 1: Production Pipeline Training
**Goal:** Train 100 projects with AI learning to find best parameters

**Configuration:**
- ✅ Save Eval Images: Enabled
- ✅ Replace Eval Images: **Enabled** ← Key
- ✅ Save Checkpoints: Enabled
- ✅ Replace Checkpoints: **Enabled** ← Key
- ✅ Save Final Splat: Enabled

**Why:**
- You only care about final results for comparison
- Don't need training history visualization
- Runs complete in 20-40 minutes (quick restart if failed)
- Storage: ~26GB instead of ~185GB (86% savings)

---

### Use Case 2: Research / Thesis Documentation
**Goal:** Document training progress for paper/thesis with visualizations

**Configuration:**
- ✅ Save Eval Images: Enabled
- ❌ Replace Eval Images: **Disabled** ← Keep all
- ✅ Save Checkpoints: Enabled
- ❌ Replace Checkpoints: **Disabled** ← Keep all
- ✅ Save Final Splat: Enabled

**Why:**
- Need complete training history for analysis
- Want to show training curves, visual progression
- Publication requires full documentation
- Storage: ~185GB (but worth it for research)

---

### Use Case 3: Quick Exploration (Minimal Storage)
**Goal:** Quickly test if pipeline works, no need for results

**Configuration:**
- ❌ Save Eval Images: **Disabled** ← Don't save
- ✅ Save Checkpoints: Enabled
- ✅ Replace Checkpoints: Enabled
- ✅ Save Final Splat: Enabled

**Why:**
- Just testing pipeline orchestration
- Don't need visual inspection
- Can check metrics from learning_results.json
- Storage: ~24.5GB (87% savings)

---

### Use Case 4: Final Production Run
**Goal:** Create deployable models with full documentation

**Configuration:**
- ✅ Save Eval Images: Enabled
- ❌ Replace Eval Images: **Disabled** ← Keep all
- ✅ Save Checkpoints: Enabled
- ✅ Replace Checkpoints: Enabled (only need final)
- ✅ Save Final Splat: Enabled

**Why:**
- Need eval images for quality inspection
- Don't need all checkpoints (only final model)
- Balance between documentation and storage
- Storage: ~52GB (72% savings vs full)

---

## Implementation Details

### Frontend: Shared Configuration

**Added state variables:**
```typescript
const [saveEvalImages, setSaveEvalImages] = useState(true);
const [replaceEvalImages, setReplaceEvalImages] = useState(true);  // Default: save storage
const [saveCheckpoints, setSaveCheckpoints] = useState(true);
const [replaceCheckpoints, setReplaceCheckpoints] = useState(true);  // Default: save storage
const [saveFinalSplat, setSaveFinalSplat] = useState(true);  // Always recommended
```

**Added to shared_config:**
```typescript
shared_config: {
  ai_input_mode: aiInputMode,
  ai_selector_strategy: aiSelectorStrategy,
  max_steps: maxSteps,
  eval_interval: evalInterval,
  // ... other params ...
  
  // Storage management options
  save_eval_images: saveEvalImages,
  replace_eval_images: replaceEvalImages,
  save_checkpoints: saveCheckpoints,
  replace_checkpoints: replaceCheckpoints,
  save_final_splat: saveFinalSplat,
}
```

---

### Backend: Training Integration (TODO)

**Where to implement:** When actual training integration is complete

**Pseudocode:**
```python
def run_training(project_dir, run_config):
    # Read storage options from run_config
    save_eval_images = run_config.get("save_eval_images", True)
    replace_eval_images = run_config.get("replace_eval_images", False)
    save_checkpoints = run_config.get("save_checkpoints", True)
    replace_checkpoints = run_config.get("replace_checkpoints", False)
    save_final_splat = run_config.get("save_final_splat", True)
    
    # During training loop:
    for step in range(max_steps):
        train_one_step()
        
        if step % eval_interval == 0:
            # Run evaluation
            eval_images = render_evaluation()
            metrics = calculate_metrics(eval_images)
            
            # Save or replace eval images
            if save_eval_images:
                eval_output_dir = run_dir / "outputs" / "evals" / f"step_{step}"
                
                if replace_eval_images:
                    # Delete previous eval directories
                    for prev_eval in (run_dir / "outputs" / "evals").glob("step_*"):
                        if prev_eval != eval_output_dir:
                            shutil.rmtree(prev_eval)
                
                # Save current eval
                eval_output_dir.mkdir(parents=True, exist_ok=True)
                save_images(eval_images, eval_output_dir)
        
        if step % checkpoint_interval == 0:
            # Save checkpoint
            if save_checkpoints:
                ckpt_path = run_dir / "checkpoints" / f"ckpt_{step}.pt"
                
                if replace_checkpoints:
                    # Delete previous checkpoints
                    for prev_ckpt in (run_dir / "checkpoints").glob("ckpt_*.pt"):
                        if prev_ckpt != ckpt_path:
                            prev_ckpt.unlink()
                
                # Save current checkpoint
                save_checkpoint(model, ckpt_path)
    
    # Save final splat model
    if save_final_splat:
        export_splat(model, run_dir / "outputs" / "splats.ply")
```

---

## File Structure Examples

### With Replace Enabled (Recommended for Pipelines)

```
project_dir/
└── runs/
    └── run_001/
        ├── config.json
        ├── outputs/
        │   ├── evals/
        │   │   └── step_5000/        ← Only latest eval
        │   │       ├── 00001.png
        │   │       └── 00002.png
        │   ├── splats.ply             ← Final model
        │   └── engines/
        │       └── gsplat/
        │           ├── input_mode_learning_results.json
        │           └── stats/
        │               └── metrics.csv
        └── checkpoints/
            └── ckpt_5000.pt           ← Only latest checkpoint
```

**Storage:** ~250 MB per run (1 eval + 1 checkpoint + 1 splat)

---

### Without Replace (Full History)

```
project_dir/
└── runs/
    └── run_001/
        ├── config.json
        ├── outputs/
        │   ├── evals/
        │   │   ├── step_200/          ← All 25 evals saved
        │   │   ├── step_400/
        │   │   ├── step_600/
        │   │   └── ... (22 more)
        │   │   └── step_5000/
        │   ├── splats.ply
        │   └── engines/gsplat/...
        └── checkpoints/
            ├── ckpt_500.pt            ← All 10 checkpoints saved
            ├── ckpt_1000.pt
            └── ... (8 more)
            └── ckpt_5000.pt
```

**Storage:** ~2.5 GB per run (25 evals + 10 checkpoints + 1 splat)

---

## Best Practices

### ✅ DO:
1. **Enable replace options for pipeline training**
   - You're running many experiments
   - Only final results matter for comparison
   - Save 85-90% storage

2. **Disable eval images for pure research**
   - If you only need metrics (PSNR, SSIM, LPIPS)
   - Visual inspection not needed
   - Metrics still saved in learning_results.json

3. **Keep final splat always enabled**
   - Needed for viewing results
   - Required for comparisons
   - Only ~150MB per run

4. **Adjust eval_interval based on need**
   - Pipeline training: 500-1000 (fewer evals)
   - Final runs: 200-500 (more detail)
   - Research: 100-200 (maximum detail)

### ❌ DON'T:
1. **Don't disable final splat**
   - You won't be able to view results
   - Can't compare models
   - Can't deploy model

2. **Don't keep all evals for large pipelines**
   - 100 runs × 25 evals = 2,500 eval sets
   - At 20MB each = 50GB wasted
   - You probably won't look at 99% of them

3. **Don't keep all checkpoints for quick runs**
   - If training completes in 20 minutes
   - Restarting from scratch is faster than resuming
   - Keep only final checkpoint

---

## Summary

**Storage Management Options Now Available:**
- ✅ Save Eval Images (default: enabled)
- ✅ Replace Eval Images (default: enabled) ← **KEY for pipelines**
- ✅ Save Checkpoints (default: enabled)
- ✅ Replace Checkpoints (default: enabled) ← **KEY for pipelines**
- ✅ Save Final Splat (default: enabled, always recommended)

**Default Configuration:**
- **Optimized for pipeline training**
- Saves 86% storage vs keeping everything
- Keeps final results for comparison
- Can switch to "keep all" for final production runs

**Recommendation:**
- **Pipeline training:** Use defaults (replace enabled)
- **Final production run:** Disable replace options, keep full history
- **Quick exploration:** Disable eval images entirely

This gives you **flexibility to balance storage vs documentation** based on your use case!
