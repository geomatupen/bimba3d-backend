# Pipeline AI Learning Table Implementation

**Date:** 2026-04-24  
**Status:** ✅ **COMPLETED**

---

## Summary

Implemented AI Learning Table in pipeline details page and addressed simulation issues.

---

## Changes Made

### Step 1: Document Simulation Issue

**File:** [`training_pipeline_orchestrator.py:273-305`](d:\bimba3d-re\bimba3d_backend\app\services\training_pipeline_orchestrator.py)

**Issue:** The orchestrator was using `_simulate_training_run()` which generates random rewards instead of actual training.

**Status:** ⚠️ **DOCUMENTED** (Not yet integrated with actual training)

**What needs to be done:**
```python
# TODO - ACTUAL TRAINING INTEGRATION:
# This function needs to:
# 1. Call the same training pipeline that project sessions use
# 2. Integrate with worker system (docker or local)
# 3. Wait for training completion (monitor status.json)
# 4. Read results from:
#    - outputs/engines/gsplat/input_mode_learning_results.json (for rewards)
#    - outputs/engines/gsplat/stats/*.csv (for eval metrics)
# 5. Calculate reward using the same logic as learner.py:
#    - For baseline (phase 1): reward = None
#    - For AI phases (2+): reward = s_run - s_base
```

**Current behavior:**
- ✅ Phase 1 (baseline): Returns `None` for rewards (correct)
- ✅ Phase 2+ (AI learning): Returns simulated rewards between -0.2 and 0.3
- ✅ Success rate: 90% (random simulation)
- ⚠️ **Not running actual training** - just simulating for testing pipeline orchestration

**Fixed:**
- ✅ Baseline phase now correctly returns `None` for rewards
- ✅ Statistics calculation fixed to only include non-None rewards
- ✅ Success rate now based on run status, not reward values

---

### Step 2: Create Pipeline Learning Table API Endpoint

**File:** [`training_pipeline.py:368-487`](d:\bimba3d-re\bimba3d_backend\app\api\training_pipeline.py)

**New Endpoint:** `GET /api/training-pipeline/{pipeline_id}/learning-table`

**What it does:**
1. Loads pipeline configuration
2. Iterates through all project directories in pipeline folder
3. Scans each project's runs directory
4. Reads `input_mode_learning_results.json` from each run
5. Aggregates all learning data into unified table
6. Returns sorted results by project name and run ID

**Response format:**
```json
{
  "pipeline_id": "pipeline_abc123",
  "pipeline_name": "training_2026-04-23",
  "rows": [
    {
      "project_name": "podoli_oblique",
      "run_id": "run_001",
      "run_name": "baseline_run",
      "is_baseline_row": true,
      "selected_preset": "balanced",
      "best_loss": 0.0123,
      "best_loss_step": 15000,
      "best_psnr": 28.5,
      "best_psnr_step": 14000,
      "best_ssim": 0.89,
      "best_ssim_step": 14500,
      "best_lpips": 0.12,
      "best_lpips_step": 14000,
      "s_best": 0.85,
      "s_run": 0.82,
      "s_base": 0.80,
      "reward": 0.02,
      "learned_input_params": {...},
      "remarks": "..."
    }
  ],
  "total_rows": 45
}
```

**Data collected per run:**
- Project name and run identification
- AI mode and baseline reference
- Selected preset and learned parameters
- Quality metrics (Loss, PSNR, SSIM, LPIPS) - best and final values with steps
- Learning scores (S_best, S_end, S_run, S_base_best, S_base_end, S_base)
- Score components (l, q, t, s) for best and end states
- Reward and remarks

---

### Step 3: Add AI Learning Table to Pipeline Details Page

**File:** [`PipelineDetailsPage.tsx`](d:\bimba3d-re\bimba3d_frontend\src\pages\PipelineDetailsPage.tsx)

**Changes:**

1. **Added interface** (Lines 29-87):
   - `AILearningTableRow` with all learning data fields

2. **Added state** (Lines 87-88):
   ```typescript
   const [learningRows, setLearningRows] = useState<AILearningTableRow[]>([]);
   const [learningLoading, setLearningLoading] = useState(false);
   ```

3. **Added data loading** (Lines 102-112):
   ```typescript
   const loadLearningTable = async () => {
     if (!id) return;
     setLearningLoading(true);
     try {
       const res = await api.get(`/training-pipeline/${id}/learning-table`);
       setLearningRows(res.data.rows || []);
     } catch (err) {
       console.error("Failed to load learning table", err);
       setLearningRows([]);
     } finally {
       setLearningLoading(false);
     }
   };
   ```

4. **Added effect hook** (Lines 125-129):
   - Loads learning table when logs tab is activated

5. **Replaced logs tab placeholder** (Lines 466-599):
   - Full AI Learning Table implementation
   - Shows all runs from all projects in pipeline
   - Displays quality metrics, scores, and rewards
   - Sticky header for easy scrolling
   - Baseline rows highlighted in amber
   - Refresh button to reload data
   - Loading and empty states

**Table columns:**
1. Project
2. Run (name + ID)
3. Preset
4. Learned Input Params (JSON + source)
5. Best Loss @ step
6. Final Loss @ step
7. Best PSNR @ step
8. Final PSNR @ step
9. Best SSIM @ step
10. Final SSIM @ step
11. Best LPIPS @ step
12. Final LPIPS @ step
13. Run Best (l, q, t, s)
14. Run End (l, q, t, s)
15. S Best
16. S End
17. S Run
18. S Base Best
19. S Base End
20. S Base
21. **Reward** (bold)
22. Remarks

**UI Features:**
- ✅ Sticky header stays visible while scrolling
- ✅ Baseline rows highlighted (amber background)
- ✅ Loading spinner while fetching data
- ✅ Empty state message when no data
- ✅ Refresh button
- ✅ Horizontal scrolling for wide table
- ✅ Max height with vertical scrolling
- ✅ Monospace font for learned params JSON
- ✅ Proper number formatting (6 decimals for scores, 4 for quality)

---

## Testing

1. **Start backend:**
   ```bash
   cd bimba3d_backend
   python -m bimba3d_backend.app.main
   ```

2. **Navigate to pipeline:**
   - Go to Dashboard
   - Click "Pipelines" tab
   - Click "Details" on any pipeline
   - Click "Logs" tab

3. **Expected behavior:**
   - If no runs completed: Shows "No learning data available yet"
   - If runs completed: Shows full table with all learning metrics
   - Baseline rows (Phase 1): No reward, highlighted in amber
   - AI learning rows (Phase 2+): Shows rewards, normal white background
   - Refresh button reloads data

---

## Current State

### ✅ Working:
1. Dashboard has "Projects" and "Pipelines" tabs
2. Pipelines list shows all pipelines with status, progress, actions
3. Pipeline details page with 5 tabs (Overview, Projects, Runs, Configuration, Logs)
4. AI Learning Table in Logs tab aggregates data from all projects
5. Baseline runs correctly show no rewards
6. Statistics exclude baseline runs from reward calculations
7. Frontend builds successfully

### ⚠️ Known Limitations:
1. **Simulation only**: Orchestrator doesn't actually run training yet
2. **No actual learning data**: Until training integration is complete, the learning table will be empty
3. **Projects tab**: Still shows "coming soon" placeholder

---

## Next Steps (Future Work)

### High Priority:
1. **Integrate actual training execution** in orchestrator:
   - Replace `_simulate_training_run()` with real training call
   - Monitor status.json for completion
   - Read actual results from learning_results.json
   - Calculate real rewards from quality metrics

2. **Projects tab implementation**:
   - List all projects in pipeline
   - Show per-project stats (runs, phases completed, best reward)
   - Click to navigate to project detail page

### Medium Priority:
3. **Pipeline resume from checkpoint**:
   - Save current state on pause/stop
   - Restore state on resume after backend restart
   - Track which runs completed successfully

4. **Enhanced logging**:
   - Stream real-time logs from training
   - Show current run progress
   - Display errors and warnings

### Low Priority:
5. **Export functionality**:
   - Export learning table as CSV
   - Export pipeline configuration
   - Download full run logs

---

## Files Modified

### Backend:
1. [`training_pipeline_orchestrator.py`](d:\bimba3d-re\bimba3d_backend\app\services\training_pipeline_orchestrator.py)
   - Added detailed TODO for actual training integration
   - Fixed baseline reward handling (Phase 1 returns None)

2. [`training_pipeline_storage.py`](d:\bimba3d-re\bimba3d_backend\app\services\training_pipeline_storage.py)
   - Fixed success_rate calculation (based on run status, not rewards)
   - Fixed reward statistics to exclude None values

3. [`training_pipeline.py`](d:\bimba3d-re\bimba3d_backend\app\api\training_pipeline.py)
   - Added `/list` endpoint (moved before `/{pipeline_id}` for correct routing)
   - Added `/{pipeline_id}/learning-table` endpoint

### Frontend:
1. [`Dashboard.tsx`](d:\bimba3d-re\bimba3d_frontend\src\pages\Dashboard.tsx)
   - Added tabs for "Projects" and "Pipelines"
   - Integrated PipelinesListPage component

2. [`PipelineDetailsPage.tsx`](d:\bimba3d-re\bimba3d_frontend\src\pages\PipelineDetailsPage.tsx)
   - Added AILearningTableRow interface
   - Added state and loading for learning table
   - Implemented full AI Learning Table in Logs tab

3. [`PipelinesListPage.tsx`](d:\bimba3d-re\bimba3d_frontend\src\pages\PipelinesListPage.tsx)
   - Fixed refresh button syntax error

4. [`App.tsx`](d:\bimba3d-re\bimba3d_frontend\src\App.tsx)
   - Added route for `/pipelines/:id`

---

## API Endpoints Summary

### Already Existed:
- `GET /api/training-pipeline/list` - List all pipelines
- `GET /api/training-pipeline/{id}` - Get pipeline details
- `POST /api/training-pipeline/{id}/start` - Start pipeline
- `POST /api/training-pipeline/{id}/pause` - Pause pipeline
- `POST /api/training-pipeline/{id}/resume` - Resume pipeline
- `POST /api/training-pipeline/{id}/stop` - Stop pipeline
- `GET /api/training-pipeline/{id}/runs` - Get run history
- `DELETE /api/training-pipeline/{id}` - Delete pipeline

### Newly Added:
- `GET /api/training-pipeline/{id}/learning-table` - Get aggregated AI learning data

---

## Reward Calculation (Current vs Intended)

### Current (Simulation):
```python
# Phase 1 (Baseline)
reward = None  # ✅ Correct

# Phase 2+ (AI Learning)
reward = random.uniform(-0.2, 0.3)  # ⚠️ Simulated
```

### Intended (Actual):
```python
# Phase 1 (Baseline)
reward = None  # No learning, just reference run

# Phase 2+ (AI Learning)
# Read from input_mode_learning_results.json:
reward = s_run - s_base

# Where:
# s_run = composite score of current run
# s_base = composite score of baseline
# s = 0.75 * q + 0.25 * t  (75% quality, 25% time)
# q = 0.4 * psnr_norm + 0.3 * ssim_norm + 0.3 * lpips_norm
```

---

## Status: ✅ Ready for Testing

The AI Learning Table is fully implemented and ready to display data once actual training runs complete. Currently shows simulated data from the orchestrator.

**To see real data:**
1. Pipeline orchestrator needs training integration
2. Complete actual training runs (Phase 1 baseline + Phase 2+ learning)
3. Runs will generate `input_mode_learning_results.json` files
4. Learning table will automatically aggregate and display the data

**Current test scenario:**
- Create a pipeline with 2-3 projects
- Start the pipeline
- Orchestrator will simulate runs (2 seconds each)
- Check pipeline status and mean reward (will be -0.2 to 0.3)
- Learning table will be empty until real training writes learning results
