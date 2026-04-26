# Remaining Issues - Baseline Comparison & Eval Interval Config

**Date:** 2026-04-23  
**Status:** 🔧 **NEEDS FIX**

---

## Issue 1: Baseline Comparison Logic (CRITICAL)

### Problem

**Current code:**
```python
# Find best step in run (by quality)
t_eval_best = step_with_best_quality  # e.g., step 16000

# Find baseline comparison step (WRONG!)
def _anchor_step(target: int) -> int:
    ge = [s for s in b_steps if s >= target]
    return int(min(ge)) if ge else int(max(b_steps))

b_best_anchor_step = _anchor_step(t_eval_best)  # Find baseline step >= 16000
s_base_best = b_by_step[b_best_anchor_step]["s"]
```

**The issue:**
- We find run's best by QUALITY (step 16000 has best PSNR/SSIM)
- We find baseline's comparison by STEP NUMBER (step >= 16000)
- **These are NOT comparable!**

**Example:**
```
Run steps:
  Step 14000: quality=59.5
  Step 16000: quality=61.2 ← BEST quality
  Step 18000: quality=60.8

Baseline steps:
  Step 14000: quality=55.0
  Step 16000: quality=57.0 ← Anchored here (step >= 16000)
  Step 18000: quality=58.5 ← Actually BEST quality!

Current: Compares run's BEST (16000) vs baseline's STEP-MATCHED (16000)
Correct: Should compare run's BEST vs baseline's BEST
```

---

### Solution

**Option 1: Compare Best to Best (Recommended)**

```python
# Find best quality step in run
run_quality_scores = {step: quality_composite for step in eval_steps}
t_run_best = max(run_quality_scores, key=run_quality_scores.get)

# Find best quality step in baseline (independently!)
baseline_quality_scores = {step: quality_composite for step in b_steps}
t_base_best = max(baseline_quality_scores, key=baseline_quality_scores.get)

# Compare best-to-best
s_run_best = by_step[t_run_best]["s"]
s_base_best = b_by_step[t_base_best]["s"]

reward = s_run_best - s_base_best
```

**Why:** Both sides optimized for quality, fair comparison.

---

**Option 2: Compare at Same Step (Time-matched)**

```python
# Find common evaluated steps
common_steps = set(eval_steps) & set(b_steps)

# For each common step, calculate improvement
improvements = {}
for step in common_steps:
    s_run = by_step[step]["s"]
    s_base = b_by_step[step]["s"]
    improvements[step] = s_run - s_base

# Best improvement across all steps
best_step = max(improvements, key=improvements.get)
reward = improvements[best_step]
```

**Why:** Fair time comparison (same training step), shows improvement trajectory.

---

**Option 3: Area Under Curve (Most comprehensive)**

```python
# Calculate total improvement over all steps
total_improvement = 0
for step in common_steps:
    improvement = by_step[step]["s"] - b_by_step[step]["s"]
    total_improvement += improvement

reward = total_improvement / len(common_steps)  # Average improvement
```

**Why:** Considers entire training trajectory, not just single point.

---

### Recommendation: **Option 1 (Best-to-Best)**

**Rationale:**
1. ✅ Aligns with goal: "Match or exceed baseline quality"
2. ✅ Simple: One clear comparison point
3. ✅ Fair: Both sides get their best performance
4. ✅ Matches user expectation: "Did I beat baseline?"

**Implementation:**
```python
# In contextual_continuous_learner.py, continuous_learner.py, learner.py

# After calculating by_step and b_by_step:

# Find run's best quality step (already done above)
t_run_best = t_eval_best  # Already set to best quality step

# Find baseline's best quality step (NEW!)
if baseline_rows:
    baseline_quality_scores = {}
    for idx, row in enumerate(baseline_rows):
        step = int(row["step"])
        q = 0.4 * b_psnr_norm[idx] + 0.3 * b_ssim_norm[idx] + 0.3 * b_lpips_norm[idx]
        baseline_quality_scores[step] = q
    
    t_base_best = max(baseline_quality_scores, key=baseline_quality_scores.get)
else:
    t_base_best = None

# Compare best-to-best
s_run_best = by_step[t_run_best]["s"]
s_run_end = by_step[t_end]["s"]
s_run = max(s_run_best, s_run_end)  # Keep end comparison as fallback

if baseline_rows and t_base_best:
    s_base_best = b_by_step[t_base_best]["s"]
    s_base_end = b_by_step[b_steps[-1]]["s"]
    s_base = max(s_base_best, s_base_end)
    
    reward_signal = s_run - s_base
    
    baseline_comparison = {
        "run_best_step": t_run_best,
        "baseline_best_step": t_base_best,
        "s_run_best": s_run_best,
        "s_run_end": s_run_end,
        "s_base_best": s_base_best,
        "s_base_end": s_base_end,
        "s_run": s_run,
        "s_base": s_base,
        "reward": reward_signal,
        "score_weights": {
            "quality": 0.75,
            "time": 0.25,
        },
    }
```

---

## Issue 2: Eval Interval Configuration

### Current Status

**✅ Project Config (Manual Projects):**
- Has `eval_interval` in [ProcessTab.tsx](d:\bimba3d-re\bimba3d_frontend\src\components\tabs\ProcessTab.tsx)
- Default: 1000 steps
- User can adjust per project

**❌ Pipeline Shared Config:**
- Has `eval_interval` in state (line 43)
- **But NOT exposed in UI!**
- Users can't adjust it for pipeline

---

### Solution: Add Eval Interval to Pipeline UI

**Step 2 of Pipeline Wizard (Shared Configuration)**

Add input field for `eval_interval`:

```tsx
// In TrainingPipelinePage.tsx, Step 2 section

<div className="space-y-4">
  <h3 className="text-lg font-semibold">Shared Training Configuration</h3>
  <p className="text-sm text-gray-600">
    These parameters apply to all projects in this pipeline.
  </p>

  {/* Existing fields... */}
  
  <div>
    <label className="block text-sm font-medium text-gray-700">
      Max Steps
    </label>
    <input
      type="number"
      value={maxSteps}
      onChange={(e) => setMaxSteps(parseInt(e.target.value) || 5000)}
      className="..."
    />
  </div>

  {/* NEW FIELD */}
  <div>
    <label className="block text-sm font-medium text-gray-700">
      Evaluation Interval
      <span className="ml-2 text-xs text-gray-500">
        (Steps between quality evaluations)
      </span>
    </label>
    <input
      type="number"
      value={evalInterval}
      onChange={(e) => setEvalInterval(parseInt(e.target.value) || 1000)}
      className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
      min={100}
      max={5000}
      step={100}
    />
    <p className="mt-1 text-xs text-gray-500">
      More frequent = better quality tracking, slower training.
      Default: 1000 steps.
    </p>
  </div>

  <div>
    <label className="block text-sm font-medium text-gray-700">
      Log Interval
    </label>
    <input
      type="number"
      value={logInterval}
      onChange={(e) => setLogInterval(parseInt(e.target.value) || 100)}
      className="..."
    />
  </div>

  {/* Rest of fields... */}
</div>
```

---

### Why This Matters

**Current situation:**
```
eval_interval: 1000 steps
max_steps: 5000
Evaluations: 1000, 2000, 3000, 4000, 5000 = 5 eval points
```

**If user wants more granular quality tracking:**
```
eval_interval: 500 steps
max_steps: 5000
Evaluations: 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000 = 10 eval points
```

**Benefits:**
- ✅ More eval points → better "best quality step" detection
- ✅ Better quality curves for analysis
- ✅ More accurate reward calculation

**Tradeoffs:**
- ⚠️ More evaluations = slower training (PSNR/SSIM/LPIPS calculation)
- ⚠️ Typical: 500-1000 is good balance

---

## Summary

### Issue 1: Baseline Comparison (CRITICAL - Must Fix)

**Current:** Anchors baseline step to run's best step number  
**Problem:** Not comparable (run optimized by quality, baseline by step)  
**Solution:** Find baseline's best quality step independently  
**Impact:** Correct reward signals, better learning  

---

### Issue 2: Eval Interval Config (Enhancement)

**Current:** Hardcoded in pipeline (1000 steps)  
**Problem:** Users can't adjust granularity  
**Solution:** Add eval_interval input to Step 2 of pipeline wizard  
**Impact:** Users can test with more/fewer eval points  

---

## Implementation Order

1. **CRITICAL: Fix baseline comparison** (Lines 582-610 in all 3 learner files)
2. **Enhancement: Add eval_interval UI** (TrainingPipelinePage.tsx Step 2)

Would you like me to implement these fixes now?
