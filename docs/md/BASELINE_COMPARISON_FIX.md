# Baseline Comparison Fix + Eval Interval Config

**Date:** 2026-04-23  
**Status:** ✅ **IMPLEMENTED**

---

## Fix 1: Baseline Comparison Logic (CRITICAL)

### Problem

**Before:**
```python
# Find run's best step by quality
t_run_best = step_16000  # Best PSNR/SSIM/LPIPS

# Find baseline comparison step (WRONG!)
def _anchor_step(target: int):
    ge = [s for s in b_steps if s >= target]
    return min(ge) if ge else max(b_steps)

b_best_anchor = _anchor_step(16000)  # Find baseline step >= 16000
s_base = b_by_step[b_best_anchor]["s"]

# Compares run's BEST QUALITY vs baseline's STEP-MATCHED (not best quality!)
```

**Example of the problem:**
```
Run (AI-guided):
  Step 14000: quality=59.5
  Step 16000: quality=61.2 ← BEST quality (selected)
  Step 18000: quality=60.8

Baseline (preset):
  Step 14000: quality=55.0
  Step 16000: quality=57.0 ← Anchored here (step >= 16000)
  Step 18000: quality=58.5 ← Actually BEST quality!

OLD: Compare run's best (16000) vs baseline's step-matched (16000)
  reward = 61.2 - 57.0 = +4.2

CORRECT: Compare run's best (16000) vs baseline's best (18000)
  reward = 61.2 - 58.5 = +2.7

Issue: Artificially inflated reward because baseline wasn't at its best!
```

---

### Solution

**After:**
```python
# Find run's best quality step
run_quality_scores = {step: quality_composite for step in eval_steps}
t_run_best = max(run_quality_scores, key=run_quality_scores.get)

# Find baseline's best quality step (INDEPENDENT!)
baseline_quality_scores = {step: quality_composite for step in b_steps}
t_base_best = max(baseline_quality_scores, key=baseline_quality_scores.get)

# Compare best-to-best
s_run_best = by_step[t_run_best]["s"]
s_base_best = b_by_step[t_base_best]["s"]

reward = s_run_best - s_base_best
```

**Why this is correct:**
- ✅ Both sides optimized for quality (fair comparison)
- ✅ Answers: "Did AI beat baseline's best performance?"
- ✅ Accurate reward signal for learning
- ✅ No artificial inflation/deflation

---

### Implementation

**Files Modified:**
1. [contextual_continuous_learner.py](d:\bimba3d-re\bimba3d_backend\worker\ai_input_modes\contextual_continuous_learner.py) (Lines 580-603)
2. [continuous_learner.py](d:\bimba3d-re\bimba3d_backend\worker\ai_input_modes\continuous_learner.py) (Lines 337-360)
3. [learner.py](d:\bimba3d-re\bimba3d_backend\worker\ai_input_modes\learner.py) (Lines 313-336)

**Changes:**
```python
# NEW: Track quality scores for baseline
baseline_quality_scores: dict[int, float] = {}
for idx, row in enumerate(baseline_rows):
    step = int(row["step"])
    q = 0.4 * b_psnr_norm[idx] + 0.3 * b_ssim_norm[idx] + 0.3 * b_lpips_norm[idx]
    # ... calculate s ...
    baseline_quality_scores[step] = q

# NEW: Find baseline's best quality step (not anchored!)
t_base_best = int(max(baseline_quality_scores.keys(), key=lambda s: baseline_quality_scores[s]))
t_base_end = int(max(b_steps))

s_base_best = b_by_step[t_base_best]["s"]
s_base_end = b_by_step[t_base_end]["s"]
s_base = max(s_base_best, s_base_end)

reward_signal = s_run - s_base
```

**Updated metadata:**
```python
baseline_comparison = {
    "run_best_step": t_eval_best,        # Run's best quality step
    "run_end_step": t_end,               # Run's final step
    "baseline_best_step": t_base_best,   # Baseline's best quality step
    "baseline_end_step": t_base_end,     # Baseline's final step
    "s_run_best": s_best,                # Run's best score
    "s_run_end": s_end,                  # Run's end score
    "s_base_best": s_base_best,          # Baseline's best score
    "s_base_end": s_base_end,            # Baseline's end score
    "s_run": s_run,                      # Run's final (max of best/end)
    "s_base": s_base,                    # Baseline's final (max of best/end)
    "reward": reward_signal,             # s_run - s_base
    "score_weights": {
        "quality": 0.75,
        "time": 0.25,
    },
}
```

---

## Fix 2: Eval Interval Configuration

### Problem

**Before:**
- ✅ Manual projects: eval_interval configurable in ProcessTab
- ❌ Pipeline: eval_interval hardcoded (line 43: `const [evalInterval, setEvalInterval] = useState(1000)`)
- ❌ Not exposed in UI

**Impact:**
- Users can't adjust evaluation granularity for pipelines
- Since reward calculation now depends on quality metrics at eval steps, this matters more!

---

### Solution

**After:**
- ✅ Added eval_interval input field to Step 2 of pipeline wizard
- ✅ Added helpful description and tooltip
- ✅ Added min/max/step validation
- ✅ Users can now test different eval frequencies

**UI Changes:** [TrainingPipelinePage.tsx](d:\bimba3d-re\bimba3d_frontend\src\pages\TrainingPipelinePage.tsx) (Lines 361-382)

```tsx
<div>
  <label style={{ display: "block", marginBottom: "5px" }}>
    Eval Interval:
    <span style={{ fontSize: "11px", color: "#666", fontWeight: "normal", marginLeft: "5px" }}>
      (Quality evaluation frequency)
    </span>
  </label>
  <input
    type="number"
    value={evalInterval}
    onChange={(e) => setEvalInterval(Number(e.target.value))}
    style={{ width: "100%", padding: "8px" }}
    min={100}
    max={5000}
    step={100}
  />
  <p style={{ fontSize: "11px", color: "#888", marginTop: "3px" }}>
    More frequent = better quality tracking. Default: 1000
  </p>
</div>
```

---

### Why This Matters

**Current reward calculation depends on eval steps:**

```python
# Find best quality step
for row in eval_rows:  # Only evaluated steps have PSNR/SSIM/LPIPS!
    step = int(row["step"])
    psnr = row["convergence_speed"]
    ssim = row["sharpness_mean"]
    lpips = row["lpips_mean"]
    quality_score = psnr + ssim + (1 - lpips)

t_best = max(quality_scores, key=quality_scores.get)
```

**More eval points = better quality detection:**

| eval_interval | max_steps | Eval Points | Detection Quality |
|---------------|-----------|-------------|-------------------|
| 1500 | 5000 | 1500, 3000, 4500 | 3 points (coarse) |
| 1000 | 5000 | 1000, 2000, 3000, 4000, 5000 | 5 points (default) |
| 500 | 5000 | 500, 1000, 1500, ..., 5000 | 10 points (fine) |
| 250 | 5000 | 250, 500, 750, ..., 5000 | 20 points (very fine) |

**Tradeoffs:**
- ✅ More points: Better best-step detection, more accurate curves
- ⚠️ More points: Slower training (PSNR/SSIM/LPIPS calculation overhead)
- **Recommended:** 500-1000 for good balance

---

## Testing

### Test 1: Baseline Comparison

**Setup:**
1. Create project with AI mode
2. Configure: `ai_input_mode=exif_only`, `ai_selector_strategy=contextual_continuous`
3. Run training with baseline

**Check `input_mode_learning_results.json`:**
```json
{
  "baseline_comparison": {
    "run_best_step": 16000,       // Run's best quality step
    "baseline_best_step": 18000,  // Baseline's best quality step (NOT same!)
    "s_run_best": 0.85,
    "s_base_best": 0.78,
    "reward": 0.07
  }
}
```

**Verify:**
- ✅ `run_best_step` matches step with highest run PSNR/SSIM
- ✅ `baseline_best_step` matches step with highest baseline PSNR/SSIM
- ✅ These can be different steps (independent optimization)
- ✅ Reward = s_run_best - s_base_best

---

### Test 2: Eval Interval

**Setup:**
1. Create pipeline
2. In Step 2, set `eval_interval = 500`
3. Set `max_steps = 5000`
4. Start pipeline

**Expected:**
- Evaluations at: 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000
- 10 quality data points per run
- Better best-step detection

**Try different values:**
- `eval_interval = 250`: Very fine (20 points) - slower but most accurate
- `eval_interval = 500`: Fine (10 points) - good balance
- `eval_interval = 1000`: Default (5 points) - fast
- `eval_interval = 1500`: Coarse (3 points) - fastest but may miss peaks

---

## Summary

### Fix 1: Baseline Comparison

**Before:** Anchored baseline step to run's best step (incorrect)  
**After:** Find baseline's best quality step independently (correct)  
**Impact:** Accurate reward signals, better learning  

**Changed:**
- ✅ Removed `_anchor_step()` function
- ✅ Added `baseline_quality_scores` tracking
- ✅ Find `t_base_best` by quality (not step number)
- ✅ Updated metadata with more detail

---

### Fix 2: Eval Interval Config

**Before:** Hardcoded in pipeline (1000 steps), not adjustable  
**After:** User-configurable in Step 2 UI with helpful hints  
**Impact:** Users can optimize evaluation granularity for testing  

**Changed:**
- ✅ Added description tooltip
- ✅ Added validation (min=100, max=5000, step=100)
- ✅ Added helper text explaining tradeoffs

---

## Result

✅ **Correct baseline comparison** (best-to-best, not step-matched)  
✅ **Configurable eval interval** (users can test different granularities)  
✅ **Better quality detection** (more eval points option)  
✅ **Accurate learning** (fair reward signals)

Both fixes are now live! 🎯
