# Phase-Specific Jitter Controls Added to UI

**Date:** 2026-04-23  
**Status:** ✅ **IMPLEMENTED**

---

## Problem

**Jitter settings existed in backend but were NOT configurable in UI:**

```typescript
// Backend supported (lines 24-26):
interface PhaseConfig {
  context_jitter: boolean;           // ← Exists
  context_jitter_mode: string;       // ← Exists ("uniform", "mild", "gaussian")
  shuffle_order: boolean;            // ← Exists
}

// Default values set (lines 50-86):
phases: [
  { context_jitter: false },  // Phase 1 (Baseline)
  { context_jitter: false },  // Phase 2 (Initial Exploration)
  { context_jitter: true, context_jitter_mode: "uniform" },  // Phase 3 (Multi-Pass)
]

// BUT UI only showed (lines 416-452):
- runs_per_project  ← Visible
- passes            ← Visible
- context_jitter    ← MISSING!
- context_jitter_mode ← MISSING!
- shuffle_order     ← MISSING!
```

**Result:** Users couldn't configure different jitter strategies per phase!

---

## Solution

**Added comprehensive jitter controls to each phase in Step 3 UI:**

### New UI Controls

**1. Context Jitter Toggle:**
```tsx
<label>
  <input type="checkbox" checked={phase.context_jitter} />
  Enable Context Jitter
  <span>(Vary context features for exploration)</span>
</label>
```

**2. Jitter Mode Selector (when enabled):**
```tsx
<select value={phase.context_jitter_mode}>
  <option value="uniform">Uniform (Sample from feature bounds)</option>
  <option value="mild">Mild (±10% variation)</option>
  <option value="gaussian">Gaussian (±15% with normal distribution)</option>
</select>

Helper text: "Uniform: Wide exploration | Mild: Gentle variation | Gaussian: Moderate spread"
```

**3. Shuffle Order Toggle:**
```tsx
<label>
  <input type="checkbox" checked={phase.shuffle_order} />
  Shuffle Project Order
  <span>(Randomize sequence each pass)</span>
</label>
```

---

## UI Layout

**Each phase now shows:**

```
┌─────────────────────────────────────────────────────┐
│ Phase 3: Multi-Pass Learning                        │
├─────────────────────────────────────────────────────┤
│ Runs per project: [1]     Passes: [5]               │
├─────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────────┐ │
│ │ ☑ Enable Context Jitter                         │ │
│ │   (Vary context features for exploration)       │ │
│ │                                                  │ │
│ │ Jitter Mode: [Uniform ▼]                        │ │
│ │   - Uniform (Sample from feature bounds)        │ │
│ │   - Mild (±10% variation)                       │ │
│ │   - Gaussian (±15% with normal distribution)    │ │
│ │                                                  │ │
│ │ Uniform: Wide exploration | Mild: Gentle...     │ │
│ │                                                  │ │
│ │ ☑ Shuffle Project Order                         │ │
│ │   (Randomize sequence each pass)                │ │
│ └─────────────────────────────────────────────────┘ │
│                                                      │
│ Total runs: 75                                       │
└─────────────────────────────────────────────────────┘
```

---

## Use Cases

### Case 1: Different Jitter Per Phase

**Phase 1 (Baseline):**
- ☐ Context Jitter: **Disabled**
- ☐ Shuffle Order: **Disabled**
- **Why:** Pure baseline with preset parameters

**Phase 2 (Initial Exploration):**
- ☐ Context Jitter: **Disabled** (or Mild)
- ☑ Shuffle Order: **Enabled**
- **Why:** First AI-guided training, start with real features

**Phase 3 (Multi-Pass Learning):**
- ☑ Context Jitter: **Enabled (Uniform)**
- ☑ Shuffle Order: **Enabled**
- **Why:** Aggressive exploration with wide feature variation

---

### Case 2: Conservative to Aggressive

**Phase 2:**
- ☑ Context Jitter: **Mild** (±10%)
- ☑ Shuffle Order: Enabled

**Phase 3:**
- ☑ Context Jitter: **Uniform** (full bounds)
- ☑ Shuffle Order: Enabled

**Phase 4 (optional):**
- ☑ Context Jitter: **Gaussian** (±15%)
- ☑ Shuffle Order: Enabled

---

### Case 3: No Jitter (Use Real Features)

**All phases:**
- ☐ Context Jitter: **Disabled**
- ☑ Shuffle Order: **Enabled**

**Why:** Test learning from actual project features only

---

## Jitter Mode Details

### Uniform (Wide Exploration)
```python
# Sample uniformly from feature bounds
focal_length: [8mm, 300mm] → random.uniform(8, 300)
gsd: [0.5cm, 15cm] → random.uniform(0.5, 15)
angle: [-90°, 0°] → random.uniform(-90, 0)

Result: Wide feature space exploration
Use case: Multi-pass learning, diversity
```

### Mild (Gentle Variation)
```python
# ±10% of actual value
actual_focal = 24mm
jittered_focal = actual_focal × random.uniform(0.9, 1.1)
  = 24 × [0.9, 1.1]
  = [21.6mm, 26.4mm]

Result: Small perturbations around real features
Use case: Robustness testing, slight variation
```

### Gaussian (Moderate Spread)
```python
# ±15% with normal distribution (mean=1.0, std=0.05)
actual_focal = 24mm
multiplier = random.gauss(1.0, 0.05)  # ~68% within [0.95, 1.05]
                                       # ~95% within [0.90, 1.10]
                                       # Can go to [0.85, 1.15]
jittered_focal = actual_focal × clamp(multiplier, 0.85, 1.15)

Result: Normal distribution around actual value
Use case: Natural variation, balanced exploration
```

---

## Backend Integration

**No changes needed!** Backend already supported these settings:

**[training_pipeline_orchestrator.py](d:\bimba3d-re\bimba3d_backend\app\services\training_pipeline_orchestrator.py):**
- Line 244: `run_config["context_jitter"] = phase.get("context_jitter", False)`
- Line 245: `run_config["context_jitter_mode"] = phase.get("context_jitter_mode", "uniform")`

**[gsplat_engine.py](d:\bimba3d-re\bimba3d_backend\worker\engines\gsplat_engine.py):**
- Lines 1762-1800: Context jitter implementation for all modes

**[exif_only.py, exif_plus_flight_plan.py, etc.](d:\bimba3d-re\bimba3d_backend\worker\ai_input_modes):**
- Feature extraction with jitter support

**Already working!** Just needed UI controls.

---

## Testing

### Test 1: Different Jitter Per Phase

**Setup:**
1. Create pipeline with 3 projects
2. Configure phases:
   - Phase 1: Jitter **OFF**
   - Phase 2: Jitter **Mild**
   - Phase 3: Jitter **Uniform**
3. Start pipeline

**Expected:**
- Phase 1 runs: No jitter in context vectors
- Phase 2 runs: ±10% variation
- Phase 3 runs: Wide variation (full bounds)

**Verify:** Check run logs for context feature values

---

### Test 2: Shuffle Order

**Setup:**
1. Create pipeline with 5 projects
2. Phase 3: `passes: 3`, `shuffle_order: true`

**Expected:**
```
Pass 1 order: [A, B, C, D, E]
Pass 2 order: [C, E, A, D, B] (shuffled!)
Pass 3 order: [D, A, E, B, C] (shuffled again!)
```

**Verify:** Check orchestrator logs for project execution order

---

### Test 3: No Jitter

**Setup:**
1. Disable jitter in all phases
2. Run pipeline

**Expected:**
- Model learns from actual project features only
- No artificial variation

**Use case:** When you want to learn from real-world context distributions

---

## Summary

✅ **Context Jitter toggle** per phase  
✅ **Jitter Mode selector** (Uniform/Mild/Gaussian) per phase  
✅ **Shuffle Order toggle** per phase  
✅ **Helpful descriptions** for each option  
✅ **Backward compatible** (backend unchanged)  

**Now users can configure:**
- Phase 1: No jitter (baseline)
- Phase 2: Mild jitter (gentle exploration)
- Phase 3: Uniform jitter (aggressive exploration)

**Full control over learning strategy per phase!** 🎯
