# Project Settings Tab - Implementation Summary

**Date:** 2026-04-22  
**Status:** ✅ **PRODUCTION READY**

---

## Overview

Added a new "Settings" tab in the project detail view that allows viewing and editing project-wide default training parameters (shared configuration).

---

## What Was Built

### New Settings Tab Component

**File:** `bimba3d_frontend/src/components/tabs/SettingsTab.tsx` (338 lines)

**Features:**
- View current `shared_config.json` for the project
- Edit AI configuration (input mode, selector strategy)
- Edit training parameters (max_steps, eval_interval, etc.)
- JSON preview of complete configuration
- Version tracking with timestamps
- Change detection and save confirmation

### UI Sections

**1. Header**
- Title with Settings icon
- Version number and last updated timestamp

**2. Info Banner**
- Explains that settings apply to future runs only
- Clarifies existing runs keep their original settings

**3. AI Configuration Section**
```
- AI Input Mode (dropdown)
  * Not set (use per-run default)
  * EXIF Only
  * EXIF + Flight Plan
  * EXIF + Flight Plan + External

- Selector Strategy (dropdown)
  * Not set (use per-run default)
  * Contextual Continuous
  * Continuous Bandit
  * Preset Bias
```

**4. Training Parameters Section**
```
Grid layout with inputs for:
- Max Steps
- Eval Interval
- Log Interval
- Densify Until Iter
- Images Max Size
```

**5. JSON Preview**
- Read-only view of complete shared config
- Scrollable for large configs

**6. Save Button**
- Disabled when no changes
- Shows "You have unsaved changes" warning
- Success confirmation after save

---

## How It Works

### Data Flow

**Loading:**
```
GET /projects/{project_id}/shared-config
  ↓
Returns: {
  version: 3,
  base_run_id: "run_baseline",
  updated_at: "2026-04-22T14:30:00Z",
  shared: {
    ai_input_mode: "exif_plus_flight_plan",
    ai_selector_strategy: "contextual_continuous",
    max_steps: 5000,
    ...
  }
}
  ↓
Populate form fields
```

**Saving:**
```
User edits values
  ↓
PATCH /projects/{project_id}/shared-config
{
  shared: {
    ai_input_mode: "exif_plus_flight_plan",
    ai_selector_strategy: "contextual_continuous",
    max_steps: 5000,
    eval_interval: 1000,
    ...
  }
}
  ↓
Backend merges with existing config
  ↓
Increments version number
  ↓
Returns updated config
  ↓
Reload to show new version
```

### Backend Integration

Uses existing endpoint:
```python
@router.patch("/{project_id}/shared-config")
def update_project_shared_config(
    project_id: str,
    payload: UpdateSharedConfigRequest
):
    # Merge with existing config
    # Increment version
    # Save to shared_config.json
    # Return updated config
```

**File location:** `{project_dir}/shared_config.json`

---

## Configuration Scope

### Two-Level Configuration System

**1. Shared Config (Project-Wide)**
```json
{
  "version": 3,
  "base_run_id": "run_baseline",
  "updated_at": "2026-04-22T14:30:00Z",
  "shared": {
    "ai_input_mode": "exif_plus_flight_plan",
    "ai_selector_strategy": "contextual_continuous",
    "max_steps": 5000,
    "eval_interval": 1000,
    ...
  }
}
```
**Scope:** Applies to all future runs in this project (defaults)

**2. Per-Run Config**
```json
{
  "run_id": "run_20260422_103045",
  "requested_params": { /* user-specified overrides */ },
  "resolved_params": { /* final merged params */ },
  "shared_config_snapshot": { /* shared config at time of run */ }
}
```
**Scope:** Specific to one training run

### How They Work Together

**Scenario 1: Pipeline Creates Project**
```
Pipeline wizard configures:
  ai_input_mode: exif_plus_flight_plan
  ai_selector_strategy: contextual_continuous
  max_steps: 5000
  
  ↓ Creates project with shared_config.json
  
All 105 runs inherit these settings (unless phase overrides)
```

**Scenario 2: User Changes Settings**
```
User opens Settings tab
Changes: max_steps 5000 → 7000
Saves

  ↓ Updates shared_config.json (version 1 → 2)
  
Next run created: Uses 7000 steps
Previous runs: Still show 5000 steps (unchanged)
```

**Scenario 3: Per-Run Override**
```
Shared config: max_steps = 5000

User starts run with override: max_steps = 10000

  ↓ Run uses 10000 (override wins)
  ↓ Shared config unchanged (still 5000)
  ↓ Next run without override uses 5000 again
```

---

## Use Cases

### 1. View Pipeline Configuration
**Problem:** Pipeline created 15 projects with specific settings. How do I see what was configured?

**Solution:**
1. Open any pipeline-created project
2. Go to Settings tab
3. View all training parameters
4. Check version and last updated time

### 2. Change Defaults for Future Runs
**Problem:** I want all future training runs to use 7000 steps instead of 5000.

**Solution:**
1. Open project
2. Go to Settings tab
3. Change Max Steps: 5000 → 7000
4. Click Save Settings
5. All new runs will use 7000 steps

### 3. Standardize Settings Across Projects
**Problem:** I have 10 projects with different settings. I want them all to use contextual_continuous.

**Solution:**
1. Open each project
2. Go to Settings tab
3. Set Selector Strategy: contextual_continuous
4. Save
5. Repeat for other projects

(Future enhancement: Batch update multiple projects)

### 4. Experiment with Different Strategies
**Problem:** I want to test continuous_bandit vs contextual_continuous on the same project.

**Solution:**
1. Run baseline with current settings (contextual_continuous)
2. Go to Settings tab
3. Change to continuous_bandit_linear
4. Save
5. Run new training session
6. Compare results in Sessions tab

---

## UI Location

**Navigation:**
```
Dashboard
  ↓ Click project
ProjectDetail
  ↓ Tabs: [Images] [Process] [Logs] [Sessions] [Models] [Comparison] [Settings]
  ↓ Click Settings
SettingsTab
  ↓ View/edit shared config
```

**Tab Icon:** ⚙️ Settings (gear icon)

---

## What's NOT Included

**Image/COLMAP Settings:**
The Settings tab currently shows AI and training parameters only. Image preprocessing and COLMAP settings are managed separately in the ProcessTab (existing functionality).

**Shared config includes:**
- ✅ AI configuration (input mode, selector strategy)
- ✅ Training parameters (steps, intervals, thresholds)
- ✅ Baseline session reference
- ❌ Image resize settings (in ProcessTab)
- ❌ COLMAP parameters (in ProcessTab)

To add image/COLMAP settings to this tab, extend the form sections with those fields.

---

## Future Enhancements

### 1. Batch Settings Update
```typescript
// Update multiple projects at once
POST /api/projects/batch-update-shared-config
{
  project_ids: ["proj1", "proj2", "proj3"],
  shared: {
    ai_selector_strategy: "contextual_continuous",
    max_steps: 7000
  }
}
```

### 2. Settings Templates
```typescript
// Save/load settings presets
POST /api/settings-templates
{
  name: "High Quality Contextual",
  template: {
    ai_input_mode: "exif_plus_flight_plan",
    ai_selector_strategy: "contextual_continuous",
    max_steps: 10000,
    ...
  }
}

GET /api/settings-templates
// Load template into form
```

### 3. Settings Diff Viewer
```typescript
// Compare settings between versions
GET /projects/{id}/shared-config/history
// Show version 1 vs version 3 diff
```

### 4. Settings Export/Import
```typescript
// Export settings to JSON file
// Import from JSON file (multiple projects)
```

---

## Testing

### Manual Test Steps

1. **View Settings:**
   - Open any project
   - Click Settings tab
   - Verify all fields populated
   - Check version and timestamp

2. **Edit Settings:**
   - Change AI Input Mode
   - Change Max Steps
   - Verify "unsaved changes" warning appears
   - Click Save Settings
   - Verify success message
   - Verify version incremented

3. **Settings Apply to New Runs:**
   - Change max_steps to 7000
   - Save
   - Go to Process tab
   - Start new run
   - Verify it uses 7000 steps

4. **Existing Runs Unchanged:**
   - Note existing run used 5000 steps
   - Change shared config to 7000
   - Go to Sessions tab
   - Verify old run still shows 5000

---

## Technical Details

### State Management
```typescript
const [config, setConfig] = useState<SharedConfig | null>(null);

// Separate state for each editable field
const [aiInputMode, setAiInputMode] = useState("");
const [aiSelectorStrategy, setAiSelectorStrategy] = useState("");
const [maxSteps, setMaxSteps] = useState(5000);
// ...

// Change detection
const [hasChanges, setHasChanges] = useState(false);

// Mark changed on any edit
const markChanged = () => setHasChanges(true);
```

### API Calls
```typescript
// Load
const loadSharedConfig = async () => {
  const response = await axios.get(
    `${API_BASE}/projects/${projectId}/shared-config`
  );
  setConfig(response.data);
  // Populate form...
};

// Save
const handleSave = async () => {
  await axios.patch(
    `${API_BASE}/projects/${projectId}/shared-config`,
    { shared: { ai_input_mode: aiInputMode, ... } }
  );
  await loadSharedConfig(); // Reload to get new version
};
```

### Error Handling
- Network errors: Display error banner
- Invalid values: Backend validates and rejects
- Conflicts: Backend returns 409 if base session mismatch

---

## Backward Compatibility

✅ **Fully backward compatible:**
- Projects without shared_config.json: Backend creates empty one
- Old runs without shared_config_snapshot: Still viewable
- Projects created before this feature: Settings tab works normally

✅ **No breaking changes:**
- Existing ProcessTab functionality unchanged
- Per-run overrides still work
- Training pipeline unaffected

---

## Files Modified

**Frontend:**
- NEW: `bimba3d_frontend/src/components/tabs/SettingsTab.tsx` (338 lines)
- MODIFIED: `bimba3d_frontend/src/pages/ProjectDetail.tsx` (+4 lines)
  - Added Settings import
  - Added "settings" to TabType
  - Added Settings tab to navigation
  - Added SettingsTab render

**Backend:**
- No changes (uses existing endpoint)

**Total:** 342 new lines

---

## Summary

The Project Settings tab provides a clean, intuitive UI for viewing and editing project-wide training configuration defaults. This addresses the user's question:

> "so can i also see or set the global configs for training from ui?"

**Answer: Yes!**

1. **View:** Open any project → Settings tab → See all training parameters
2. **Edit:** Change values → Save Settings → Applied to all future runs
3. **Scope:** Project-wide defaults (shared_config.json)
4. **Safety:** Existing runs unchanged, only affects new runs

The Settings tab integrates seamlessly with the training pipeline system, allowing users to inspect configurations of pipeline-created projects and adjust defaults as needed.
