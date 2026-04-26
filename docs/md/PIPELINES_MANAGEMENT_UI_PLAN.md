# Pipelines Management UI Plan

**Date:** 2026-04-23  
**Status:** 📋 **PLANNED** (Not Yet Implemented)

---

## Problem

**Current state:**
- ✅ User can create pipelines (wizard at `/training-pipeline`)
- ✅ Individual projects show in Dashboard with pipeline badges
- ❌ **No way to view pipeline list**
- ❌ **No way to see pipeline status/progress**
- ❌ **No way to control pipelines** (pause, resume, stop)
- ❌ **No way to see which projects belong to which pipeline**

**User question:** *"Can I see the summary of this config from anywhere? List of past pipeline runs and projects associated with it?"*

---

## Solution: Create Pipelines Management Page

### New Route: `/pipelines`

**Navigation:**
```
Dashboard
  ├─ "Projects" (current default)
  └─ "Pipelines" ← NEW button/tab
      └─ Shows list of all pipelines
```

---

## UI Design

### Page Structure

```
┌─────────────────────────────────────────────────────────────┐
│  Pipelines                                     [+ New Pipeline] │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 🟢 training_2026-04-23                                │  │
│  │                                                        │  │
│  │ Status: Running | Progress: 45/105 runs (42%)        │  │
│  │ Phase 2/3: Initial Exploration (Pass 1/1)            │  │
│  │ Current: bilovec_nadir (Project 7/15)                │  │
│  │                                                        │  │
│  │ 📊 Stats: Success: 42 | Failed: 3 | Mean Reward: 0.08│  │
│  │ ⏱️  Started: 2h 30m ago | Est. remaining: 5h 15m     │  │
│  │                                                        │  │
│  │ Projects (15): podoli_oblique, bilovec_nadir, ...    │  │
│  │                                                        │  │
│  │ [⏸️ Pause] [⏹️ Stop] [📊 Details] [🔄 Refresh]       │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 🟡 contextual_learning_experiment                     │  │
│  │                                                        │  │
│  │ Status: Paused | Progress: 23/90 runs (25%)          │  │
│  │ Phase 1/3: Baseline Collection (Pass 1/1)            │  │
│  │                                                        │  │
│  │ Projects (10): oblique_1, oblique_2, ...             │  │
│  │                                                        │  │
│  │ [▶️ Resume] [⏹️ Stop] [📊 Details]                    │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ ✅ exif_baseline_comparison                           │  │
│  │                                                        │  │
│  │ Status: Completed | Runs: 105/105 (100%)             │  │
│  │ Completed: 2 days ago | Duration: 18h 30m            │  │
│  │                                                        │  │
│  │ Projects (15): dataset_a, dataset_b, ...             │  │
│  │                                                        │  │
│  │ [📊 Details] [🔄 Restart] [🗑️ Delete]                 │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Pipeline Card Details

### Card Header
```typescript
{
  icon: statusIcon,  // 🟢 Running, 🟡 Paused, ✅ Completed, 🔴 Failed, ⏸️ Stopped
  name: "training_2026-04-23",
  status: "running",
}
```

### Progress Section
```typescript
{
  progress: "45/105 runs (42%)",
  currentPhase: "Phase 2/3: Initial Exploration",
  currentPass: "Pass 1/1",
  currentProject: "bilovec_nadir (Project 7/15)",
}
```

### Statistics
```typescript
{
  completedRuns: 42,
  failedRuns: 3,
  meanReward: 0.08,
  successRate: 93.3,
  bestReward: 0.15,
}
```

### Time Information
```typescript
{
  createdAt: "2026-04-23T14:30:00Z",
  startedAt: "2026-04-23T14:35:00Z",
  elapsedTime: "2h 30m",
  estimatedRemaining: "5h 15m",
  completedAt: null,  // For completed pipelines
}
```

### Projects List
```typescript
{
  projects: [
    { name: "podoli_oblique", status: "completed" },
    { name: "bilovec_nadir", status: "processing" },
    { name: "terrain_rough", status: "pending" },
    // ... 12 more
  ],
  totalProjects: 15,
  displayLimit: 5,  // Show first 5, then "... and 10 more"
}
```

### Actions (Status-dependent)
```typescript
// Running pipeline
actions: [
  { label: "Pause", icon: "⏸️", action: "pause" },
  { label: "Stop", icon: "⏹️", action: "stop", confirm: true },
  { label: "Details", icon: "📊", action: "viewDetails" },
]

// Paused pipeline
actions: [
  { label: "Resume", icon: "▶️", action: "resume" },
  { label: "Stop", icon: "⏹️", action: "stop", confirm: true },
  { label: "Details", icon: "📊", action: "viewDetails" },
]

// Completed pipeline
actions: [
  { label: "Details", icon: "📊", action: "viewDetails" },
  { label: "Restart", icon: "🔄", action: "restart", confirm: true },
  { label: "Delete", icon: "🗑️", action: "delete", confirm: true },
]

// Pending pipeline
actions: [
  { label: "Start", icon: "▶️", action: "start" },
  { label: "Edit", icon: "✏️", action: "edit" },
  { label: "Delete", icon: "🗑️", action: "delete", confirm: true },
]
```

---

## Details Page: `/pipelines/:id`

### Tab Structure

```
┌─────────────────────────────────────────────────────────────┐
│  ← Back to Pipelines                                         │
│                                                               │
│  training_2026-04-23                    🟢 Running           │
│  Created: 2 days ago | Progress: 45/105 runs (42%)          │
│                                                               │
│  [Overview] [Projects] [Runs] [Configuration] [Logs]        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  [Tab content here]                                          │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

### Tab 1: Overview

```
┌────────────────────────────────────────────────────────────┐
│ Pipeline Status                                             │
├────────────────────────────────────────────────────────────┤
│ Status:          Running                                    │
│ Phase:           2/3 (Initial Exploration)                  │
│ Pass:            1/1                                        │
│ Current Project: bilovec_nadir (7/15)                       │
│ Next Run:        In 8 minutes (cooldown active)             │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│ Progress                                                    │
├────────────────────────────────────────────────────────────┤
│ Total Runs:      45/105 (42%)                               │
│ ▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░ 42%                               │
│                                                             │
│ Completed:       42 runs                                    │
│ Failed:          3 runs                                     │
│ Remaining:       60 runs                                    │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│ Learning Statistics                                         │
├────────────────────────────────────────────────────────────┤
│ Mean Reward:     0.08                                       │
│ Best Reward:     0.15 (bilovec_nadir, run_023)             │
│ Success Rate:    93.3% (42/45)                              │
│ Model Updates:   39 (batch-aware)                           │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│ Time                                                        │
├────────────────────────────────────────────────────────────┤
│ Created:         Apr 23, 2026 14:30:00                      │
│ Started:         Apr 23, 2026 14:35:00                      │
│ Elapsed:         2h 30m                                     │
│ Est. Remaining:  5h 15m                                     │
│ Est. Completion: Apr 23, 2026 22:20:00                      │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│ Actions                                                     │
├────────────────────────────────────────────────────────────┤
│ [⏸️ Pause Pipeline] [⏹️ Stop Pipeline] [🔄 Refresh]        │
└────────────────────────────────────────────────────────────┘
```

---

### Tab 2: Projects

**Table showing all projects in pipeline:**

| Project Name | Status | Runs | Phase | Last Run | Best Reward |
|--------------|--------|------|-------|----------|-------------|
| podoli_oblique | ✅ Completed | 3/3 | All | 2h ago | 0.12 |
| bilovec_nadir | 🟢 Processing | 2/3 | Phase 2 | Now | 0.08 |
| terrain_rough | ⏸️ Pending | 0/3 | - | - | - |
| ... 12 more | | | | | |

**Actions:**
- Click project → Navigate to project detail page
- Filter by status (All, Completed, Processing, Pending, Failed)
- Sort by name, status, reward

---

### Tab 3: Runs

**Chronological list of all training runs:**

| # | Project | Phase | Pass | Status | Reward | Duration | Completed |
|---|---------|-------|------|--------|--------|----------|-----------|
| 45 | bilovec_nadir | 2 | 1 | 🟢 Running | - | 15m | - |
| 44 | podoli_oblique | 2 | 1 | ✅ Done | +0.12 | 18m | 5m ago |
| 43 | terrain_rough | 2 | 1 | ✅ Done | +0.08 | 16m | 23m ago |
| 42 | oblique_15 | 2 | 1 | ✅ Done | +0.09 | 17m | 41m ago |
| 41 | dataset_14 | 1 | 1 | ✅ Done | 0.00 | 15m | 1h ago |
| ... 40 more | | | | | | | |

**Actions:**
- Click run → View run details (logs, checkpoints, metrics)
- Filter by status, phase, project
- Export run history as CSV

---

### Tab 4: Configuration

**Show pipeline configuration (read-only):**

```yaml
Pipeline Name: training_2026-04-23
Strategy: contextual_continuous

Shared Configuration:
  - AI Input Mode: exif_plus_flight_plan
  - Max Steps: 5000
  - Eval Interval: 1000
  - Densify Until: 4000

Projects (15):
  1. podoli_oblique (245 images)
  2. bilovec_nadir (312 images)
  ... 13 more

Phases:
  Phase 1: Baseline Collection
    - Runs per project: 1
    - Passes: 1
    - Context Jitter: Disabled
    - Preset: balanced
    - Update Model: No

  Phase 2: Initial Exploration
    - Runs per project: 1
    - Passes: 1
    - Context Jitter: Disabled
    - Strategy: contextual_continuous
    - Update Model: Yes

  Phase 3: Multi-Pass Learning
    - Runs per project: 1
    - Passes: 5
    - Context Jitter: Enabled (uniform)
    - Shuffle Order: Yes
    - Update Model: Yes

Thermal Management:
  - Cooldown: 10 minutes between runs
  - Strategy: fixed_interval

Total Runs: 105 (15 projects × 7 total passes)
```

**Actions:**
- [📋 Copy Config] - Copy as JSON
- [💾 Export] - Download config.json

---

### Tab 5: Logs

**Real-time log streaming:**

```
[2026-04-23 16:45:23] Pipeline started
[2026-04-23 16:45:25] Phase 1: Baseline Collection
[2026-04-23 16:45:30] Project 1/15: podoli_oblique
[2026-04-23 16:45:32] Starting run_001 (baseline)
[2026-04-23 17:03:15] Run completed: PSNR=28.5, Duration=18m
[2026-04-23 17:13:20] Cooldown complete, resuming
[2026-04-23 17:13:22] Project 2/15: bilovec_nadir
...
```

**Features:**
- Auto-scroll (toggle)
- Filter by level (INFO, WARN, ERROR)
- Search logs
- Download full log file

---

## API Endpoints Needed

### Already Exist ✅
```
GET  /api/training-pipeline/list?limit=50
POST /api/training-pipeline/create
GET  /api/training-pipeline/{id}
POST /api/training-pipeline/{id}/start
POST /api/training-pipeline/{id}/pause
POST /api/training-pipeline/{id}/resume
POST /api/training-pipeline/{id}/stop
GET  /api/training-pipeline/{id}/runs
DELETE /api/training-pipeline/{id}
POST /api/training-pipeline/{id}/elevate-learner-model
```

### Need to Add ❌
```
GET  /api/training-pipeline/{id}/logs
     → Stream or paginated logs

GET  /api/training-pipeline/{id}/projects
     → List projects with their run status in this pipeline

GET  /api/training-pipeline/{id}/stats
     → Aggregated statistics (mean/best reward, success rate, etc.)
```

---

## Implementation Plan

### Phase 1: Basic List View
1. Create `PipelinesListPage.tsx`
2. Add route `/pipelines` to App.tsx
3. Add "Pipelines" button to Dashboard header
4. Fetch pipeline list from `/api/training-pipeline/list`
5. Display cards with basic info (name, status, progress)
6. Add Start/Pause/Stop buttons

### Phase 2: Pipeline Details
1. Create `PipelineDetailsPage.tsx`
2. Add route `/pipelines/:id`
3. Implement Overview tab
4. Implement Projects tab
5. Implement Runs tab
6. Add navigation from list → details

### Phase 3: Advanced Features
1. Implement Configuration tab
2. Implement Logs tab (if streaming added)
3. Add filters and sorting
4. Add export functionality
5. Add real-time updates (polling or WebSockets)

### Phase 4: Polish
1. Add loading states
2. Add error handling
3. Add confirmation dialogs for destructive actions
4. Add toast notifications
5. Add responsive design

---

## File Structure

```
bimba3d_frontend/src/
  ├── pages/
  │   ├── Dashboard.tsx              ← Existing
  │   ├── TrainingPipelinePage.tsx   ← Existing (Create wizard)
  │   ├── PipelinesListPage.tsx      ← NEW (List all pipelines)
  │   └── PipelineDetailsPage.tsx    ← NEW (Pipeline details)
  │
  ├── components/
  │   └── pipelines/
  │       ├── PipelineCard.tsx       ← NEW (Pipeline summary card)
  │       ├── PipelineProgress.tsx   ← NEW (Progress bar)
  │       ├── PipelineStats.tsx      ← NEW (Statistics display)
  │       └── ProjectsList.tsx       ← NEW (Projects table)
  │
  └── App.tsx                        ← Add /pipelines routes
```

---

## Navigation Flow

```
Dashboard
  │
  ├─ [Projects Tab] (default)
  │   └─ Shows individual projects with pipeline badges
  │
  ├─ [Pipelines Tab] ← NEW
  │   ├─ List of all pipelines
  │   │   └─ Click pipeline → /pipelines/{id}
  │   │
  │   └─ [+ New Pipeline] button → /training-pipeline
  │
  └─ /pipelines/{id}
      ├─ Overview tab
      ├─ Projects tab → Click project → /projects/{id}
      ├─ Runs tab → Click run → /projects/{project_id}/runs/{run_id}
      ├─ Configuration tab
      └─ Logs tab
```

---

## Summary

**What's missing:**
- ❌ Pipelines list page
- ❌ Pipeline details page
- ❌ Pipeline management UI (start/pause/stop)
- ❌ View projects grouped by pipeline
- ❌ View run history for pipeline
- ❌ View pipeline configuration
- ❌ View pipeline logs

**What exists:**
- ✅ Pipeline creation wizard
- ✅ Pipeline execution backend
- ✅ Pipeline API endpoints
- ✅ Individual projects show pipeline badge

**Recommendation:** Implement Phase 1 (Basic List View) first to give you visibility into pipeline status and control!

Would you like me to implement the Pipelines List Page now?
