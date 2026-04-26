# Bimba3D Documentation

This directory contains technical documentation for the Bimba3D Gaussian Splatting training platform.

---

## Table of Contents

### Training Pipeline
- [CROSS_PROJECT_TRAINING_PIPELINE_PLAN.md](CROSS_PROJECT_TRAINING_PIPELINE_PLAN.md) - Overall pipeline architecture and design
- [TRAINING_PIPELINE_IMPLEMENTATION.md](TRAINING_PIPELINE_IMPLEMENTATION.md) - Implementation details and status
- [TRAINING_PIPELINE_IMPROVEMENTS.md](TRAINING_PIPELINE_IMPROVEMENTS.md) - Enhancements and optimizations
- [PIPELINE_DIRECTORY_STRUCTURE.md](PIPELINE_DIRECTORY_STRUCTURE.md) - File structure and organization
- [PIPELINE_COLMAP_STRATEGY.md](PIPELINE_COLMAP_STRATEGY.md) - COLMAP reuse strategy (run once per project)
- [STORAGE_MANAGEMENT.md](STORAGE_MANAGEMENT.md) - Storage configuration options

### UI Implementation
- [PIPELINES_MANAGEMENT_UI_PLAN.md](PIPELINES_MANAGEMENT_UI_PLAN.md) - Pipeline list and details pages design
- [PIPELINE_AI_LEARNING_TABLE.md](PIPELINE_AI_LEARNING_TABLE.md) - AI learning table implementation
- [SETTINGS_TAB_IMPLEMENTATION.md](SETTINGS_TAB_IMPLEMENTATION.md) - Project settings tab
- [PHASE_JITTER_CONTROLS_ADDED.md](PHASE_JITTER_CONTROLS_ADDED.md) - Context jitter UI controls

### AI Learning System
- [CONTEXTUAL_CONTINUOUS_SUMMARY.md](CONTEXTUAL_CONTINUOUS_SUMMARY.md) - Overview of contextual continuous bandit learner
- [CONTEXTUAL_CONTINUOUS_GUIDE.md](CONTEXTUAL_CONTINUOUS_GUIDE.md) - How to use the learner
- [CONTEXTUAL_CONTINUOUS_AUDIT.md](CONTEXTUAL_CONTINUOUS_AUDIT.md) - System audit and verification
- [CONTEXTUAL_FEATURES_COMPLETE.md](CONTEXTUAL_FEATURES_COMPLETE.md) - Feature implementation status
- [REWARD_CALCULATION_UPDATE.md](REWARD_CALCULATION_UPDATE.md) - Quality-driven reward system (75% quality, 25% time)
- [BASELINE_COMPARISON_FIX.md](BASELINE_COMPARISON_FIX.md) - Independent baseline comparison fix

### Model Management
- [SHARED_MODEL_ARCHITECTURE.md](SHARED_MODEL_ARCHITECTURE.md) - Cross-project learning architecture
- [LEARNER_MODEL_ELEVATION.md](LEARNER_MODEL_ELEVATION.md) - Promoting pipeline models to global registry
- [MODEL_LINEAGE_COMPARISON.md](MODEL_LINEAGE_COMPARISON.md) - Model provenance and comparison
- [BATCH_MODEL_UPDATE.md](BATCH_MODEL_UPDATE.md) - Batch-aware model updates (last run only)

### Bug Fixes & Issues
- [PIPELINE_FOLDER_FIX.md](PIPELINE_FOLDER_FIX.md) - Pipeline folder 404 error fix
- [REMAINING_ISSUES_PLAN.md](REMAINING_ISSUES_PLAN.md) - Outstanding issues and planned work

---

## Quick Start Guide

1. **Understanding the Pipeline:**
   - Start with [CROSS_PROJECT_TRAINING_PIPELINE_PLAN.md](CROSS_PROJECT_TRAINING_PIPELINE_PLAN.md)
   - Read [PIPELINE_DIRECTORY_STRUCTURE.md](PIPELINE_DIRECTORY_STRUCTURE.md)
   - Review [PIPELINE_COLMAP_STRATEGY.md](PIPELINE_COLMAP_STRATEGY.md) (IMPORTANT!)

2. **AI Learning System:**
   - Read [CONTEXTUAL_CONTINUOUS_SUMMARY.md](CONTEXTUAL_CONTINUOUS_SUMMARY.md)
   - Follow [CONTEXTUAL_CONTINUOUS_GUIDE.md](CONTEXTUAL_CONTINUOUS_GUIDE.md)
   - Understand [REWARD_CALCULATION_UPDATE.md](REWARD_CALCULATION_UPDATE.md)

3. **Using the UI:**
   - See [PIPELINES_MANAGEMENT_UI_PLAN.md](PIPELINES_MANAGEMENT_UI_PLAN.md)
   - Check [STORAGE_MANAGEMENT.md](STORAGE_MANAGEMENT.md) for storage options

4. **Model Management:**
   - Read [SHARED_MODEL_ARCHITECTURE.md](SHARED_MODEL_ARCHITECTURE.md)
   - Review [LEARNER_MODEL_ELEVATION.md](LEARNER_MODEL_ELEVATION.md)

---

## Document Status Legend

- ✅ **COMPLETED** - Implementation finished and tested
- 📋 **DOCUMENTED** - Planned but not yet implemented
- ⚠️ **PARTIAL** - Partially implemented
- 🔄 **IN PROGRESS** - Currently being worked on
- 📝 **PLAN** - Design document

---

## Key Concepts

### Training Pipeline
Automated system for training multiple projects with AI-driven parameter selection across multiple phases.

### Contextual Continuous Bandit Learner
Machine learning system that learns optimal training parameters based on project context (EXIF data, flight plan, etc.) using ridge regression and Thompson sampling.

### Cross-Project Learning
Shared model directory enables knowledge accumulated from training one project to benefit subsequent projects.

### Baseline Comparison
Each project has a baseline run (fixed parameters) that AI learning runs are compared against to calculate rewards.

### COLMAP Reuse Strategy
COLMAP sparse reconstruction is created once per project (baseline phase), then reused by all subsequent training runs with different parameters.

### Storage Management
Configuration options to control what gets saved (eval images, checkpoints) with replace mode to minimize storage usage for pipeline training.

---

## Contributing

When adding new documentation:
1. Place `.md` files in this directory (`docs/md/`)
2. Use descriptive, ALL_CAPS filenames
3. Update this README with a link and brief description
4. Include status badge (✅/📋/⚠️/🔄/📝)
5. Add creation/update dates at the top of the document

---

Last Updated: 2026-04-24
