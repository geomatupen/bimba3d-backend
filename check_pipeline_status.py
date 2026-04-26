#!/usr/bin/env python
"""Quick script to check the status of the latest real_training pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "bimba3d_backend"))

from bimba3d_backend.app.services import training_pipeline_storage

pipelines = training_pipeline_storage.list_pipelines()
real_pipelines = [p for p in pipelines if 'real_training' in p['name']]

if not real_pipelines:
    print("No real_training pipelines found")
    sys.exit(1)

p = real_pipelines[-1]  # Latest
print(f"Pipeline: {p['name']}")
print(f"ID: {p['id']}")
print(f"Status: {p['status']}")
print(f"Progress: {p['completed_runs']}/{p['total_runs']} runs")
print(f"Current Phase: {p['current_phase']}/2")
print(f"Failed runs: {p['failed_runs']}")

if p.get('mean_reward') is not None:
    print(f"Mean Reward: {p['mean_reward']:.4f}")
if p.get('best_reward') is not None:
    print(f"Best Reward: {p['best_reward']:.4f}")

if p.get('last_error'):
    print(f"\nLast Error: {p['last_error']}")

if p.get('runs'):
    print(f"\nRuns ({len(p['runs'])}):")
    for i, run in enumerate(p['runs'], 1):
        reward_str = f", reward={run['reward']:.4f}" if run.get('reward') is not None else ""
        print(f"  {i}. {run.get('run_name', 'N/A')}: {run['status']}{reward_str}")

print(f"\nFolder: E:\\Thesis\\PipelineProjects\\{p['name']}")
