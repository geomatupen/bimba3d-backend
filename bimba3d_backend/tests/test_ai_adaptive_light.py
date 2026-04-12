import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from bimba3d_backend.worker.ai_adaptive_light import (
    ACTION_DENSIFY_UP,
    ACTION_KEEP,
    ACTION_LR_UP,
    ACTION_PRUNE_DOWN,
    ACTIONS,
    CoreAIAdaptiveController,
)


class DummyOptimizer:
    def __init__(self, lr: float):
        self.param_groups = [{"lr": lr}]


class DummyStrategy:
    def __init__(self):
        self.grow_grad2d = 1e-3
        self.prune_opa = 5e-3
        self.refine_every = 100
        self.reset_every = 2000


def make_runner():
    optimizers = {
        "means": DummyOptimizer(0.01),
        "opacities": DummyOptimizer(0.02),
        "scales": DummyOptimizer(0.03),
        "quats": DummyOptimizer(0.04),
        "sh0": DummyOptimizer(0.05),
        "shN": DummyOptimizer(0.06),
    }
    cfg = SimpleNamespace(strategy=DummyStrategy())
    means = SimpleNamespace(shape=(1000, 3))
    return SimpleNamespace(optimizers=optimizers, cfg=cfg, splats={"means": means})


class CoreAIAdaptiveControllerTests(unittest.TestCase):
    def _controller(self, project_dir: Path) -> CoreAIAdaptiveController:
        return CoreAIAdaptiveController(
            project_dir=project_dir,
            run_id="test-run",
            max_steps=10000,
            tune_start_step=100,
            tune_end_step=9000,
            strategy_start_step=500,
            strategy_end_step=8000,
            base_min_improvement=0.01,
            decision_interval=100,
        )

    def test_lrp_action_can_be_selected_after_warmup(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = self._controller(Path(tmp))
            runner = make_runner()

            def fake_forward(_x):
                logits = np.zeros((len(ACTIONS),), dtype=np.float64)
                logits[ACTIONS.index(ACTION_LR_UP)] = 5.0
                return np.zeros((64,), dtype=np.float64), np.zeros((64,), dtype=np.float64), logits

            controller.model.forward = fake_forward  # type: ignore[assignment]

            first = controller.decide_and_apply(step=200, loss=1.0, runner_obj=runner, apply_lr=True, apply_strategy=True)
            second = controller.decide_and_apply(step=300, loss=1.1, runner_obj=runner, apply_lr=True, apply_strategy=True)

            self.assertEqual(first.action, ACTION_KEEP)
            self.assertEqual(second.action, ACTION_LR_UP)
            self.assertTrue((Path(tmp) / "runs" / "test-run" / "adaptive_ai" / "runs" / "test-run.summary.json").exists())

    def test_strategy_action_blocked_outside_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = self._controller(Path(tmp))
            runner = make_runner()

            def fake_forward(_x):
                logits = np.zeros((len(ACTIONS),), dtype=np.float64)
                logits[ACTIONS.index(ACTION_DENSIFY_UP)] = 5.0
                return np.zeros((64,), dtype=np.float64), np.zeros((64,), dtype=np.float64), logits

            controller.model.forward = fake_forward  # type: ignore[assignment]

            controller.decide_and_apply(step=200, loss=1.0, runner_obj=runner, apply_lr=True, apply_strategy=True)
            decision = controller.decide_and_apply(step=300, loss=1.1, runner_obj=runner, apply_lr=True, apply_strategy=False)

            self.assertEqual(decision.action, ACTION_KEEP)
            self.assertEqual(decision.reason, "outside_window")

    def test_small_change_band_keeps_action_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = self._controller(Path(tmp))
            runner = make_runner()

            def fake_forward(_x):
                logits = np.zeros((len(ACTIONS),), dtype=np.float64)
                logits[ACTIONS.index(ACTION_LR_UP)] = 5.0
                return np.zeros((64,), dtype=np.float64), np.zeros((64,), dtype=np.float64), logits

            controller.model.forward = fake_forward  # type: ignore[assignment]

            controller.decide_and_apply(step=200, loss=1.0, runner_obj=runner, apply_lr=True, apply_strategy=True)
            decision = controller.decide_and_apply(step=300, loss=0.995, runner_obj=runner, apply_lr=True, apply_strategy=True)

            self.assertEqual(decision.action, ACTION_KEEP)
            self.assertEqual(decision.reason, "stable_small_change")

    def test_late_phase_quality_priority_blocks_risky_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = self._controller(Path(tmp))
            runner = make_runner()

            def fake_forward(_x):
                logits = np.zeros((len(ACTIONS),), dtype=np.float64)
                logits[ACTIONS.index(ACTION_LR_UP)] = 5.0
                return np.zeros((64,), dtype=np.float64), np.zeros((64,), dtype=np.float64), logits

            controller.model.forward = fake_forward  # type: ignore[assignment]

            controller.decide_and_apply(step=7900, loss=1.0, runner_obj=runner, apply_lr=True, apply_strategy=True)
            decision = controller.decide_and_apply(step=8000, loss=1.005, runner_obj=runner, apply_lr=True, apply_strategy=True)

            self.assertEqual(decision.action, ACTION_KEEP)
            self.assertEqual(decision.reason, "late_phase_quality_priority")

    def test_late_phase_quality_priority_allows_risky_action_on_strong_gain(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = self._controller(Path(tmp))
            runner = make_runner()

            def fake_forward(_x):
                logits = np.zeros((len(ACTIONS),), dtype=np.float64)
                logits[ACTIONS.index(ACTION_LR_UP)] = 5.0
                return np.zeros((64,), dtype=np.float64), np.zeros((64,), dtype=np.float64), logits

            controller.model.forward = fake_forward  # type: ignore[assignment]

            controller.decide_and_apply(step=7900, loss=1.0, runner_obj=runner, apply_lr=True, apply_strategy=True)
            decision = controller.decide_and_apply(step=8000, loss=0.95, runner_obj=runner, apply_lr=True, apply_strategy=True)

            self.assertEqual(decision.action, ACTION_LR_UP)
            self.assertEqual(decision.reason, "late_phase_gate_allow")

    def test_prune_down_action_reduces_prune_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = self._controller(Path(tmp))
            runner = make_runner()

            def fake_forward(_x):
                logits = np.zeros((len(ACTIONS),), dtype=np.float64)
                logits[ACTIONS.index(ACTION_PRUNE_DOWN)] = 5.0
                return np.zeros((64,), dtype=np.float64), np.zeros((64,), dtype=np.float64), logits

            controller.model.forward = fake_forward  # type: ignore[assignment]

            controller.decide_and_apply(step=500, loss=1.0, runner_obj=runner, apply_lr=True, apply_strategy=True)
            decision = controller.decide_and_apply(step=600, loss=1.05, runner_obj=runner, apply_lr=True, apply_strategy=True)

            self.assertEqual(decision.action, ACTION_PRUNE_DOWN)
            self.assertLess(runner.cfg.strategy.prune_opa, 5e-3)


if __name__ == "__main__":
    unittest.main()
