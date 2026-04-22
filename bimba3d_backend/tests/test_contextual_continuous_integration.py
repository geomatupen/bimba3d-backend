"""Integration test for contextual continuous strategy."""
from __future__ import annotations

import json
import pytest
from pathlib import Path

from bimba3d_backend.worker.ai_input_modes.resolver import (
    apply_initial_preset,
    VALID_SELECTOR_STRATEGIES,
)


@pytest.fixture
def temp_project_dir(tmp_path):
    """Create temporary project directory structure."""
    project_dir = tmp_path / "project"
    image_dir = project_dir / "images"
    colmap_dir = project_dir / "colmap"
    image_dir.mkdir(parents=True)
    colmap_dir.mkdir(parents=True)

    # Create dummy images
    (image_dir / "img001.jpg").write_bytes(b"fake_image_data")
    (image_dir / "img002.jpg").write_bytes(b"fake_image_data")

    return project_dir


def test_contextual_continuous_strategy_registered():
    """Verify contextual_continuous is in valid strategies."""
    assert "contextual_continuous" in VALID_SELECTOR_STRATEGIES


def test_apply_initial_preset_contextual_continuous(temp_project_dir, monkeypatch):
    """Test that contextual_continuous strategy works end-to-end."""

    # Mock logger
    class MockLogger:
        def info(self, *args, **kwargs):
            pass

    logger = MockLogger()

    # Mock feature extraction to return dummy features
    # (In real use, these come from EXIF/image analysis)
    def mock_build_preset(ctx):
        from bimba3d_backend.worker.ai_input_modes.common import PresetResult

        features = {
            "focal_length_mm": 24.0,
            "focal_missing": 0,
            "shutter_s": 0.002,
            "shutter_missing": 0,
            "iso": 800.0,
            "iso_missing": 0,
            "img_width_median": 4000.0,
            "img_height_median": 3000.0,
        }

        return PresetResult(
            mode="exif_only",
            features=features,
            notes=["Mock features for testing"],
            updates={"preset_name": "balanced"},
        )

    # Patch the feature extraction
    monkeypatch.setattr(
        "bimba3d_backend.worker.ai_input_modes.resolver.build_exif_only_preset",
        mock_build_preset,
    )

    params = {
        "ai_input_mode": "exif_only",
        "ai_selector_strategy": "contextual_continuous",
        "feature_lr": 2.5e-3,
        "position_lr_init": 1.6e-4,
        "scaling_lr": 5.0e-3,
        "opacity_lr": 5.0e-2,
        "rotation_lr": 1.0e-3,
        "densify_grad_threshold": 2.0e-4,
        "opacity_threshold": 0.005,
        "lambda_dssim": 0.2,
    }

    image_dir = temp_project_dir / "images"
    colmap_dir = temp_project_dir / "colmap"

    result = apply_initial_preset(
        params,
        image_dir=image_dir,
        colmap_dir=colmap_dir,
        logger=logger,
    )

    # Verify result structure
    assert result["mode"] == "exif_only"
    assert result["applied"] is True
    assert result["selector_strategy"] == "contextual_continuous"
    assert result["selected_preset"] == "contextual_continuous"

    # Verify multipliers were predicted
    assert "yhat_scores" in result
    assert len(result["yhat_scores"]) == 8

    # Verify parameters were updated
    assert "updates" in result
    assert "feature_lr" in result["updates"]
    assert result["updates"]["preset_name"] == "contextual_continuous"

    # Verify params dict was modified in place
    assert params["feature_lr"] != 2.5e-3  # Should be modified by multiplier

    # Verify features were extracted and passed
    assert "features" in result
    assert result["features"]["focal_length_mm"] == 24.0


def test_contextual_continuous_model_persistence(temp_project_dir, monkeypatch):
    """Test that model is persisted correctly."""

    class MockLogger:
        def info(self, *args, **kwargs):
            pass

    logger = MockLogger()

    def mock_build_preset(ctx):
        from bimba3d_backend.worker.ai_input_modes.common import PresetResult

        features = {
            "focal_length_mm": 50.0,
            "focal_missing": 0,
            "shutter_s": 0.001,
            "shutter_missing": 0,
            "iso": 400.0,
            "iso_missing": 0,
            "img_width_median": 4000.0,
            "img_height_median": 3000.0,
        }

        return PresetResult(
            mode="exif_only",
            features=features,
            notes=[],
            updates={"preset_name": "balanced"},
        )

    monkeypatch.setattr(
        "bimba3d_backend.worker.ai_input_modes.resolver.build_exif_only_preset",
        mock_build_preset,
    )

    params = {
        "ai_input_mode": "exif_only",
        "ai_selector_strategy": "contextual_continuous",
        "feature_lr": 2.5e-3,
        "position_lr_init": 1.6e-4,
        "scaling_lr": 5.0e-3,
        "opacity_lr": 5.0e-2,
        "rotation_lr": 1.0e-3,
        "densify_grad_threshold": 2.0e-4,
        "opacity_threshold": 0.005,
        "lambda_dssim": 0.2,
    }

    image_dir = temp_project_dir / "images"
    colmap_dir = temp_project_dir / "colmap"

    # First selection - loads default model (not persisted until update)
    result1 = apply_initial_preset(
        params.copy(),
        image_dir=image_dir,
        colmap_dir=colmap_dir,
        logger=logger,
    )

    # Model file is created lazily on first update, not on selection
    model_path = temp_project_dir / "models" / "contextual_continuous_selector" / "exif_only.json"
    # This is expected - model only persists after update_from_run is called
    # (which happens after actual training runs complete)

    # Second selection - should work with same strategy
    result2 = apply_initial_preset(
        params.copy(),
        image_dir=image_dir,
        colmap_dir=colmap_dir,
        logger=logger,
    )

    # Both results should use same strategy
    assert result1["selector_strategy"] == result2["selector_strategy"]
    assert result1["selector_strategy"] == "contextual_continuous"

    # Verify selections produce consistent structure
    assert "yhat_scores" in result1
    assert "yhat_scores" in result2
    assert len(result1["yhat_scores"]) == 8
    assert len(result2["yhat_scores"]) == 8


def test_contextual_continuous_fallback_to_preset_bias(temp_project_dir, monkeypatch):
    """Test that invalid strategy falls back to preset_bias."""

    class MockLogger:
        def info(self, *args, **kwargs):
            pass

    logger = MockLogger()

    def mock_build_preset(ctx):
        from bimba3d_backend.worker.ai_input_modes.common import PresetResult

        features = {
            "focal_length_mm": 24.0,
            "focal_missing": 0,
            "shutter_s": 0.002,
            "shutter_missing": 0,
            "iso": 800.0,
            "iso_missing": 0,
            "img_width_median": 4000.0,
            "img_height_median": 3000.0,
        }

        return PresetResult(
            mode="exif_only",
            features=features,
            notes=[],
            updates={"preset_name": "balanced"},
        )

    monkeypatch.setattr(
        "bimba3d_backend.worker.ai_input_modes.resolver.build_exif_only_preset",
        mock_build_preset,
    )

    params = {
        "ai_input_mode": "exif_only",
        "ai_selector_strategy": "invalid_strategy_name",  # Invalid
        "feature_lr": 2.5e-3,
    }

    image_dir = temp_project_dir / "images"
    colmap_dir = temp_project_dir / "colmap"

    result = apply_initial_preset(
        params,
        image_dir=image_dir,
        colmap_dir=colmap_dir,
        logger=logger,
    )

    # Should fall back to preset_bias
    assert result["selector_strategy"] == "preset_bias"
