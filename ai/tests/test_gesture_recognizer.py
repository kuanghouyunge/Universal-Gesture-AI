import numpy as np
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

import src.recognition.gesture_recognizer as recognizer_module
from src.recognition.gesture_recognizer import GestureRecognizer


def create_landmarks():
    return [
        SimpleNamespace(x=float(i), y=float(i + 1), z=float(i + 2))
        for i in range(21)
    ]


def create_mock_model():
    model = MagicMock()
    model.n_features_in_ = 68
    model.classes_ = np.array([0, 1, 2, 3])

    model.predict.return_value = np.array([1])
    model.predict_proba.return_value = np.array(
        [[0.1, 0.7, 0.1, 0.1]]
    )

    return model


def test_model_file_not_found(monkeypatch):
    monkeypatch.setattr(
        recognizer_module,
        "MODEL_PATH",
        recognizer_module.Path("missing_model.pkl"),
    )

    with pytest.raises(FileNotFoundError):
        GestureRecognizer()


def test_model_feature_dimension_invalid(monkeypatch):
    model = create_mock_model()
    model.n_features_in_ = 63

    monkeypatch.setattr(
        recognizer_module.MODEL_PATH.__class__,
        "exists",
        lambda self: True,
    )
    monkeypatch.setattr(
        recognizer_module.joblib,
        "load",
        lambda path: model,
    )

    with pytest.raises(ValueError):
        GestureRecognizer()


def test_model_initialization(monkeypatch):
    model = create_mock_model()

    monkeypatch.setattr(
        recognizer_module.MODEL_PATH.__class__,
        "exists",
        lambda self: True,
    )
    monkeypatch.setattr(
        recognizer_module.joblib,
        "load",
        lambda path: model,
    )

    recognizer = GestureRecognizer()

    assert recognizer.model is model


def test_predict(monkeypatch):
    model = create_mock_model()

    monkeypatch.setattr(
        recognizer_module.MODEL_PATH.__class__,
        "exists",
        lambda self: True,
    )
    monkeypatch.setattr(
        recognizer_module.joblib,
        "load",
        lambda path: model,
    )

    monkeypatch.setattr(
        recognizer_module,
        "normalize_landmarks",
        lambda landmarks: np.zeros(63, dtype=np.float32),
    )

    monkeypatch.setattr(
        recognizer_module,
        "extract_joint_angles",
        lambda landmarks: np.zeros(5, dtype=np.float32),
    )

    recognizer = GestureRecognizer()
    result = recognizer.predict(create_landmarks())

    assert result["gesture_id"] == 1
    assert result["gesture"] == "FIST"
    assert result["confidence"] == 0.7
    assert result["probabilities"] == {
        "OPEN_PALM": 0.1,
        "FIST": 0.7,
        "POINT": 0.1,
        "THUMBS_UP": 0.1,
    }

    features = model.predict.call_args[0][0]

    assert features.shape == (1, 68)
    assert model.predict_proba.called