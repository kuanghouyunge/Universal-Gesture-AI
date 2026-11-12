from unittest.mock import MagicMock, patch

import pytest

import src.vision.hand_landmarker as module
from src.vision.hand_landmarker import HandLandmarker


def test_model_file_not_found(monkeypatch):
    monkeypatch.setattr(
        module.MODEL_PATH.__class__,
        "exists",
        lambda self: False,
    )

    with pytest.raises(FileNotFoundError):
        HandLandmarker()


@patch("src.vision.hand_landmarker.vision.HandLandmarker.create_from_options")
def test_initialization(mock_create, monkeypatch):
    mock_detector = MagicMock()
    mock_create.return_value = mock_detector

    monkeypatch.setattr(
        module.MODEL_PATH.__class__,
        "exists",
        lambda self: True,
    )

    landmarker = HandLandmarker()

    assert landmarker.detector is mock_detector
    assert landmarker.timestamp_ms == 0

    options = mock_create.call_args[0][0]
    assert options.num_hands == 2
    assert options.running_mode == module.vision.RunningMode.VIDEO
    assert options.min_hand_detection_confidence == 0.6
    assert options.min_hand_presence_confidence == 0.6
    assert options.min_tracking_confidence == 0.6


@patch("src.vision.hand_landmarker.vision.HandLandmarker.create_from_options")
def test_detect_timestamp(mock_create, monkeypatch):
    mock_detector = MagicMock()
    mock_detector.detect_for_video.return_value = "result"
    mock_create.return_value = mock_detector

    monkeypatch.setattr(
        module.MODEL_PATH.__class__,
        "exists",
        lambda self: True,
    )

    landmarker = HandLandmarker()

    image = MagicMock()

    result1 = landmarker.detect(image)
    result2 = landmarker.detect(image)

    assert result1 == "result"
    assert result2 == "result"
    assert landmarker.timestamp_ms == 66

    assert mock_detector.detect_for_video.call_args_list[0].args == (
        image,
        33,
    )

    assert mock_detector.detect_for_video.call_args_list[1].args == (
        image,
        66,
    )


@patch("src.vision.hand_landmarker.vision.HandLandmarker.create_from_options")
def test_close(mock_create, monkeypatch):
    mock_detector = MagicMock()
    mock_create.return_value = mock_detector

    monkeypatch.setattr(
        module.MODEL_PATH.__class__,
        "exists",
        lambda self: True,
    )

    landmarker = HandLandmarker()
    landmarker.close()

    mock_detector.close.assert_called_once()
    assert landmarker.detector is None


@patch("src.vision.hand_landmarker.vision.HandLandmarker.create_from_options")
def test_close_without_detector(mock_create, monkeypatch):
    mock_create.return_value = MagicMock()

    monkeypatch.setattr(
        module.MODEL_PATH.__class__,
        "exists",
        lambda self: True,
    )

    landmarker = HandLandmarker()
    landmarker.detector = None

    landmarker.close()

    assert landmarker.detector is None


@patch("src.vision.hand_landmarker.vision.HandLandmarker.create_from_options")
def test_context_manager(mock_create, monkeypatch):
    mock_detector = MagicMock()
    mock_create.return_value = mock_detector

    monkeypatch.setattr(
        module.MODEL_PATH.__class__,
        "exists",
        lambda self: True,
    )

    with HandLandmarker() as landmarker:
        assert landmarker.detector is mock_detector

    mock_detector.close.assert_called_once()
    assert landmarker.detector is None