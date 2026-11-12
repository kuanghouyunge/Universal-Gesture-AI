import cv2
from unittest.mock import MagicMock, patch

from src.vision.camera_manager import CameraManager


def test_initialization():
    camera = CameraManager()

    assert camera.camera_index == 1
    assert camera.cap is None


@patch("src.vision.camera_manager.cv2.VideoCapture")
def test_open_success(mock_video_capture):
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_video_capture.return_value = mock_cap

    camera = CameraManager(camera_index=1)

    assert camera.open() is True

    mock_video_capture.assert_called_once_with(1,  cv2.CAP_DSHOW)
    assert camera.cap is mock_cap


@patch("src.vision.camera_manager.cv2.VideoCapture")
def test_open_failure(mock_video_capture):
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    mock_video_capture.return_value = mock_cap

    camera = CameraManager()

    assert camera.open() is False


def test_read_without_camera():
    camera = CameraManager()

    success, frame = camera.read()

    assert success is False
    assert frame is None


def test_read_with_camera():
    camera = CameraManager()
    mock_cap = MagicMock()
    mock_cap.read.return_value = (True, "frame")
    camera.cap = mock_cap

    success, frame = camera.read()

    assert success is True
    assert frame == "frame"


def test_release():
    camera = CameraManager()
    mock_cap = MagicMock()
    camera.cap = mock_cap

    camera.release()

    mock_cap.release.assert_called_once()
    assert camera.cap is None


def test_release_without_camera():
    camera = CameraManager()

    camera.release()

    assert camera.cap is None