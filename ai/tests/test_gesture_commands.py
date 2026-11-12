import pytest

from src.control.gesture_commands import map_gesture_to_command


@pytest.mark.parametrize(
    "gesture, confidence, expected",
    [
        ("OPEN_PALM", 0.9, "PAUSE"),
        ("FIST", 0.9, "STOP"),
        ("POINT", 0.9, "SELECT"),
        ("THUMBS_UP", 0.9, "CONFIRM"),
        ("UNKNOWN_GESTURE", 0.9, "UNKNOWN"),
        ("FIST", 0.5, "UNKNOWN"),
    ],
)
def test_map_gesture_to_command(gesture, confidence, expected):
    assert map_gesture_to_command(gesture, confidence) == expected