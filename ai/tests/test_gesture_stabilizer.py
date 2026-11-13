
from src.features.gesture_stabilizer import GestureStabilizer


def test_initial_state():
    stabilizer = GestureStabilizer()

    assert len(stabilizer.history) == 0


def test_command_requires_three_occurrences():
    stabilizer = GestureStabilizer()

    assert stabilizer.update("STOP") == "UNKNOWN"
    assert stabilizer.update("STOP") == "UNKNOWN"
    assert stabilizer.update("STOP") == "STOP"


def test_different_commands_do_not_stabilize():
    stabilizer = GestureStabilizer()

    assert stabilizer.update("STOP") == "UNKNOWN"
    assert stabilizer.update("SELECT") == "UNKNOWN"
    assert stabilizer.update("CONFIRM") == "UNKNOWN"


def test_unknown_command_returns_unknown():
    stabilizer = GestureStabilizer()

    assert stabilizer.update("UNKNOWN") == "UNKNOWN"


def test_sliding_window():
    stabilizer = GestureStabilizer()

    stabilizer.update("STOP")
    stabilizer.update("STOP")
    stabilizer.update("SELECT")
    stabilizer.update("SELECT")

    assert stabilizer.update("STOP") == "STOP"


def test_reset():
    stabilizer = GestureStabilizer()

    stabilizer.update("STOP")
    stabilizer.update("STOP")
    stabilizer.reset()

    assert len(stabilizer.history) == 0
    assert stabilizer.update("STOP") == "UNKNOWN"
