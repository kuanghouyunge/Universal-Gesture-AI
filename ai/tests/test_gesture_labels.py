from src.data.gesture_labels import GESTURE_LABELS


def test_gesture_labels():
    assert GESTURE_LABELS == {
        0: "OPEN_PALM",
        1: "FIST",
        2: "POINT",
        3: "THUMBS_UP",
        4: "V_SIGN",
    }


def test_gesture_labels_count():
    assert len(GESTURE_LABELS) == 5


def test_gesture_labels_unique():
    assert len(set(GESTURE_LABELS.values())) == 5