import numpy as np
import pytest
from types import SimpleNamespace

from src.vision.hand_features import (
    normalize_landmarks,
    normalize_hand_rotation,
    extract_joint_angles,
)


def create_landmarks(points=None):
    if points is None:
        points = np.zeros((21, 3), dtype=np.float32)

    return [
        SimpleNamespace(x=float(p[0]), y=float(p[1]), z=float(p[2]))
        for p in points
    ]


def test_normalize_landmarks():
    points = np.zeros((21, 3), dtype=np.float32)
    points[:, 0] = np.arange(21)

    result = normalize_landmarks(create_landmarks(points))

    assert result.shape == (63,)
    assert result.dtype == np.float32
    assert np.allclose(result[:3], [0, 0, 0])
    assert np.isclose(np.max(np.linalg.norm(result.reshape(21, 3), axis=1)), 1.0)


def test_normalize_landmarks_zero_input():
    result = normalize_landmarks(create_landmarks())

    assert np.all(result == 0)


def test_normalize_landmarks_invalid_input():
    with pytest.raises(ValueError):
        normalize_landmarks(create_landmarks(np.zeros((20, 3))))


def test_normalize_hand_rotation():
    points = np.zeros((21, 3), dtype=np.float32)
    points[9] = [0, 1, 0]
    points[5] = [-1, 0, 0]
    points[17] = [1, 0, 0]

    result = normalize_hand_rotation(
        create_landmarks(points),
        "Left",
        640,
        480,
    )

    assert result.shape == (63,)
    assert result.dtype == np.float32
    assert np.all(np.isfinite(result))


def test_normalize_hand_rotation_degenerate():
    result = normalize_hand_rotation(
        create_landmarks(),
        "Left",
        640,
        480,
    )

    assert np.all(result == 0)


def test_extract_joint_angles():
    points = np.zeros((21, 3), dtype=np.float32)

    joint_indices = [
        (2, 3, 4),
        (6, 7, 8),
        (10, 11, 12),
        (14, 15, 16),
        (18, 19, 20),
    ]

    for a, b, c in joint_indices:
        points[a] = [1, 0, 0]
        points[b] = [0, 0, 0]
        points[c] = [0, 1, 0]

    result = extract_joint_angles(create_landmarks(points))

    assert result.shape == (5,)
    assert result.dtype == np.float32
    assert np.allclose(result, 0.5)


def test_extract_joint_angles_degenerate():
    result = extract_joint_angles(create_landmarks())

    assert np.all(result == 0)


def test_extract_joint_angles_invalid_input():
    with pytest.raises(ValueError):
        extract_joint_angles(create_landmarks(np.zeros((20, 3))))