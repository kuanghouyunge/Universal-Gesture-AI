
from src.features.sequence_buffer import SequenceBuffer


def test_add_and_get_sequence():
    buffer = SequenceBuffer()

    buffer.add([1, 2, 3])
    buffer.add([4, 5, 6])

    assert buffer.get_sequence() == [
        [1, 2, 3],
        [4, 5, 6],
    ]


def test_max_length():
    buffer = SequenceBuffer(max_length=3)

    for i in range(5):
        buffer.add(i)

    assert buffer.get_sequence() == [2, 3, 4]


def test_clear():
    buffer = SequenceBuffer()

    buffer.add([1, 2, 3])
    buffer.clear()

    assert buffer.get_sequence() == []


def test_is_ready():
    buffer = SequenceBuffer(max_length=20)

    for i in range(14):
        buffer.add(i)

    assert buffer.is_ready() is False

    buffer.add(14)

    assert buffer.is_ready() is True


def test_custom_min_frames():
    buffer = SequenceBuffer()

    for i in range(5):
        buffer.add(i)

    assert buffer.is_ready(min_frames=5) is True
    assert buffer.is_ready(min_frames=6) is False
