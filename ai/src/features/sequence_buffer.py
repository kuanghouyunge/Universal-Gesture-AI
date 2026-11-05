from collections import deque


class SequenceBuffer:

    def __init__(self, max_length=60):
        self.frames = deque(maxlen=max_length)

    def add(self, features):
        self.frames.append(features)

    def get_sequence(self):
        return list(self.frames)

    def clear(self):
        self.frames.clear()

    def is_ready(self, min_frames=15):
        return len(self.frames) >= min_frames