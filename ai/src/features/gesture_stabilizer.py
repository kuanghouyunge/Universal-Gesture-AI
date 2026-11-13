
from collections import deque
from collections import Counter


class GestureStabilizer:
    def __init__(self, window_size=5, min_count=3):
        self.history = deque(maxlen=window_size)
        self.min_count = min_count

    def update(self, command):
        self.history.append(command)

        if command == "UNKNOWN":
            return "UNKNOWN"

        counts = Counter(
            item for item in self.history
            if item != "UNKNOWN"
        )

        stable_command, count = counts.most_common(1)[0]

        if count >= self.min_count:
            return stable_command

        return "UNKNOWN"

    def reset(self):
        self.history.clear()
