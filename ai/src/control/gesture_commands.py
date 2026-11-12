
GESTURE_COMMANDS = {
    "OPEN_PALM": "PAUSE",
    "FIST": "STOP",
    "POINT": "SELECT",
    "THUMBS_UP": "CONFIRM",
}

def map_gesture_to_command(gesture, confidence, threshold=0.7):
    """根据手势和置信度映射控制指令。"""

    if confidence < threshold:
        return "UNKNOWN"

    return GESTURE_COMMANDS.get(gesture, "UNKNOWN")