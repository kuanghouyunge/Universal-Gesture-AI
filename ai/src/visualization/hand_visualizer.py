
import cv2


class HandVisualizer:
    def __init__(self, connections):
        self.connections = connections

    def draw_hand(
        self,
        frame,
        landmarks,
        handedness,
        gesture,
        confidence,
        command,
        hand_index=0,
    ):
        """绘制手部关键点、骨架和识别结果。"""

        height, width = frame.shape[:2]

        points = [
            (int(landmark.x * width), int(landmark.y * height))
            for landmark in landmarks
        ]

        # 绘制手部骨架
        if self.connections is not None:
            for connection in self.connections:
                start = points[connection.start]
                end = points[connection.end]
                cv2.line(frame, start, end, (255, 0, 0), 2)

        # 绘制关键点
        for x, y in points:
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # 每只手分配独立的信息显示区域
        text_x = 20 + hand_index * 300

        gesture_text = f"{gesture} {confidence:.0%}"

        cv2.putText(
            frame,
            gesture_text,
            (text_x, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Command: {command}",
            (text_x, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            f"Landmarks: {len(points)}",
            (text_x, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # 绘制左右手标签
        cv2.putText(
            frame,
            handedness,
            (points[0][0], points[0][1] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        return frame
