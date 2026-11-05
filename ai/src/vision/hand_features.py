
import numpy as np


def normalize_landmarks(landmarks):
    """
    将 MediaPipe 的 21 个手部关键点
    转换为归一化的 63 维特征。

    输入：21 个 landmark
    输出：长度为 63 的 NumPy 数组
    """

    points = np.array(
        [[lm.x, lm.y, lm.z] for lm in landmarks],
        dtype=np.float32
    )

    if points.shape != (21, 3):
        raise ValueError("Expected 21 landmarks with XYZ coordinates")

    # 以手腕（第 0 个点）作为坐标原点
    points = points - points[0]

    # 根据手掌最大关键点距离进行缩放
    scale = np.max(np.linalg.norm(points, axis=1))

    if scale < 1e-6:
        return np.zeros(63, dtype=np.float32)

    points = points / scale

    # 展平为 63 维特征
    return points.flatten()
