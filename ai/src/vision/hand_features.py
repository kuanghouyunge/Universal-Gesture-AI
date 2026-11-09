
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

def extract_joint_angles(landmarks):
    """
    提取五根手指的主要关节角度。

    输出：长度为 5 的 NumPy 数组。
    """

    points = np.array(
        [[lm.x, lm.y, lm.z] for lm in landmarks],
        dtype=np.float32
    )

    if points.shape != (21, 3):
        raise ValueError("Expected 21 landmarks with XYZ coordinates")

    # 五根手指主要关节的位置
    joint_indices = [
        (2, 3, 4),       # 拇指
        (6, 7, 8),       # 食指
        (10, 11, 12),    # 中指
        (14, 15, 16),    # 无名指
        (18, 19, 20)     # 小指
    ]

    angles = []

    for a, b, c in joint_indices:
        # 以中间关节为顶点构建两个向量
        v1 = points[a] - points[b]
        v2 = points[c] - points[b]

        # 计算向量夹角
        denominator = np.linalg.norm(v1) * np.linalg.norm(v2)

        if denominator < 1e-6:
            angles.append(0.0)
            continue

        cos_angle = np.dot(v1, v2) / denominator
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        angle = np.arccos(cos_angle)

        # 归一化到 0~1
        angles.append(angle / np.pi)

    return np.array(angles, dtype=np.float32)