
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

def normalize_hand_rotation(landmarks, handedness, image_width, image_height):
    """
    将手部关键点转换到手掌局部坐标系。

    消除平移、尺度和大部分整体旋转影响。
    输出：63 维特征。
    """

    points = np.array(
        [[lm.x, lm.y, lm.z] for lm in landmarks],
        dtype=np.float32
    )

    if points.shape != (21, 3):
        raise ValueError("Expected 21 landmarks with XYZ coordinates")

    # 修正 MediaPipe X/Y 坐标的尺度差异
    points[:, 1] *= image_height / image_width

    # 以手腕为原点
    points = points - points[0]

    # 建立手掌纵向轴：手腕 -> 中指 MCP
    forward = points[9]

    print("手掌纵向向量:", np.round(forward, 3))

    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-6:
        return np.zeros(63, dtype=np.float32)

    forward = forward / forward_norm
    # 建立手掌横向轴：小指 MCP -> 食指 MCP
    side = points[5] - points[17]

    print(
    "食指到小指方向:",
    np.round(points[5] - points[17], 3)
)

    # 去除横向轴在纵向轴上的分量
    side = side - np.dot(side, forward) * forward

    side_norm = np.linalg.norm(side)
    if side_norm < 1e-6:
        return np.zeros(63, dtype=np.float32)

    side = side / side_norm

    # 建立第三个轴
    normal = np.cross(forward, side)
    normal_norm = np.linalg.norm(normal)

    print("坐标轴点积:")
    print("side · forward:", np.dot(side, forward))
    print("side · normal:", np.dot(side, normal))
    print("forward · normal:", np.dot(forward, normal))

    if normal_norm < 1e-6:
        return np.zeros(63, dtype=np.float32)

    normal = normal / normal_norm

    # 统一左右手的法向轴方向
    if handedness == "Right":
        normal = -normal

    print(
    "局部坐标轴:",
    "side =", np.round(side, 3),
    "forward =", np.round(forward, 3),
    "normal =", np.round(normal, 3)
)

    # 投影到手掌局部坐标系
    local_points = np.column_stack((
        np.dot(points, side),
        np.dot(points, forward),
        np.dot(points, normal)
    ))

    # 根据手掌尺寸缩放
    scale = np.linalg.norm(points[9])

    if scale < 1e-6:
        return np.zeros(63, dtype=np.float32)

    local_points = local_points / scale

    return local_points.flatten()


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