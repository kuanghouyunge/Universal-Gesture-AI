import csv
from pathlib import Path

import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

AI_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = AI_DIR / "datasets" / "gestures" / "gesture_data.csv"
MODEL_DIR = AI_DIR / "models"
MODEL_PATH = MODEL_DIR / "gesture_classifier.pkl"

LABEL_NAMES = [
    "OPEN_PALM",
    "FIST",
    "POINT",
    "THUMBS_UP",
]

def load_dataset():
    features = []
    labels = []

    with open(DATA_PATH, "r", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        for row in reader:
            label_id = int(row["label_id"])

            # 首版只训练 0~3 四类
            if label_id not in range(4):
                continue

            feature = [
                float(row[f"feature_{i}"])
                for i in range(63)
            ]

            angles = [
                float(row[f"angle_{i}"])
                for i in range(5)
            ]

            features.append(feature + angles)
            labels.append(label_id)

    return np.array(features, dtype=np.float32), np.array(labels)


def main():
    print("正在加载数据...")

    X, y = load_dataset()

    print("样本数量:", len(X))
    print("特征维度:", X.shape[1])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("训练集:", len(X_train))
    print("测试集:", len(X_test))

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    print("正在训练模型...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\n准确率:", accuracy_score(y_test, predictions))

    print("\n分类报告:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1, 2, 3],
            target_names=LABEL_NAMES,
            zero_division=0,
        )
    )

    print("\n混淆矩阵:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1, 2, 3]))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("\n模型已保存:", MODEL_PATH)


if __name__ == "__main__":
    main()