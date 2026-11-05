from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI(
    title="Universal Gesture AI",
    version="0.1.0"
)


class Prediction(BaseModel):
    mode: str
    language: str
    label: Optional[str] = None
    confidence: float = 0.0
    status: str = "waiting"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "Universal Gesture AI"
    }


@app.post("/predict", response_model=Prediction)
def predict():
    # TODO:
    # 1. 获取摄像头帧或特征序列
    # 2. 进行关键点提取
    # 3. 送入 CSL 时序模型
    # 4. 返回预测结果

    return Prediction(
        mode="sign_recognition",
        language="CSL",
        label=None,
        confidence=0.0,
        status="model_not_ready"
    )