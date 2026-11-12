# Universal Gesture AI

一个基于计算机视觉与机器学习的通用手势识别和交互控制项目，旨在通过摄像头识别用户手势，并将其转换为对应的控制指令。

## 1. 项目简介

Universal Gesture AI 使用 MediaPipe 提取手部关键点，通过特征归一化和机器学习模型识别手势，实现从视觉输入到交互指令的转换。

项目采用模块化架构，后续计划扩展中国手语识别、人体姿态检测、人脸检测以及跨平台交互应用。

## 2. 已实现功能

- 摄像头实时画面读取
- MediaPipe 手部关键点检测
- 手部特征提取与归一化
- 基于机器学习模型的手势分类
- 手势与控制指令映射
- 手部骨架及识别结果实时可视化

### 当前支持的手势

| 手势 | 识别类别 | 对应指令 |
|---|---|---|
| 张开手掌 | OPEN_PALM | PAUSE |
| 握拳 | FIST | STOP |
| 食指指向 | POINT | SELECT |
| 竖起大拇指 | THUMBS_UP | CONFIRM |

## 3. 技术栈

- Python
- MediaPipe
- OpenCV
- NumPy
- Scikit-learn
- PyTorch（后续模型扩展）
- FastAPI（后续接口扩展）
- Flutter（计划中的跨平台应用）

## 4. 项目结构

```text
Universal Gesture AI/
├── ai/
│   ├── main.py
│   ├── requirements.txt
│   ├── models/
│   ├── datasets/
│   ├── data/
│   ├── training/
│   └── src/
│       ├── application/
│       ├── control/
│       ├── data/
│       ├── features/
│       ├── recognition/
│       ├── sign_language/
│       ├── vision/
│       └── visualization/
└── README.md
```

## 5. 环境配置

建议使用 Python 3.13 及独立虚拟环境。

进入 AI 项目目录：

```bash
cd ai
```

创建虚拟环境：

```bash
python -m venv .venv
```

激活虚拟环境（Windows PowerShell）：

```powershell
.venv\Scripts\Activate.ps1
```

安装依赖：

```bash
pip install -r requirements.txt
```

## 6. 运行项目

在 `ai` 目录下执行：

```bash
python main.py
```

启动摄像头实时手势识别。

按 `Q` 键退出程序。

注意：运行前需要确保 `ai/models/` 中存在所需的 MediaPipe 手部检测模型和手势分类模型。

## 7. 后续开发计划

- [ ] 完善手势识别与交互控制逻辑
- [ ] 建立自动化测试体系
- [ ] 优化手势识别准确率与鲁棒性
- [ ] 扩展动态手势识别
- [ ] 开发中国手语识别功能
- [ ] 接入 FastAPI 服务
- [ ] 开发 Flutter 跨平台交互应用

## 8. 项目目标

构建一个模块化、可扩展的通用手势交互系统，使计算机能够通过自然手势理解用户意图，并执行相应操作。