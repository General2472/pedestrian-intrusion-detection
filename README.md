# 基于离线视频的人员危险区域入侵检测

## 1. 项目简介

使用 OpenCV 逐帧读取本地视频，YOLOv8n 检测人员目标，结合多边形 ROI 区域判断与连续帧防抖机制，实现人员入侵报警、证据截图保存与结构化 CSV 事件日志。PC 端离线运行，本阶段不接摄像头、不接硬件。

处理流程：

```text
本地视频文件 → OpenCV 逐帧读取 → YOLOv8n 人员检测 → ROI 入侵判断
→ 连续帧防抖 → 报警/解除状态机 → 截图 + CSV 日志
```

## 2. 环境依赖

- Python 3.12 虚拟环境，解释器路径：`D:\venvs\embedded-ai\Scripts\python.exe`
- 安装依赖：

```bash
pip install ultralytics opencv-python numpy
```

- `torch` 由 ultralytics 自动安装（本项目使用 CUDA 版，可调用 GPU）；
- `csv`、`datetime`、`time` 为 Python 标准库，无需 pip 安装。

## 3. 运行方法

1. 选用上述虚拟环境解释器运行脚本；
2. 离线加载权重：`YOLO(r"本地路径\yolov8n.pt")`，不要只写模型名，避免检测不到时联网下载；
3. 修改 `cap = cv2.VideoCapture(...)` 中的视频路径；
4. 运行后窗口播放视频，按 `q` 退出。

## 4. 参数说明

| 参数 | 取值 | 说明 |
|---|---|---|
| 置信度阈值 | `conf >= 0.5` | 低于该分数的检测框丢弃 |
| 检测类别 | 仅 person（类别号 0） | 其他类别忽略 |
| 报警触发 | 连续 5 帧人员在 ROI 内 | NORMAL → ALARM |
| 报警解除 | 连续 10 帧无人在 ROI 内 | ALARM → NORMAL |
| ROI | `[(300,150),(500,150),(500,310),(300,310)]` | 按 768×576 画面设定的固定矩形 |
| 入侵判断 | 目标框中心点 | 中心点落在 ROI 内判定为进入 |

## 5. 测试结果

| 视频 | 场景 | ALARM | CLEAR | 结果 |
|---|---|---|---|---|
| sample_bbb.mp4 | 无人卡通画面 | 0 | 0 | 不误报 |
| seg1_enter.mp4 | 单人进入并停留 | 1 | 0 | 通过 |
| seg2_crowd.mp4 | 多人同时出现 | 2 | 1 | 通过 |
| seg3_leave.mp4 | 离开后再次进入 | 2 | 1 | 通过 |
| pedestrians.avi | 长视频多人场景 | 6 | 5 | 通过 |

## 6. 性能（RTX 5060 Laptop，YOLOv8n）

- 平均推理约 9~11 ms/帧；单帧处理总耗时约 10~12 ms/帧；
- 推理理论约 90~110 FPS，远高于测试视频的 10 FPS；
- 瓶颈在模型推理，后处理与绘制开销约 1ms，占比很小。

## 7. 输出说明

- 报警截图：`shortcuts/` 下按场景分目录保存，仅在刚触发 ALARM 那一帧保存（沿触发，不重复存图）；
- 事件日志：`logs/events*.csv`，字段为 `time, frame, event, confidence, bbox`，记录触发与解除事件，支持离线复盘。
