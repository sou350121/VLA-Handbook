# 手势控制灵巧手：MediaPipe + WujiHand 实战项目 (MediaPipe + WujiHand Project)

> **定位**：用视觉手势（MediaPipe Hands）实时驱动灵巧手（以 WujiHand 为例）的端到端工程落地笔记。
> **适用**：面试中回答“你如何做真机闭环 / 低延迟遥操作 / 多模态对齐”的项目型问题。
> **状态**：本文为最小可读骨架；细节与数据会持续补齐（不确定处以 TODO 标注）。
## 1. 系统总览 (System Overview)

核心链路可以抽象成“感知 → 映射 → 控制 → 反馈”的闭环：

```text
Camera
  │
  ▼
MediaPipeHands(landmarks,handedness,score)
  │
  ├─(optional) Filtering/Smoothing/OutlierReject
  │
  ▼
Retargeting(landmarks→hand_pose→joint_targets)
  │
  ▼
HandController(CAN/EtherCAT/TCP)
  │
  ▼
WujiHand(20+DOF)
```

## 2. 关键工程问题 (Key Engineering Problems)

### 2.1 低延迟 (Latency)
- **目标**：端到端延迟（摄像头→手指动作）尽量 <100ms（视场景而定）。
  - TODO：补充你/项目的实际测量方法与数据（例如 P50/P95、端到端 vs 分段）。
- **常见瓶颈**：相机采集、推理线程阻塞、串行 retargeting、控制链路频率不足、USB/网络抖动。
### 2.2 稳定性与抖动 (Jitter)
- **过滤**：对 landmarks 做 EMA/OneEuro/Kalman（按帧率与任务需求取舍）。
- **死区/限速**：对关节命令设置 deadband、slew-rate limit，避免小抖动引发高频抖动。
### 2.3 手势→关节映射 (Retargeting)
- **最小可行方案**：从关键点几何构造指关节屈伸角/张开度，再映射到手的关节空间。
- **更稳方案**：学习型 retargeting（以标定数据拟合），并加入物理/关节限位约束。
- TODO：补充 WujiHand 的关节定义、零位、限位与控制接口细节。
## 3. 实现建议 (Implementation Notes)

### 3.1 进程/线程模型
- **建议**：采集、推理、控制分线程/分进程，避免相互阻塞；用无锁队列或 ring buffer 传递最新状态。
- **时间戳**：每个阶段打点，便于定位 P95 变差来自哪里。
### 3.2 控制接口与频率
- **控制频率**：理想情况下，手端控制回路应高于视觉帧率（例如视觉 30Hz，控制 100-200Hz 进行插值/保持）。
- TODO：补充 WujiHand 的通信协议与推荐控制频率（CANFD/EtherCAT 等）。
## 4. 常见坑 (Pitfalls)
- **镜像/左右手**：摄像头镜像导致 handedness 反转，映射会“反手”。
- **坐标系**：相机坐标/手部局部坐标/手模型关节坐标需统一。
- **遮挡**：遮挡时 landmarks 漂移，需要置信度门限 + fallback（保持上次稳定姿态）。
## 5. 参考 (References)
- MediaPipe Hands 官方文档：`https://developers.google.com/mediapipe/solutions/vision/hand_landmarker`
- MediaPipe GitHub：`https://github.com/google-ai-edge/mediapipe`

---
[← Back to Deployment](./README.md)
