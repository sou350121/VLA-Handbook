# Sensing System（感知系统）总索引 (Sensing System Index)

> **定位**：面向 VLA 真机落地的“感知系统工程”入口索引。\n+> 覆盖：传感器选型 → 标定与坐标系 → 多模态同步 → 数据质量与观测性 → 在线监控与评估。\n+> **写作规范**：遵循仓库根目录 `AGENT.md`（部署类文档结构：环境/硬件 → 步骤 → 配置/参数 → 常见坑 → 参考）。\n+
---

## 目录（会持续扩展）

### 1) 传感器与拓扑 (Sensors & Topology)
- TODO：相机（RGB/RGB-D/事件相机）、IMU、力/力矩、触觉、编码器、外部定位（Vicon/UWB）等选型与系统拓扑。

### 2) 标定与坐标系 (Calibration & Frames)
- TODO：内参/外参、手眼（eye-in-hand/eye-to-hand）、时间偏移估计、温漂与重标定策略。

### 3) 同步与时间戳 (Sync & Timestamping)
- 已有入口：[`multimodal_data_synchronization.md`](../multimodal_data_synchronization.md)
- TODO：硬同步（触发线/PTP） vs 软同步（时间戳对齐），以及“1000Hz 控制 + 30Hz 视觉”的桥接策略。

### 4) 数据质量、观测性与失败模式 (Data Quality & Failure Modes)
- TODO：遮挡、反光/透明、运动模糊、滚快门、触觉飘移、IMU bias；如何把这些转成可监控指标与 fallback。

### 5) 在线监控与评估 (Monitoring & Evaluation)
- TODO：感知健康度、延迟分布（P50/P95）、丢帧率、重投影误差、漂移检测、自动报警与回放复盘。

---

## 未来你给我论文/资料时，我会怎么落地（workflow）

你只要把论文链接/笔记丢给我，我会按以下步骤做增量更新：
- **归类**：判断它属于“选型/标定/同步/质量/监控”等哪个子主题，并决定落在 `deployment/sensing-system/` 还是复用现有 `deployment/*.md` 文档。
- **落文档**：按 `AGENT.md` 的部署文档结构写（环境/硬件 → 步骤 → 配置/参数 → 常见坑 → 参考），不确定的数据用 `TODO/待证` 标记，绝不编造指标。
- **同步索引**：更新本文件（`deployment/sensing-system/README.md`）以及必要时更新 `deployment/README.md` 的入口链接。
- **链接自检**：确保新增/修改的内部链接都可点击且目标存在。

---
[← Back to Deployment](../README.md)

