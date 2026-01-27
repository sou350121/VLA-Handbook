# Perception（感知）总索引：Sensing × Calibration × Sync × Data Quality (Perception Index)

> **定位**：面向 VLA 真机落地的“感知工程”主入口。  
> 覆盖：传感器选型与拓扑 → 标定与坐标系 → 多模态同步 → 数据质量/失败模式 → 在线监控与评估。  
> **写作规范**：遵循仓库根目录 `AGENT.md`（部署类文档结构：环境/硬件 → 步骤 → 配置/参数 → 常见坑 → 参考）。

> 本页是**唯一主入口**；原 `deployment/sensing-system/` 已迁移并删除（不再保留旧索引）。

---

## 1) 传感器与拓扑 (Sensors & Topology)

本节聚焦“传感器与拓扑”的可落地条目（会持续扩展）：

- **[Hyperpacked piezoelectric-powered capacitive sensor array（Nat. Sens. 2026）](./hyperpacked_piezocapacitive_vibration_sensor_array.md)**：自供能电容振动传感阵列，80–5,000 Hz 平坦频响，用于语音/呼吸/音乐等高保真振动检测。
- **[SuperTac + DOVE：仿生多模态触觉传感与触觉语言模型（Nat. Sens. 2026）](./supertac_dove_biomimetic_multimodal_tactile_sensing.md)**：多光谱（UV/VIS/NIR/MIR）+ 摩擦电 + IMU 的 1mm 触觉皮肤，并用 8.5B 触觉语言模型做语义理解与推理。
- **[触觉阵列算法：电容阵列 vs 压阻阵列（含视触觉）](./tactile_array_algorithms_capacitive_piezoresistive.md)**：从原始 taxel/触觉图像到 CoP/面积/法向代理 + 预滑/滑移特征与状态机；覆盖漂移、串扰、滞回/蠕变与视触觉（marker tracking / 深度重建）。

---

## 2) 标定与坐标系 (Calibration & Frames)

- **[相机标定与手眼对齐 (Camera Calibration)](../camera_calibration_eye_in_hand.md)**：Eye-in-Hand vs Eye-to-Hand 标定实战。

---

## 3) 同步与时间戳 (Sync & Timestamping)

- **[多模态数据同步技术 (Multimodal Sync)](../multimodal_data_synchronization.md)**：解决 RGB-D 与高频控制（1000Hz）的时间对齐难题。

---

## 4) 数据质量、观测性与失败模式 (Data Quality & Failure Modes)

- **[触觉集成挑战 (Tactile Integration)](../tactile_sensor_integration_challenges.md)**：触觉传感器与夹爪集成的工程难点（硬件/布线/噪声/耐久）。
- **[LingBot-Depth：用 MDM 修复透明/反光深度失效，让 RGB-D “看见玻璃”](./lingbot_depth_transparent_reflective_depth_enhancement.md)**：把透明/反光的深度孔洞当作学习信号，提升深度覆盖率与下游抓取/追踪可用性（含启示与不足）。

---

## 5) 在线监控与评估 (Monitoring & Evaluation)

- TODO：感知健康度、延迟分布（P50/P95）、丢帧率、漂移检测、自动报警与回放复盘。

---

## workflow：你给我论文/资料时，我会怎么落地

你只要把论文链接/笔记丢给我，我会按以下步骤做增量更新：

- **归类**：判断它属于“选型/标定/同步/质量/监控”等哪个子主题，并决定落在 `deployment/perception/` 或复用现有 `deployment/*.md`。
- **落文档**：按 `AGENT.md` 的部署文档结构写（环境/硬件 → 步骤 → 配置/参数 → 常见坑 → 参考），不确定的数据用 `TODO/待证` 标记，绝不编造指标。
- **同步索引**：更新本文件（`deployment/perception/README.md`）以及必要时更新 `deployment/README.md` 的入口链接。
- **链接自检**：确保新增/修改的内部链接都可点击且目标存在。

---
[← Back to Deployment](../README.md)

