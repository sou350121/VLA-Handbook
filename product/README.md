# 机器人产品大百科 (Product Encyclopedia)

本章节以**产品形态**为维度，深度解析全球主流具身智能硬件。

## 📂 产品分类

### 1. **[具身智能本体 (Humanoid Robots)](./humanoids.md)**
> Tesla Optimus, Figure 02, Unitree G1, Agibot A2, Fourier GR-1, 1X NEO...

### 2. **[灵巧手 (Dexterous Hands)](./hands.md)**
> Shadow Hand, Inspire, Unitree Dex3, Agibot X1, LEAP Hand, Parsen DexH13, Daimon Hand...

### 3. **[平行夹爪 (Parallel Grippers)](./grippers.md)**
> Robotiq 2F-85, DH Robotics AG-95, Daimon Tac-G...

### 4. **[科研机械臂 (Research Arms)](./arms.md)**
> WidowX 250, UR5e, Unitree Z1, Kinova Gen3, xArm 6, Franka, Agile Robots Diana 7...

### 5. **[移动底盘 (Mobile Bases)](./mobile_bases.md)**
> AgileX LIMO/Scout, Unitree Go2...

### 6. **[触觉与感知 (Tactile & Sensors)](./sensors.md)**
> Qianjue XENSE, GelSight, Xela, Daimon Tac-W, Parsen PX-6AX, Tashan, Digit...

---

## 🔧 部署与选型指南（新增）

- **控制与驱动 (Actuation & Drives)**  
  - 伺服/关节模组、力矩传感器，EtherCAT/CAN 支持矩阵  
  - 与机械臂/手/夹爪的接口、电源、电流冗余

- **末端快换与 EOAT 生态 (Tooling & Quick-Change)**  
  - 快换板规格、法兰兼容性；吸盘/螺丝刀/焊接等 EOAT 选型  
  - 参考: 手/夹爪 [hands.md](./hands.md) · [grippers.md](./grippers.md)

- **视觉与感知套件 (Vision Kits)**  
  - RGB-D/双目/结构光/激光雷达选型，安装位姿与时间同步  
  - 参考: 触觉/传感 [sensors.md](./sensors.md)

- **边缘算力与主机 (Edge Compute)**  
  - Jetson/NUC/工控机配置，功耗/重量/接口 (PoE/USB3/CSI)  
  - 与移动底盘/本体的供电与网络拓扑  
  - 参考: 本体/底盘 [humanoids.md](./humanoids.md) · [mobile_bases.md](./mobile_bases.md)

- **通信与布线 (Comms & Cabling)**  
  - EtherCAT/CAN/RS485/Modbus/USB3/PoE 适配注意事项  
  - 线缆长度、屏蔽、应力释放与拖链路径

- **电源与安全 (Power & Safety)**  
  - 24V/48V 供电规划、急停/安全继电器/光栅  
  - 接地、防反接、防浪涌、保险丝选型

- **软件与中间件 (Software & Middleware)**  
  - ROS 2/MoveIt/Isaac/厂商 SDK 支持矩阵，固件升级风险提示  
  - 标定工具链：手眼标定、力控标定、相机内外参

- **典型组合方案 (Reference Bundles)**  
  - 场景化套装：桌面实验、移动+操作、灵巧手+触觉  
  - 列出推荐部件与关键注意事项，便于快速配置

- **维护与备件 (Maintenance & Spares)**  
  - 易耗件/备件清单：指套、线缆、减速机油脂、吸盘垫片  
  - 常见故障与排查 checklist

---
[← Back to Root](../README.md)
