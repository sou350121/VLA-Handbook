## 1. 入门科研与教育 (Entry-level Research & Education)

### WidowX 250
- **厂商**: Trossen Robotics
- **自由度**: 6 DOF。
- **特点**: **ALOHA 标配**。Python API 友好，适合复现 Mobile ALOHA，入门首选。
- **参考价格**: ~$1,900 USD。

### myCobot 280
- **厂商**: Elephant Robotics (大象机器人)
- **自由度**: 6 DOF。
- **特点**: 桌面级教育机械臂，适合极低预算入门 (~3000 RMB)。
- **生态**: 支持 ROS2, Python, myBlockly。

### myArm 300 Pi
- **厂商**: Elephant Robotics
- **自由度**: **7 DOF** (桌面级罕见)。
- **特点**: 世界最小的 7 轴桌面机械臂，搭载 Raspberry Pi 4B。
- **参数**: 负载 200g，半径 300mm，重复定位精度 ±0.5mm。
- **参考价格**: ~$1,500 USD。

---

## 2. 轻量级与移动操作 (Lightweight & Mobile Manipulation)

### Unitree Z1 (Air / Pro)
- **厂商**: Unitree (宇树科技)
- **自由度**: 6 DOF。
- **定位**: **四足机器人伴侣**。专为安装在 B2/Go2 等机器狗上设计。
- **核心参数**:
    - **负载**: 2kg (Air) / 3kg (Pro)。
    - **自重**: 仅 4.3kg (Air)，极轻。
    - **精度**: ±0.1mm。
- **参考价格**: ~$12,000 - $14,000 USD。

### Kinova Gen3
- **厂商**: Kinova (加拿大)
- **自由度**: 6 / 7 DOF。
- **特点**: **超轻量化 + 嵌入式视觉**。
    - 自带 2D/3D 视觉模块 (Depth Camera) 和力矩传感器。
    - 无控制柜设计 (控制器集成在底座)，非常适合移动底盘集成。
- **参数**: 负载 2kg/4kg，自重 7.2kg/8.2kg，无限旋转关节。
- **参考价格**: ~$30,000 - $40,000 USD。

---

## 3. 协作与自适应 (Collaborative & Adaptive)

### UFACTORY xArm 6
- **厂商**: UFACTORY (由于科技)
- **自由度**: 6 DOF。
- **特点**: **碳纤维材质**，高性价比中端机械臂。
- **参数**: 负载 5kg，半径 700mm，自重 12.2kg。
- **优势**: 极其完善的 Python/ROS SDK，广泛用于科研和轻工业。
- **参考价格**: ~$8,500 - $11,000 USD。

### Franka Emika (Research 3 / Production)
- **厂商**: Franka Emika (德国)
- **自由度**: 7 DOF。
- **地位**: **力控研究的金标准**。
- **特点**: 每个关节都配备高精度力矩传感器，支持极低阻抗控制 (Impedance Control)。
- **夹爪**: **Franka Hand** (自带)。深度集成，API 调用极其简单，支持灵敏力反馈。
- **参数**: 负载 3kg，半径 855mm，重复精度 ±0.1mm。
- **参考价格**: ~€25,000 (Research 版)。

### Agile Robots Diana 7
- **厂商**: Agile Robots (思灵机器人)
- **自由度**: 7 DOF。
- **特点**: 高精度力控，对标 Franka。
- **夹爪**: **Adaptive Gripper** (灵巧夹爪)。手指具有柔性/连杆机构，能自适应包裹物体，适合精密装配。
- **参数**: **负载 7kg** (同级别中较高)，半径 923mm，力控精度 0.5N。
- **应用**: 精密装配，医疗手术辅助。

### Flexiv Rizon 4 (拂晓)
- **厂商**: Flexiv (非夕科技)
- **自由度**: 7 DOF。
- **核心技术**: **自适应力控 (Adaptive Force Control)**。
- **参数**: 负载 4kg，精度 ±0.05mm，力控精度 0.1N。
- **特点**: 能够像人手一样处理复杂的不确定性接触任务 (如打磨、插拔)。
- **参考价格**: ~$18,000 - $24,000 USD。

---

## 4. 工业级协作 (Industrial Collaborative)

### UR5e
- **厂商**: Universal Robots (丹麦)
- **自由度**: 6 DOF。
- **地位**: **协作机器人鼻祖**。
- **特点**: 极其稳定，生态最丰富 (UR+ 插件库)，ROS 驱动最成熟。
- **参数**: 负载 5kg，半径 850mm。
- **缺点**: 力控性能不如 Franka/Flexiv (基于电流而非关节力矩传感器)。
- **参考价格**: ~$25,000 - $35,000 USD。

---
[← Back to Product Index](./README.md)
