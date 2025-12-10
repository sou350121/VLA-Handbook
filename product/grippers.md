# 平行夹爪 (Parallel Grippers)

本章节汇总了主流的二指平行夹爪，它们是目前最具身智能落地最常用的末端执行器 (End Effector)。

## Robotiq 2F-85 / 2F-140
- **厂商**: Robotiq (加拿大)
- **地位**: **工业界标准**。几乎所有的科研论文 (如 Google RT-1) 和高端协作臂都默认适配 Robotiq。
- **核心参数**:
    - **行程 (最大开口)**: 85mm (2F-85) / 140mm (2F-140)。
    - **抓持力**: 20-235N (可调)。
    - **反馈**: 内置抓取检测 (Grip Detection)，支持位置、速度、力控制。
- **优点**: 极其稳定，ROS 驱动完善 (官方支持)，即插即用。
- **缺点**: 价格较贵。

## DH Robotics (大寰机器人)
- **厂商**: DH Robotics (中国)
- **地位**: **高性价比替代方案**。
- **AG-95**:
    - **对标**: 直接对标 Robotiq 2F-85/140。
    - **行程 (最大开口)**: 95mm。
    - **特点**: 兼容性好，价格亲民，广泛用于国内高校和科研机构。
- **PGE 系列**:
    - **特点**: 工业级电动夹爪，体积小，精度高。

## Daimon DM-Tac G (Visuotactile Gripper)
- **厂商**: Daimon Robotics (戴盟机器人)
- **定位**: 视触觉一体化夹爪。
- **特点**: 将视触觉传感器直接集成在平行夹爪指尖，开箱即用。
- **能力**: 集成高精度视触觉，每秒采集 **900万** 组数据，支持微米级接触变化识别 (如硬度辨识、滑移检测)。

---

> **Note**: 部分机械臂厂商提供深度集成的原厂夹爪，例如：
> - **Franka Hand**: 详见 [Franka Emika](../product/arms.md#franka-emika-research-3--production)
> - **Agile Robots Adaptive Gripper**: 详见 [Agile Robots Diana 7](../product/arms.md#agile-robots-diana-7)
>
> **Technical Insight**: 想要了解夹爪与传感器集成的工程难点？请阅读 **[触觉传感器集成难点](../deployment/sensor_integration.md)**。

---

## 📐 统一条目模板（便于扩展）
- **型号 | 厂商 | 年份**
- **核心规格**：行程/开口 | 抓力 | 重复精度 | 自重 | IP 等级
- **传感**：力/位置/视触觉集成
- **接口与总线**：EtherCAT/CAN/RS485/Ethernet/Modbus，I/O，供电 (24/48V)
- **法兰与兼容性**：安装法兰/尺寸、兼容的机械臂/快换
- **软件栈**：SDK/ROS 2/Modbus/插件支持
- **安全**：力限、软碰撞、机械限位
- **典型搭配**：推荐臂/传感器/算力/快换
- **维护与备件**：指套/垫片/线缆等易耗，保养周期
- **Last Update**：YYYY-MM-DD

## 📊 速查表骨架
| 型号 | 行程 | 抓力 | 重复精度 | 自重 | 传感 | 接口/总线 | 法兰/兼容 | 供电 | 备注 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| TODO |  |  |  |  |  |  |  |  |  |

---
[← Back to Product Index](./README.md)
