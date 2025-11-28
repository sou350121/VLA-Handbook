# 触觉与感知 (Tactile & Sensors)

本章节汇总了赋予机器人"触觉"的关键传感器组件。

## Qianjue (千觉机器人)
> **Slogan**: "赋予机器人比人类更灵敏的触觉。"
> **官网**: [http://qianjue.jinjiayun.net.cn/](http://qianjue.jinjiayun.net.cn/)

### **XENSE-G1 (多模态触觉传感器系列)**
- **详情**: [产品页面](http://qianjue.jinjiayun.net.cn/product/367)
- **定位**: 通用型高精度触觉传感器平台。
- **核心参数**:
    - **分辨率**: **50,000 点/cm²** (信息密度是人类手指的 800 倍)。
    - **精度**: X/Y 轴 **0.03mm**, Z 轴 **0.06mm**。
    - **延迟**: 深度场处理时间仅 **10ms**。
- **多模态感知**:
    - **三维力**: 能同时感知法向力 (Pressure) 和切向力 (Shear/Friction)。
    - **纹理与材质**: 能识别物体表面的微小纹理 (如织物、金属拉丝)。
    - **滑移检测**: 毫秒级检测物体滑落趋势，实现动态握力调整。

### **G1-WS (Wedge Shape / 夹爪专用型)**
- **详情**: [产品页面](http://qianjue.jinjiayun.net.cn/product/386)
- **设计**: **楔形结构**。
    - 前端最薄处仅 **5mm**，专为狭窄空间作业设计 (如密集排布的线缆插拔)。
- **适配**: 完美适配 Robotiq 2F-85/140, Agile Robots 等主流夹爪。
- **应用**:
    - **盲插**: 在视觉被遮挡的情况下，靠触觉反馈完成 USB/网线插入。
    - **精密装配**: 轴承、齿轮等精密零部件的装配。

### **XENSE-Fingertip (指尖型)**
- **详情**: [产品页面](http://qianjue.jinjiayun.net.cn/product/368)
- **设计**: 仿生指尖曲面设计，模拟人类手指腹。
- **用途**: 适用于灵巧手 (Dexterous Hand) 指尖，增强抓取的稳定性和感知能力。

### **G1-OS (柔性层替换件)**
- **定义**: G1 系列传感器的**可更换柔性表皮**。
- **特点**:
    - **耗材化设计**: 表面磨损后可快速更换 (30秒内)，无需更换整个传感器。
    - **自标定**: 更换后支持自动校准，保证数据一致性。
    - **低成本维护**: 大幅降低了工业场景下的长期使用成本。

### **Xense_Sim (触觉仿真工具)**
- **平台**: 基于 NVIDIA Isaac Sim。
- **功能**:
    - **高保真仿真**: 精确模拟切向力 (摩擦力) 和法向力。
    - **Sim-to-Real**: 填补了触觉数据在仿真中的空白，加速策略训练。

## GelSight Family
- **官网**: [https://www.gelsight.com/](https://www.gelsight.com/)
- **厂商**: GelSight
- **原理**: **视触觉 (Visuo-tactile)**。基于光学原理，生成极高精度的 3D 表面拓扑图。
- **产品线**:
    - **GelSight Mini**: 标准科研版，高精度。
    - **GelSight Slim**: 更紧凑，适合集成到灵巧手。

## Daimon (戴盟机器人)
- **官网**: [http://www.dmrobot.com/](http://www.dmrobot.com/)
- **产品**: **DM-Tac 系列** (W/F/G)。
- **特点**:
    - **高分辨率**: >40,000 感测单元/cm²。
    - **多模态**: 具备 DM-Tac G (视触觉夹爪)，集成度高。
- **参考**: [RobotShop](https://www.robotshop.com/products/daimon-robotics-dm-tac-w-tactile-sensor-large)

## 9DTact
- **来源**: 学术成果 / 开源项目
- **特点**: **超紧凑 (Compact)**。体积仅为 32.5mm x 25.5mm x 25.5mm，适合灵巧手集成。
- **能力**: 6D 力估计 (1D 法向 + 2D 切向 + 3D 扭矩) + 3D 重建。
- **参考**: [Project Page](https://github.com/Rwsl/9DTact)

## Tac3D
- **来源**: 学术成果
- **原理**: **虚拟双目视觉 (Virtual Binocular)**。通过镜面反射实现单相机双视角，解决深度感知问题。
- **参考**: [arXiv Paper](https://arxiv.org/abs/2308.08866)

## Digit (Open Source)
- **官网**: [https://digit.ml/](https://digit.ml/)
- **来源**: Meta AI (Facebook) 开源。
- **特点**:
    - **低成本**: 易于制造，BOM 成本低。
    - **生态**: 配套 PyTouch 开源库，社区活跃。
- **参考**: [GitHub](https://github.com/facebookresearch/digit-interface)

## Parsen (帕西尼感知)
- **官网**: [https://www.paxini.com/](https://www.paxini.com/)
- **产品**: PX 系列 (PX-6AX, PX-3AX)。
- **特点**:
    - **多维触觉**: 专注于多维触觉阵列。
    - **高性价比**: 消费级产品 (PX-3AX) 价格极具竞争力 (约 ~200 RMB)。
- **参考**: [36Kr 报道](https://36kr.com/p/2312636662486405)

## Xela uSkin
- **官网**: [https://xelarobotics.com/](https://xelarobotics.com/)
- **厂商**: Xela Robotics
- **原理**: **磁触觉**。
- **特点**: 柔性皮肤，三维力感知，常用于 Allegro Hand 指尖。

---
[← Back to Product Index](./README.md)
