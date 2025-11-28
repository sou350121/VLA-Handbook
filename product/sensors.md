# 触觉与感知 (Tactile & Sensors)

本章节汇总了赋予机器人"触觉"的关键传感器组件，涵盖了从工业级高精度传感器到开源低成本方案。

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

---

## GelSight Family
> **官网**: [https://www.gelsight.com/](https://www.gelsight.com/)
> **原理**: **视触觉 (Visuo-tactile)**。通过高分辨率相机拍摄弹性体表面的变形，生成极高精度的 3D 表面拓扑图。

### **GelSight Mini**
- **定位**: 标准科研版，即插即用。
- **核心参数**:
    - **尺寸**: 32mm x 28mm (紧凑型)。
    - **分辨率**: **微米级 (Micron-level)**，远超人类手指分辨率。
    - **采样率**: >30Hz (3D map), >60Hz (2D image)。
- **参考价格**: ~$500 USD。
- **应用**: 表面缺陷检测、纹理识别、精密操作研究。

### **GelSight Slim**
- **定位**: 集成型传感器。
- **特点**: 更轻薄的外形设计，专为集成到灵巧手 (如 Wonik Allegro, Shadow Hand) 指尖而优化。

---

## Daimon (戴盟机器人)
> **官网**: [http://www.dmrobot.com/](http://www.dmrobot.com/)
> **核心技术**: 光学视触觉技术。

### **DM-Tac W Series (W/M/S)**
- **定位**: 通用视触觉传感器，提供大/中/小三种尺寸。
- **核心参数 (Medium版)**:
    - **尺寸**: 57.4 × 35.0 × 25.0 mm。
    - **重量**: 61g。
    - **分辨率**: **>40,000 感测单元/cm²**。
    - **采样率**: **120Hz** (极高帧率，适合动态控制)。
- **参考价格**: ~$1,000 USD (Medium)。
- **参考**: [RobotShop Link](https://www.robotshop.com/products/daimon-robotics-dm-tac-w-tactile-sensor-large)

### **DM-Tac G (Visuotactile Gripper)**
- **定位**: 视触觉一体化夹爪。
- **特点**: 将视触觉传感器直接集成在平行夹爪指尖，开箱即用。

---

## 9DTact
> **来源**: 学术成果 / 开源项目
> **参考**: [Project Page](https://github.com/Rwsl/9DTact)

- **特点**: **超紧凑 (Compact)**。
    - 尺寸仅为 **32.5mm x 25.5mm x 25.5mm**，比传统视触觉传感器小得多。
- **能力**:
    - **6D 力估计**: 能同时解算 1D 法向力 + 2D 切向力 + 3D 扭矩。
    - **3D 重建**: 高精度的接触面几何重建。
- **优势**: 适合安装在空间受限的灵巧手手指上。

---

## Tac3D
> **来源**: 学术成果
> **参考**: [arXiv Paper](https://arxiv.org/abs/2308.08866)

- **原理**: **虚拟双目视觉 (Virtual Binocular)**。
    - 创新性地利用镜面反射，在单相机视野内构建双目视角，从而解决传统单目视触觉传感器的深度感知难题。
- **优势**: 结构简单，但能提供更准确的 3D 变形信息。

---

## Digit (Open Source)
> **官网**: [https://digit.ml/](https://digit.ml/)
> **来源**: Meta AI (Facebook) 开源。

- **定位**: **低成本、可制造性强**的视触觉传感器。
- **核心参数**:
    - **分辨率**: Digit 360 版本高达 **8.3 Million taxels**。
    - **灵敏度**: 可检测 **1mN** 的微小力。
- **生态**:
    - **PyTouch**: Meta 提供的开源触觉处理库，支持机器学习模型训练。
    - **社区**: 广泛用于学术界 (如 CMU, MIT) 的大规模触觉数据采集。
- **参考**: [GitHub Repository](https://github.com/facebookresearch/digit-interface)

---

## Parsen (帕西尼感知)
> **官网**: [https://www.paxini.com/](https://www.paxini.com/)
> **核心技术**: 多维触觉阵列 (6D Hall Array)。

### **PX 系列**
- **PX-6AX**: 工业级高精度六维力触觉传感器。
- **PX-3AX (消费级)**:
    - **价格**: **~200 RMB** (极具破坏力的价格)。
    - **特点**: 专为大规模部署设计，适合低成本机器人或教育用途。
- **应用**: 人形机器人皮肤、灵巧手触觉反馈。
- **参考**: [36Kr 报道](https://36kr.com/p/2312636662486405)

---

## Xela uSkin
> **官网**: [https://xelarobotics.com/](https://xelarobotics.com/)
> **原理**: **磁触觉 (Magnetic Tactile)**。

- **特点**:
    - **柔性 (Flexible)**: 像皮肤一样柔软，可弯曲贴合在曲面指尖上 (如 Allegro Hand, iCub)。
    - **三维力**: 每个触点 (Taxel) 都能独立测量 X/Y/Z 三轴力。
- **核心参数**:
    - **灵敏度**: <1gf (极高灵敏度)。
    - **量程**: 法向力可达 18N。
- **优势**: 相比视触觉传感器，它更薄、更软，且没有相机的体积限制。

---
[← Back to Product Index](./README.md)
