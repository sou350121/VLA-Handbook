# 硬件选型与价格参考 (Hardware & Pricing)

本章节汇总了 VLA 算法落地常用的硬件设备，重点关注 **灵巧手 (Dexterous Hands)**，并提供参考价格（人民币）和官方链接。

> [!NOTE]
> 价格仅供参考，实际成交价受汇率、关税、代理商折扣及配置（如是否含触觉传感器）影响较大。

## 1. 灵巧手 (Dexterous Hands) [核心关注]
灵巧手是 VLA 算法（尤其是精细操作）落地的关键执行单元。

| 厂商 | 型号 | 自由度 (DOF) | 参考价格 (RMB) | 特点 | 链接 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Inspire Robots**<br>(因时机器人) | **RH56DFX** | 6 | ~7.4万 - 15.5万 | 商业化成熟，集成度高，常用于科研与人形机器人集成。支持力控。 | [官网](https://www.inspire-robots.com/en/product/dexterous-hands) |
| **Unitree**<br>(宇树科技) | **Dex3-1** | 7 | ~5.7万 - 6万 | 三指设计，力控，适合抓取任务。 | [官网](https://www.unitree.com/products/dex3-1) |
| | **G1 Hand** | 6 | ~7.2万 | Unitree G1 人形机器人的配套手，性价比高。 | [官网](https://www.unitree.com/products/g1) |
| **Agibot**<br>(智元机器人) | **OmniHand** | 12 (Active) | ~1.45万 | **高性价比**。适合低成本科研验证，模块化设计。 | [官网](https://www.agibot.com/) |
| **LEAP Hand** | **LEAP Hand** | 16 | ~1.4万 - 2.1万 | **开源项目**。成本低，可自行 3D 打印组装。Sim-to-Real 友好，学术界常用。 | [官网](https://www.leaphand.com/) |
| **Wonik Robotics** | **Allegro Hand** | 16 (4指) | ~11万 - 18万 | 经典的科研用手，资料丰富，但价格较贵。 | [官网](https://www.wonikrobotics.com/allegro-hand) |
| **Shadow Robot** | **Shadow Hand** | 20 | ~85万+ | **高端触觉**。接近人手灵活度，常用于遥操作 (Teleoperation) 和高端触觉研究。 | [官网](https://www.shadowrobot.com/) |

### 选型建议
- **低成本科研 / 学生**: 推荐 **LEAP Hand** (开源自制) 或 **Agibot OmniHand**。
- **通用抓取 / 工业验证**: 推荐 **Unitree Dex3-1** 或 **Inspire RH56DFX**。
- **高端触觉 / 遥操作**: 推荐 **Shadow Hand** (预算充足) 或 **Wonik Allegro Hand**。

## 2. 机械臂 (Robotic Arms)
VLA 算法通常需要机械臂作为载体。

| 厂商 | 型号 | 自由度 | 参考价格 (RMB) | 特点 | 链接 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Trossen Robotics** | **WidowX 250** | 6 | ~1.9万 | **入门科研首选**。ALOHA 项目标配，Python API 友好，适合复现 Mobile ALOHA。 | [官网](https://www.trossenrobotics.com/widowx-250-robot-arm.aspx) |
| **Universal Robots** | **UR5e** | 6 | ~20万 - 30万 | **工业级协作**。稳定性极强，精度高，ROS 支持完善。 | [官网](https://www.universal-robots.com/products/ur5-robot/) |
| **Elephant Robotics** | **myCobot 280** | 6 | ~3000 | 桌面级教育机械臂，适合极低预算入门。 | [官网](https://www.elephantrobotics.com/en/mycobot-en/) |

## 3. 移动底盘 (Mobile Bases)
用于实现移动操作 (Mobile Manipulation)。

| 厂商 | 型号 | 参考价格 (RMB) | 特点 | 链接 |
| :--- | :--- | :--- | :--- | :--- |
| **AgileX** (松灵) | **LIMO** | ~2万 - 4万 | 多模态（四轮/履带），适合教育和小型实验。 | [官网](https://www.agilex.ai/) |
| | **Scout Mini** | ~6万 | 越野能力强，适合室外环境。 | [官网](https://www.agilex.ai/) |
| | **Tracer** | ~7万 | 室内 AGV 底盘，适合平整地面。 | [官网](https://www.agilex.ai/) |

## 4. 人形/四足机器人 (Humanoid & Legged)
终极形态的具身智能载体。

| 厂商 | 型号 | 参考价格 (RMB) | 特点 | 链接 |
| :--- | :--- | :--- | :--- | :--- |
| **Unitree** (宇树) | **Go2** | ~2万 (Air) / ~8-16万 (Edu) | 四足狗。Edu 版支持高算力开发。 | [官网](https://www.unitree.com/go2/) |
| | **G1** | ~10万 | **量产人形**。性价比极高，适合作为人形机器人研究平台。 | [官网](https://www.unitree.com/g1/) |
| **Fourier** (傅利叶) | **GR-1** | ~90万 - 100万 | 高端人形，主要用于康复和科研。 | [官网](https://www.fftai.com/) |
| **Astribot** (星尘) | **S1** | ~50万 - 60万 (预计) | 高性能，动作极快，演示效果惊人。 | [官网](https://astribot.com/) |

## 5. 触觉传感器 (Tactile Sensors)
赋予机器人"触觉"的关键组件，对于盲盒摸索、材质识别至关重要。

| 厂商 | 产品 | 参考价格 | 特点 | 链接 |
| :--- | :--- | :--- | :--- | :--- |
| **Qianjue**<br>(千觉机器人) | **XENSE-G1** | 询价 (高端定制) | **多模态感知**。能同时感知三维力、滑动、纹理和物体形状。分辨率极高 (5万点/cm²)，适合精细装配。 | [官网](http://qianjue.jinjiayun.net.cn/product/35) |
| **GelSight** | **GelSight Mini** | ~$3,000 | **视触觉 (Visuo-tactile)**。基于光学原理，能生成极高精度的 3D 表面拓扑图。 | [官网](https://www.gelsight.com/) |
| **Xela Robotics** | **uSkin** | ~2万 - 4万 | **磁触觉**。柔性皮肤，三维力感知，常用于 Allegro Hand 指尖。 | [官网](https://xelarobotics.com/) |

## 6. 机器人公司与求职参考

> **Note**: 为了提供更详细的求职与行业信息，我们将公司列表移动到了独立的目录。

请访问 **[机器人公司与求职指南](../companies/README.md)** 查看：
- [中国头部机器人公司](../companies/china.md)
- [国际机器人公司 (Tesla, Figure, etc.)](../companies/international.md)
- [亚洲机器人公司 (SG/JP/TW/KR)](../companies/asia.md)
- [具身智能软件与平台](../companies/embodied_ai.md)

---
[← Back to Deployment](./README.md)


