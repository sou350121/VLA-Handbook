# 机器人产品汇总之 (Product Showcase)

本章节针对全球主流机器人公司的核心产品进行汇总与技术参数解析。面试中常被问到："你了解 Optimus Gen 2 的手有多少个自由度吗？" 或者 "Unitree G1 的关节扭矩是多少？"

## 目录
- **[国际巨头 (International Giants)](#1-国际巨头-international-giants)**
    - [Tesla (Optimus)](#tesla-optimus)
    - [Figure AI (Figure 01/02)](#figure-ai-figure-0102)
    - [1X Technologies (NEO)](#1x-technologies-neo)
    - [Sanctuary AI (Phoenix)](#sanctuary-ai-phoenix)
- **[中国力量 (China Power)](#2-中国力量-china-power)**
    - [Unitree (宇树科技)](#unitree-宇树科技)
    - [Agibot (智元机器人)](#agibot-智元机器人)
    - [Fourier (傅利叶智能)](#fourier-傅利叶智能)
    - [Astribot (星尘智能)](#astribot-星尘智能)
    - [Galbot (银河通用)](#galbot-银河通用)
- **[核心零部件 (Core Components)](#3-核心零部件-core-components)**
    - [Qianjue (千觉机器人)](#qianjue-千觉机器人)

---

## 1. 国际巨头 (International Giants)

### Tesla (Optimus)
> **Slogan**: "A general purpose, bi-pedal, humanoid robot capable of performing tasks that are unsafe, repetitive or boring."

#### **Optimus Gen 2 (Dec 2023)**
- **身高/体重**: ~173cm / ~57kg (比 Gen 1 轻了 10kg)。
- **自由度 (DOF)**: 全身 >40 DOF。
- **灵巧手 (The Hand)**:
    - **11 DOF**: 11 个自由度，设计注重鲁棒性。
    - **触觉**: 指尖配备触觉传感器 (Tactile Sensors)，能捏起鸡蛋。
    - **驱动**: 采用 **空心杯电机 + 丝杠** (Actuators integrated in forearm)。
- **核心技术**:
    - **End-to-End Neural Net**: 视觉输入 -> 关节输出，完全复用 FSD (Full Self-Driving) 的算法架构。
    - **2D 摄像头**: 仅依赖视觉，无 LiDAR。

### Figure AI (Figure 01/02)
> **Slogan**: "Deploying autonomous humanoid workers."

#### **Figure 02 (Aug 2024)**
- **外观**: 采用哑光黑外壳，设计现代。
- **核心升级**:
    - **VLM**: 与 OpenAI 深度合作，集成了 GPT-4o，具备极强的语音交互和推理能力 (Speech-to-Speech)。
    - **灵巧手**: 16 DOF，与人类手掌尺寸 1:1。
    - **电池**: 2.25 kWh，续航 >5 小时 (实战进工厂的关键)。
- **场景**: 已经在 BMW 工厂实测打工 (搬运箱子)。

### 1X Technologies (NEO)
> **Backed by**: OpenAI, Tiger Global.

#### **NEO (Beta)**
- **特点**: **软体肌肉 (Soft Robotics)**。
    - 采用软体机器人技术，电机设计注重柔顺性 (Compliance)。
    - **安全**: 即使撞到人也不会造成伤害，适合家庭场景。
- **操作**: 极其安静，动作像人类一样自然。

### Sanctuary AI (Phoenix)
> **Focus**: "Creating the world's first human-like intelligence in general-purpose robots."

#### **Phoenix (Gen 7)**
- **特点**: **注重遥操作 (Teleoperation Focused)**。
    - 拥有目前世界上最灵巧的手 (20 DOF, 仿生液压/气动混合驱动)，能做极其精细的动作 (如贴标签、分拣小零件)。
    - **策略**: 先通过高质量遥操作收集数据 (Data Engine)，再训练 AI。

---

## 2. 中国力量 (China Power)

### Unitree (宇树科技)
> **地位**: 知名足式与人形机器人公司。

#### **G1 (Humanoid Agent)**
- **价格**: **9.9万 RMB** (极具竞争力的价格)。
- **身高/体重**: 127cm / 35kg (偏小，像个大童)。
- **自由度**: 23-43 DOF (取决于是否选配灵巧手)。
- **关节**: 自研高扭矩关节电机，甚至能做"鲤鱼打挺"、"后空翻"。
- **用途**: 科研教育、二次开发、数据采集。

#### **H1 (Full-size Humanoid)**
- **特点**: **注重移动能力**。
- **速度**: 3.3 m/s，具备较强的鲁棒性。
- **驱动**: 纯电驱，爆发力极强。

### Agibot (智元机器人)
> **Founder**: 稚晖君 (华为天才少年)。

#### **远征 A2 (Yuanzheng A2)**
- **定位**: 工业与家庭通用的具身智能机器人。
- **特点**: 模块化设计，下半身可以换成轮式、足式。
- **灵巧手**: **灵犀 X1**。
    - 模块化设计，主动自由度 12 个。
    - 刚柔耦合传动，低成本高可靠。

### Fourier (傅利叶智能)
> **Background**: 康复机器人起家。

#### **GR-1 / GR-2**
- **核心优势**: **自研执行器 (FSA)**。
    - 既然买不到合适的电机，就自己造。FSA 关节集成了电机、减速器、驱动器，力矩密度极高。
- **负重**: GR-1 能负重 50kg (自身重量 55kg)，几乎是 1:1 的负重比。

### Astribot (星尘智能)
> **Focus**: "AI + Robot" 极致速度。

#### **S1**
- **特点**: **以高动态性能著称**。
- **演示**: 叠衣服、削苹果、甚至**颠锅**。动作速度达到了人类水平 (10m/s 末端速度)。
- **技术**: 采用基于模仿学习 (Imitation Learning) 的优化算法。

### Galbot (银河通用)
> **Focus**: 泛化抓取与大模型。

#### **Galbot G1**
- **构型**: 轮式底盘 + 升降躯干 + 双臂 + 灵巧手 + 多模态大模型。
- **特点**: 专注于**抓取 (Grasping)**。
    - 能够 Zero-shot 抓取透明物体、反光物体、堆叠物体。
    - 其核心竞争力在于**泛化抓取能力**。

---

## 3. 核心零部件 (Core Components)

### Qianjue (千觉机器人)
> **Focus**: "专注于高精度触觉感知技术。"

#### **XENSE-G1 (多模态触觉传感器)**
- **定位**: 通用型高精度触觉传感器，适用于指尖、掌心等多种位置。
- **核心参数**:
    - **分辨率**: **50,000 点/cm²** (信息密度是人类手指的 800 倍)。
    - **精度**: X/Y 轴 **0.03mm**, Z 轴 **0.06mm**。
    - **延迟**: 深度场处理时间仅 **10ms**。
- **多模态感知**:
    - **三维力**: 能同时感知法向力 (Pressure) 和切向力 (Shear/Friction)。
    - **纹理与材质**: 能识别物体表面的微小纹理 (如织物、金属拉丝)。
    - **滑移检测**: 毫秒级检测物体滑落趋势，实现动态握力调整。

#### **G1-WS (夹爪专用型)**
- **设计**: **楔形结构 (Wedge Shape)**。
    - 前端最薄处仅 **5mm**，专为狭窄空间作业设计 (如密集排布的线缆插拔)。
- **适配**: 完美适配 Robotiq 2F-85/140, Agile Robots 等主流夹爪。
- **应用**:
    - **盲插**: 在视觉被遮挡的情况下，靠触觉反馈完成 USB/网线插入。
    - **精密装配**: 轴承、齿轮等精密零部件的装配。

#### **XENSE-Fingertip (指尖型)**
- **设计**: 仿生指尖曲面设计，模拟人类手指腹。
- **用途**: 适用于灵巧手 (Dexterous Hand) 指尖，增强抓取的稳定性和感知能力。


---
[← Back to Root](../README.md)
