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

---

## 1. 国际巨头 (International Giants)

### Tesla (Optimus)
> **Slogan**: "A general purpose, bi-pedal, humanoid robot capable of performing tasks that are unsafe, repetitive or boring."

#### **Optimus Gen 2 (Dec 2023)**
- **身高/体重**: ~173cm / ~57kg (比 Gen 1 轻了 10kg)。
- **自由度 (DOF)**: 全身 >40 DOF。
- **灵巧手 (The Hand)**:
    - **11 DOF**: 只有 11 个自由度 (相比 Shadow Hand 的 20+ 少)，但极其鲁棒。
    - **触觉**: 指尖配备触觉传感器 (Tactile Sensors)，能捏起鸡蛋。
    - **驱动**: 采用 **空心杯电机 + 丝杠** (Actuators integrated in forearm)。
- **核心技术**:
    - **End-to-End Neural Net**: 视觉输入 -> 关节输出，完全复用 FSD (Full Self-Driving) 的算法架构。
    - **2D 摄像头**: 仅依赖视觉，无 LiDAR。

### Figure AI (Figure 01/02)
> **Slogan**: "Deploying autonomous humanoid workers."

#### **Figure 02 (Aug 2024)**
- **外观**: 哑光黑外壳，极其科幻。
- **核心升级**:
    - **VLM**: 与 OpenAI 深度合作，集成了 GPT-4o，具备极强的语音交互和推理能力 (Speech-to-Speech)。
    - **灵巧手**: 16 DOF，与人类手掌尺寸 1:1。
    - **电池**: 2.25 kWh，续航 >5 小时 (实战进工厂的关键)。
- **场景**: 已经在 BMW 工厂实测打工 (搬运箱子)。

### 1X Technologies (NEO)
> **Backed by**: OpenAI, Tiger Global.

#### **NEO (Beta)**
- **特点**: **软体肌肉 (Soft Robotics)**。
    - 不像其他机器人是硬邦邦的金属，NEO 穿着类似运动服的软壳，且电机设计注重柔顺性 (Compliance)。
    - **安全**: 即使撞到人也不会造成伤害，适合家庭场景。
- **操作**: 极其安静，动作像人类一样自然。

### Sanctuary AI (Phoenix)
> **Focus**: "Creating the world's first human-like intelligence in general-purpose robots."

#### **Phoenix (Gen 7)**
- **特点**: **遥操作之王 (Teleoperation First)**。
    - 拥有目前世界上最灵巧的手 (20 DOF, 仿生液压/气动混合驱动)，能做极其精细的动作 (如贴标签、分拣小零件)。
    - **策略**: 先通过高质量遥操作收集数据 (Data Engine)，再训练 AI。

---

## 2. 中国力量 (China Power)

### Unitree (宇树科技)
> **地位**: 机器人界的 "大疆" (DJI of Robotics)。

#### **G1 (Humanoid Agent)**
- **价格**: **9.9万 RMB** (震惊业界的低价)。
- **身高/体重**: 127cm / 35kg (偏小，像个大童)。
- **自由度**: 23-43 DOF (取决于是否选配灵巧手)。
- **关节**: 自研高扭矩关节电机，甚至能做"鲤鱼打挺"、"后空翻"。
- **用途**: 科研教育、二次开发、数据采集。

#### **H1 (Full-size Humanoid)**
- **特点**: 它是用来**跑**的。
- **速度**: 3.3 m/s (世界纪录保持者)，能跑能跳，抗踹。
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
- **特点**: **天下武功，唯快不破**。
- **演示**: 叠衣服、削苹果、甚至**颠锅**。动作速度达到了人类水平 (10m/s 末端速度)。
- **技术**: 可能是基于模仿学习 (Imitation Learning) 的极致优化。

### Galbot (银河通用)
> **Focus**: 泛化抓取与大模型。

#### **Galbot G1**
- **构型**: 轮式底盘 + 升降躯干 + 双臂 + 灵巧手 + 多模态大模型。
- **特点**: 专注于**抓取 (Grasping)**。
    - 能够 Zero-shot 抓取透明物体、反光物体、堆叠物体。
    - 这里的"产品"更多是指它的**泛化抓取能力**，而不是单纯的硬件。

---
[← Back to Root](../README.md)
