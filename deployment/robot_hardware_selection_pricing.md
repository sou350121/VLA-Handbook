# 硬件选型与价格参考 (Hardware Selection & Pricing)

本章节提供 VLA 算法落地常用硬件的**参考价格**与**选型建议**。
详细的技术参数与产品解析，请移步 **[机器人产品大百科](../product/README.md)**。

> [!NOTE]
> 价格仅供参考，实际成交价受汇率、关税、代理商折扣及配置影响较大。

## 1. 灵巧手 (Dexterous Hands)

### 1.1 技术流派深度对比 (Technical Lineages)

在 VLA 研究中，选择灵巧手本质上是在选择其**动力传动流派**，这直接影响了算法的复杂度和示教数据的质量。

#### A. 电机集成与直驱/准直驱方案 (Integrated & Direct-Drive)
*   **代表产品**：**Wuji (舞肌)**, **Sharpa Wave**, **Tesla Optimus Hand**, **Agibot OmniHand**。
*   **技术特点**：电机通过行星减速器或齿轮直接嵌入指骨/关节处。
*   **VLA 意义**：
    *   **高透明度**：没有拉索的摩擦和滞后，动作极其精准且线性，模型更容易学习。
    *   **低延迟**：力矩传递路径短，适合 1000Hz 级高频实时闭环。
    *   **高鲁棒性**：结构紧凑，维护简单，没有断绳风险。
*   **挑战**：由于电机在手指上，手指自重较大，对电机的功率密度要求极高。

#### B. 拉索与线驱动方案 (Tendon-Driven)
*   **代表产品**：**Shadow Hand**, **LEAP Hand**, **Wonik Allegro**, **Inspire Hand**。
*   **技术特点**：电机通常集中在手掌或前臂，通过拉索（钢丝绳或高强线）远程牵引关节。
*   **VLA 意义**：
    *   **仿生形态**：手指可以做得非常细长轻盈，惯性小。
    *   **过载保护**：拉索天然具备一定的柔性缓冲区，不容易因为硬碰撞烧毁电机。
*   **挑战**：
    *   **摩擦力非线性**：拉索与导管的摩擦、线缆拉伸导致的弹性，会引入难以建模的动力学噪声（Hysteresis）。
    *   **维护成本高**：长时间高强度使用后拉索会松动甚至断裂，严重影响数据的一致性。

### 1.2 价格与选型表
| **Inspire** | RH56DFX | ~7.4万 - 15.5万 | 科研, 人形集成 | [Link](../product/hands.md#inspire-rh56dfx) |
| **Unitree** | Dex3-1 | ~5.7万 - 6万 | 通用抓取 | [Link](../product/hands.md#unitree-dex3-1) |
| **Agibot** | OmniHand | ~1.45万 | **高性价比**, 科研 | [Link](../product/hands.md#agibot-灵犀-x1) |
| **LEAP** | LEAP V2 Adv | ~$3,000 (约2.1万) | **开源低成本**, 仿生数据采集 | [Link](#leap-hand-v2-advanced) |
| **Sharpa** | Sharpa Wave | 询价 | **高性能**, 1:1 同构, 极速控制 | [Link](#sharpa-wave) |
| **Wonik** | Allegro | ~11万 - 18万 | 经典科研 | [Link](../product/hands.md#wonik-allegro-hand) |
| **Shadow** | Shadow Hand | ~85万+ | **高端触觉**, 遥操作 | [Link](../product/hands.md#shadow-hand) |
| **Daimon** | DM-Hand1 | 询价 | **视触觉**, 仿人构型 | [Link](../product/hands.md#daimon-dm-hand1-视触觉灵巧手) |

## 2. 前沿灵巧手深度调研

### 2.1 Sharpa Wave (新加坡 Sharpa Robotics)
*   **核心优势**：极致的 **1:1 同构 (Isomorphism)**。其手掌宽/长比严格遵循 0.618 黄金比例。
*   **硬件参数**：
    *   **22 个主动自由度**：目前行业领先，支持转笔、使用剪刀等精细动作。
    *   **指尖压力 > 20N**：具备重载抓取能力。
    *   **4Hz 全手势频率**：响应极快，适配高频 VLA 闭环（如 100Hz+ 控制）。
*   **黑科技：DTA (Dynamic Tactile Array)**：
    *   集成了基于神经网络算法的动态触觉阵列，能够感知物体表面的微小滑移和纹理细节，是训练端到端“触觉感知策略”的理想硬件。
*   **鲁棒性**：100 万次循环寿命，全关节支持 **倒驱 (Backdrivable)**，抗意外冲击能力极强。

### 2.2 LEAP Hand V2 Advanced (CMU)
*   **核心优势**：**刚柔耦合 (Hybrid Rigid-Soft)** 与 **可动掌骨**。
*   **技术亮点**：
    *   **Articulated Palm**：手掌具有 2 个动力关节，可以像人手一样“收拢”手掌，极大地提升了抓握异形工具（如电钻、锥形瓶）的稳定性。
    *   **TPU 仿生外壳**：PLA 骨架 + TPU 柔性外皮，天然具备物理防滑和力学顺应性，降低了对感知精度的极端依赖。
*   **数据采集适配**：完美适配 **Manus 触觉手套**，是目前性价比最高的专家示教（Expert Demonstration）硬件方案。

## 3. 平行夹爪 (Grippers)

| 厂商 | 型号 | 参考价格 (RMB) | 特点 | 详情 |
| :--- | :--- | :--- | :--- | :--- |
| **Robotiq** | 2F-85 | ~2.5万 - 3.5万 | **行业标准**, 极其稳定 | [Link](../product/grippers.md#robotiq-2f-85--2f-140) |
| **DH** | AG-95 | ~0.8万 - 1.2万 | **高性价比**, 完美平替 | [Link](../product/grippers.md#dh-robotics-大寰机器人) |
| **Franka** | Hand | (随臂赠送) | 深度集成, 高灵敏 | [Link](../product/arms.md#franka-emika-research-3--production) |
| **Daimon** | DM-Tac G | 询价 | **视触觉夹爪**, 高精度 | [Link](../product/grippers.md#daimon-dm-tac-g-visuotactile-gripper) |

## 3. 机械臂 (Robotic Arms)

| 厂商 | 型号 | 参考价格 (RMB) | 适用场景 | 详情 |
| :--- | :--- | :--- | :--- | :--- |
| **Trossen** | WidowX 250 | ~1.9万 | **ALOHA 复现**, 入门 | [Link](../product/arms.md#widowx-250) |
| **UR** | UR5e | ~20万 - 30万 | 工业级, 高精度 | [Link](../product/arms.md#ur5e) |
| **Elephant** | myCobot 280 | ~3000 | 极低预算, 教育 | [Link](../product/arms.md#mycobot-280) |

## 4. 移动底盘 (Mobile Bases)

| 厂商 | 型号 | 参考价格 (RMB) | 适用场景 | 详情 |
| :--- | :--- | :--- | :--- | :--- |
| **AgileX** | LIMO | ~2万 - 4万 | 教育, 桌面实验 | [Link](../product/mobile_bases.md#agilex-limo--scout) |
| **AgileX** | Scout Mini | ~6万 | 室外越野 | [Link](../product/mobile_bases.md#agilex-limo--scout) |
| **Unitree** | Go2 (Edu) | ~8万 - 16万 | 四足移动, 高算力 | [Link](../product/mobile_bases.md#unitree-go2) |

## 5. 具身智能本体 (Humanoids)

| 厂商 | 型号 | 参考价格 (RMB) | 特点 | 详情 |
| :--- | :--- | :--- | :--- | :--- |
| **Unitree** | G1 | ~9.9万 | **入门首选**, 量产 | [Link](../product/humanoids.md#unitree-g1--h1) |
| **Fourier** | GR-1 | ~90万 - 100万 | 康复, 负重强 | [Link](../product/humanoids.md#fourier-gr-1) |
| **Astribot** | S1 | ~50万 - 60万 (Est) | 极致速度 | [Link](../product/humanoids.md#astribot-s1) |

## 6. 触觉传感器 (Tactile Sensors)

| 厂商 | 型号 | 参考价格 | 特点 | 详情 |
| :--- | :--- | :--- | :--- | :--- |
| **Qianjue** | XENSE-G1 | 询价 | **多模态**, 高精度 | [Link](../product/sensors.md#qianjue-xense-g1) |
| **GelSight** | Mini | ~$3,000 | 视触觉, 3D拓扑 | [Link](../product/sensors.md#gelsight-mini) |
| **Daimon** | DM-Tac W | ~$1,000 | **视触觉**, 120Hz | [Link](../product/sensors.md#daimon-dm-tac-w-series-通用视触觉传感器) |
| **Tashan** | AI Tactile | 询价 | **数模混合芯片**, 接近觉 | [Link](../product/sensors.md#tashan-他山科技) |
| **Xela** | uSkin | ~2万 - 4万 | 磁触觉, 柔性 | [Link](../product/sensors.md#xela-uskin) |

---
[← Back to Deployment](./README.md)
