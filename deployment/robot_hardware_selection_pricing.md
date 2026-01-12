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

#### C. 液压驱动流派 (Hydraulic Actuation)
*   **代表产品**：**Sanctuary AI - Phoenix**。
*   **技术特点**：利用微型液压泵和伺服阀驱动关节。Sanctuary AI 成功将液压组件缩小至硬币尺寸，并通过了 20 亿次循环测试以解决传统液压的泄漏难题。
*   **VLA 意义**：
    *   **极致力量 (Heavy Duty)**：指尖力可达 **100N**（是普通电手的 5-10 倍），能轻松捏碎易拉罐，也能执行穿针引线等精细任务。
    *   **高功率密度**：相同体积下动力输出是电机的 3-5 倍，非常接近人手的“肌肉”爆发力。
    *   **天然顺应性**：液压介质具有物理阻尼特性，在高速接触碰撞时比刚性齿轮传动更具安全性。
*   **挑战**：造价极高（单手系统或超 $5万），存在漏油风险，且工作时伴有明显的泵机噪音。

#### D. 合成肌肉/生物启发流派 (Synthetic Muscles / Bio-inspired)
*   **代表产品**：**Clone Robotics - Clone Hand**。
*   **技术特点**：采用**热液压 (Thermal-Hydraulic)** 或水基合成肌肉。通过加热水介质使管道膨胀产生拉力，模拟生物肌肉纤维。
*   **VLA 意义**：
    *   **极致轻量化**：手指部分几乎没有电机和金属齿轮，重量极轻（约 1kg 以内）。
    *   **解剖学一致性**：肌肉排布完全遵循人手解剖学（约 36 个驱动器），是目前形态上最接近人手的方案。
*   **挑战**：控制逻辑极其复杂且非线性，响应速度受热循环限制，仍处于实验室/极少数科研阶段。

### 1.2 触觉感知流派 (Tactile Sensing Lineages)

在 VLA 闭环中，触觉反馈是解决“视觉遮挡”和“柔性抓取”的核心。

#### A. 视触觉 (Vision-based / Optical Tactile)
*   **代表**：**GelSight**, **DIGIT**, **Daimon (视触觉)**。
*   **特点**：利用摄像头捕捉内部凝胶表面的弹性形变。
*   **优势**：空间分辨率极高（可看清指纹/螺纹纹理），适合训练基于图像的端到端感知模型。
*   **劣势**：体积通常较大，难以嵌入指侧；凝胶易损，需定期更换。

#### B. 阵列式/电信号类 (Electronic Array / MEMS)
*   **代表**：**Xela (uSkin)**, **SynTouch (BioTac)**, **Optimus Gen 2 (自研阵列)**。
*   **特点**：利用电容、压阻或磁场变化感知压力分布。
*   **优势**：超薄可贴合，可覆盖全手（如 uSkin）；响应频率极高（kHz 级）。
*   **劣势**：分辨率远低于视觉类；存在布线难题。

#### C. 动态触觉阵列 (Dynamic Tactile Array / DTA)
*   **代表**：**Sharpa Wave**。
*   **特点**：高集成度的微型传感器阵列，配合神经网络进行动态滑移检测。
*   **价值**：专门针对高速闭环控制优化，是目前工业级灵巧手与 AI 结合的最前沿方向。

### 1.3 典型任务挑战与硬件适配 (Task Challenges)

在 VLA 算法开发中，灵巧手的“难点”往往集中在以下四类任务，不同流派的硬件表现各异：

#### A. 抓取小物件 (Small Objects, e.g., 螺丝、针)
*   **挑战**：**精度与遮挡**。指尖接触面积小，视觉传感器容易被机械结构遮挡。
*   **硬件要求**：
    *   **直驱/集成流派**：如 Sharpa Wave 的高自由度能确保指尖对齐。
    *   **触觉**：必须具备高分辨率触觉（如 GelSight 或 DTA），依靠视觉往往无法完成最后 1mm 的精准闭合。
    *   **算法**：需要模型具备高频反馈能力（100Hz+），处理微小的接触力平衡。

#### B. 抓取大物件与重载 (Large/Heavy Objects, e.g., 哑铃、大水壶)
*   **挑战**：**负载能力与包络空间**。大物件要求手指有足够的跨度（Workspace）和抓握力。
*   **核心痛点**：**末端关节（Distal Joints）性能要求极高**。在抓取重物时，末端指节承担了最大的静力矩负载，若末端电机扭矩不足或结构件刚性不够，手指会在重力作用下被动“掰开”导致物体滑落。
*   **硬件要求**：
    *   **高扭矩密度**：要求末端指节集成高减速比的行星减速器，或采用高压液压驱动。
    *   **可折叠掌骨**：如 LEAP V2 的掌骨关节，能增加有效包络半径，通过手掌辅助支撑。
    *   **算法**：关注多指协同的接触稳定性预测，防止因重力矩导致的物体滑移。

#### C. “开可乐”等精细协同 (Complex Tool Use)
*   **挑战**：**爆发力、空间限制与杠杆原理**。拉开易拉罐环的核心难点在于：
    1.  **指甲的利用 (Fingernail Utility)**：需要末端指节具备极薄且坚硬的“指甲”结构，以便插入扣环与罐盖之间的微米级缝隙。
    2.  **极小空间的大力矩输出**：在扣环被勾起的一瞬间，手指处于极端弯曲状态，此时需要利用**杠杆原理**，以指尖为支点输出巨大的爆发拉力。
*   **硬件要求**：
    *   **末端指节构型**：要求指尖具有类似指甲的硬质薄边缘设计，而非全软包覆，否则无法插入缝隙。
    *   **高带宽控制**：要求硬件能瞬间响应力矩突变（扣环弹开时的载荷突降）。
    *   **鲁棒齿轮箱**：直驱方案中的行星减速器需具备高抗冲击性，防止在杠杆受力瞬间崩齿。
    *   **算法**：典型的多阶段长程任务，通常需要 **Hierarchical VLA** 或 **CoT** 来拆解“按压-勾取-杠杆拉开”的子目标。

#### D. 抓取手机等光滑薄片 (Flat & Smooth Objects)
*   **挑战**：**摩擦力管理与桌面碰撞对抗**。手机表面极滑且薄，单指很难从桌面扣起；更难的是，手指在插入物体底部时会与桌面发生强烈的**物理碰撞与对抗**。
*   **工程洞察**：**绳驱流派（Tendon-Driven）的柔顺优势**。在这种需要与环境“硬碰硬”的场景下，绳驱手依靠线缆的微小弹性和非刚性传动，具备天然的**物理自适应能力（Passive Compliance）**，能更好地化解与桌面接触时的冲击力。相比之下，直驱派若无精确的阻抗控制，容易因碰撞过硬导致损坏。
*   **硬件要求**：
    *   **刚柔耦合外壳**：如 LEAP 的 TPU 软皮或 Wuji 的海绵包覆，利用物理摩擦力弥补算法控制的精度不足。
    *   **算法**：需要学习“非接触式策略”（如利用手指将手机推至桌面边缘再抓取），这对 VLA 的逻辑推理能力提出了极高要求。

### 1.4 价格与选型表
| **Inspire** | RH56DFX | ~7.4万 - 15.5万 | 科研, 人形集成 | [Link](../product/hands.md#inspire-rh56dfx) |
| **Unitree** | Dex3-1 | ~5.7万 - 6万 | 通用抓取 | [Link](../product/hands.md#unitree-dex3-1) |
| **Agibot** | OmniHand | ~1.45万 | **高性价比**, 科研 | [Link](../product/hands.md#agibot-灵犀-x1) |
| **Sanctuary AI** | Phoenix Hand | ~$5万+ | **液压重载**, 极致力量 | [Link](#液压驱动流派-hydraulic-actuation) |
| **Clone Robotics** | Clone Hand | 询价 | **合成肌肉**, 仿生极点 | [Link](#d-合成肌肉生物启发流派-synthetic-muscles--bio-inspired) |
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
