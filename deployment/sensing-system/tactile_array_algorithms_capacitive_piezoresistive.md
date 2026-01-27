# 触觉阵列算法：电容阵列 vs 压阻阵列 (Tactile Array Algorithms: Capacitive vs Piezoresistive)

> **发布时间**：2026-01-27  
> **核心定位**：把“触觉阵列”从原始 taxel 信号变成可闭环的 **接触状态 / 滑移风险 / 力与几何代理量**，并说明电容阵列与压阻阵列在 **漂移、串扰、滞回/蠕变** 上的差异与对应算法。  
> **一句话 takeaway**：先把触觉做成 **稳健的几何量（CoP/面积/法向代理）+ 早期滑移特征（高频能量/CoP 速度）+ 状态机**；学习模型应服务于闭环 KPI，而不是替代基础的时间同步与标定。

这份笔记面向 **真机落地**：你可以先按“规则 teacher”把系统跑稳，再在局部模块引入 student（TCN/CNN）提升泛化与延迟。

---

## 0. 术语与符号（保持一致）

- **Taxel**：触觉阵列的一个像素/单元。
- $x_i$：第 $i$ 个 taxel 的原始读数（电容/电阻/电压）。
- $b_i$：基线（零点）/offset。
- $g_i$：增益（scale）。
- $p_i$：归一化后的“压力/形变代理量”（不一定是真压力单位）。
- $\mathbf{r}_i$：taxel 的平面坐标（在指尖坐标系）。
- **CoP**：Center of Pressure，压力中心。
- **HF energy**：高频能量（滑移/预滑常用特征）。

---

## 1. 电容阵列 vs 压阻阵列：算法关注点总览

| 维度 | 电容阵列 (Capacitive array) | 压阻阵列 (Piezoresistive array) | 你应该怎么选算法 |
|---|---|---|---|
| 主要噪声来源 | EMI/寄生电容/走线耦合、接地/人体电容、湿度 | 材料非线性、温漂、像素一致性差 | 电容：先做 **屏蔽/guard + 去串扰**；压阻：先做 **标定 + 蠕变/滞回补偿** |
| 主要慢变化 | 环境/接近效应导致基线漂移 | 蠕变（creep）+ 滞回（hysteresis）+ 老化 | 两者都要 **基线跟踪**，但压阻更需要 **动态补偿模型** |
| 采样结构 | 常见多路复用扫描，taxel 有相位差 | 也可能扫描，但更常见“幅值直接读” | 只要多路复用：要记 **扫描相位**（否则滑移特征会被抹掉/错位） |
| 动态能力 | 小形变分辨率好，适合几何/微接触 | 大范围压力响应好但饱和/滞回明显 | 预滑/振动：两者都可做，但压阻要分离 slow/fast 分量 |
| 工程落地风险 | 串扰/静电/线缆布置影响巨大 | 长期稳定性与一致性更难 | **先建 QA 指标**（漂移、坏点、饱和率、噪声谱） |

---

## 2. 环境/硬件：你需要记录哪些“不可省略”的元数据

### 2.1 最小硬件信息（写进日志）

- **阵列尺寸**：例如 16×16/32×32/64×64
- **帧率与扫描方式**：整帧频率、逐行/逐列、多路复用周期
- **封装材料**：表面软胶厚度、硬度（Shore）、纹理层（是否有皮纹/微结构）
- **接口与电气**：采样 ADC 位宽、滤波器、屏蔽/接地方案（电容尤其关键）

### 2.2 最小时间同步要求（闭环要用）

- 每帧至少需要一个 **可信时间戳**（最好设备端打戳）。
- 如果是多路复用：建议额外记录 **scan_start_ts / scan_end_ts** 或 **行/列时间表**，以便“对齐/插值”。

---

## 3. 步骤：把原始 taxel 变成可闭环的信号（推荐流水线）

下面这条流水线适用于两类阵列；不同点会在每步注明。

### 3.1 Step A：输入对齐与相位校正（多路复用必做）

**目标**：让“同一时刻”的触觉图在时间上可比。

- **做法 1（工程最常用）**：把整帧当作同一 timestamp，但在做滑移/振动特征时只使用低频量（CoP/面积/总和）。
- **做法 2（更准确）**：为每一行/列补一个时间戳，在计算 HF 特征时按行/列进行相位对齐（或把每个 taxel 的时间偏移用于插值）。

> 经验法则：如果你要做 **<50ms 级别的预滑提前量**，做法 2 的收益通常明显。

### 3.2 Step B：每像素基线与增益（最小可用标定）

将原始信号归一化为代理量：

$$
p_i(t)=g_i\cdot(x_i(t)-b_i(t))
$$

- **电容阵列**：
  - $b_i(t)$ 受环境/接近影响大，建议用 **慢时标基线跟踪**，并在“接触状态”为真时冻结基线：
    - `if contact: b_i <- b_i`  
    - `else: b_i <- (1-α)b_i + α x_i`
- **压阻阵列**：
  - $b_i(t)$ 同样漂移，但更典型的是 **蠕变**：接触后信号缓慢变化，即使力不变也会“爬”。
  - 常见工程策略是把信号拆成 slow/fast 两路（见 3.3）。

### 3.3 Step C：分离 slow/fast（压阻强烈建议，电容可选）

用简单滤波把“准静态压力”与“动态事件（预滑/振动）”分开：

- $p^{slow} = \text{LPF}(p)$
- $p^{fast} = p - p^{slow}$（或对 $p$ 直接做带通）

**理由**：压阻阵列的 creep/hysteresis 会污染低频；把 fast 分量拿出来，能显著提升滑移检测稳定性。

### 3.4 Step D：几何量（闭环最值钱的 4 个量）

1) **接触面积**（阈值可用分位数/自适应）  
$$
A=\sum_i \mathbf{1}[p_i>\tau]\cdot a_{taxel}
$$

2) **总法向代理**  
$$
N\propto \sum_i p_i
$$

3) **CoP（压力中心）**  
$$
\mathbf{c}=\frac{\sum_i p_i\mathbf{r}_i}{\sum_i p_i+\epsilon}
$$

4) **形状二阶矩（接触椭圆/方向）**  
$$
\mathbf{M}=\frac{\sum_i p_i(\mathbf{r}_i-\mathbf{c})(\mathbf{r}_i-\mathbf{c})^\top}{\sum_i p_i+\epsilon}
$$

> 这 4 个量具有“跨硬件鲁棒性”：即使换阵列材料/分辨率，闭环结构也能复用。

### 3.5 Step E：预滑/滑移特征（inc. slip）

推荐先做 **可解释特征 + 状态机**，再决定是否上学习。

**特征 1：高频能量（HF energy）**  
对 fast 分量做带通（例如 20–200Hz 或 30–300Hz，取决于采样率与封装），再算能量：

$$
E_{HF}(t)=\sum_i \left(\text{BPF}(p_i^{fast})(t)\right)^2
$$

**特征 2：CoP 速度**  
$$
v_{cop}(t)=\|\mathbf{c}(t)-\mathbf{c}(t-\Delta t)\|/\Delta t
$$

**特征 3：形状突变（边缘先滑）**  
接触面积 $A$ 下降 + 峰值/分位数上升，通常意味着压力重新分布、边缘微滑。

### 3.6 Step F：接触模式与滑移状态机（强烈建议先上）

**目标**：让触觉输出变成“可解释、可验收、抗抖动”的事件流。

推荐最小状态机（可按任务扩展）：

```text
no_contact
  └──(contact_detected)──> contact_stick
contact_stick
  ├──(incipient_slip)──> contact_preslip
  └──(contact_lost)──> no_contact
contact_preslip
  ├──(slip_confirmed)──> contact_slip
  ├──(recovered)──> contact_stick
  └──(contact_lost)──> no_contact
contact_slip
  ├──(recovered)──> contact_stick
  └──(contact_lost)──> no_contact
```

实现要点（部署常见坑基本都在这）：
- **阈值要滞回**（两套阈值 `enter/exit`），避免在边界抖动。
- **最小持续时间 debounce**：例如 `incipient_slip` 需连续满足 30–80ms 才触发。
- **多速率闭环**：触觉特征可以在 50–200Hz 上跑，控制指令在 200–1000Hz 的本体环执行（参考 [`end_effector_control.md`](../end_effector_control.md)）。

### 3.7 Step G：切向载荷/摩擦裕度的“工程代理量”

很多阵列只能可靠测到“法向压力分布”，但工程闭环并不一定需要精确 $F_t$：

#### 3.7.1 Shear proxy（无显式剪切传感时的做法）

可用组合特征构造剪切/滑移风险代理：

- $v_{cop}$：CoP 速度（切向相对运动的强信号）
- $\Delta \mathbf{c}$ 的方向稳定性：连续朝同一方向漂移更像滑移
- 接触形状二阶矩 $\mathbf{M}$ 的主轴旋转/拉伸：滚动 vs 滑移的粗分辨
- `HF energy`：微滑的高频振动能量

一个可部署的风险打分（示意）：

$$
r(t)=w_1\cdot \text{norm}(E_{HF}) + w_2\cdot \text{norm}(v_{cop}) + w_3\cdot \text{norm}(\Delta A)
$$

其中 `norm(.)` 是按运行中统计的 P50/P95 做归一化（比固定量纲更稳）。

#### 3.7.2 摩擦裕度（margin）的“闭环版本”

你最终想控的是“别滑”，可用：

- **目标**：保持 $r(t) < r_{max}$（风险阈值）  
- **控制手段**：
  - 增加法向（加力/夹紧）
  - 降低切向（改轨迹/降低速度/提高阻尼）
  - 改接触几何（滚动/重抓取/换指面）

这比直接估 $\mu$ 更容易落地（尤其在早期原型阶段）。

### 3.8 Step H：从规则 teacher 到学习 student（不翻车版本）

**推荐路径**（先稳再强）：

1) **Teacher（规则/轻模型）**：输入 $A,N,\mathbf{c},E_{HF},v_{cop}$ 输出 `contact_mode/slip_risk`
2) **Student（小模型）**：输入更原始的触觉小窗（例如 0.2–0.5s 的触觉图序列）拟合 teacher 输出 + 少量人工标签
3) **验收绑定闭环 KPI**：掉物率/平均夹紧力/误触发/延迟，而不是离线准确率

数据与回放建议参考：
- 多模态对齐：[`deployment/multimodal_data_synchronization.md`](../multimodal_data_synchronization.md)
- 灵巧手采数与回放验证：[`deployment/dexterous_hand_data_collection.md`](../dexterous_hand_data_collection.md)

---

## 4. 配置/参数：一个可部署的默认配置表

> 下面默认值是“工程起步点”，需要你按 **阵列帧率、封装材料、任务速度** 调参。

| 参数 | 建议默认 | 解释 |
|---|---:|---|
| `baseline_alpha` | 0.001–0.01 | 无接触时基线 EMA 更新率（越小越稳，越大越快） |
| `tau_contact` | 分位数 P90 或 `mean+3σ` | 接触阈值（建议自适应，避免温漂/个体差） |
| `lpf_cutoff_hz` | 2–10 Hz | slow 分量截止频率（准静态） |
| `bpf_low_hz` | 20–50 Hz | 预滑/滑移的低端（视封装与采样率） |
| `bpf_high_hz` | 150–300 Hz | 高频上限（低采样率时要降低） |
| `debounce_ms` | 30–80 ms | 事件去抖（避免瞬时噪声误触发） |
| `slip_risk_threshold` | 经验阈值 + 滞回 | 用 HF energy / CoP 速度触发 `incipient_slip` |
| `cop_vel_enter` / `cop_vel_exit` | P95/P80（自适应） | CoP 速度滞回阈值 |
| `hf_enter` / `hf_exit` | P95/P80（自适应） | 高频能量滞回阈值 |
| `saturation_ratio_max` | 1–5% | 饱和 taxel 占比上限（超出要降增益或报警） |
| `dead_taxel_std_min` | 近零 | 低方差坏点（几乎不动）检测阈值 |
| `noisy_taxel_std_max` | P99 | 高方差坏点（噪声大）检测阈值 |

---

## 5. 常见坑与对策（按传感器类型）

### 5.1 电容阵列：串扰/寄生电容导致“假接触”

- **现象**：靠近物体但未接触就有明显响应；或电机动作时整图漂。
- **对策**：
  - 硬件侧：屏蔽层/guard/主动屏蔽（AC shield）与走线优化。
  - 软件侧：基线冻结 + 自适应阈值 + 频谱监控（电机频点）。

参考：寄生电容与屏蔽的工程讨论可见 Analog Devices “AC shield”文章与 Microchip 设计指南。  
另见仓库中的集成难点综述：[`tactile_sensor_integration_challenges.md`](../tactile_sensor_integration_challenges.md)。

### 5.2 压阻阵列：蠕变/滞回导致“力读数不可信”

- **现象**：恒定力下读数缓慢变化；加/减力路径不一致；重复性差。
- **对策**：
  - 算法：slow/fast 分离，把滑移检测主要建立在 fast 上；对准静态用更保守的滤波与再标定。
  - 建模：引入滞回补偿模型（如外环/分段线性等），但要明确部署复杂度。

### 5.3 两类阵列都躲不开：坏点/饱和/健康度指标缺失

- **坏点类型**：
  - **死点**：几乎不响应（std 很低）
  - **噪点**：抖动很大（std 很高）
  - **漂点**：无接触时基线持续漂（baseline drift 很高）
- **建议你在系统里显式输出 3 个监控指标**（上线后会救命）：
  - `taxel_health_ratio`：健康 taxel 占比
  - `baseline_drift_rate`：无接触时基线漂移速度（/min）
  - `saturation_ratio`：饱和 taxel 占比

这些指标建议接入你的监控面板（见 `deployment/sensing-system/README.md` 的 Monitoring 方向）。

---

## 6. 一个玩具例子：从 raw → CoP/面积 → slip_risk

假设 4×4 阵列（示意），我们用 1 帧算几何量、用 0.3s 窗算 HF 能量（滑移风险）。

```python
import numpy as np

P = np.array([
  [0, 0, 0, 0],
  [0, 2, 3, 0],
  [0, 4, 6, 0],
  [0, 0, 0, 0],
], dtype=float)

tau = 1.0
mask = (P > tau)
A = mask.sum()  # taxel-count as area proxy
N = P.sum()

yy, xx = np.mgrid[0:4, 0:4]
eps = 1e-6
cx = (P * xx).sum() / (N + eps)
cy = (P * yy).sum() / (N + eps)

print("A=", A, "N=", N, "CoP=", (cx, cy))
```

滑移风险（示意，窗口能量）：

```python
# P_seq: shape [T, H, W], already baseline-corrected and bandpassed
E_hf = (P_seq ** 2).sum(axis=(1,2))      # per-frame energy
E_win = E_hf[-30:].mean()               # e.g., last 30 frames (~0.3s at 100Hz)
slip_risk = E_win > threshold
```

真实系统里：
- $A, N, \mathbf{c}$ 用于 **抓取力调节、接触保持**；
- `HF energy + CoP 速度` 用于 **预滑触发与恢复动作**。

---

## 7. 视触觉（Visuotactile）：把“触觉”变成“可计算的形变场”

这一类传感器（典型思路：相机 + 光源 + 弹性介质 +（可选）marker）把接触造成的形变编码成图像。你可以把它看成：

- **输入**：RGB / 灰度触觉图像序列（30–200Hz，取决于相机/ISP）
- **输出**：接触几何（接触区域、法向代理）、切向/剪切代理、预滑/滑移事件

### 7.1 视触觉的“算法流水线”（最常见 3 条路）

#### 路线 A：marker tracking → 位移/剪切 → slip

适合：有 dot marker / speckle 的弹性层。

- **步骤**：
  - 追踪 marker（光流/KLT/模板匹配）
  - 得到位移场 $\mathbf{u}(x,y)$、应变场 $\nabla \mathbf{u}$
  - 从 $\|\mathbf{u}\|$ 或其高频分量做 `incipient_slip`（边缘先动）

工程要点：
- 相机延迟（USB/MIPI）和帧抖动是关键瓶颈；预滑触发通常要靠 **短窗统计 + 滞回 + debounce**。

#### 路线 B：photometric stereo / learned depth → 高度图 $h(x,y)$ → 接触几何

适合：无 marker 但有可控光照的设计。

- **步骤**：
  - 图像 $\rightarrow$ 表面法线 / 高度图 $h(x,y)$（传统光度法或 CNN）
  - 接触区域：阈值/分割得到 mask
  - 法向代理：$\sum h$ 或 $\sum \max(h,0)$
  - 形状/边缘：从 $h$ 的梯度/曲率抽特征，用于插拔/边缘接触识别

工程要点：
- 在真机上，光照漂移/污染会把模型带偏；要把 **重标定**（白平衡、曝光锁定、参考帧）写进 SOP。

#### 路线 C：端到端（image sequence → state / risk）

适合：数据多、追求最短延迟/最强泛化。

- **输入**：最近 $T$ 帧触觉图序列（可拼上本体 q/dq、电流）
- **输出**：`contact_mode / slip_risk / shear_proxy` 等
- **模型**：小 CNN + TCN / ConvLSTM / ViT-lite（部署时以延迟为约束）

工程要点：
- 仍建议保留可解释的几何量（CoP/面积）作为监控与 fallback。

### 7.2 视触觉与“电容/压阻阵列”共用的闭环结构

不论传感器类型，闭环结构可以统一成：

- **几何量**：接触面积、CoP、法向代理
- **动态量**：高频能量 / 位移场高频分量 / CoP 速度
- **状态机**：`stick → preslip → slip → recover`

> 这就是为什么本文把视触觉也归到“触觉阵列算法”：你最终要的是可控的状态与风险，而不是某个传感器的 raw 值。

---

## 8. 评测与验收：不要只看分类准确率

建议把“算法验收”绑定到闭环指标：

- **稳定抓取**：掉物率、平均夹紧力（越小越好）、误触发率、反应延迟（ms）
- **预滑提前量**：在滑移发生前能提前多少 ms 触发（并且误报可控）
- **跨表面鲁棒性**：干/湿/粉尘/不同材料下阈值是否需要重调
- **可回放性**：同一 episode 重放时，关键触觉事件（接触/预滑）是否一致

滑移检测方法的系统化综述可参考 IEEE 的 survey 与 NIST 报告（见参考链接）。

---

## 参考链接

- Slip detection survey（IEEE）：[Methods and Sensors for Slip Detection in Robotics: A Survey](https://ieeexplore.ieee.org/ielx7/6287639/8948470/09066937.pdf)  
- 摩擦/预滑综述线索（Semantic Scholar）：[Tactile Sensors for Friction Estimation and Incipient Slip Detection—Toward Dexterous Robotic Manipulation: A Review](https://www.semanticscholar.org/paper/Tactile-Sensors-for-Friction-Estimation-and-Slip-A-Chen-Khamis/6dd6bd45f8de70e69fcae1aa0e658d2685997615)  
- NIST（触觉作为滑移检测器的标定与分析）：[Calibration and Analysis of Tactile Sensors as Slip Detectors](https://www.nist.gov/publications/calibration-and-analysis-tactile-sensors-slip-detectors)  
- 压阻触觉滞回补偿（Sensors/MDPI 示例之一）：[A New Model Based on Adaptation of the External Loop to Compensate the Hysteresis of Tactile Sensors](https://mdpi-res.com/d_attachment/sensors/sensors-15-26170/article_deploy/sensors-15-26170.pdf?version=1444903761)  
- 电容传感屏蔽工程实践（Analog Devices）：[AC Shield Enhances Remote Capacitive Sensing](https://www.analog.com/en/resources/analog-dialogue/articles/ac-shield-enhances-remote-capacitive-sensing.html)  
- 电容触摸/寄生电容设计指南（Microchip）：[Capacitive Touch Sensor Design Guide](https://ww1.microchip.com/downloads/en/Appnotes/Capacitive-Touch-Sensor-Design-Guide-DS00002934-B.pdf)  
- 视触觉（GelSight，MIT 项目页）：[GelSight (MIT CSAIL)](https://people.csail.mit.edu/kimo/gelsight/)  
- 视触觉（DIGIT，论文页）：[DIGIT (Lambeta et al., 2020)](https://www.seas.upenn.edu/~dineshj/publication/lambeta-2020-digit/)  
- 视触觉（DIGIT，arXiv）：[arXiv:2005.14679](https://arxiv.org/abs/2005.14679)  
- 视触觉（DIGIT，开源设计）：[facebookresearch/digit-design](https://github.com/facebookresearch/digit-design)  
- 视触觉（TacTip Family，论文页）：[The TacTip Family (Abdn)](https://aura.abdn.ac.uk/handle/2164/13296)  
- 视触觉（TacTip，工具书入口）：[TacTip (Soft Robotics Toolkit)](https://softroboticstoolkit.com/tactip)  

---
[← Back to Sensing System Index](./README.md)

