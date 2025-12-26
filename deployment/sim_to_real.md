# Sim-to-Real (仿真到真机)

在仿真环境中训练策略，然后迁移到真机，是解决机器人数据匮乏问题的核心路径。

## 1. 为什么需要 Sim-to-Real?
- **安全性**: 避免真机在探索过程中损坏。
- **效率**: 仿真可以并行加速 (e.g., Isaac Gym 可以同时跑数千个环境)。
- **成本**: 获取真机数据的成本远高于仿真数据。

## 2. 核心挑战：Reality Gap (虚实鸿沟)
- **视觉差异 (Visual Gap)**: 仿真渲染的图像与真实摄像头图像在光照、纹理、噪声上存在差异。
- **动力学差异 (Dynamics Gap)**: 摩擦力、质量分布、电机响应特性难以完美建模。

## 3. 常用技术

### Domain Randomization (域随机化)
> **核心思想**: "如果仿真环境足够多样化，真实世界就只是其中的一种情况。"

- **视觉随机化**:
    - 随机调整光照、纹理、颜色、摄像机位置。
    - 在图像中添加高斯噪声、椒盐噪声。
- **动力学随机化**:
    - 随机调整物体质量、摩擦系数、关节阻尼、电机 PID 参数。
    - 随机施加外力扰动。

### Domain Adaptation (域自适应)
> **核心思想**: 学习特征的不变性，或将一种域的数据映射到另一种域。

- **Pixel-level Adaptation**: 使用 GAN (Generative Adversarial Networks) 将仿真图像转化为真实风格的图像 (Sim2RealGAN)。
- **Feature-level Adaptation**: 强制模型学习到的特征在仿真和真机下分布一致 (Domain Invariant Features)。

### System Identification (系统辨识)
- 在真机上收集少量数据，估算真实的物理参数 (摩擦力、质量)，然后更新仿真环境参数，使其更接近真实世界。

## 4. 实战技巧
1. **使用深度图 (Depth)**: 深度图比 RGB 图像对光照和纹理变化更鲁棒，Sim-to-Real 成功率通常更高。
2. **动作空间选择**: 使用 **Delta Position Control** (增量位置控制) 通常比 Torque Control (力矩控制) 更容易迁移，因为底层的 PID 控制器屏蔽了部分动力学差异。
3. **混合训练**: 在仿真数据中混入少量 (e.g., 10%) 的真机数据进行 Co-training，可以显著提升迁移效果。

## 5. 案例分析: Pi0.5 的 Co-training 策略
Pi0.5 展示了目前最先进的 Sim-to-Real 范式，它不再单纯依赖 Domain Randomization，而是采用 **Data Co-training**。

- **数据配比**:
    - **Simulation Data**: 用于学习长序列逻辑 (Long-horizon Logic)。仿真数据完美标注，适合学习"先开门，再拿杯子"这种因果关系。
    - **Internet Videos**: 用于学习物理常识 (World Model)。
    - **Robot Data**: 用于学习具体的动作执行 (Motor Control)。
- **效果**:
    - 仿真数据教会了模型"做什么" (What to do)。
    - 真机数据教会了模型"怎么做" (How to do)。
    - 这种组合使得 Pi0.5 能够 Zero-shot 迁移到从未见过的厨房环境。

## 面试高频考点
1. **Domain Randomization**: 随机化范围越大越好吗？(答: 不是。范围过大会导致任务过于困难，模型无法收敛。需要根据实际情况调整)
2. **Visual Gap**: 除了随机化，还有什么方法减小视觉差异？(答: 使用 Real-to-Sim，即用真实图像作为纹理贴图到仿真物体上；或者使用从不依赖纹理的特征，如 Keypoints)
3. **失败案例**: 举一个 Sim-to-Real 失败的例子。(答: 仿真中摩擦力建模不准，导致抓取物体时滑落；或者线缆的柔性动力学未建模，导致真机运动受阻)

---
[← Back to Deployment](./README.md)
