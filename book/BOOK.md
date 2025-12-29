# VLA Handbook：从理论到实践

> **Vision-Language-Action 完全指南**
> 
> 本书系统性地介绍 VLA（视觉-语言-动作）模型的核心理论、关键技术与工程实践，适合准备机器人/具身智能方向面试的工程师和研究者。

---

# 目录

## 第一部分：基础架构
1. [Transformer vs CNN](#第1章-transformer-vs-cnn)
2. [Flash Attention 与推理优化](#第2章-flash-attention-与推理优化)
3. [多模态模型基础](#第3章-多模态模型基础)
4. [VLA 架构总览](#第4章-vla-架构总览)

## 第二部分：策略生成与动作表示
5. [动作表示方法](#第5章-动作表示方法)
6. [Tokenization 专题](#第6章-tokenization-专题)
7. [Diffusion Policy](#第7章-diffusion-policy)
8. [Action Chunking Transformer (ACT)](#第8章-action-chunking-transformer-act)
9. [Flow Matching](#第9章-flow-matching)
10. [FAST 动作序列编码](#第10章-fast-动作序列编码)

## 第三部分：训练技术与优化
11. [参数高效微调 (PEFT/LoRA)](#第11章-参数高效微调-peftlora)
12. [强化学习基础与 RLHF](#第12章-强化学习基础与-rlhf)
13. [数据飞轮与跨模态迁移](../theory/frontier/data_flywheel_and_cross_modal.md) 🆕
14. [自主奖励发现 (Nature Comm 2025)](../theory/frontier/reward_discovery_rl.md)
15. [知识蒸馏](#第14章-知识蒸馏)
15. [自监督学习](#第14章-自监督学习)
16. [迁移学习与 Co-training](#第15章-迁移学习与-co-training)
17. [量化技术](#第16章-量化技术)

## 第四部分：感知与空间智能
18. [空间数学基础](#第17章-空间数学基础)
19. [机器人动力学系统分类](../theory/robot_dynamics_classification.md)
20. [机器人控制方法](#第18章-机器人控制方法)
21. [感知技术](#第19章-感知技术)
22. [点云与 SLAM](#第20章-点云与-slam)
23. [状态估计](#第21章-状态估计)
24. [具身导航 (VLN) / DualVLN 快慢系统](#第22章-具身导航-vln--dualvln-快慢系统)

## 第五部分：抓取与运动规划
24. [抓取算法](#第23章-抓取算法)
25. [运动规划](#第24章-运动规划)
26. [全模态共享 Token 空间 (MM-ACT)](../theory/frontier/vla_unified_token_space.md)
27. [触觉 VLA](#第26章-触觉-vla)

## 第六部分：前沿模型解析
28. [RDT (Robotics Diffusion Transformer)](#第27章-rdt-robotics-diffusion-transformer)
29. [π0 系列解析](#第28章-π0-系列解析)
30. [Galaxea G0](#第28章-galaxea-g0)
31. [WALL-OSS](#第30章-wall-oss)
32. [GR00T-N1.6 (NVIDIA)](#第31章-gr00t-n16-nvidia)
33. [OneTwoVLA: 自适应推理切换](../theory/frontier/onetwovla.md)

## 第七部分：评估与推理
34. [Chain-of-Thought 推理](#第33章-chain-of-thought-推理)
35. [评估方法论](#第34章-评估方法论)

## 第八部分：真机部署与工程实战
36. [UR5 Python 控制实战](#第35章-ur5-python-控制实战)
37. [ROS 集成与算法优化](#第36章-ros-集成与算法优化)
38. [Sim-to-Real 技术](#第37章-sim-to-real-技术)
39. 灵巧手硬件与部署
    *   [39.1 Wuji 手深度解析 (独立驱动)](../deployment/dexterous_hand_wuji.md)
    *   [39.2 Tesla Optimus V2 手解析 (肌腱驱动)](../deployment/optimus_hand_v2.md)
40. [模型量化与边缘部署](#第39章-模型量化与边缘部署)

## 第九部分：安全与对齐
41. [VLA 本质安全 (SGTM)](../theory/frontier/vla_intrinsic_safety.md)
42. [具身对齐与伦理](#第41章-具身对齐与伦理)

## 第十部分：系统设计与扩展
43. [AI Coding 智能体设计](../system-design/ai_coding_agent_design.md)
44. [大规模分布式训练](#第44章-大规模分布式训练)
45. [评估系统设计](#第45章-评估系统设计)

## 附录
- [数据格式与处理](#附录a-数据格式与处理)
- [文献综述](#附录b-文献综述)
- [ASCII 图表速查](#附录c-ascii-图表速查)

---

\newpage

# 第一部分：基础架构

---

\newpage






