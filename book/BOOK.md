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
6. [Diffusion Policy](#第6章-diffusion-policy)
7. [Action Chunking Transformer (ACT)](#第7章-action-chunking-transformer-act)
8. [Flow Matching](#第8章-flow-matching)
9. [FAST 动作序列编码](#第9章-fast-动作序列编码)

## 第三部分：训练技术与优化
10. [参数高效微调 (PEFT/LoRA)](#第10章-参数高效微调-peftlora)
11. [强化学习基础与 RLHF](#第11章-强化学习基础与-rlhf)
12. [知识蒸馏](#第12章-知识蒸馏)
13. [自监督学习](#第13章-自监督学习)
14. [迁移学习与 Co-training](#第14章-迁移学习与-co-training)
15. [量化技术](#第15章-量化技术)

## 第四部分：感知与空间智能
16. [空间数学基础](#第16章-空间数学基础)
17. [机器人控制方法](#第17章-机器人控制方法)
18. [感知技术](#第18章-感知技术)
19. [点云与 SLAM](#第19章-点云与-slam)
20. [状态估计](#第20章-状态估计)
21. [具身导航 (VLN) / DualVLN 快慢系统](#第21章-具身导航-vln--dualvln-快慢系统)

## 第五部分：抓取与运动规划
22. [抓取算法](#第22章-抓取算法)
23. [运动规划](#第23章-运动规划)
24. [全模态共享 Token 空间 (MM-ACT)](#第24章-全模态共享-token-空间-mm-act)
25. [触觉 VLA](#第25章-触觉-vla)

## 第六部分：前沿模型解析
26. [RDT (Robotics Diffusion Transformer)](#第26章-rdt-robotics-diffusion-transformer)
27. [π0 系列解析](#第26章-π0-系列解析)
28. [Galaxea G0](#第27章-galaxea-g0)
29. [WALL-OSS](#第29章-wall-oss)

## 第七部分：评估与推理
29. [Chain-of-Thought 推理](#第29章-chain-of-thought-推理)
30. [评估方法论](#第30章-评估方法论)

## 第八部分：真机部署与工程实战
31. [UR5 Python 控制实战](#第31章-ur5-python-控制实战)
32. [ROS 集成与算法优化](#第32章-ros-集成与算法优化)
33. [Sim-to-Real 技术](#第33章-sim-to-real-技术)
34. [灵巧手部署指南](#第34章-灵巧手部署指南)
35. [模型量化与边缘部署](#第35章-模型量化与边缘部署)

## 第九部分：安全与对齐
36. [VLA 本质安全 (SGTM)](#第36章-vla-本质安全-sgtm)
37. [具身对齐与伦理](#第37章-具身对齐与伦理)

## 附录
- [数据格式与处理](#附录a-数据格式与处理)
- [文献综述](#附录b-文献综述)
- [ASCII 图表速查](#附录c-ascii-图表速查)

---

\newpage

# 第一部分：基础架构

---

\newpage






