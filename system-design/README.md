# 系统设计 (System Design)

本模块关注 VLA 系统的宏观架构设计，这是 Tech Lead 和 Staff Engineer 面试的核心考点。

## 目录

1.  **[数据闭环设计 (Data Pipeline Design)](./data_pipeline.md)**
    - 如何构建一个自动化的数据飞轮？
    - Auto-labeling (VLM 标注)
    - Active Learning (主动学习与难例挖掘)
    - Human-in-the-loop (人机回环)

2.  **[云端基础设施 (Cloud Infrastructure)](./cloud_infrastructure.md)**
    - 分布式训练架构 (FSDP, Megatron-LM)
    - 存储系统选型 (S3 vs Lustre)
    - 持续评估 (Continuous Evaluation)
    - 车队管理 (Fleet Management & OTA)

3.  **[大规模模型训练 (Large-Scale Training)](./large_scale_training.md)** 🆕
    - GPU 集群选型与网络架构
    - 分布式训练策略 (DDP, FSDP, TP, PP, 3D 并行)
    - 训练优化 (混合精度, Gradient Checkpointing)
    - 训练稳定性与调试

4.  **[评估系统设计 (Evaluation System)](./evaluation.md)**
    - Simulation Benchmark (仿真基准)
    - Real-world Proxy (真机代理指标)
    - A/B Testing & Canary Deployment

5.  **[AI Coding 智能体设计 (AI Coding Agent Design)](./ai_coding_agent_design.md)** 🆕
    - 提示词预处理 (@路径, Slash 命令)
    - MCP 协议与工具发现链路
    - SubAgent 架构与上下文隔离
    - 规约驱动开发 (Spec-driven Development)

6.  **[Text-to-SQL 可靠架构（Palantir 视角）](./text_to_sql_reliable_architecture_palantir_aip.md)** 🆕
    - 语义层 (Ontology) + 工具化执行 + 审计
    - 低置信度反问与可追溯性

## 学习建议
- **关注 Scalability**: 所有的设计都要考虑 "如果机器人数从 10 台变成 1000 台，这个系统还能跑吗？"
- **关注 Automation**: 尽量减少人工介入。最好的系统是机器人自己收集数据、自己训练、自己变强。

---
[← Back to Main README](../README.md)
