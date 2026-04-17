# VLA Daily Digest — 2026-04-17

> **日级强信号日**。Physical Intelligence 发布 π0.7（2026-04-16），小红书社区采集 15 篇新帖形成高密度社区共识。
> **B0 (数据策略 > 模型架构) 上调 72% → 77%**（raw 80→85%），理由见 §3。

---

## 1. 扫描范围

| 源 | 扫描窗口 | 状态 |
|---|---|---|
| arxiv cs.RO (2604.14+) | 04-16/04-17 submissions | ✅ 无新 April 17 论文；04-15/16 主要论文 (HiVLA/VLAJS/WOMBET/EEAgent/Sim-Real Co-Training) 已在 04-16 digest 覆盖 |
| Physical Intelligence 官方 | 04-16 博客/TechCrunch | ✅ **π0.7 发布确认** |
| 小红书 VLA 社区 | 2026-04-17-auto.md | ✅ 15 篇新帖 + 1 篇标题登记（累计 60 篇） |
| VLA 社交情报 | 2026-04-16.md | ✅ 无新增（顶级实验室信号连续 13 天缺席） |

---

## 2. 核心信号

### 🔺 π0.7 发布（2026-04-16, PI）— 今日最大信号

**事实**：
- TechCrunch (2026-04-16) 报道 Physical Intelligence 发布 π0.7
- 核心声称：`compositional generalization` — 通过 language coaching 解决未训练任务
- 匹配 specialist 模型性能（coffee / laundry / box assembly）
- 小红书 NotBot (04-17) 拆解技术细节：**Episode Metadata**（Quality 1-5 分 / Mistake 布尔 / Speed）+ **Knowledge Insulation** 梯度隔离
- 推理时用 `"Quality: 5, Mistake: false"` 之类元数据精确引导策略

**作者评价**（NotBot 帖 11）：
> "架构上没啥特别，增大的 VLM 更像是为了提升语义容量。**功夫在数据工程上**——越是严肃搞模型的公司越是严肃对待数据。"
> "具身智能已经准备好迎接自己的 GPT3 时刻了。"

### 三视角辩论

🔴 **Bull**：PI 是 VLA 领域最被关注的 lab，以产业级发布亲自背书"数据工程 > 架构创新"。Episode Metadata 首次把 LLM 领域"质量标签控生成"成功迁移到 VLA——数据标注成为下一个差异化战场。"匹配 specialist 性能"意味着通用模型窗口打开，pi0.7 可能是 VLA 的 GPT-3 时刻开端。

🔵 **Bear**：评估都是 PI 自己做的，第三方验证需 3-6 月。社区复现规律极其残酷——LeRobot 版 pi0 官方承认 30% 成功率（pi0.5 Lingbot 55/100 vs paper 更高）。"论文 vs 实测 2-3x 衰减"在 2026 年依然是系统性现象。Episode Metadata 是数据工程创新而非新范式（LLM 领域早有 RLAIF/DPO quality labels）。"语言 coaching 解决未训练任务"在 RT-2 时代就部分存在。

🟢 **Arbiter**：π0.7 对 B0 是强正向产业信号，但校准纪律要求谨慎（>80% 区间 LLM 系统性过度自信）。决定 B0 上调 +5%（72→77% / raw 80→85%）——恢复 B0=B1=B2=B3=77% 的父子一致性，记录充分反方弹药（复现困境）作未来 3 月内校验条件。

**行动建议**：
1. 若走 VLA 模型复现路线：**别硬刚 π0.7 原版**，等 HuggingFace 或清华/MIT 的独立复现数据出来
2. 在自家数据里提前标注 Quality/Mistake/Speed 元数据（成本低、潜在回报高，无论 π0.7 能否复现）
3. 30 天内关注预测 #9 #10（见 §4）

**相关深度解读（04-17 paper-verified update）**：
- [π0.7 架构 paper-verified 解读](../theory/vla-core/pi0_7_steerable_compositional_generalization_2026.md)（主文件，含 5B 主体 + 14B BAGEL WM 架构拆解、新增件 vs 继承件对比、推理成本表）
- [完整 HTML deep-dive 报告](2026-04-17-pi07-paper-verified-report.html)（9 节完整版：护城河拆解 · 威胁矩阵 · 12 月预测 · 审计日志 · 繁/简中切换）

---

## 3. Belief Graph 变更

### B0: 72% → **77%** ↑（raw 80→85%）

**理由**：π0.7 (2026-04-16) 以 Episode Metadata + Knowledge Insulation + "架构上没啥特别，功夫在数据工程"的产业级背书，是 B0 的强正向信号。上次变更 04-06（B0 从 85→80% 下调，因 DIAL/DFM-VLA/DiT4DiT 三条架构信号），此次是反向修正且依据更强（PI 是产业级信号 vs 学术端三条独立论文）。

**最小更新纪律**：按"Bull 和 Bear 同意方向时 ≥10%"规则，实际上 Bear 强烈反对（复现困境），只满足"强证据但双方分歧"条件，适用普通最小更新 ±5%。

**父子一致性**：B0 原 72% < B1 77% 存在子>父不一致，此次上调消除。

### 其他节点：维持不变，但累积弹药更新

| 节点 | 影响 | 处理 |
|---|---|---|
| B1 (77%, 43d 保守偏误) | π0.7 metadata 机制对飞轮双向影响（质量维度扩大 vs 工程门槛降低）| 维持；保守偏误持续追踪 |
| B2 (77%, 16d) | 反方弹药增加（real-world RL 物料被干烂 3583 赞 + "大规模 VLA RL 做不了"共识）| 04-01 刚下调不重复；累积至下次审查 |
| B3 (77%, 24d) | 弱正向（π0.7 coaching 是在线自改进雏形）| 维持 |
| B6 (75%, 43d 保守偏误) | 无新信号（04-16 HiVLA 已累积）| 维持 |
| B7 (75%, 33d 保守偏误) | 弱正向（Knowledge Insulation 由 PI 再度背书解耦正统）| 不足以上调 |
| B8 (70%, 31d 保守偏误) | **弱正向累积**（TouchAnything+DECO+TaSA+视觉力矩 4 条）| **抵消 04-15 审计"下调至 65%"建议，维持 70%** |
| B9 (75%, 2d) | **反方弱信号**（VLA 推理频率虚标=frequency×action_chunk_size 社区共识）| 维持；未来复核优先检查 |

**逆共识**：
- **C1 (35%, 距升格 5%)**：π0.7 "架构上没啥特别"是直接反对 C1 的产业级信号。但逆共识保护机制（阈值低 1/3）防止单 lab 信号触发下调。C1 维持 35%，记录反方弹药。
- **C3 (24%)**：π0.7 "听得到语言并据此行动——甚至违反训练数据视觉偏差"强化语言 grounding 必要性（反 C3 弹药）。维持。

---

## 4. 其他信号（未改变 Belief Graph）

### 社区共识信号（小红书 15 篇新帖）

| 信号 | 来源（互动量） | 影响 |
|---|---|---|
| 🔺 LeRobot pi0 官方承认 30% 成功率；个人实测 20% vs SmolVLA 3 万步全成功 | 帖 12（105 赞；28 评论含工程共识）| **论文 vs 复现 2-3x 衰减**系统性确认；未来 benchmark 数据必须打折 |
| 🔺 Lingbot-VLA 复现：个人 20% vs 官方复测 55% vs paper 更高 | 帖 3（436 赞；Lingbot 团队官方回应） | 同上；再次验证 |
| 🔺 Jupiter Zhai（PI 组内）"real-world RL 物料被干烂" | 帖 1（**3583 赞本轮最高**）| B2 反方弹药；**逆共识候选**："real-world RL ROI < 仿真 RL + domain adaptation" |
| SimpleVLA-RL：单轨迹 SFT + RL 将 LIBERO-10 从 17→91%，LIBERO-Avg 从 48.9→94.1% | 帖 8（206 赞；清华+上海 AI Lab） | B2 弱正向（单轨迹数据稀缺场景路线） |
| VLA 论文推理频率=frequency×action_chunk_size 虚标 | 帖 14（26 赞评论区给出根因）| B9 反方弱信号；**校准规则**：未来 paper 引用推理频率时 /chunk_size 还原 |
| TouchAnything（EgoTouch 数据集 + 首个视频→双手触觉估计模型）| 帖 15（204 赞，杨朔 SJTU）| B8 弱正向；视频→触觉新数据路径 |
| pi*0.6 本质 = advantage conditioned SFT（非 RL 约束）| 帖 9 评论（小白学具身 5 赞）| B2 定义澄清；VLA+RL 分类需更细 |
| GR-RL（字节 2025-11）：distributional RL for flow-based model + hindsight + task progress | 帖 7（210 赞 RetrievalAG）| 未改变 B5/B2，但"distributional RL × Flow"是潜在新范式 |
| 人形 sim2real 四条路线（DR/SI/delta-action/offline RL）都是"缓解不是解决" | 帖 10（180 赞，皮卡丘学术版） | 对 Phase 5（跨具身泛化 45%）间接弱负向 |

### Arxiv 04-16/04-17 论文
无新增 April 17 论文；04-15/16 主论文已被 04-16 digest 覆盖（HiVLA、VLAJS、WOMBET、EEAgent、Sim-Real Co-Training Analysis、RoboLab）。

### 社交情报
- 灵初智能再融资（国投先导 + 京西瑞瓴），继 20 亿天使/Pre-A 后再融资
- 智元酷拓 04-14 上海浦东产品发布
- **顶级实验室信号连续 13 天缺席**（结构性信号：产业资本热度 vs 学术端沉默背离持续扩大）

---

## 5. 纪律检查

### 致命实验状态
- 无 7 天内到期的致命实验
- 最早到期组（2026-12）距今 ~7-8 个月：B0（纯架构创新 >30%）、B4（无团队真机>1000ep WM>BC+RL）、B5（>10B FAST 达 π0.6 / AR 超 FM humanoid）

### 保守偏误
- **B1 (43d) / B6 (43d) / B7 (33d) / B8 (31d)** 四节点持续超阈值
- B0 刚上调重置
- **B8 重新评估**：04-15 审计建议下调至 65%，但 TouchAnything 等 4 条近期累积提供正向证据，**维持 70%**；致命实验"连续 3 月 VLA+tactile 占比 <5%"继续追踪
- **B1 审查建议**：43d 未变更，π0.7 效应对 B1 不确定；下次周审查优先

### 预测追踪（30 天内）
- **#9 (新)**：π0.7 第三方独立复现能否达到 PI 宣称的 "match specialist" 水平？若 < 70% PI 值 → B0 需重新校准
- **#10 (新)**：HuggingFace / LeRobot / 清华 / MIT 中任一团队公布 π0.7 微调结果（到期 2026-05-17）

---

## 6. 今日 1 句话总结

**π0.7 以"数据工程就是主战场"正式定调 — B0 上调至 77% 恢复父子一致性；但社区复现困境（论文 vs 实测 2-3x 衰减）是系统性阻力，等待 3-6 月的独立复现数据。**

---

Sources:
- [Physical Intelligence — pi0.7 can direct robots to solve unfamiliar tasks (TechCrunch, 2026-04-16)](https://techcrunch.com/2026/04/16/physical-intelligence-a-hot-robotics-startup-says-its-new-robot-brain-can-figure-out-tasks-it-was-never-taught/)
- [Physical Intelligence Blog](https://www.physicalintelligence.company/blog)
- [小红书社区采集 2026-04-17-auto.md (本地)](computer://C:\Users\sou35\KW_VLA\memory\blog\archives\xiaohongshu-community\2026-04-17-auto.md)
- [VLA 社交情报 2026-04-16.md (本地)](computer://C:\Users\sou35\KW_VLA\memory\blog\archives\vla-social-intel\2026-04-16.md)
- [BELIEF_GRAPH.md (已更新)](computer://C:\Users\sou35\KW_VLA\docs\system\BELIEF_GRAPH.md)
