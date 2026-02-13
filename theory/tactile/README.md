# 🖐️ 触觉专题入口（Tactile / VTLA / Contact-Rich Control）

> 目标：把“触觉相关内容”从各处笔记里 **单独拎出**，形成一个可复用的导航页：**先看什么、再看什么、落地要看哪里**。

---

## 0. 先给一句话结论（你可以直接复述）

**触觉的核心价值不是“更细的图像”，而是把操作里最难的那段——“接触之后的闭环纠错（滑移/卡死/力调制/遮挡）”变成可观测、可学习、可验收。**

---

## 1. 3 条阅读路径（按你的目标选）

### A) 你要“先跑通能用”（工程优先）

- **主入口（总览 + Checklist）**：[`../tactile_vla.md`](../tactile_vla.md)
- **触觉如何让策略更稳（单目 + 二值触觉）**：[`../frontier/visual_tactile_pretraining_online_multitask_learning_2026.md`](../frontier/visual_tactile_pretraining_online_multitask_learning_2026.md)
- **System 0 / 接触闭环在全身系统里的位置**：[`../frontier/figure_helix_02_full_body_autonomy_2026.md`](../frontier/figure_helix_02_full_body_autonomy_2026.md)

### B) 你要“研究路线图”（表征/对齐/泛化）

- **触觉为何不可替代（研究问题→可计算变量）**：[`../frontier/tactile_irreplaceable.md`](../frontier/tactile_irreplaceable.md)
- **SaTA / 空间锚定触觉（把触觉变成有坐标语义的 token）**：见 [`../tactile_vla.md`](../tactile_vla.md) 的 SaTA 小节
- **TaF-VLA（触觉-力对齐）**：[`../frontier/taf_vla_tactile_force_alignment_2026.md`](../frontier/taf_vla_tactile_force_alignment_2026.md)
- **UniTacHand（人手→机器人触觉技能迁移）**：[`../frontier/unitachhand.md`](../frontier/unitachhand.md)
- **UniVTAC（统一视触觉仿真平台 + benchmark）**：[`../frontier/univtac_unified_visuo_tactile_simulation_platform_2026.md`](../frontier/univtac_unified_visuo_tactile_simulation_platform_2026.md)
- **SuperTac + DOVE（多模态电子皮肤 + 触觉语言模型）**：[`../../deployment/perception/supertac_dove_biomimetic_multimodal_tactile_sensing.md`](../../deployment/perception/supertac_dove_biomimetic_multimodal_tactile_sensing.md)
- **GenForce（跨触觉传感器的可迁移力感知）**：[`./genforce_tactile_force_transfer_2026.md`](./genforce_tactile_force_transfer_2026.md)

### C) 你要“硬件/传感器选型”（产品化优先）

- **触觉传感器谱系与选型建议**：[`../tactile_vla.md`](../tactile_vla.md)
- **（部署）触觉阵列与算法口径**：[`../../deployment/perception/tactile_array_algorithms_capacitive_piezoresistive.md`](../../deployment/perception/tactile_array_algorithms_capacitive_piezoresistive.md)
- **（部署）触觉集成常见坑**：[`../../deployment/tactile_sensor_integration_challenges.md`](../../deployment/tactile_sensor_integration_challenges.md)

---

## 2. 一张 ASCII 图：触觉在系统里的“自然位置”

```text
              (slow)                     (fast)                     (ultra-fast)
Language/Task ───► S2 (plan/latents) ───► S1 (targets/policy) ──────► S0 (contact reflex)
                                        ▲                               ▲
                                        │                               │
Vision (global) ────────────────────────┘                               │
Tactile/Force (contact) ────────────────────────────────────────────────┘

直觉：视觉负责“去哪/大致怎么做”；触觉负责“接触后怎么稳住、怎么纠错、怎么把误差当场消化”。
```

---

## 3. “二值触觉 vs 高分辨率触觉”怎么选？（一句话决策）

- **先追求“稳定做完任务”**：二值触觉（contact/no-contact）+ 时间对齐 + 阈值漂移管理，往往比直接堆高分辨率更划算  
  - 参考：[`../frontier/visual_tactile_pretraining_online_multitask_learning_2026.md`](../frontier/visual_tactile_pretraining_online_multitask_learning_2026.md)
- **追求“精细力控/抗滑移/装配精度/低损伤”**：需要更丰富的剪切/力/接触分布（2–4bit 强度、阵列触觉、F/T、或更强 proprio 代理）  
  - 参考：[`../tactile_vla.md`](../tactile_vla.md) 的“触觉 vs 本体（Proprioception）”与工程 checklist

---

## 4. 工程落地入口（不在 theory 侧重复维护）

- **触觉工程主入口（Perception）**：[`../../deployment/perception/README.md`](../../deployment/perception/README.md)  
  - 推荐从 `4.1 触觉落地的最小验收点（Checklist）` 开始。

---

[← Back to Theory](../README.md)

