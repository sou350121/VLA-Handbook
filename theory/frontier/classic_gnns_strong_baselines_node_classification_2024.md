# 经典 GNN 仍然很强：节点分类基线再评估 (Classic GNNs are Strong Baselines for Node Classification)

> **发布时间**：2024（NeurIPS Datasets & Benchmarks）  
> **论文题目**：Classic GNNs are Strong Baselines: Reassessing GNNs for Node Classification  
> **核心定位**：通过系统调参与消融，证明经典 GCN/GAT/GraphSAGE 在节点分类上仍是强基线，GT 的优势很大程度来自 GNN 调参不足。

社区普遍认为 Graph Transformer 在节点分类上压过传统 GNN，但这篇工作显示：把归一化、dropout、残差与层数等关键超参调到位后，经典 GNN 仍可实现非常强的结果。

**一手来源**：
- 论文 PDF：`https://proceedings.neurips.cc/paper_files/paper/2024/file/b10ed15ff1aa864f1be3a75f1ffc021b-Paper-Datasets_and_Benchmarks_Track.pdf`  
- 代码实现：`https://github.com/LUOyk1999/tunedGNN`

---

## 0. 先把“可复述结论”写清楚（1 分钟版）
- **一句话**：GT 的优势未必来自架构本身，经典 GNN 在充分调参后仍是强基线。  
- **方法**：统一对比 GCN/GAT/GraphSAGE 与多种 GT，覆盖同质/异质/大图，系统调参 + 消融。  
- **主要发现**：调参后经典 GNN 在 18 个数据集里 17 个达到或超过 GT（论文口径，[PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/b10ed15ff1aa864f1be3a75f1ffc021b-Paper-Datasets_and_Benchmarks_Track.pdf)）。  
- **工程启示**：归一化、dropout、残差与层数选择对性能影响巨大，评测必须交代搜索空间与训练细节。  
- **局限**：仅覆盖节点分类，未讨论图级/链接预测任务；调参成本高。

---

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 维度 | 经典 GNN (GCN/GAT/GraphSAGE) | Graph Transformer (GT) | 本文评估设置 |
|---|---|---|---|
| 输入 → 输出 | 图结构 + 节点特征 → 节点类别 | 同上 | 同一数据集/划分 |
| 关键机制 | 局部消息传递 | 全局注意力/结构增强 | 统一评测下比较 |
| 主要超参 | Norm/Dropout/Residual/Depth | Heads/PE/Hops/Global tokens | 分别按各自搜索空间调参 |
| 训练策略 | 小图可全批 | 常需采样/分批 | 按图规模选择全批/采样/分区 |
| 指标 | Accuracy / ROC-AUC | 同上 | 多次运行取均值±方差 |

### 1.2 关键机制 (Key Mechanism)

1) **系统调参**：对 GCN/GAT/GraphSAGE 的归一化、dropout、残差、层数、隐藏维度与学习率进行系统搜索。  
2) **广覆盖评测**：同质/异质/大规模图共 18 个数据集，避免单一数据集偏差。  
3) **消融验证**：拆解归一化、dropout、残差和层深对性能的贡献（论文口径）。

### 1.3 信息流/架构图 (Flow / Diagram)

```
Datasets (homophily / heterophily / large-scale)
            │
            ├─ Classic GNNs (GCN / GAT / GraphSAGE)
            └─ Graph Transformers (GraphGPS / SGFormer / Polynormer ...)
                       │
             Hyperparameter search + ablation
                       │
        Train (full-batch / sampling / partition)
                       │
      Metrics (Accuracy / ROC-AUC, mean ± std)
                       │
                  Comparison & conclusions
```

---

## 2. 数学核心：消息传递如何实现节点分类 (Math Core)

**目标**：通过消息传递聚合邻居特征，得到节点表示并完成分类。

**通用消息传递**：

$$
\mathbf{h}_{v}^{l} = \text{UPDATE}^l\Big(\mathbf{h}_{v}^{l-1},\ \text{AGG}^l(\{\mathbf{h}_{u}^{l-1}\mid u\in\mathcal{N}(v)\})\Big)
$$

**GCN 形式**：

$$
\mathbf{h}_{v}^{l}=\sigma\Big(\sum_{u\in\mathcal{N}(v)\cup\{v\}}\frac{1}{\sqrt{\hat{d}_u\hat{d}_v}}\mathbf{h}_{u}^{l-1}\mathbf{W}^{l}\Big)
$$

**分类目标**：

$$
\mathcal{L}=\sum_{v\in\mathcal{V}_{\text{train}}}\text{CE}(\mathbf{y}_v,\ \text{softmax}(\mathbf{h}_v^{L}\mathbf{W}_{\text{out}}))
$$

- $\mathcal{N}(v)$：节点 $v$ 的邻居集合  
- $\hat{d}_v$：加自环后的度数  
- $\mathbf{W}^{l}$：第 $l$ 层可学习权重  
- $\sigma(\cdot)$：非线性激活（如 ReLU）

**直觉**：每一层把“邻居的信息”混入自身表示，层数越深感受野越大；归一化和残差用于稳定深层训练并抑制过平滑。

---

## 3. 带数字走一遍：玩具例子 (Worked Example)

**场景**：节点 $v$ 有两个邻居 $u_1, u_2$，用简单均值聚合。

- 初始特征：$h_v^0=0.2,\ h_{u_1}^0=1.0,\ h_{u_2}^0=-0.5$  
- 聚合（含自环均值）：$\bar{h} = (0.2+1.0-0.5)/3 = 0.233$  
- 设 $W=1$、ReLU，则 $h_v^1=\max(0,\bar{h})=0.233$

**若加残差**：$h_v^1 = 0.2 + 0.233 = 0.433$  

---

## 4. 工程视角：评测与训练折中 (Engineering View)

### 4.1 训练与资源权衡
- **小图可全批**：Cora/CiteSeer 等可直接 full-batch 训练。  
- **大图需采样/分区**：论文采用邻居采样或随机分区来训练大图（如 OGB 与 pokec），避免显存爆炸（论文口径）。  
- **多次运行**：报告均值与方差，降低偶然性（论文口径）。

### 4.2 超参启发
- **归一化**：大图更依赖 BN/LN 来稳定训练。  
- **Dropout**：跨多类数据集表现一致有效。  
- **残差与层深**：异质图上残差与更深层数更关键（论文口径）。

### 4.3 可复现实践
- 报告搜索空间与训练细节，避免“默认超参”造成的误判。  
- 开源代码与配置让比较更公平（见 [tunedGNN](https://github.com/LUOyk1999/tunedGNN)）。

---

## 5. 数据与评测 (Data & Eval)

> 论文覆盖 18 个数据集，包含同质/异质与大规模图（论文口径）。

| 类别 | 代表数据集 | 常用指标 |
|---|---|---|
| 同质图 (Homophily) | Cora / CiteSeer / PubMed / WikiCS 等 | Accuracy |
| 异质图 (Heterophily) | Squirrel / Chameleon / Roman-Empire / Questions 等 | Accuracy / ROC-AUC |
| 大规模图 | ogbn-arxiv / ogbn-products / ogbn-proteins / pokec | Accuracy / ROC-AUC |

---

## 6. 能力与失败模式 (Capabilities & Failure Modes)

**能力**：
- 经典 GNN 在充分调参后仍能达到强基线甚至 SOTA。  
- 消融给出清晰的工程指导（归一化、dropout、残差与层深的重要性）。  
- 评测覆盖广，减少“单一数据集偏差”。

**失败模式 / 局限**：
- 只验证节点分类，未覆盖图级/链接预测任务（论文局限）。  
- 性能对超参极其敏感，调参成本高。  
- 并未提出新模型，更多是“评测与工程方法论”贡献。

---

## 7. 与相关工作对比 (Comparison)

| 方向 | 优势 | 局限 | 适用场景 |
|---|---|---|---|
| 经典 GNN（本文调参） | 架构简单、消息传递高效、调参后竞争力强 | 依赖调参、深层易过平滑 | 节点分类强基线 |
| Graph Transformer | 全局依赖建模强 | 计算/显存昂贵，需结构增强 | 长程依赖强的图任务 |
| 异质图专用 GNN | 针对 heterophily 设计 | 适用范围窄、结构复杂 | 强异质图场景 |

**面试 Tip**：  
“被问 GT vs GNN 时，先强调**评测公平性与超参敏感性**；经典 GNN 在充分调参后仍是强基线，GT 的优势需要在统一搜索空间下再比较。”

---

## 参考链接
- 论文 PDF：`https://proceedings.neurips.cc/paper_files/paper/2024/file/b10ed15ff1aa864f1be3a75f1ffc021b-Paper-Datasets_and_Benchmarks_Track.pdf`  
- 代码实现：`https://github.com/LUOyk1999/tunedGNN`

---
[← Back to Theory](../README.md)
