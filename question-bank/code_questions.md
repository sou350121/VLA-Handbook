# 代码实战题 (Code Questions with Answers)

面向 VLA 具身算法岗的高频代码/工程题，共 30 题，含要点式解答。按主题分组，可用于突击与自测。  
**需要更详细代码与解释**: 见 `code_answers.md`。

---

## 推荐刷题顺序（逻辑索引）
1) [Python 基础](#1-python-基础)  
2) [Python 数学与数值计算](#7-python-数学与数值计算)  
3) [数据格式与解析](#8-数据格式与解析)  
4) [PyTorch 训练](#2-pytorch-训练)  
5) [学习算法与模型](#6-学习算法与模型-xgboost--cnn--resnet)  
6) [Git 协作](#3-git-协作)  
7) [SLAM / 视觉里程计](#4-slam--视觉里程计)  
8) [运动控制 / 轨迹规划](#5-运动控制--轨迹规划)  

---

## 1) Python 基础

**Q1 LRU Cache（O(1) get/put）**  
哈希 + 双向链表；get/put 需移到表头；超限删尾；注意空缓存与重复 key。

**Q2 流式统计 1GB+ 日志字段频次**  
行迭代 + `Counter.update`；避免一次性加载；提供 top-k；支持过滤字段。

**Q3 滑动窗口生成器**  
`yield seq[i:i+win]`，可选 stride；窗口不足提前退出；测试空序列/窗口大于长度。

**Q4 有限状态机解析指令流**  
`Enum` 状态 + 转移表；非法转移抛自定义异常；单测覆盖正常/异常路径。

**Q5 CLI 扫描大文件并删除确认**  
`argparse` + `pathlib.rglob`；dry-run；按大小排序输出；删除前交互确认。

---

## 2) PyTorch 训练

**Q6 自定义 Dataset + collate_fn 处理变长序列**  
`pad_sequence` 对齐；返回 lengths 供 mask；保留原始数据 id 便于调试。

**Q7 训练循环含 AMP、梯度裁剪、LR 调度**  
`autocast` + `GradScaler`；`clip_grad_norm_`；scheduler.step 时机在 optimizer.step 之后。

**Q8 checkpoint 保存/恢复**  
保存 `model/optimizer/scheduler/scaler` + epoch/step；加载映射到设备；`model.train()` 续训。

**Q9 DDP 最小示例**  
`init_process_group`，`DistributedSampler`，`DDP(model, device_ids=[rank])`；对齐随机种子；只在 rank0 记录日志。

**Q10 自定义 Autograd Function (可微 clamp)**  
前向保存 mask；反向对越界梯度置零；用 `gradcheck` 验证。

**Q11 推理性能分析与优化**  
`torch.profiler` 定位热点；优化：减少 `.to()`/`.cpu()`，合并小 kernel，`pin_memory`，`non_blocking`，`torch.compile`/JIT。

**Q12 权重衰减与梯度分组**  
分组排除 `bias/LayerNorm`；param_groups 校验；记录组大小避免漏配。

---

## 3) Git 协作

**Q13 脏工作树同步远端**  
`git stash push -u` → `git fetch` → `git rebase origin/main` 或新分支 → `git stash pop` 解决冲突 → 提交。

**Q14 设计 `.gitignore` 防泄漏**  
忽略 `*.pt/*.ckpt/*.h5`、数据目录、`__pycache__`、`*.so`、日志；说明体积/隐私/可重建原因。

**Q15 `rebase -i` 压缩/改写提交**  
`git rebase -i HEAD~N` 选 `squash/fixup/edit`；冲突后 `git rebase --continue`；需要时 `--abort`。

**Q16 pre-commit 钩子示例**  
`.pre-commit-config.yaml` 运行 `black/ruff/mypy`；`pre-commit install`；阻止未格式化提交。

---

## 4) SLAM / 视觉里程计

**Q17 特征匹配 + RANSAC 单应/本质矩阵**  
ORB/SIFT + `BFMatcher`；`findHomography` 或 `findEssentialMat` with RANSAC；输出内点掩码后恢复姿态。

**Q18 PnP 位姿估计**  
`solvePnP/solvePnPRansac`；给内参/畸变；EPnP 初值；检查重投影误差。

**Q19 双目/两帧位姿恢复 (E 矩阵)**  
归一化相机坐标→`findEssentialMat`→`recoverPose`；尺度需外源（IMU/里程计）。

**Q20 简易闭环检测**  
ORB BoW/NetVLAD 向量；余弦相似度阈值 + 时间一致性；触发回环约束。

**Q21 Pose Graph 优化**  
节点=位姿，边=相对约束；误差项 `log(T_ij^-1 * T_i^-1 * T_j)`；鲁棒核抑制外点；用 G2O/Ceres。

**Q22 关键帧策略**  
新帧加入：视差阈值/内点数/时间间隔；淘汰：冗余、覆盖度低；共视图更新边。

**Q23 立体匹配 + 视差优化**  
代价体（SAD/Census）+ WTA/SGM；子像素插值；左右一致性检查 + 中值滤波。

---

## 5) 运动控制 / 轨迹规划

**Q24 离散 PID（抗饱和）**  
`u = kp*e + ki*Σe*dt + kd*Δe/dt`；积分分离/限幅；死区/微分先行；仿真阶跃响应。

**Q25 Pure Pursuit 跟踪**  
选前视点；曲率 `2*y_l/Ld^2`；转向 `delta = atan(L*curvature)`；前视距离随速度调节。

**Q26 Stanley 控制器**  
转向 = 航向误差 + `atan(k * crosstrack / v)`；低速抖动需限幅或前馈。

**Q27 A* / Hybrid A* 路径规划**  
栅格 + 启发式；Hybrid A* 采样朝向与运动学步长；记录父节点复原路径；障碍膨胀。

**Q28 MPC (单轨模型 QP)**  
线性化单轨，`x_{k+1}=Ax+Bu`；代价含轨迹误差/控制增量；用 OSQP/qpOASES；软约束防不可行。

**Q29 碰撞检测与安全距离**  
车体矩形近似圆/多边形；障碍膨胀 `inflate_radius`；SAT 或圆距离检测；早停优化。

**Q30 控制回路延迟/噪声仿真**  
在仿真中注入时延/噪声；观察超调/震荡；改进：滤波、前馈、降低增益、预测补偿。

---

## 6) 学习算法与模型 (XGBoost / CNN / ResNet)

**Q31 用 XGBoost 训练二分类并评估 AUC**  
加载 CSV，划分 train/valid，`xgboost.XGBClassifier` 训练；监控 eval_set AUC；保存模型与特征重要性。

**Q32 XGBoost 参数调优脚本**  
网格/随机搜索 max_depth、eta、subsample、colsample_bytree；早停；输出最优参数与曲线。

**Q33 手写一个最小 CNN 分类器 (PyTorch)**  
Conv-BN-ReLU-MaxPool 堆叠 + 全连接；实现训练/验证循环；在 CIFAR-10 子集过拟合。

**Q34 ResNet block 前向实现**  
BasicBlock: 3x3 Conv-BN-ReLU-3x3 Conv-BN，加残差，最后 ReLU；确保通道/步幅对齐时用下采样分支。

**Q35 迁移学习微调 ResNet**  
加载 torchvision resnet18 预训练；冻结前层，替换最后 FC；仅训练头部后再全量解冻小 LR 微调；评估 Top-1。

---

## 7) Python 数学与数值计算

**Q36 大数阶乘与对数阶乘**  
实现迭代阶乘与 `math.lgamma` 对数阶乘；比较溢出与性能。

**Q37 矩阵乘法与广播**  
用 `numpy`/`torch` 实现矩阵乘；演示广播规则与形状检查。

**Q38 数值稳定的 softmax 与 log-sum-exp**  
实现减最大值的稳定写法；验证与 naive 版本的差异。

**Q39 随机采样与设定随机种子**  
`random`/`numpy`/`torch` 统一设种；采样均值方差对齐；演示 reproducibility。

**Q40 线性回归的闭式解与梯度下降**  
推导正规方程 `(X^T X)^{-1} X^T y`，实现数值解与 GD 迭代，比较误差。

---

## 8) 数据格式与解析

**Q41 解析与验证 JSON/NDJSON**  
流式读取 NDJSON，验证必需字段与类型，异常行计数并跳过。

**Q42 CSV/Parquet 互转与类型保真**  
读取 CSV 指定 dtype、缺失值处理；转换为 Parquet 并校验行数/列类型。

**Q43 时间戳与时区处理**  
把 ISO8601/Unix 秒互转，统一为 UTC；处理夏令时/时区偏移并排序。

**Q44 图像批处理的格式与元数据**  
批量读取 PNG/JPEG，保留 EXIF/尺寸，统一通道顺序与 dtype；写出压缩参数可控。

**Q45 NumPy/Torch dtype 与内存占用检查**  
演示 float64→float32/bfloat16 转换的精度/内存对比；防止隐式 float64。

**Q46 时间序列重采样与对齐**  
用 pandas 将不同频率的传感器流重采样到统一步长；选择 `mean/ffill/bfill` 并对缺口报警。

**Q47 按时间戳近邻对齐多源数据**  
`merge_asof` 基于时间对齐，设 tolerance 防止跨大窗口匹配；对未匹配行计数并输出。

**Q48 数据 Schema 校验 (Pandera/Pydantic/JsonSchema)**  
定义字段类型、取值范围、非空约束；批处理数据帧前先校验并输出错误样本。

**Q49 视触传感器数据对齐与可视化**  
对齐相机帧与触觉/力矩传感器流（时间戳近邻或重采样），归一化力/扭矩并在同一时间轴可视化（图像 + 力曲线）。

---

使用建议：按模块刷，先 Python/PyTorch，再 Git，再 SLAM/控制；每题配最小可运行 demo/单测可快速验证掌握度。

