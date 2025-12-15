# 代码题详细解答 (Code Answers & Explanations)

为 `code_questions.md` 的 30 题提供更详细的思路、示例代码与常见坑。示例以 Python 为主，强调可运行的最小片段与验证要点。

---

## 推荐刷题顺序（逻辑索引）
1) [Python 基础](#1-python-基础)  
2) [Python 数学与数值计算](#7-python-数学与数值计算)  
3) [数据格式与解析](#8-数据格式与解析)  
4) [Python 面向对象与类设计 (OOP)](#9-python-面向对象与类设计-oop)
5) [PyTorch 训练](#2-pytorch-训练)  
6) [学习算法与模型](#6-学习算法与模型-xgboost--cnn--resnet)  
7) [Git 协作](#3-git-协作)  
8) [SLAM / 视觉里程计](#4-slam--视觉里程计)  
9) [运动控制 / 轨迹规划](#5-运动控制--轨迹规划)  

---

## 术语速览（本答案中出现的关键术语，解释到可操作）
- **行迭代 / 流式统计**: 用 `for line in f:` 或分块读取，边读边更新计数/聚合，不一次性载入全文件，内存恒定，适合 GB 级日志。
- **重采样 (resample)**: 将时间序列按统一步长重排，同一窗口做聚合（`mean/max`）或插值（`ffill/bfill/asfreq`），用于不同频率传感器对齐。
- **近邻对齐 (`merge_asof`)**: 按时间找最近记录，设 `tolerance` 限定最大间隔，`direction` 控制前/后/最近；常用于多源时间戳对齐。
- **Schema 校验**: 先定义字段类型/范围/非空，再批量校验数据帧或记录，提前发现脏数据（`pandera`/`pydantic`/`jsonschema`）。
- **数值稳定 softmax / logsumexp**: 先减去最大值再指数，避免溢出；`logsumexp` 同理。
- **AMP (自动混合精度)**: `autocast + GradScaler`，低精度算、高精度累积，减显存、提速。
- **梯度裁剪**: `clip_grad_norm_`/`clip_grad_value_` 限制梯度大小，防止梯度爆炸。
- **DDP**: `torch.distributed` 的多卡数据并行，需 `init_process_group`、`DistributedSampler`、随机种子对齐，`DDP(model, device_ids=[rank])`。
- **权重衰减分组**: 对 `bias/LayerNorm` 设 `weight_decay=0`，其他参数正常衰减，防过正则。
- **闭式解 / GD**: 线性回归解析解 `(X^T X)^{-1}X^T y` vs 梯度下降迭代；带正则用 `(XtX + λI)^{-1}Xt y`（岭回归）。

---

## 库解法 / 自实现 提示
- **Q1 LRU**: 自实现双向链表 + dict；库解法可用 `cachetools.LRUCache`。
- **Q38 softmax / logsumexp**: 自实现减最大值；库解法用 `scipy.special.softmax/logsumexp` 或 `torch.nn.functional.softmax/logsumexp`。
- **Q40 线性回归**: 自实现闭式解/GD；库解法用 `sklearn.linear_model.LinearRegression` 或 `Ridge`。
- **Q41 JSON/NDJSON 校验**: 自实现逐行解析；库解法用 `jsonschema.validate` 或 `pydantic`/`marshmallow`。
- **Q42 CSV/Parquet**: pandas 自带 `read_csv`/`to_parquet`；高性能可用 `pyarrow` 或 `polars`。
- **Q47 时间对齐**: 自实现遍历/二分；库解法用 `pandas.merge_asof`（已示例）或 `polars.join_asof`。
- **Q48 Schema 校验**: 自实现字段检查；库解法用 `pandera`（已示例）/`pydantic`/`jsonschema`。

---

## 1) Python 基础

### Q1 LRU Cache（O(1) get/put）
- **概念/目标**: O(1) 时间 get/put 的最近最少使用缓存。
- **为什么**: 频繁访问热点需快速命中；淘汰长尾节省内存。
- **自实现步骤**: dict 做索引；双向链表维护顺序；get/put 时移动到表头；超限删尾。
- **库方案**: `cachetools.LRUCache(maxsize)` 或 `functools.lru_cache`（装饰器形式）。
- **示例（自实现）**:
```python
from collections import OrderedDict

class LRU:
    def __init__(self, cap: int):
        self.cap = cap
        self.od = OrderedDict()
    def get(self, k):
        if k not in self.od:
            return None
        self.od.move_to_end(k, last=False)
        return self.od[k]
    def put(self, k, v):
        if k in self.od:
            self.od.move_to_end(k, last=False)
        self.od[k] = v
        if len(self.od) > self.cap:
            self.od.popitem(last=True)
```
- **验证要点**: 空缓存返回 None；覆盖写后保持顺序；容量=1 边界；超限自动淘汰。
- **常见坑**: 忘记访问后移动到头；put 未处理已有 key；超限未删尾。

### Q2 流式统计 1GB+ 日志字段频次
- **概念/目标**: 大文件逐行读取，统计某字段频次，内存恒定。
- **为什么**: 1GB+ 日志不可一次性载入；流式可边读边聚合。
- **自实现步骤**: `for line in f:` 行迭代；split 取字段；Counter 更新。
- **库方案**: `pandas.read_csv(chunksize=...)` 分块聚合；`polars.scan_csv` 懒加载。
- **示例（行迭代）**:
```python
from collections import Counter

def count_field(path, field_idx=0, sep="\t", topk=10):
    c = Counter()
    with open(path, "r", buffering=1024*1024) as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split(sep)
            if field_idx < len(parts):
                c[parts[field_idx]] += 1
    return c.most_common(topk)
```
- **验证要点**: 空行/缺列被跳过；`wc -l` 行数对齐；topk 合理。
- **常见坑**: 未 strip 导致空 key；分隔符不对；大文件未设较大缓冲。

### Q3 滑动窗口生成器
- **概念/目标**: 按固定窗口和步长产出子序列。
- **为什么**: 时序特征提取/批量切片常用。
- **自实现步骤**: 检查 win/stride 正；range 迭代切片；不足窗口不输出。
- **库方案**: `more_itertools.windowed(iterable, n, step)`。
- **示例**:
```python
def sliding(seq, win, stride=1):
    if win <= 0 or stride <= 0:
        raise ValueError
    n = len(seq)
    for i in range(0, n - win + 1, stride):
        yield seq[i:i+win]
```
- **验证要点**: win>len 无输出；stride>win 正常；空序列返回空迭代器。
- **常见坑**: 包含末尾不足窗口；win/stride 非正未检查。

### Q4 有限状态机解析指令流
- **概念/目标**: 定义状态转移表，解析指令序列并校验合法性。
- **为什么**: 控制流程需显式状态管理，避免非法顺序。
- **自实现步骤**: Enum 定义状态；dict 定义转移；遍历指令查转移，不存在则抛错。
- **库方案**: `transitions` 库可快速定义状态机。
- **示例**:
```python
from enum import Enum

class State(Enum):
    IDLE = 0
    ARMED = 1
    RUN = 2

TRANS = {
    State.IDLE: {"arm": State.ARMED},
    State.ARMED: {"start": State.RUN, "disarm": State.IDLE},
    State.RUN: {"stop": State.IDLE},
}

def run_fsm(cmds):
    s = State.IDLE
    for cmd in cmds:
        if cmd not in TRANS[s]:
            raise ValueError(f"illegal transition {s}->{cmd}")
        s = TRANS[s][cmd]
    return s
```
- **验证要点**: 合法序列终态正确；非法指令抛异常；空指令留在初始态。
- **常见坑**: 忘记处理未知指令；转移表漏状态。

### Q5 CLI 扫描大文件并删除确认
- **概念/目标**: 扫描目录大文件，支持 dry-run 和安全删除。
- **为什么**: 排查磁盘占用，防误删需确认。
- **自实现步骤**: argparse 解析参数；Path.rglob 遍历；统计/排序；交互确认再删除。
- **库方案**: 亦可用 `click` 写 CLI；删除用 `send2trash` 更安全（回收站）。
- **示例**:
```python
import argparse, pathlib

def scan(root, threshold_mb):
    files = []
    for p in pathlib.Path(root).rglob("*"):
        if p.is_file():
            sz = p.stat().st_size
            if sz >= threshold_mb * 1024 * 1024:
                files.append((sz, p))
    files.sort(reverse=True)
    return files

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--threshold-mb", type=int, default=100)
    ap.add_argument("--delete", action="store_true")
    args = ap.parse_args()
    files = scan(args.root, args.threshold_mb)
    for sz, p in files:
        print(f"{sz/1024/1024:.1f}MB\t{p}")
    if args.delete:
        confirm = input("Delete all? [y/N] ")
        if confirm.lower().startswith("y"):
            for _, p in files:
                p.unlink()

if __name__ == "__main__":
    main()
```
- **验证要点**: dry-run 时不删除；无匹配提示；确认输入大小写。
- **常见坑**: 未处理权限错误；阈值单位误用；递归时忽略隐藏文件需求未确认。

---

## 2) PyTorch 训练

### Q6 Dataset + collate_fn 变长序列
- **示例**:
```python
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class SeqDS(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return torch.tensor(self.data[i], dtype=torch.long)

def collate(batch):
    lengths = torch.tensor([len(x) for x in batch])
    padded = pad_sequence(batch, batch_first=True, padding_value=0)
    return padded, lengths

dl = DataLoader(SeqDS([[1,2],[3,4,5]]), batch_size=2, collate_fn=collate)
```
- **验证**: padding 值正确；lengths 递减可用于 pack_padded_sequence。

### Q7 训练循环含 AMP/梯度裁剪/LR 调度
- **示例**:
```python
scaler = torch.cuda.amp.GradScaler()
for step, (x,y) in enumerate(dl):
    opt.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast():
        logits = model(x)
        loss = criterion(logits, y)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    scheduler.step()
```
- **验证**: 无梯度爆 NaN；scheduler 调用时机在 step 之后。

### Q8 checkpoint 保存/恢复
- **示例**:
```python
def save_ckpt(path, model, opt, sched, scaler, epoch, step):
    torch.save({
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "sched": sched.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch, "step": step
    }, path)

def load_ckpt(path, model, opt, sched, scaler, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["opt"])
    sched.load_state_dict(ckpt["sched"])
    scaler.load_state_dict(ckpt["scaler"])
    return ckpt["epoch"], ckpt["step"]
```
- **验证**: 不同设备加载；缺文件处理；恢复后继续下降。

### Q9 DDP 最小示例
- **示例**:
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world, backend="nccl"):
    dist.init_process_group(backend, rank=rank, world_size=world)
    torch.cuda.set_device(rank)

def main_worker(rank, world):
    setup(rank, world)
    model = DDP(MyModel().cuda(rank), device_ids=[rank])
    ds = MyDataset()
    sampler = torch.utils.data.distributed.DistributedSampler(ds)
    dl = DataLoader(ds, batch_size=32, sampler=sampler)
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        for x,y in dl:
            ...
```
- **验证**: 各 rank 数据不重复；仅 rank0 评估/保存。

### Q10 自定义 Autograd Function (可微 clamp)
- **示例**:
```python
class SoftClamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lo, hi):
        ctx.save_for_backward(x, torch.tensor(lo), torch.tensor(hi))
        return x.clamp(lo, hi)
    @staticmethod
    def backward(ctx, grad_out):
        x, lo, hi = ctx.saved_tensors
        mask = (x >= lo) & (x <= hi)
        return grad_out * mask, None, None
```
- **验证**: `torch.autograd.gradcheck`；边界值梯度为 0。

### Q11 推理性能分析与优化
- **示例要点**:
```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    with torch.no_grad():
        model(x)
print(prof.key_averages().table(sort_by="cuda_time_total"))
```
- **优化清单**: 避免重复 `.to()`；合并小张量操作；`torch.compile` / `torch.jit.trace`；`pin_memory=True`、`non_blocking=True`。

### Q12 权重衰减与梯度分组
- **示例**:
```python
decay, no_decay = [], []
for n, p in model.named_parameters():
    if not p.requires_grad: continue
    if p.ndim == 1 or n.endswith("bias"):
        no_decay.append(p)
    else:
        decay.append(p)
opt = torch.optim.AdamW([
    {"params": decay, "weight_decay": 0.01},
    {"params": no_decay, "weight_decay": 0.0},
], lr=3e-4)
```
- **验证**: 组数与参数量对齐；无漏配。

---

## 3) Git 协作

### Q13 脏工作树安全同步
- **步骤**: `git status` → `git stash push -u` → `git fetch` → `git rebase origin/main` (或新分支) → `git stash pop` 解决冲突 → `git add` → `git rebase --continue` → `git push --force-with-lease`（若改历史）。
- **验证**: stash 空/非空路径；冲突解决后工作树干净。

### Q14 `.gitignore` 设计
- **示例片段**:
```
__pycache__/
*.pt
*.ckpt
*.h5
logs/
data/
build/
*.so
```
- **说明**: 体积大、可重建、隐私/合规风险。

### Q15 `rebase -i` 整理提交
- **流程**: `git rebase -i HEAD~N` 选 `squash/fixup/edit`；冲突解决后 `git rebase --continue`；异常用 `git rebase --abort`。
- **验证**: `git log --oneline` 检查历史；CI 通过。

### Q16 pre-commit 钩子
- **示例** `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.3.0
    hooks: [{id: black}]
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.8
    hooks: [{id: ruff}]
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.10.0
    hooks: [{id: mypy, additional_dependencies: ["types-requests"]}]
```
- **使用**: `pip install pre-commit && pre-commit install`。

---

## 4) SLAM / 视觉里程计

### Q17 特征匹配 + RANSAC 单应/本质
- **示例**:
```python
kp, des = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(des, des2)
pts1 = np.float32([kp[m.queryIdx].pt for m in matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
```
- **注意**: 本质矩阵需内参归一化；检查内点比例。

### Q18 PnP 位姿估计
- **示例**:
```python
ret, rvec, tvec, inliers = cv2.solvePnPRansac(
    objectPoints=pts3d, imagePoints=pts2d, cameraMatrix=K,
    distCoeffs=None, reprojectionError=3.0, flags=cv2.SOLVEPNP_EPNP
)
```
- **验证**: 重投影误差；inlier 数量；尺度单位一致。

### Q19 两帧位姿恢复 (E 矩阵)
- **示例**:
```python
E, mask = cv2.findEssentialMat(pts1n, pts2n, focal=1.0, pp=(0,0), method=cv2.RANSAC, prob=0.999, threshold=1e-3)
_, R, t, mask_pose = cv2.recoverPose(E, pts1n, pts2n)
```
- **注意**: 需归一化相机坐标；尺度不可观。

### Q20 简易闭环检测
- **思路**: ORB-BOW 或全局描述子 (NetVLAD)；阈值 + 时间一致性；候选通过几何校验。
- **伪码**:
```python
scores = sim(query_vec, db_vecs)   # 余弦相似
cands = [i for i,s in enumerate(scores) if s > thr and abs(i - curr_idx) > gap]
if cands:
    best = max(cands, key=lambda i: scores[i])
    # 触发回环，添加约束
```

### Q21 Pose Graph 优化
- **思路**: 节点=位姿，边=相对约束；鲁棒核抑制外点；用 G2O/Ceres/ISAM2。
- **伪码**:
```python
for (i,j, Tij, info) in edges:
    err_ij = Log(Tij.inv() * Ti.inv() * Tj)
    cost += rho(err_ij.T @ info @ err_ij)
optimize(cost)
```

### Q22 关键帧策略
- **规则示例**: 视差 > p_thresh 或内点数 < n_thresh 或时间间隔 > t；淘汰冗余帧（覆盖度、视差小）。

### Q23 立体匹配 + 视差优化
- **思路**: 代价体 (SAD/Census) → WTA/SGM → 子像素插值 → 左右一致性检查 → 中值滤波。
- **验证**: 边缘噪声、无纹理区域；填充空洞。

---

## 5) 运动控制 / 轨迹规划

### Q24 离散 PID（抗饱和）
- **示例**:
```python
class PID:
    def __init__(self, kp, ki, kd, dt, i_limit=1.0, u_limit=1.0):
        self.kp, self.ki, self.kd, self.dt = kp, ki, kd, dt
        self.i_limit, self.u_limit = i_limit, u_limit
        self.i = 0.0
        self.prev_e = 0.0
    def step(self, target, curr):
        e = target - curr
        self.i = max(-self.i_limit, min(self.i + e*self.dt, self.i_limit))
        d = (e - self.prev_e) / self.dt
        u = self.kp*e + self.ki*self.i + self.kd*d
        self.prev_e = e
        return max(-self.u_limit, min(u, self.u_limit))
```
- **验证**: 阶跃响应不过冲；积分分离效果；限幅生效。

### Q25 Pure Pursuit
- **核心**: 选前视点 (lookahead)；曲率 `k = 2*y_l/Ld^2`；转向 `delta = atan(L*k)`。
- **验证**: 低速适当缩短 Ld；避免路径反复跳点。

### Q26 Stanley 控制器
- **公式**: `delta = heading_error + atan(k * crosstrack / v)`；低速抖动时对 `v` 加下限或加前馈转角。

### Q27 A* / Hybrid A*
- **要点**: 栅格启发式 (Manhattan/Euclid) 一致性；Hybrid 采样朝向、车辆长度约束；父节点存储便于回溯。
- **验证**: 障碍膨胀后路径仍可行；启发式不高估。

### Q28 MPC (单轨模型 QP)
- **步骤**: 线性化 → 构建大矩阵 (A_bar, B_bar) → 代价 `Σ (x_ref - x)^T Q (x_ref - x) + Δu^T R Δu` → QP 求解 → 施加第 0 步控制。
- **验证**: QP 可行性；软约束处理越界；时延补偿。

### Q29 碰撞检测与安全距离
- **要点**: 车体矩形 → 多圆或 SAT；障碍膨胀安全距离；路径逐段检测，命中即提前退出。

### Q30 控制回路延迟/噪声仿真
- **思路**: 在仿真中对观测/执行加入 `delay` 与 `noise`；观察超调/震荡；对策：滤波 (LPF/卡尔曼)、前馈、降低增益、Smith 预估补偿。

---

## 6) 学习算法与模型 (XGBoost / CNN / ResNet)

### Q31 用 XGBoost 训练二分类并评估 AUC
- **思路**: 读取 CSV → train/valid 划分 → `XGBClassifier` 训练，eval_set 开启 AUC 监控与早停。
- **示例**:
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

df = pd.read_csv("data.csv")
X, y = df.drop("label", axis=1), df["label"]
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = XGBClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, eval_metric="auc"
)
model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
pred = model.predict_proba(Xva)[:,1]
print("AUC:", roc_auc_score(yva, pred))
```
- **注意**: 缺失值可直接喂给 XGBoost；类别特征需编码；早停可用 `early_stopping_rounds`。

### Q32 XGBoost 参数调优脚本
- **思路**: 随机/网格搜索核心超参；以 AUC/Logloss 选优；早停避免过拟合。
- **示例伪码**:
```python
param_grid = [
    {"max_depth": d, "eta": lr, "subsample": ss, "colsample_bytree": cs}
    for d in [4,6,8] for lr in [0.03,0.05,0.1]
    for ss in [0.7,0.9] for cs in [0.7,0.9]
]
best = None
for p in param_grid:
    m = XGBClassifier(
        n_estimators=800, eval_metric="auc",
        **p
    )
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False, early_stopping_rounds=50)
    auc = roc_auc_score(yva, m.predict_proba(Xva)[:,1])
    if best is None or auc > best[0]:
        best = (auc, p)
print("best", best)
```
- **注意**: 控制搜索规模；固定随机种子；特征重要性（gain/cover）用于分析。

### Q33 手写最小 CNN 分类器 (PyTorch)
- **思路**: Conv-BN-ReLU-池化 堆叠，最后全连接；小数据先过拟合检验链路。
- **示例**:
```python
import torch, torch.nn as nn, torch.nn.functional as F

class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64*8*8, num_classes)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)         # 32x16x16
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)         # 64x8x8
        x = x.flatten(1)
        return self.fc(x)
```
- **验证**: 在少量样本上过拟合；检查输入归一化与数据增强关闭对齐。

### Q34 ResNet BasicBlock 前向
- **思路**: 两个 3x3 Conv-BN，中间 ReLU，残差加和后 ReLU；步幅或通道变更时用下采样。
- **示例**:
```python
class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.down is not None:
            identity = self.down(x)
        out += identity
        return F.relu(out)
```
- **验证**: 输出形状对齐；当 stride=2 或通道变更时残差分支匹配。

### Q35 迁移学习微调 ResNet
- **思路**: 载入预训练 resnet18；冻结特征层，替换 FC；训练头部后再小 LR 全量微调。
- **示例**:
```python
import torchvision.models as models

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
for p in model.parameters():
    p.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, num_classes)

# 第一阶段：仅训练头
opt = torch.optim.AdamW(model.fc.parameters(), lr=1e-3)
# 若验证收敛，再解冻全网，低 LR 微调
for p in model.parameters():
    p.requires_grad = True
opt_finetune = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)
```
- **验证**: 冻结时只更新头部参数；解冻后学习率更小；评估 Top-1/Top-5。

---

## 7) Python 数学与数值计算

### Q36 大数阶乘与对数阶乘
- **概念/目标**: 计算 n! 及 ln(n!)，避免溢出。
- **为什么**: 大 n 时阶乘极大；概率/组合常需 log 形式防溢出。
- **自实现步骤**: 迭代乘积；log 用 `math.lgamma`；检查 n>=0。
- **库方案**: `math.factorial`（精确整数），`scipy.special.gammaln`（ln Γ）。
- **示例**:
```python
import math

def fact(n: int) -> int:
    if n < 0: raise ValueError
    out = 1
    for i in range(2, n+1):
        out *= i
    return out

def log_fact(n: int) -> float:
    return math.lgamma(n + 1)  # ln(n!)
```
- **验证要点**: 与 `math.factorial` 对比；n>1000 使用 log；负数抛异常。
- **常见坑**: 直接计算大数溢出；忽略输入校验。

### Q37 矩阵乘法与广播
- **概念/目标**: 正确使用矩阵乘与广播规则。
- **为什么**: 形状错易导致 silent broadcast 或计算错误。
- **自实现步骤**: 检查内积维度；广播从尾维对齐。
- **库方案**: NumPy `@`/`matmul`，PyTorch `torch.matmul`。
- **示例**:
```python
import numpy as np

A = np.random.randn(2,3)
B = np.random.randn(3,4)
C = A @ B  # (2,4)

x = np.ones((5,1,4))
y = np.arange(3).reshape(1,3,1)
z = x + y  # 形状 (5,3,4)
```
- **验证要点**: 内积维度一致；广播期望形状；对不期望的广播需显式 `reshape`。
- **常见坑**: 维度搞反；无意间触发广播；混用 HWC/CHW。

### Q38 数值稳定的 softmax 与 log-sum-exp
- **概念/目标**: 计算 softmax / logsumexp 时防止指数溢出。
- **为什么**: 大数直接 exp 会 inf，导致 NaN。
- **自实现步骤**: 先减最大值；exp 再归一化；logsumexp 同理。
- **库方案**: `scipy.special.softmax/logsumexp`；`torch.nn.functional.softmax/logsumexp`。
- **示例**:
```python
import numpy as np

def softmax(x):
    x = np.asarray(x)
    m = x.max()
    e = np.exp(x - m)
    return e / e.sum()

def logsumexp(x):
    x = np.asarray(x)
    m = x.max()
    return m + np.log(np.exp(x - m).sum())
```
- **验证要点**: 与 naive 结果在小数值上一致；大数不溢出；softmax 和约等于 1。
- **常见坑**: 忘记减最大值；对批次维度未指定 axis。

### Q39 随机采样与设定随机种子
- **概念/目标**: 多库一致设种，确保结果可复现。
- **为什么**: 不设种导致实验不可复现，调试困难。
- **自实现步骤**: 分别设 `random`/`numpy`/`torch` 种子；关闭 cudnn benchmark。
- **库方案**: PyTorch `torch.manual_seed`，`torch.use_deterministic_algorithms(True)`（需权衡性能）。
- **示例**:
```python
import random, numpy as np, torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```
- **验证要点**: 多次运行结果一致；统计特性合理。
- **常见坑**: 只设一个库；GPU 未设；开启 deterministic 导致不支持的算子需替换。

### Q40 线性回归的闭式解与梯度下降
- **概念/目标**: 线性回归求解参数，理解解析 vs 迭代。
- **为什么**: 小规模用闭式快，大规模/流式用 GD 更灵活。
- **自实现步骤**: 解析用正态方程；GD 用学习率迭代；可加 L2 正则。
- **库方案**: `sklearn.linear_model.LinearRegression` / `Ridge`；PyTorch 自动求梯度。
- **示例**:
```python
import numpy as np

def linreg_closed_form(X, y):
    # X: (n,d)
    XtX = X.T @ X
    w = np.linalg.solve(XtX, X.T @ y)
    return w

def linreg_gd(X, y, lr=1e-2, steps=500):
    n, d = X.shape
    w = np.zeros(d)
    for _ in range(steps):
        pred = X @ w
        grad = (1/n) * X.T @ (pred - y)
        w -= lr * grad
    return w
```
- **验证要点**: 无噪声时与闭式接近；加入 L2 后误差下降；学习率过大不发散。
- **常见坑**: `X^T X` 奇异需正则；学习率过大爆炸；未标准化导致条件数差。

---

## 8) 数据格式与解析

### Q41 解析与验证 JSON/NDJSON
- **概念/目标**: 大量 JSON 行流式读取并校验必需字段/类型。
- **为什么**: NDJSON 常用于日志/流数据，需内存友好与健壮性。
- **自实现步骤**: 行迭代 `json.loads`；检查必需字段；异常行计数。
- **库方案**: `jsonschema.validate` 做结构校验；`pydantic` 定义模型后 `.model_validate_json`。
- **示例**:
```python
import json

def load_ndjson(path, required=("id","ts")):
    good, bad = [], 0
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if not all(k in obj for k in required):
                    raise ValueError("missing fields")
                good.append(obj)
            except Exception:
                bad += 1
    return good, bad
```
- **验证要点**: 空行跳过；缺字段/类型错计入 bad；统计总行与有效行。
- **常见坑**: 未 strip 导致空行解析异常；未限制字段类型；错误未计数。

### Q42 CSV/Parquet 互转与类型保真
- **概念/目标**: CSV↔Parquet 转换时保持列类型和行数一致。
- **为什么**: 默认推断易把数值读成 object；缺失值处理不一致会污染数据。
- **自实现步骤**: 读 CSV 显式 dtype/na_values；写 Parquet；读回校验形状与 dtypes。
- **库方案**: pandas + pyarrow；高性能可用 `polars.scan_csv().collect().write_parquet()`
- **示例**:
```python
import pandas as pd

df = pd.read_csv("data.csv", dtype={"id": "int64"}, na_values=["", "NA"])
df.to_parquet("data.parquet", index=False)
df2 = pd.read_parquet("data.parquet")
assert df.shape == df2.shape
assert df.dtypes.equals(df2.dtypes)
```
- **验证要点**: 行列数一致；数值列未变成 object；缺失值数量一致。
- **常见坑**: 未设 dtype 导致类型漂移；NaN/空字符串混淆；Parquet 写入未指定 `index=False`。

### Q43 时间戳与时区处理
- **概念/目标**: 多种时间表示（ISO8601、Unix 秒）统一成 UTC 可感知时间戳。
- **为什么**: 避免时区/夏令时导致排序和对齐错误。
- **自实现步骤**: `pd.to_datetime(..., utc=True)`；非法转换为 NaT；排序用 UTC。
- **库方案**: pandas；`arrow`/`pendulum` 也可处理时区。
- **示例**:
```python
import pandas as pd

def normalize_ts(s):
    # s: pandas Series of str/int
    ts = pd.to_datetime(s, utc=True, errors="coerce")
    return ts

ts = normalize_ts(pd.Series(["2024-01-01T00:00:00Z", 1704067200]))  # Unix 秒
ts_sorted = ts.sort_values()
```
- **验证要点**: 含时区字符串正确解析；非法值为 NaT；排序按 UTC。
- **常见坑**: 未设 utc=True 导致本地偏移；混用秒/毫秒时间戳。

### Q44 图像批处理的格式与元数据
- **概念/目标**: 批量读写图像，保持元数据与通道一致。
- **为什么**: 训练数据需统一通道/尺寸；保存需控制压缩与信息丢失。
- **自实现步骤**: PIL 读取→转 np array；检查通道；保存时设质量/优化。
- **库方案**: `Pillow`；高性能批处理可用 `opencv`；元数据处理可用 `piexif`.
- **示例**:
```python
from PIL import Image
import numpy as np

def load_image(path):
    img = Image.open(path)
    arr = np.array(img)        # HWC, uint8
    return arr, img.info       # info 含 EXIF/icc_profile

def save_image(arr, path, quality=90):
    img = Image.fromarray(arr)
    img.save(path, quality=quality, optimize=True)
```
- **验证要点**: 通道/模式不变；写出文件大小随质量变化；元数据可选保留。
- **常见坑**: RGBA 直接保存到 JPEG 导致 alpha 丢失；uint8/float32 混用导致显示异常。

### Q45 NumPy/Torch dtype 与内存占用检查
- **概念/目标**: 控制 dtype 以权衡精度与内存。
- **为什么**: 默认 float64 占用大且算慢；训练常用 float32/bfloat16。
- **自实现步骤**: 创建数组张量时显式 dtype；比较 `nbytes`/`element_size`。
- **库方案**: NumPy、PyTorch 原生；可用 `torch.set_default_dtype` 设置默认。
- **示例**:
```python
import numpy as np, torch

a64 = np.ones((1024,1024), dtype=np.float64)
a32 = a64.astype(np.float32)

t64 = torch.ones((1024,1024), dtype=torch.float64)
tbf = t64.bfloat16()

print(a64.nbytes/1e6, a32.nbytes/1e6)
print(t64.element_size(), tbf.element_size())
```
- **验证要点**: 核心算子输出 dtype 正确；内存占用符合预期；精度需求满足。
- **常见坑**: 混合 dtype 导致隐式 upcast；默认 float64 未显式控制。

### Q46 时间序列重采样与对齐
- **思路**: 将多路不同频率的流对齐到统一时间栅格；用 `resample`/`asfreq`/`ffill`/`bfill`；对缺口做质量告警。
- **示例**:
```python
import pandas as pd

def resample_stream(df, freq="100ms", method="ffill"):
    # df: columns [ts, value]
    df = df.set_index("ts").sort_index()
    if method == "mean":
        out = df.resample(freq).mean()
    else:
        out = df.resample(freq).asfreq()
        if method == "ffill":
            out = out.ffill()
        elif method == "bfill":
            out = out.bfill()
    gaps = out["value"].isna().sum()
    return out.reset_index(), gaps
```
- **验证**: 乱序时间戳先排序；统计插值/缺失个数；大缺口时报警。

### Q47 按时间戳近邻对齐多源数据
- **思路**: `merge_asof` 按时间近邻匹配，设 `tolerance` 防止跨过大窗口；`direction='nearest'/'backward'` 控制方向；统计未匹配行。
- **示例**:
```python
import pandas as pd

def align_by_time(df_main, df_aux, tol="200ms"):
    df_main = df_main.sort_values("ts")
    df_aux = df_aux.sort_values("ts")
    merged = pd.merge_asof(
        df_main, df_aux, on="ts", direction="nearest", tolerance=pd.Timedelta(tol), suffixes=("", "_aux")
    )
    unmatched = merged["value_aux"].isna().sum()
    return merged, unmatched
```
- **验证**: 方向与容差是否合理；对 unmatched 行单独输出检查。

### Q48 数据 Schema 校验 (Pandera/Pydantic/JsonSchema)
- **思路**: 先定义 Schema，再批量校验，提前发现类型/范围/缺失问题；pandera 适合 DataFrame，pydantic/JsonSchema 适合记录。
- **示例 (pandera)**:
```python
import pandera as pa
from pandera import Column, DataFrameSchema, Check

schema = DataFrameSchema({
    "id": Column(int, Check.greater_than_or_equal_to(0), nullable=False),
    "value": Column(float, Check.in_range(-1e3, 1e3), nullable=False),
    "ts": Column(pa.Timestamp, nullable=False),
})

def validate_df(df):
    return schema.validate(df, lazy=True)  # 返回校验后的 DF，lazy 收集全部错误
```
- **验证**: 缺字段/类型不符/越界应报错；lazy 模式收集多条错误；对错误样本记录行号与原因。

### Q49 视触传感器数据对齐与可视化
- **思路**: 相机帧与触觉/力矩流按时间对齐（重采样或 `merge_asof` 近邻匹配）；归一化力/扭矩；联合可视化：左侧图像，右侧力/扭矩曲线标注当前帧时刻。
- **示例**:
```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def load_tactile(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["ts"])
    df = df.sort_values("ts")
    return df

def align_frame_to_force(df_force, frame_ts, tol="30ms"):
    df_force = df_force.sort_values("ts")
    # 取离 frame_ts 最近的一条力/扭矩记录
    m = pd.merge_asof(
        pd.DataFrame({"ts": [frame_ts]}).sort_values("ts"),
        df_force,
        on="ts",
        direction="nearest",
        tolerance=pd.Timedelta(tol),
    )
    return m.iloc[0]

def plot_frame_with_force(img_path, df_force, frame_ts):
    row = align_frame_to_force(df_force, frame_ts)
    forces = df_force[["fx","fy","fz"]].to_numpy()
    torques = df_force[["tx","ty","tz"]].to_numpy()
    t = (df_force["ts"] - df_force["ts"].iloc[0]).dt.total_seconds()

    fig, ax = plt.subplots(1, 2, figsize=(10,4))
    ax[0].imshow(Image.open(img_path))
    ax[0].set_title(f"Frame @ {frame_ts}")
    ax[0].axis("off")

    ax[1].plot(t, forces[:,0], label="fx")
    ax[1].plot(t, forces[:,1], label="fy")
    ax[1].plot(t, forces[:,2], label="fz")
    ax[1].axvline((frame_ts - df_force['ts'].iloc[0]).total_seconds(), color="k", ls="--", label="frame ts")
    ax[1].set_xlabel("time (s)")
    ax[1].legend()
    plt.tight_layout()
    plt.show()
```
- **验证**: 乱序时间戳需排序；未匹配时返回 NaN 并提示；力/扭矩单位统一后再归一化（如除以量程）。

---

## 9) Python 面向对象与类设计 (OOP)

### Q50 抽象基类 (ABC) 定义统一接口
- **概念/目标**: 强制子类实现特定方法，定义统一的 API 契约。
- **为什么**: 大型项目中确保不同硬件驱动/模块遵循相同的接口。
- **自实现步骤**: 继承 `abc.ABC`；装饰器 `@abstractmethod`；子类必须覆盖。
- **示例**:
```python
from abc import ABC, abstractmethod

class BaseRobot(ABC):
    @abstractmethod
    def connect(self): pass
    @abstractmethod
    def move(self, x, y): pass

class Arm(BaseRobot):
    def connect(self): print("Arm connected")
    def move(self, x, y): print(f"Arm move to {x},{y}")

# r = BaseRobot()  # TypeError: Can't instantiate abstract class
a = Arm()
a.connect()
```
- **验证要点**: 实例化基类报错；子类漏写抽象方法报错。

### Q51 继承与 super() 初始化
- **概念/目标**: 正确调用父类初始化逻辑，支持多重继承。
- **为什么**: 避免重复代码；Mixin 模式增加功能。
- **自实现步骤**: `super().__init__()` 传递必要参数；注意 MRO (Method Resolution Order)。
- **示例**:
```python
class Robot:
    def __init__(self, name): self.name = name

class LoggableMixin:
    def log(self, msg): print(f"[{self.name}] {msg}")

class HexapodRobot(Robot, LoggableMixin):
    def __init__(self, name, legs=6):
        super().__init__(name)
        self.legs = legs

h = HexapodRobot("Spider", 6)
h.log(f"has {h.legs} legs")
```
- **验证要点**: 父类属性正确初始化；Mixin 方法可用。

### Q52 @property 属性校验与封装
- **概念/目标**: 像访问属性一样调用 getter/setter，增加校验逻辑。
- **为什么**: 防止设置非法物理参数（如关节角度越界）。
- **自实现步骤**: `@property` 定义 getter；`@xxx.setter` 定义 setter。
- **示例**:
```python
class Joint:
    def __init__(self, angle=0.0, limits=(-1.0, 1.0)):
        self._limits = limits
        self.angle = angle  # 触发 setter

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, value):
        lo, hi = self._limits
        if not (lo <= value <= hi):
            raise ValueError(f"Angle {value} out of bounds {self._limits}")
        self._angle = value
```
- **验证要点**: 读取正常；越界赋值抛异常。

### Q53 魔术方法实现自定义序列
- **概念/目标**: 实现 `__len__`, `__getitem__` 等，使对象表现得像 Python 内置类型。
- **为什么**: 增加代码可读性，支持 `len()`, `for`, `+` 等操作。
- **自实现步骤**: 实现对应 dunder methods。
- **示例**:
```python
class Trajectory:
    def __init__(self, points=None):
        self.points = points or []
    def __len__(self): return len(self.points)
    def __getitem__(self, i): return self.points[i]
    def __add__(self, other):
        if not isinstance(other, Trajectory): return NotImplemented
        return Trajectory(self.points + other.points)
    def __repr__(self): return f"Trajectory(len={len(self)})"
```
- **验证要点**: `len(t)`；`t[0]`；`t1 + t2`。

### Q54 类方法工厂与静态方法
- **概念/目标**: `@classmethod` 作工厂构造；`@staticmethod` 作工具函数。
- **为什么**: 提供多种实例化方式（如从 JSON/Config）；将相关工具函数组织在类中。
- **自实现步骤**: `cls` 参数构造实例；无 `self/cls` 参数做纯函数。
- **示例**:
```python
import json

class Config:
    def __init__(self, host, port):
        self.host, self.port = host, port

    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        return cls(data["host"], data["port"])

    @staticmethod
    def validate_port(port):
        return 0 <= port <= 65535
```
- **验证要点**: `Config.from_json(...)` 返回实例；`Config.validate_port` 可直接调用。

---

## 使用与自测
- 建议对每题写最小可运行脚本与单测（pytest），验证边界与异常。
- 训练题建议跑一个 toy 数据集（如随机分类）验证 loss 降、checkpoint 续训、DDP 正常。
- SLAM/控制类可用小型数据或仿真 (OpenCV 示例图、matplotlib/pybullet) 验证。

