# DCP：凸性检测规则与 CVX/CVXPY 建模心法 (Disciplined Convex Programming)

> **定位**：优化建模方法论 / 工程速记  
> **核心定位**：解释 CVX/CVXPY 如何用 DCP 规则判断凸性、为什么“数学上凸”仍会被拒绝、以及如何重构问题通过 DCP。

**参考入口**：
- CVX：`https://cvxr.com/cvx/`  
- CVXPY：`https://github.com/cvxpy/cvxpy`  
- DCP 主页：`https://dcp.stanford.edu/`  
- Convex.jl：`https://github.com/jump-dev/Convex.jl`  
- CVXR：`https://github.com/cvxgrp/CVXR`

---

## 0) 1 分钟版结论

- **DCP 是规则系统，不是求导验算**：CVX/CVXPY 不会去算 Hessian，而是用“原子函数曲率 + 复合规则”做判定。  
- **DCP 是充分非必要条件**：数学上凸的函数也可能被拒绝。  
- **正确思路**：优先把目标/约束改写成 DCP 可识别的“原子 + 组合”；否则考虑引入辅助变量、使用等价的凸表达式，或降级为非凸求解器。

---

## 1) 背景：为什么要用 DCP？

**凸优化的关键好处**：任何局部最优都是全局最优。  
因此，只要问题是凸的，求解器只需收敛到局部最优即可。

现实中，我们通常希望把实际问题尽可能改写为凸优化，并使用下列工具：
- **CVX**（MATLAB）  
- **CVXPY**（Python）  
- **Convex.jl**（Julia）  
- **CVXR**（R）

这些工具的共同基础就是 **DCP（Disciplined Convex Programming）规则系统**。

---

## 2) 凸集与凸函数（最小复习）

**凸集定义**：集合 $C$ 若对任意 $x,y\\in C$ 与 $\\theta\\in[0,1]$，  
满足 $\\theta x + (1-\\theta) y \\in C$，则 $C$ 为凸集。

**凸函数定义**：函数 $f$ 在凸域内满足  
$f(\\theta x + (1-\\theta) y) \\le \\theta f(x) + (1-\\theta) f(y)$。

**常见凸函数**：仿射函数、二次函数（$Q\\succeq 0$）、范数、最大值函数、指数、负对数、log-sum-exp、核范数等。

---

## 3) 凸性判断准则（工程常用）

1) **一阶条件**：  
若 $f$ 可微，则 $f$ 凸当且仅当  
$f(y) \\ge f(x) + \\nabla f(x)^T (y-x)$。

2) **二阶条件**：  
若 $f$ 二阶可微，则 $\\nabla^2 f(x) \\succeq 0$。

3) **保凸运算**：  
- 非负加权和、仿射变换、逐点上确界仍凸  
- 复合规则（单调性是关键）：  
  - 凸且非减 ∘ 凸 → 凸  
  - 凸且非增 ∘ 凹 → 凸

4) **限制到任意直线**：  
对任意方向 $d$，$f(x+td)$ 是一元凸函数。

---

## 4) DCP 的核心规则（CVX/CVXPY 的判定逻辑）

**DCP 的本质**：  
系统只允许你用“已标注曲率与单调性”的**原子函数**进行组合。

### 4.1 原子函数（Atoms）
每个原子函数都有预定义属性：
- **曲率**：constant / affine / convex / concave  
- **单调性**：nondecreasing / nonincreasing / nonmonotonic

示例：
- `square(x)`：凸，但 **nonmonotonic**  
- `exp(x)`：凸且单调递增  
- `log(x)`：凹且单调递增  
- `norm(x)`：凸且单调递增（在范数意义下）

### 4.2 组合规则（递归推理）
DCP 会从内到外递归判断表达式是否满足组合规则。  

**目标函数与约束的硬规则**：
- 最小化：目标必须是 **convex**  
- 最大化：目标必须是 **concave**  
- 等式约束：必须是 **affine == affine**  
- 不等式约束：  
  - $f(x) \\le g(x)$ 需要 $f$ 凸、$g$ 凹  
  - 等价于 $f(x) - g(x) \\le 0$，要求 $f-g$ 凸

**关键事实**：  
CVX/CVXPY **不会做 Hessian 检测**，只做规则推理。  
这意味着：**DCP 是充分条件，不是必要条件**。

---

## 5) 一个常见“数学上凸但 DCP 拒绝”的例子

考虑目标中的正则项：

$$
\\frac{\\rho}{1+\\|x\\|^2}
$$

很多人用二阶条件证明它在某些域内是凸的，但 **CVX 会报错**：

```
Disciplined convex programming error:
Cannot perform the operation:
{positive constant} ./ {convex}
```

**原因**：  
在 DCP 规则中，“正数 ÷ 凸函数”不是允许的组合。  
CVX 不会去判断它是否真的凸，只要不符合规则就拒绝。

---

## 6) DCP 通过的常见重构策略

1) **引入辅助变量（Epigraph / Hypograph）**  
把复杂函数拆成原子函数的组合，再用约束连接。

2) **使用 CVX 原子函数替换**  
例如 `quad_over_lin`、`log_sum_exp`、`inv_pos`、`-log_det` 等。

3) **换等价的凸表达式**  
例如把 $x^2$ 改写成 `quad_over_lin(x,1)`，或使用范数表达。

4) **若本质是准凸**  
可以考虑 CVXPY 的 DQCP 模式，或降级为非凸求解器（Ipopt 等）。

---

## 7) 最佳实践清单

- 建模前先查 **CVX Atom Library**：`https://cvxr.com/cvx/doc/funcref.html`  
- 尽量使用已知原子函数（如 `norm`、`quad_over_lin`、`log_sum_exp`、`-log_det`）  
- 遇到 DCP error：先考虑**引入辅助变量**重构  
- 不要指望 CVX 帮你做“数学证明”；它只做规则推理

---

## 8) 一句话回答：CVX 如何检测凸性？

**答案**：  
CVX 通过 DCP 规则系统进行**语法级别的曲率推理**：  
每个原子函数预设曲率/单调性，表达式按规则递归组合；若满足规则则接受，否则报错。  
它不做 Hessian 计算，也不做数值验证。

---

## 参考链接
- CVX：`https://cvxr.com/cvx/`  
- CVXPY：`https://github.com/cvxpy/cvxpy`  
- DCP 主页：`https://dcp.stanford.edu/`  
- Convex.jl：`https://github.com/jump-dev/Convex.jl`  
- CVXR：`https://github.com/cvxgrp/CVXR`

---
[← Back to Theory](./README.md)
