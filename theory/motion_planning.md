# 运动规划 (Motion Planning)

> **面试场景**: “请你介绍一下机器人运动规划的常见方法，并说明在实际工程中如何选择？”

本文总结采样式规划、轨迹优化、混合范式以及工程落地 (MoveIt / cuRobo) 的要点，帮助快速回答机器人运动规划相关问题。

## 0. 主要數學思想 (Main Mathematical Idea)

> **第一性原理**: **Connectivity and Smoothness (连通性与平滑性)**

规划问题本质上是在高维空间中寻找一条"连接起点和终点"且"无碰撞"的曲线。

- **核心数学工具**: **Graph Search (离散图搜索)** vs **Functional Optimization (连续泛函优化)**。
- **解题逻辑**:
    1.  **采样式 (Sampling-based)**: 将连续的几何空间**离散化**为图 (Graph)。利用概率论中的"稠密性" (当采样点足够多时，一定会覆盖可行解)，将几何问题转化为图论中的连通性问题 (如 DFS/BFS)。
    2.  **优化式 (Optimization-based)**: 将路径视为一根有弹性的"橡皮筋"。通过最小化能量函数 (Energy Functional = 平滑度代价 + 障碍物斥力)，让路径在物理约束下自然收敛到最优形状。

---

## 🧭 运动规划全景

```
┌──────────────────────────────────────────────────────────────────────────┐
│                       规划决策树                                           │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   起点 + 目标 + 约束                                                     │
│          │                                                               │
│          ├── 低维 (2D/3D) → 栅格搜索 (A*, D*)                             │
│          │                                                               │
│          └── 高维 (机械臂, 7DoF+)                                         │
│                 │                                                         │
│                 ├── 采样式规划 (Sampling-Based)                           │
│                 │     • RRT, RRT*, PRM, BIT*                              │
│                 │                                                         │
│                 ├── 轨迹优化 (Trajectory Optimization)                    │
│                 │     • CHOMP, TrajOpt, STOMP, GPMP                       │
│                 │                                                         │
│                 └── 混合方法                                              │
│                       • RRT + TrajOpt (先可行再优化)                      │
│                       • Learning + Planning (策略热启动)                  │
│                                                                          │
│   输出: 可行路径/轨迹 (Piecewise Linear / Polynomial / Spline)           │
└──────────────────────────────────────────────────────────────────────────┘
```

--- 
# 运动规划 (Motion Planning)

> **面试场景**: “请你介绍一下机器人运动规划的常见方法，并说明在实际工程中如何选择？”

本文总结采样式规划、轨迹优化、混合范式以及工程落地 (MoveIt / cuRobo) 的要点，帮助快速回答机器人运动规划相关问题。

## 0. 主要數學思想 (Main Mathematical Idea)

> **第一性原理**: **Connectivity and Smoothness (连通性与平滑性)**

规划问题本质上是在高维空间中寻找一条"连接起点和终点"且"无碰撞"的曲线。

- **核心数学工具**: **Graph Search (离散图搜索)** vs **Functional Optimization (连续泛函优化)**。
- **解题逻辑**:
    1.  **采样式 (Sampling-based)**: 将连续的几何空间**离散化**为图 (Graph)。利用概率论中的"稠密性" (当采样点足够多时，一定会覆盖可行解)，将几何问题转化为图论中的连通性问题 (如 DFS/BFS)。
    2.  **优化式 (Optimization-based)**: 将路径视为一根有弹性的"橡皮筋"。通过最小化能量函数 (Energy Functional = 平滑度代价 + 障碍物斥力)，让路径在物理约束下自然收敛到最优形状。

---

## 🧭 运动规划全景

```
┌──────────────────────────────────────────────────────────────────────────┐
│                       规划决策树                                           │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   起点 + 目标 + 约束                                                     │
│          │                                                               │
│          ├── 低维 (2D/3D) → 栅格搜索 (A*, D*)                             │
│          │                                                               │
│          └── 高维 (机械臂, 7DoF+)                                         │
│                 │                                                         │
│                 ├── 采样式规划 (Sampling-Based)                           │
│                 │     • RRT, RRT*, PRM, BIT*                              │
│                 │                                                         │
│                 ├── 轨迹优化 (Trajectory Optimization)                    │
│                 │     • CHOMP, TrajOpt, STOMP, GPMP                       │
│                 │                                                         │
│                 └── 混合方法                                              │
│                       • RRT + TrajOpt (先可行再优化)                      │
│                       • Learning + Planning (策略热启动)                  │
│                                                                          │
│   输出: 可行路径/轨迹 (Piecewise Linear / Polynomial / Spline)           │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 1. 采样式规划 (Sampling-Based Planning)

### 1.1 核心理念

- 直接在配置空间 \( \mathcal{C} \) 中随机采样节点，避免显式离散化高维空间。
- 避免对障碍物几何进行显式建模，只需要碰撞检测函数 `collisionFree(q)`。
- 随采样次数增加，算法趋近于找到可行路径；带有 *-star* 的算法还能保证渐进最优 (asymptotically optimal)。

### 1.2 RRT / RRT*

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     RRT 扩展示意                                           │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   q_start ●                                                              │
│          │＼                                                             │
│          │  ＼                                                           │
│          │    ● q_new                                                    │
│          │    / │                                                        │
│          │  ／  │                                                       │
│       ● ─┘     │                                                        │
│    q_rand      │                                                        │
│                │                                                        │
│           障碍物                                                         │
│                                                                          │
│   步骤:                                                                   │
│   1. 随机采样 q_rand                                                      │
│   2. 找到树中最近节点 q_near                                              │
│   3. 朝 q_rand 方向扩展 step → q_new                                      │
│   4. 若连线无碰撞，将 q_new 加入树                                        │
│                                                                          │
│   RRT*: 在插入新节点时重新连线 (rewire)，选择代价更低的父节点，并更新子树    │
└──────────────────────────────────────────────────────────────────────────┘
```

| 算法 | 优点 | 缺点 | 适用场景 |
|:-----|:-----|:-----|:---------|
| **RRT** | 实现简单，快速找到可行路径 | 路径质量差，抖动 | 高维配置空间，快速避障 |
| **RRT\*** | 渐进最优，路径更平滑 | 计算量大 (rewire) | 需要较高质量路径 |
| **Informed RRT\*** | 采样限制在椭球区域，加速收敛 | 需要启发式 | 起止点已知、需加速 |
| **BIT\*** | 批量采样 + 最优 | 实现复杂 | 需要兼顾速度和最优性 |

### 1.3 PRM / PRM\*

- **PRM (Probabilistic Roadmap)**: 先离线采样大量节点并构建无向图，在线阶段只需连接起点/终点。
- 适用于多次查询的场景 (同一空间多次规划)。
- PRM\*: 渐进最优版本，连接半径随样本数量动态调整。

---

## 2. 轨迹优化 (Trajectory Optimization)

### 2.1 思路

- 将轨迹离散化为 \( \mathbf{x}_{0:T} \)，定义优化目标 + 约束，转化为非线性优化问题。
- 典型目标：路径长度、速度/加速度正则、与障碍物距离惩罚。
- 约束：动力学约束、接触约束、关节限制、终端姿态等。

### 2.2 典型算法

| 算法 | 核心思想 | 优势 | 局限 |
|:-----|:---------|:-----|:-----|
| **CHOMP** | 基于梯度下降 + 碰撞势能，优化离散轨迹 | 平滑、可融入障碍势 | 对初值敏感，梯度局部 |
| **TrajOpt** | 以凸优化 (sequential convex) 形式求解 | 收敛快，可加复杂约束 | 需要良好初值 |
| **STOMP** | 采样多个扰动轨迹，根据代价加权平均 | 无需梯度，鲁棒 | 计算量大 |
| **GPMP2** | 将轨迹建模为高斯过程，使用因子图优化 | 与 SLAM/估计工具链一致 | 实现复杂 |
| **MPC (Short-horizon)** | 在线滚动优化局部轨迹 | 能处理动态障碍 | 需要实时算力 |

### 2.3 TrajOpt 代价函数示例

```
J(x) = w_smooth * Σ ||x_{t+1} - 2x_t + x_{t-1}||²    (平滑项)
     + w_collision * Σ max(0, d_safe - d(x_t))²       (碰撞约束软化)
     + w_goal * ||x_T - x_goal||²                     (终点约束)

subject to: joint_limits, velocity_limits, etc.
```

---

## 3. 混合方法与工程实践

### 3.1 采样 + 优化 (Warm-Start)

1. **RRT 找到可行路径**，保证碰撞自由。
2. **TrajOpt/CHOMP** 以 RRT 结果作为初始轨迹，进一步平滑并满足动力学约束。
3. 工程中常结合 MoveIt “OMPL (采样) + TrajOpt (优化)” Pipeline。

### 3.2 学习辅助规划

- **策略热启动**: 用模仿/强化学习模型预测初始轨迹，再交给 TrajOpt 优化。
- **价值函数引导采样**: 训练一个 value network 评估节点好坏，提高采样效率 (Learning-RRT)。
- **Diffusion/LfD**: 直接生成可行轨迹，作为规划器候选。

---

## 4. MoveIt & cuRobo 实战

### 4.1 MoveIt 规划流水线

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        MoveIt Pipeline                                   │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Planning Scene ← Obstacles / Robot State                                │
│          │                                                               │
│          ▼                                                               │
│  OMPL Planner (RRTConnect, RRT*)                                         │
│          │  (可选前处理: Simplify)                                      │
│          ▼                                                               │
│  Trajectory Post-processing                                              │
│   • Time Parameterization (TOTG)                                         │
│   • Smoothing                                                            │
│                                                                          │
│  Controller Manager → FollowJointTrajectory                              │
└──────────────────────────────────────────────────────────────────────────┘
```

- **OMPL**: 包含大部分采样式规划器 (RRT, RRT*, PRM, KPIECE)。
- **Pilz**: MoveIt 内置的笛卡尔规划器，适合直线/插补。
- **TrajOpt / STOMP 插件**: 需要额外安装，提供优化型规划。

### 4.2 NVIDIA cuRobo

- 基于 GPU 的运动规划库，采用并行化的 **多种规划器候选 + 轨迹优化**。
- 可在 Jetson Orin 上实现 <100ms 的 7DoF 机械臂规划。

---

## 5. 代码参考

```python
# MoveIt Python RRTConnect 示例
import moveit_commander
import geometry_msgs.msg as geometry_msgs

moveit_commander.roscpp_initialize([])
robot = moveit_commander.RobotCommander()
group = moveit_commander.MoveGroupCommander("manipulator")

# 设置规划器
group.set_planner_id("RRTConnectkConfigDefault")
group.set_num_planning_attempts(10)
group.set_planning_time(2.0)

# 目标
pose_goal = geometry_msgs.Pose()
pose_goal.position.x = 0.4
pose_goal.position.y = 0.2
pose_goal.position.z = 0.3
pose_goal.orientation.w = 1.0

group.set_pose_target(pose_goal)
plan = group.go(wait=True)
group.stop()
group.clear_pose_targets()
```

```python
# 使用 ompl Python API 自定义 RRT*
import ompl.base as ob
import ompl.geometric as og

def is_state_valid(space_information, state):
    # TODO: 调用碰撞检测
    return True

space = ob.RealVectorStateSpace(7)  # 7 DoF
bounds = ob.RealVectorBounds(7)
bounds.setLow(-3.14)
bounds.setHigh(3.14)
space.setBounds(bounds)

si = ob.SpaceInformation(space)
si.setStateValidityChecker(ob.StateValidityCheckerFn(
    lambda state: is_state_valid(si, state)))
si.setup()

problem = ob.ProblemDefinition(si)
start = ob.State(space)
goal = ob.State(space)
# ... 设置 start/goal ...
problem.setStartAndGoalStates(start, goal)

planner = og.RRTstar(si)
planner.setRange(0.1)  # 扩展步长
planner.setProblemDefinition(problem)
planner.setup()

if planner.solve(5.0):
    path = problem.getSolutionPath()
    path.interpolate(100)
```

---

## 6. 面试 Q&A

### Q1: 采样式规划 vs 轨迹优化？

- 采样式：易找到可行解，适合复杂障碍空间；但路径抖动，需要后处理。
- 轨迹优化：可直接考虑动力学和成本，但对初值敏感，可能陷入局部最优。
- 工程实践：先采样找可行路径，再用 TrajOpt 平滑。

### Q2: 如何处理动态障碍？

1. 使用 **在线 Re-planning** (RRT Replan) 或 **MPC**。
2. 维护 **占据栅格** / **ESDF** 实时更新碰撞信息。
3. 短时 MPC + 长期规划结合：全局路径 + 局部避障。

### Q3: 高维机械臂如何加速规划？

- 合理的 **joint limits** 与 **sampling bounds**。
- 使用 **IK 解** 作为起点，减少搜索空间。
- **Task Space RRT**: 在笛卡尔空间采样，再投影回 joint space。
- 利用 GPU (cuRobo) 并行求解。

### Q4: CHOMP 为什么需要“障碍势能”？

- 通过距离场定义势能 \( U(q) = \phi(d(q)) \)，在轨迹优化时通过梯度远离障碍。
- 常用 SDF (Signed Distance Field) 或 ESDF 作为距离计算。

### Q5: 如何保证规划结果可执行？

- **时间参数化**: TOTG / Iterative Parabolic Time Parameterization。
- **速度/加速度限制**: 确保关节命令不超限。
- **轨迹跟踪控制**: Feedforward + PD/MPC。

---

## 📚 推荐阅读

- *Principles of Robot Motion* (MIT Press)
- OMPL: [ompl.kavrakilab.org](https://ompl.kavrakilab.org/)
- MoveIt Tutorials: [moveit.picknik.ai](https://moveit.picknik.ai/)
- NVIDIA cuRobo: [github.com/NVlabs/curobo](https://github.com/NVlabs/curobo)

---

[← Back to Theory Index](./README.md)

