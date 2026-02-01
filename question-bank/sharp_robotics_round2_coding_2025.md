# Sharp Robotics 二面：两道纯手撕代码（矩阵乘法优化 / TopK）(2025)

> **来源**：截图记录（用户整理）  
> **面试轮次**：Sharp Robotics 二面（纯手撕两道）  
> **记录日期**：2025-10-13（截图标题写 10.11，记录处写 10.13，以记录处为准）  
> **核心主题**：数值计算的缓存友好优化（matrix multiply） + TopK 的算法/工程优化路径

---

## 1) 题目 1：实现 $ (m\times n)\cdot(n\times k) $ 的矩阵乘法 + cache 优化追问

### 1.1 题目描述（从截图还原）

给定：
- 矩阵 $A$：形状 $m\times n$
- 矩阵 $B$：形状 $n\times k$

输出：
- $C = A\cdot B$：形状 $m\times k$

先写最基础的三重循环实现。随后面试官追问：

> “第二个数组（通常指 $B$）在内层是**列遍历**，缓存不友好，如何优化实现让访问更 cache-friendly（更像行遍历）？”

### 1.2 基础写法（直观但可能 cache 不友好）

假设使用**行主序（row-major）**存储（C/C++ 常见），直接写：

```cpp
// A: m x n, B: n x k, C: m x k
for (int i = 0; i < m; ++i) {
  for (int j = 0; j < k; ++j) {
    double sum = 0;
    for (int p = 0; p < n; ++p) {
      sum += A[i][p] * B[p][j]; // 固定 j，p 在变：对 B 是“按列走”（stride=k）
    }
    C[i][j] = sum;
  }
}
```

**为什么对 B 不友好**：在 row-major 下，`B[p][j]` 随着 `p` 递增会跨行跳（步长约为 `k`），会导致 cache line 利用率差。

### 1.3 最常见的面试优化点：改循环顺序，让 B 走“行”

把内层循环换成 `j`，让 `B[p][j]` 的 `j` 连续递增（同一行连续内存）：

```cpp
// i-p-j：让 B 按行连续访问，且复用 A[i][p]
// 注意：这个写法对 C 是累加，需保证 C 先初始化为 0
for (int i = 0; i < m; ++i) {
  for (int p = 0; p < n; ++p) {
    const double a = A[i][p];
    for (int j = 0; j < k; ++j) {
      C[i][j] += a * B[p][j];  // B[p][j] 对 j 连续；C[i][j] 也连续
    }
  }
}
```

要点：
- **B 行连续**：`B[p][j]` 的 `j` 递增是顺序访问
- **A 复用**：`A[i][p]` 作为标量 `a` 被复用 `k` 次
- **C 行连续写**：`C[i][j]` 也是顺序写（注意先把 `C` 清零）

### 1.4 另一条路线：先转置 B（把列访问变成行访问）

如果你更想保留 `i-j-p` 的结构，也可以先做 `B_T = transpose(B)`（大小 $k\times n$），使得 `B[p][j]` 访问变成 `B_T[j][p]`（对 `p` 连续）：

```text
预处理：B_T[j][p] = B[p][j]
计算：C[i][j] = Σ_p A[i][p] * B_T[j][p]
```

优点：主循环更直观；缺点：需要额外内存与一次转置成本。

### 1.5 进一步：分块（tiling / blocking）是“工业级答案”

当矩阵大到 L1/L2 装不下时，单纯换循环顺序仍会 cache miss。经典做法是分块：

```text
for i0 in [0..m) step Ti
  for p0 in [0..n) step Tp
    for j0 in [0..k) step Tj
      在 (Ti×Tp)*(Tp×Tj) 的小块上做三重循环
```

面试里不用背 SIMD/AVX 细节，但要能说出：
- 分块让工作集落到 cache
- `Ti/Tp/Tj` 选择与 cache 大小有关

### 1.6 复杂度

- **时间复杂度**：$O(mnk)$
- **空间复杂度**：$O(1)$ 额外空间（若做转置则额外 $O(nk)$）

---

## 2) 题目 2：大数组 Top10 / TopK（欢迎讨论：优化算法路线）

### 2.1 题目描述（从截图还原）

给一个很大的数组（长度 $n$），找出 **Top10 最大的元素**（可推广到 TopK）。

截图里记录了面试对话脉络：
- 直接 `sort` 最简单：$O(n\log n)$
- 面试官大概率想考察：快排/堆排/堆的差别
- 优化方向：空间换时间，维护一个只存 top10 的数据结构
- 候选：最小堆（priority queue），复杂度 $O(n\log k)$（k=10）
- 有人提 `deque`，但并不合适（除非特定单调队列场景；TopK 不是它的典型用法）
- 还提到“分治/快排”的思路（即 Quickselect / partition）

下面把“面试最常用的 4 档答案”按强度列出来。

---

## 3) TopK 的常见解法分层（从最稳到最强）

### 3.1 解法 A：全排序（最简单，但通常不是最优）

- **思路**：排序后取最后 $k$ 个
- **复杂度**：时间 $O(n\log n)$，空间取决于排序实现（原地/额外）
- **适用**：你还需要全局排序结果；或 $n$ 不大

### 3.2 解法 B：最小堆维护 TopK（面试最稳、最通用）

**核心思想**：维护一个大小为 $k$ 的最小堆，堆顶是当前 TopK 里最小的那个：

1) 前 $k$ 个元素入堆  
2) 扫描剩余元素 `x`：
   - 若 `x <= heap_min`：跳过
   - 若 `x > heap_min`：弹出堆顶并插入 `x`

- **时间复杂度**：$O(n\log k)$
- **空间复杂度**：$O(k)$
- **当 k 很小（比如 10）**：几乎线性扫描，工程上非常香

Python 模板：

```python
import heapq

def topk(nums, k):
    h = []
    for x in nums:
        if len(h) < k:
            heapq.heappush(h, x)
        else:
            if x > h[0]:
                heapq.heapreplace(h, x)
    return sorted(h, reverse=True)  # 如需有序输出
```

C++ 模板（最小堆）：

```cpp
#include <queue>
#include <vector>
using namespace std;

vector<int> topk(const vector<int>& a, int k) {
  priority_queue<int, vector<int>, greater<int>> pq;
  for (int x : a) {
    if ((int)pq.size() < k) pq.push(x);
    else if (x > pq.top()) { pq.pop(); pq.push(x); }
  }
  vector<int> ans;
  while (!pq.empty()) { ans.push_back(pq.top()); pq.pop(); }
  // ans 是从小到大；如需从大到小可 reverse 或 sort
  return ans;
}
```

### 3.3 解法 C：Quickselect（分治/partition，期望线性）

**核心思想**：用 partition 把数组按“比 pivot 大/小”分开，递归地只进入包含第 $k$ 大的那一侧。

- **期望时间复杂度**：$O(n)$
- **最坏时间复杂度**：$O(n^2)$（pivot 选得太差）
- **空间复杂度**：原地 $O(1)$（递归栈 $O(\log n)$ 期望）

工程上通常回答：
- C++ 有现成的 `nth_element`：平均线性，把第 `n-k` 个放到正确位置，左边都更小/右边都更大（不保证内部有序）
- 拿到阈值后再把 topK 部分 sort 一下得到有序输出：$O(k\log k)$

C++ 工程口径（推荐）：

```cpp
// nth_element(a.begin(), a.end()-k, a.end()) 之后
// [end-k, end) 是 topK（无序）
```

### 3.4 解法 D：流式/外存/并行（当 n 超大）

如果数组大到内存装不下，面试官可能会继续追问：
- **外部排序 / MapReduce**：分块各自取 topK，再对候选集合做 topK（候选规模约为 `num_shards * k`）
- **并行**：每个线程维护一个本地最小堆 topK，最后合并

这是“系统面试”加分项，不要求写完，但要能讲思路。

---

## 4) 常见追问点（你可以直接背的答法）

### 4.1 “为什么不是 deque？”

`deque`/单调队列常用于：
- 滑动窗口最大值（Window Max）
- 需要维护“队首永远最大/最小”的序列结构

TopK 是全局选择问题，不是窗口问题；除非题目额外约束（比如“数据流 + 只关心最近 W 个”），否则 `deque` 不是主角。

### 4.2 “我用堆取出来的 TopK 是无序的怎么办？”

两种常见回答：
- 取完再 sort 一次：$O(k\log k)$，当 $k$ 很小（10）几乎免费
- 或者维护最大堆并弹出（但最大堆大小是 k 时，插入仍 $O(\log k)$，收益不大）

### 4.3 “什么时候选堆？什么时候选 quickselect？”

一个很面试友好的判断：
- **k 很小**：最小堆 $O(n\log k)$ 最稳，代码短、稳定、可流式
- **只要 topK 但 k 可能接近 n/2**：`nth_element`/quickselect 更合适（期望线性）
- **还需要全排序**：直接 sort

---

## 5) 一句话总结（面试口径）

- **矩阵乘法优化**：row-major 下把循环改成 `i-p-j` 或先转置 `B`，让 `B` 的访问变成连续；更进一步用 blocking。  
- **TopK 优化**：从 `sort(O(n log n))` 升级到 **最小堆 $O(n\log k)$**；需要更强可讲 quickselect / `nth_element`（期望 $O(n)$）。

---

[← Back to Question Bank](./README.md)

