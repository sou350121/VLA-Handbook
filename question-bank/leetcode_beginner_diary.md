# LeetCode 小白训练日记 🚀

> **写给完全零基础的你**：用大白话讲解50道LeetCode热门题，面试突击必备！

## 📖 使用说明

- **风格**：超级口语化，像聊天一样轻松
- **目标**：零基础也能看懂，快速掌握面试高频题
- **建议**：按顺序刷，每道题都动手写一遍代码
- **进阶**：刷完后可以挑战 [VLA工程实战题](./code_questions.md)

---

## 🗺️ 学习路线图

### 第一部分：数组和字符串（15题）⭐ 必须掌握
1. [两数之和](#第1题两数之和two-sum) - 哈希表入门
2. [买卖股票最佳时机](#第2题买卖股票最佳时机best-time-to-buy-and-sell-stock)
3. [存在重复元素](#第3题存在重复元素contains-duplicate)
4. [除自身以外数组的乘积](#第4题除自身以外数组的乘积product-of-array-except-self)
5. [最大子数组和](#第5题最大子数组和maximum-subarray)
6. [合并区间](#第6题合并区间merge-intervals)
7. [三数之和](#第7题三数之和3sum)
8. [盛最多水的容器](#第8题盛最多水的容器container-with-most-water)
9. [无重复字符的最长子串](#第9题无重复字符的最长子串longest-substring-without-repeating-characters)
10. [验证回文串](#第10题验证回文串valid-palindrome)
11. [字母异位词分组](#第11题字母异位词分组group-anagrams)
12. [最长连续序列](#第12题最长连续序列longest-consecutive-sequence)
13. [轮转数组](#第13题轮转数组rotate-array)
14. [查找首尾位置](#第14题查找首尾位置find-first-and-last-position)
15. [搜索旋转排序数组](#第15题搜索旋转排序数组search-in-rotated-sorted-array)

### 第二部分：链表（8题）⭐ 指针操作
16. [反转链表](#第16题反转链表reverse-linked-list)
17. [合并两个有序链表](#第17题合并两个有序链表merge-two-sorted-lists)
18. [环形链表](#第18题环形链表linked-list-cycle)
19. [删除倒数第N个节点](#第19题删除倒数第n个节点remove-nth-node-from-end)
20. [两数相加](#第20题两数相加add-two-numbers)
21. [复制带随机指针的链表](#第21题复制带随机指针的链表copy-list-with-random-pointer)
22. [LRU缓存](#第22题lru缓存lru-cache)
23. [相交链表](#第23题相交链表intersection-of-two-linked-lists)

### 第三部分：栈和队列（5题）⭐ 数据结构基础
24. [有效的括号](#第24题有效的括号valid-parentheses)
25. [最小栈](#第25题最小栈min-stack)
26. [每日温度](#第26题每日温度daily-temperatures)
27. [逆波兰表达式求值](#第27题逆波兰表达式求值evaluate-reverse-polish-notation)
28. [用栈实现队列](#第28题用栈实现队列implement-queue-using-stacks)

### 第四部分：树（12题）⭐ 递归思维
29. [翻转二叉树](#第29题翻转二叉树invert-binary-tree)
30. [二叉树的最大深度](#第30题二叉树的最大深度maximum-depth-of-binary-tree)
31. [验证二叉搜索树](#第31题验证二叉搜索树validate-binary-search-tree)
32. [最近公共祖先](#第32题最近公共祖先lowest-common-ancestor)
33. [层序遍历](#第33题层序遍历binary-tree-level-order-traversal)
34. [序列化和反序列化](#第34题序列化和反序列化serialize-and-deserialize-binary-tree)
35. [从前序和中序构造二叉树](#第35题从前序和中序构造二叉树construct-binary-tree)
36. [路径总和](#第36题路径总和path-sum)
37. [二叉树的右视图](#第37题二叉树的右视图binary-tree-right-side-view)
38. [二叉搜索树第K小元素](#第38题二叉搜索树第k小元素kth-smallest-element-in-bst)
39. [二叉树的直径](#第39题二叉树的直径diameter-of-binary-tree)
40. [另一棵树的子树](#第40题另一棵树的子树subtree-of-another-tree)

### 第五部分：动态规划（7题）⭐ 找规律
41. [爬楼梯](#第41题爬楼梯climbing-stairs)
42. [零钱兑换](#第42题零钱兑换coin-change)
43. [最长递增子序列](#第43题最长递增子序列longest-increasing-subsequence)
44. [打家劫舍](#第44题打家劫舍house-robber)
45. [不同路径](#第45题不同路径unique-paths)
46. [单词拆分](#第46题单词拆分word-break)
47. [解码方法](#第47题解码方法decode-ways)

### 第六部分：图和搜索（3题）⭐ 遍历技巧
48. [岛屿数量](#第48题岛屿数量number-of-islands)
49. [克隆图](#第49题克隆图clone-graph)
50. [课程表](#第50题课程表course-schedule)

---

## 📚 题目详解

---

## 第1题：两数之和（Two Sum）

**难度**：简单  
**标签**：哈希表

好，小白模式开启。只讲**人话**，不装专业。

---

### 这题在干嘛？

给你一堆数字（一个数组）  
再给你一个目标数 `target`  
👉 **从里面找两个数，加起来刚好等于 target**  
👉 **返回这两个数在数组里的位置（下标）**

---

### 先解释所有专业名词（一个一个来）

#### 1️⃣ 数组（array / list）

一排数字，按顺序放好。

例子：

```text
nums = [2, 7, 11, 15]
```

位置编号（下标）从 **0 开始**：

```text
值:   2   7   11   15
下标: 0   1    2    3
```

---

#### 2️⃣ 下标（index）

数字在数组里的**位置编号**  
不是数字本身！

例：

```text
nums[1] = 7
```

👉 **1 是下标，7 是值**

---

#### 3️⃣ target（目标值）

题目说：

> 你要找的两个数，加起来 = target

比如：

```text
target = 9
```

那我们就在找：

```text
谁 + 谁 = 9
```

---

#### 4️⃣ 暴力解法（不推荐）

意思是：

> 一个一个试，全部组合都试一遍

像这样：

```text
2+7?
2+11?
2+15?
7+11?
...
```

❌ 很慢  
❌ 像没脑子一样乱试  
❌ 面试官不喜欢

---

#### 5️⃣ 哈希表 / 字典（Hash Map / dict）【重点】

你可以把它当成一个：

> **超快的查找小本子**

Python 里叫 `dict`

例子：

```python
seen = {2: 0, 7: 1}
```

意思是：

```text
我见过 2，它在位置 0
我见过 7，它在位置 1
```

👉 **用"值"去查"位置"**

---

#### 6️⃣ 时间复杂度（你现在只要这样理解）

* 慢：一个一个试（O(n²)）
* 快：看一眼就知道（O(n)）

这题要求你用 **快的方法**

---

### 正确思路（白话版）

我边走边记住我看过的数字。

每看到一个新数字，我就问自己一句话：

> **"我要凑到 target，还差多少？"**

如果这个"差的数"：

* **以前见过** → 找到了
* **没见过** → 先记下来

---

### 用生活例子讲

target = 10  
你现在手里拿着 **3**

你会想：

> "我需要一个 7 才能凑到 10"

如果你以前已经见过 7  
👉 成功配对！

---

### 代码一句一句翻译

```python
seen = {}
```

👉 准备一个小本子，记"见过的数字"

---

```python
for i in range(len(nums)):
```

👉 从第一个数开始，一个一个看

---

```python
x = nums[i]
```

👉 当前看到的数字

---

```python
need = target - x
```

👉 我还差多少才能凑到 target

---

```python
if need in seen:
    return [seen[need], i]
```

👉 如果之前见过那个"需要的数"  
👉 返回它的位置 + 现在这个位置

---

```python
seen[x] = i
```

👉 没找到，就把当前数字记下来

---

### 完整代码（不用改，直接交）

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        seen = {}
        for i in range(len(nums)):
            x = nums[i]
            need = target - x
            if need in seen:
                return [seen[need], i]
            seen[x] = i
```

---

### 你现在真正需要记住的只有 3 句话

1. **下标不是数字本身**
2. **字典 = 快速查找**
3. **target − 当前数 = 我需要的数**

---

### 🎤 面试官可能问的

**Q**: 为什么不能先把所有数字都存进字典？  
**A**: 因为会匹配到自己。比如 target=6, nums=[3,...]，3 会错误地跟自己配对。

**Q**: 如果数组是排序的呢？  
**A**: 可以用双指针（左右夹逼），空间更省，但要先排序。

**Q**: 时间复杂度是多少？  
**A**: O(n) - 只遍历一遍，每次查字典是 O(1)。

---

## 第2题：买卖股票最佳时机（Best Time to Buy and Sell Stock）

### 😰 题目长啥样
_【待补充】给你每天的股票价格数组，只能买卖一次，求最大利润_

### 💡 怎么想的
_【待补充】记录最低买入价，每天算如果今天卖能赚多少_

### 📝 代码怎么写
```python
# 【待补充】
def maxProfit(prices):
    pass
```

### 🤔 为啥这么写
_【待补充】_

### ⏱️ 复杂度
_【待补充】_

### 🎤 面试官会问啥
_【待补充】_

---

## 第3题：存在重复元素（Contains Duplicate）

### 😰 题目长啥样
_【待补充】数组里有没有重复的数字_

### 💡 怎么想的
_【待补充】用 set 记录见过的，或者直接比较长度_

### 📝 代码怎么写
```python
# 【待补充】
def containsDuplicate(nums):
    pass
```

---

## 第4题：除自身以外数组的乘积（Product of Array Except Self）

_【待补充】_

---

## 第5题：最大子数组和（Maximum Subarray）

_【待补充】_

---

## 第6题：合并区间（Merge Intervals）

_【待补充】_

---

## 第7题：三数之和（3Sum）

_【待补充】_

---

## 第8题：盛最多水的容器（Container With Most Water）

**难度**：中等  
**标签**：双指针  
**LeetCode**: 11

好，小白模式继续。  
**这题是"双指针"的经典题，必须会。**我只讲人话。

---

### 这题在干嘛？（一句话）

从一排"竖线"里选两条，  
用**短的那条当高度**，  
用**两条之间的距离当宽度**，  
算能装多少水，找最大的。

---

### 先把名词讲清楚

#### 1️⃣ height 数组

```text
height = [1,8,6,2,5,4,8,3,7]
```

意思是：

* 第 0 根线高度是 1
* 第 1 根线高度是 8
* …

---

#### 2️⃣ 容器装水怎么算？

公式只有一个：

```
水量 = 宽度 × 高度
```

* 宽度 = 右下标 − 左下标
* 高度 = **两根线里较短的那一根**

⚠️ 水会从矮的那边漏掉

---

### 为什么不能暴力？

暴力：两层循环，试所有组合

* 时间复杂度 O(n²)
* n 最大 10⁵ → 超时

---

### 核心思路（一句话，背下来）

> **左右各放一根指针，谁矮就动谁**

---

### 双指针是啥？（人话）

* 左指针 `l`：从最左边开始
* 右指针 `r`：从最右边开始

```text
l → | | | | | | | ← r
```

---

### 为什么"动矮的那根"？

这是这题的灵魂。

假设：

```text
左 = 3，右 = 10
```

高度被 **3 限制**  
你动右边（10）：

* 宽度变小
* 高度还是 ≤ 3
* 水一定更少 ❌

只有动左边，**才有可能遇到更高的线** ✅

---

### 算法流程（一步一步）

1. l = 0，r = n-1
2. 算当前水量
3. 更新最大值
4. **哪边矮，动哪边**
5. 重复直到 l >= r

---

### 用例子走一遍（关键一步）

初始：

```text
l=0 (1), r=8 (7)
面积 = min(1,7) × 8 = 8
```

左边矮 → l++

---

```text
l=1 (8), r=8 (7)
面积 = min(8,7) × 7 = 49 ✅ 最大
```

右边矮 → r--

---

继续，但再也超过不了 49。

---

### 完整代码（直接交）

```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        l, r = 0, len(height) - 1
        ans = 0

        while l < r:
            h = min(height[l], height[r])
            w = r - l
            ans = max(ans, h * w)

            if height[l] < height[r]:
                l += 1
            else:
                r -= 1

        return ans
```

---

### 代码逐行翻译成人话

```python
l, r = 0, len(height) - 1
```

👉 左右各放一根线

---

```python
h = min(height[l], height[r])
```

👉 水位被矮的限制

---

```python
w = r - l
```

👉 宽度

---

```python
if height[l] < height[r]:
    l += 1
```

👉 矮的那边往里走

---

### 小白最容易犯的 3 个错

❌ 以为动高的更好  
❌ 高度用 max  
❌ 忘记宽度是下标差

---

### 你现在掌握了什么？

✅ 双指针思想  
✅ 贪心为什么正确  
✅ O(n) 解法

---

### 🎤 面试官可能问的

**Q**: 为什么不是动高的那根？  
**A**: 因为高度被矮的限制，动高的只会让宽度变小，面积一定不会变大。

**Q**: 时间复杂度是多少？  
**A**: O(n) - 每个元素最多访问一次，左右指针各走一遍。

**Q**: 这个贪心策略为什么是正确的？  
**A**: 因为我们不会错过任何可能的最大值。每次移动矮的指针，才有机会找到更高的线来增加面积。

**Q**: 如果相同高度怎么办？  
**A**: 动哪个都行，代码里选择了动右边（else 分支）。

---

## 第9题：无重复字符的最长子串（Longest Substring Without Repeating Characters）

_【待补充】_

---

## 第10题：验证回文串（Valid Palindrome）

_【待补充】_

---

## 第11题：字母异位词分组（Group Anagrams）

**难度**：中等  
**标签**：字符串 + 哈希表

好，小白模式继续。**这题是"字符串 + 哈希表"的核心题**。  
我按顺序来：**人话 → 名词 → 思路 → 代码 → 常见坑**。

---

### 这题在干嘛？（一句话）

把**由同样字母组成、只是顺序不同**的单词，分到同一组里。

---

### 先把名词讲清楚（不跳步）

#### 1️⃣ 字符串（string）

一串字符，比如：

```text
"eat"  "tea"  "tan"
```

---

#### 2️⃣ 字母异位词（Anagram）

**字母一样，顺序不一样**

例子：

```text
"eat"  "tea"  "ate"
```

👉 都是 `a e t`  
👉 所以是一组

不是异位词：

```text
"bat" vs "tab" ✅
"bat" vs "bad" ❌
```

---

#### 3️⃣ 分组（group）

意思是：

```text
[
  ["ate","eat","tea"],
  ["tan","nat"],
  ["bat"]
]
```

---

#### 4️⃣ 排序（sort）

把字母按顺序排好：

```text
"eat" → "aet"
"tea" → "aet"
"ate" → "aet"
```

👉 **排序后一样 = 同一组**

---

#### 5️⃣ 哈希表 / 字典（dict）

还是那个**超快小本子**

我们用它来做：

```text
key   →  一组单词
```

---

### 核心思路（白话版）

#### 关键一句话（背下来）

> **把每个单词排序，排序结果一样的，放同一组**

---

### 用例子一步一步走

输入：

```text
["eat","tea","tan","ate","nat","bat"]
```

处理过程：

| 单词  | 排序后 | 放哪    |
| --- | --- | ----- |
| eat | aet | aet 组 |
| tea | aet | aet 组 |
| tan | ant | ant 组 |
| ate | aet | aet 组 |
| nat | ant | ant 组 |
| bat | abt | abt 组 |

---

### 字典长什么样？

```python
{
  "aet": ["eat","tea","ate"],
  "ant": ["tan","nat"],
  "abt": ["bat"]
}
```

最后只要：

```python
dict.values()
```

---

### 完整代码（直接交）

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        groups = {}

        for s in strs:
            key = ''.join(sorted(s))
            if key not in groups:
                groups[key] = []
            groups[key].append(s)

        return list(groups.values())
```

---

### 代码逐行翻译成人话

```python
groups = {}
```

👉 创建分组本子

---

```python
for s in strs:
```

👉 一个一个单词看

---

```python
key = ''.join(sorted(s))
```

👉 把单词字母排序，变成"身份证"

---

```python
groups[key].append(s)
```

👉 身份证一样的，放同一组

---

```python
return list(groups.values())
```

👉 只要分好的组，不要 key

---

### 为什么这样一定对？

因为：

* 字母异位词 → 排序后一定一样
* 非异位词 → 排序后一定不同

这是**一一对应**，不会错

---

### 小白最容易踩的 3 个坑

❌ 用 list 当 key（list 不能当字典 key）  
❌ 忘了 `''.join()`  
❌ 每次新建 list，没 append

---

### 你现在掌握的能力

✅ 字符串处理  
✅ 哈希表分组  
✅ 排序作为"特征"

---

### 🎤 面试官可能问的

**Q**: 除了排序，还有别的方法吗？  
**A**: 可以用字母计数，比如 "eat" → `{a:1, e:1, t:1}`，但实现更复杂。

**Q**: 时间复杂度是多少？  
**A**: O(n × k log k)，n 是单词数，k 是最长单词长度（排序的代价）。

**Q**: 如果单词很长怎么优化？  
**A**: 用字母计数代替排序，可以降到 O(n × k)。

---

## 第12题：最长连续序列（Longest Consecutive Sequence）

_【待补充】_

---

## 第13题：移动零（Move Zeroes）

**难度**：简单  
**标签**：数组 + 双指针  
**LeetCode**: 283

好，小白模式继续。**这题是"数组 + 双指针"的入门必会题**。我只讲人话。

---

### 这题在干嘛？（一句话）

把数组里的 **0 全部挪到最后**，  
**非 0 的顺序不能变**，  
而且 **不能新建数组（原地改）**。

---

### 先解释名词（不跳步）

#### 1️⃣ 数组（nums）

一排数字，有顺序、有位置（下标从 0 开始）。

例子：

```text
[0, 1, 0, 3, 12]
```

---

#### 2️⃣ 原地操作（in-place）

意思是：

> **只能在原数组上改，不能用新数组**

❌ 不允许：

```python
new = []
```

---

#### 3️⃣ 保持相对顺序

非 0 的数字，**前后顺序不能乱**

```text
1 在 3 前面，3 在 12 前面
```

---

### 核心思路（一句话，背下来）

> **把所有非 0 的数，按顺序往前塞**

---

### 用"搬箱子"来理解

想象数组是一排箱子：

```text
[0, 1, 0, 3, 12]
```

你有一个"空位指针"，专门指向**下一个该放非 0 的位置**。

---

### 双指针是啥？（人话）

不是两个数组，是**两个位置指针**：

* `i`：扫描指针（一路往右看）
* `pos`：放非 0 的位置

---

### 具体流程（一步一步）

#### 初始

```text
nums = [0, 1, 0, 3, 12]
pos = 0
```

---

#### i = 0

```text
nums[0] = 0 → 跳过
```

---

#### i = 1

```text
nums[1] = 1 → 非 0
放到 nums[pos]
```

变成：

```text
[1, 0, 0, 3, 12]
pos = 1
```

---

#### i = 3

```text
nums[3] = 3 → 非 0
```

变成：

```text
[1, 3, 0, 0, 12]
pos = 2
```

---

#### i = 4

```text
nums[4] = 12 → 非 0
```

结果：

```text
[1, 3, 12, 0, 0]
```

---

### 完整代码（直接交，别改）

```python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        pos = 0

        for i in range(len(nums)):
            if nums[i] != 0:
                nums[pos], nums[i] = nums[i], nums[pos]
                pos += 1
```

---

### 代码逐行翻译成人话

```python
pos = 0
```

👉 下一个非 0 应该放的位置

---

```python
for i in range(len(nums)):
```

👉 从头到尾看每个数

---

```python
if nums[i] != 0:
```

👉 只处理非 0

---

```python
nums[pos], nums[i] = nums[i], nums[pos]
```

👉 把非 0 放到前面该放的位置

---

```python
pos += 1
```

👉 下一个空位往右移

---

### 为什么这样不会乱顺序？

因为：

* `i` 是从左到右
* 非 0 是按出现顺序一个一个放进去的

---

### 小白最容易犯的 3 个错

❌ 新建数组（违规）  
❌ 只统计 0 的个数，不移动元素  
❌ 乱 swap，破坏顺序

---

### 你现在学会了什么？

✅ 什么是原地操作  
✅ 什么是双指针  
✅ 如何稳定移动元素

---

### 🎤 面试官可能问的

**Q**: 为什么要用交换而不是覆盖？  
**A**: 交换保证不丢失任何元素，而且自动把 0 往后移。

**Q**: 时间复杂度是多少？  
**A**: O(n) - 只扫一遍数组。

**Q**: 空间复杂度呢？  
**A**: O(1) - 只用了两个指针变量。

**Q**: 如果有多种类型的元素要移动呢？  
**A**: 可以用相同的双指针思路，只是判断条件不同。

---

## 第14题：查找首尾位置（Find First and Last Position）

_【待补充】_

---

## 第15题：搜索旋转排序数组（Search in Rotated Sorted Array）

_【待补充】_

---

# 第二部分：链表 🔗

## 第16题：反转链表（Reverse Linked List）

_【待补充】_

---

## 第17题：合并两个有序链表（Merge Two Sorted Lists）

_【待补充】_

---

## 第18题：环形链表（Linked List Cycle）

_【待补充】_

---

## 第19题：删除倒数第N个节点（Remove Nth Node From End）

_【待补充】_

---

## 第20题：两数相加（Add Two Numbers）

_【待补充】_

---

## 第21题：复制带随机指针的链表（Copy List with Random Pointer）

_【待补充】_

---

## 第22题：LRU缓存（LRU Cache）

_【待补充】提示：可以参考 [code_questions.md Q1](./code_questions.md) 的实现，这里用更白话的方式讲解_

---

## 第23题：相交链表（Intersection of Two Linked Lists）

_【待补充】_

---

# 第三部分：栈和队列 📚

## 第24题：有效的括号（Valid Parentheses）

_【待补充】_

---

## 第25题：最小栈（Min Stack）

_【待补充】_

---

## 第26题：每日温度（Daily Temperatures）

_【待补充】_

---

## 第27题：逆波兰表达式求值（Evaluate Reverse Polish Notation）

_【待补充】_

---

## 第28题：用栈实现队列（Implement Queue using Stacks）

_【待补充】_

---

# 第四部分：树 🌲

## 第29题：翻转二叉树（Invert Binary Tree）

_【待补充】_

---

## 第30题：二叉树的最大深度（Maximum Depth of Binary Tree）

_【待补充】_

---

## 第31题：验证二叉搜索树（Validate Binary Search Tree）

_【待补充】_

---

## 第32题：最近公共祖先（Lowest Common Ancestor）

_【待补充】_

---

## 第33题：层序遍历（Binary Tree Level Order Traversal）

_【待补充】_

---

## 第34题：序列化和反序列化（Serialize and Deserialize Binary Tree）

_【待补充】_

---

## 第35题：从前序和中序构造二叉树（Construct Binary Tree）

_【待补充】_

---

## 第36题：路径总和（Path Sum）

_【待补充】_

---

## 第37题：二叉树的右视图（Binary Tree Right Side View）

_【待补充】_

---

## 第38题：二叉搜索树第K小元素（Kth Smallest Element in BST）

_【待补充】_

---

## 第39题：二叉树的直径（Diameter of Binary Tree）

_【待补充】_

---

## 第40题：另一棵树的子树（Subtree of Another Tree）

_【待补充】_

---

# 第五部分：动态规划 💰

## 第41题：爬楼梯（Climbing Stairs）

_【待补充】_

---

## 第42题：零钱兑换（Coin Change）

_【待补充】_

---

## 第43题：最长递增子序列（Longest Increasing Subsequence）

_【待补充】_

---

## 第44题：打家劫舍（House Robber）

_【待补充】_

---

## 第45题：不同路径（Unique Paths）

_【待补充】_

---

## 第46题：单词拆分（Word Break）

_【待补充】_

---

## 第47题：解码方法（Decode Ways）

_【待补充】_

---

# 第六部分：图和搜索 🗺️

## 第48题：岛屿数量（Number of Islands）

_【待补充】_

---

## 第49题：克隆图（Clone Graph）

_【待补充】_

---

## 第50题：课程表（Course Schedule）

_【待补充】_

---

## 🎯 刷题建议

1. **每天3-5题**，先易后难
2. **一定要自己动手写**，不要只看答案
3. **同类型题目连着刷**，培养题感
4. **遇到不会的先想5分钟**，实在不会再看解析
5. **隔天再刷一遍**，巩固记忆

## 🔗 进阶资源

- 完成本文后，挑战 [VLA工程实战题](./code_questions.md)
- 查看详细解答 [代码答案与解释](./code_answers.md)
- LeetCode 官方题库：https://leetcode.cn/

---

**祝你刷题顺利！加油！💪**
