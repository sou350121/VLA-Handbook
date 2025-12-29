# AI Coding 智能体设计深度解析 (AI Coding Agent Design)

> **核心定位**：通过分析 Gemini-CLI、Claude Code 等顶尖工具，解构 AI 编程智能体的“大脑”与“肢体”设计。
> **关键关键词**：ReAct 框架、MCP 协议、SubAgent 隔离、规约驱动开发 (Spec-driven)。

---

## 1. 用户提示词预处理 (Prompt Preprocessing)

高效的智能体首先需要理解用户的“意图”并精准获取“上下文”。

### 1.1 `@路径`：上下文的即时注入
*   **原理**：在发送给模型前，智能体拦截 `@路径` 或 `@目录` 字符。
*   **价值**：
    *   **减少会话轮次**：一次性带入必要代码，避免模型反复调用 `read_file`。
    *   **精准度**：用户显式指定上下文，避免 RAG 检索的噪点。

### 1.2 斜线命令 (Slash Commands)
Gemini-CLI 将命令分为三类：
1.  **内置命令**：如 `/clear`（清空上下文）、`/init`（项目初始化分析）。
2.  **MCP 扩展命令**：MCP Server 提供的 `prompts` 会转化为带有 `[MCP]` 标识的命令。
3.  **本地/扩展包命令**：通过 `.toml` 文件定义的预置提示词。

---

## 2. 工具调用与 MCP 协议 (Tool Calling & MCP)

智能体通过工具与物理世界（文件系统、终端）交互。

### 2.1 MCP (Model Context Protocol) 的三种能力
1.  **Tools (工具)**：大模型决定调用，智能体执行。
2.  **Prompts (提示词)**：作为扩展命令（Slash Command）存在。
3.  **Resources (资源)**：作为只读上下文。

### 2.2 传统 MCP 的局限性：Token 爆炸
*   **问题**：当连接数十个 MCP Server 时，广播的数百个工具 Schema 会迅速消耗上下文空间（Context Window），导致成本上升和模型注意力分散。
*   **解决方案 (Claude Code 模式)**：
    *   **Skills 懒加载**：初始仅加载名称和描述（<1k），确认相关后再二次加载完整代码。
    *   **Code Execution with MCP**：大模型通过编写代码来调用 MCP，而非直接传递所有 Schema。

---

## 3. 子智能体架构 (SubAgent & Context Isolation)

基于 **“高内聚、低耦合”** 原则，处理复杂软件工程任务。

### 3.1 隔离的意义
*   **上下文空间限制**：主智能体处理主流程，子智能体（如 `CodebaseInvestigator`）在独立的上下文空间运行。
*   **权限管控**：例如，子智能体通常被设置为 **“只读”**，仅负责探索代码库（`ls`, `read_file`, `grep`），防止意外修改。

### 3.2 CodebaseInvestigatorAgent 流程
1.  **接收目标**：主智能体分配一个调查任务（如“分析认证逻辑的漏洞”）。
2.  **ReAct 循环**：子智能体自主进行多轮 `grep` 与文件读取。
3.  **提交报告**：完成后调用 `complete_task` 返回结构化 JSON 报告给主智能体。

---

## 4. 架构设计：ReAct 框架执行链路

### 4.1 主流程 ReAct 闭环
```mermaid
graph TD
    User[用户输入] --> Pre[提示词预处理/@路径/命令解析]
    Pre --> Reason[Reasoning: 模型预测动作/工具调用]
    Reason --> Action[Acting: 智能体执行工具/Shell/文件修改]
    Action --> Observe[Observing: 收集执行结果与错误]
    Observe --> Update[Updating: 将结果存入上下文历史]
    Update --> Reason
    Reason --> Final[Final Response: 任务完成输出结果]
```

### 4.2 记忆压缩 (Memory Compression)
当 Token 达到阈值（通常为模型的 20%）时触发：
1.  **分割点查找**：保留最近 30% 的对话。
2.  **摘要生成**：利用模型将前 70% 的历史压缩为固定的 XML 结构（`<state_snapshot>`）。
3.  **快照内容**：包含 `overall_goal`、`key_knowledge`、`file_system_state` 和 `current_plan`。

---

## 5. 规约驱动开发 (Spec-driven Development)

规约（Spec）是 AI 时代的核心资产，它比代码本身更具“主权”。

### 5.1 核心流程 (以 OpenSpec 为例)
1.  **Proposal 阶段**：运行 `/openspec:proposal`。模型分析需求，生成 `proposal.md` 和 `tasks.md`。
2.  **Apply 阶段**：运行 `/openspec:apply`。模型严格按照 `tasks.md` 的步骤逐一修改代码并验证。
3.  **Archive 阶段**：开发完成后归档，保持 `specs/` 目录为项目唯一的“真实真相（Source of Truth）”。

### 5.2 为什么有效？
*   **消除模糊性**：在编码前强制模型思考架构、破坏性变更和安全风险。
*   **上下文最小化**：模型只需关注当前的 Spec 增量，而非整个代码库的乱序指令。

---

## 6. 独立思考与批判性疑问 (Critical Thinking)

### 疑问一：记忆压缩的“信息熵”损失
虽然 XML 快照保留了核心目标，但细微的调试信息和失败尝试在压缩中会被抹除。这是否会导致模型在后续步骤中重复之前犯过的错误？

### 疑问二：代码执行 MCP vs 工具广播
如果采用大模型编写代码来调用 MCP 的方案，模型本身是否具备足够的安全性意识来防止生成的调用代码产生副作用（如递归删除文件）？

### 疑问三：规约与代码的“同步偏差”
当开发者手动修改了代码但没有更新 Spec 时，规约驱动开发模式会迅速失效。如何通过 Hook 机制实现 Spec 与代码的强关联校验？

---
[← 返回系统设计索引](./README.md)

