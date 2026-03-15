# Skill: xiaohongshu-vla-collector

> **用途**：自动化从小红书收集 VLA 相关复现经验、踩坑经验、个人观点
> **触发词**：小红书收集 / 社区声音 / 收集经验 / VLA经验搜集
> **前置条件**：用户已在 Chrome 中登录小红书

---

## 执行流程

### Phase 1: 初始化

```
1. 检查 Chrome 登录状态
   - 导航到 xiaohongshu.com
   - 验证是否已登录（检查用户头像元素）
   - 未登录 → 提示用户登录后重试

2. 加载去重库
   - 读取 KW_VLA/collected_urls.json
   - 如不存在则创建空库

3. 加载 VLA Expert Memory（用于质量筛选和关键词动态调整）
   - 读取 KW_VLA/VLA_EXPERT_MEMORY.md
```

### Phase 2: 搜索与提取

```
关键词矩阵（每次执行随机选择 3-5 组）：

模型复现类：
  "VLA 复现 经验" / "pi0 训练 踩坑" / "SmolVLA 微调" /
  "ACT 模型 机械臂" / "OpenVLA 部署" / "LeRobot 训练"

工程实践类：
  "VLA 推理 卡顿 优化" / "具身智能 真机 部署" /
  "机械臂 标定 踩坑" / "sim2real 失败" / "仿真环境 搭建 坑"

硬件经验类：
  "机械臂 选型 对比" / "ALOHA 组装" /
  "数据采集 遥操作" / "相机 手眼标定"

方向判断类：
  "VLA 泛化 失败" / "具身智能 技术路线" /
  "WAM VLA 对比" / "具身智能 创业 反思"

新兴话题类：
  "VLA RL 强化学习" / "触觉 力觉 具身" /
  "world model 机器人" / "具身智能 量化 部署"

执行逻辑：
  for 每组关键词:
    1. 搜索小红书
    2. 提取前 5 条结果（标题+链接）
    3. 跳过已在去重库中的 URL
    4. 对新帖子导航提取（标题、正文、作者、日期、评论）
    5. 间隔 3-5 秒
    6. 最多处理 15 条新帖
```

### Phase 3: 质量评估

```
对每条新帖子打分：

+3 分：包含具体数据（epoch数、成功率、GPU型号、训练时间）
+3 分：描述失败模式和根因分析
+2 分：作者一手经验（"我试过"而非"我觉得"）
+2 分：评论区有补充洞察
+2 分：覆盖当前知识盲区（FAST tokenization / Co-training / 视觉编码器对比 / 量化部署 / World Model / 多机器人协同）
-10 分：求职帖
-5 分：纯转发/广告

≥4 分 → 入库（完整记录）
<4 分 → 记录标题但不详细分析
```

### Phase 4: 更新输出

```
1. 追加新帖子到报告文件
2. 更新 collected_urls.json
3. 检查是否需要更新信念网络
4. 输出本次收集摘要
```

---

## DOM 提取脚本库

### 搜索结果页提取

```javascript
const items = document.querySelectorAll('.note-item, section.note-item');
const results = [];
items.forEach((item, i) => {
  const titleEl = item.querySelector('.title span, .note-text');
  const linkEl = item.querySelector('a[href*="/explore/"]');
  const title = titleEl ? titleEl.textContent.trim() : '';
  const href = linkEl ? linkEl.href : '';
  if (title || href) results.push({i, title: title.slice(0,80), href});
});
JSON.stringify(results.slice(0, 10));
```

### 帖子详情页提取

```javascript
const t = document.querySelector('#detail-title')?.textContent?.trim() || '';
const d = document.querySelector('#detail-desc')?.textContent?.trim() || '';
const a = document.querySelector('.username')?.textContent?.trim() || '';
const dt = document.querySelector('.date')?.textContent?.trim() || '';
const comments = [];
document.querySelectorAll('.comment-item, .parent-comment, [class*="comment-inner"]').forEach((c, i) => {
  if (i >= 15) return;
  const user = c.querySelector('.name, .user-name, .author-name')?.textContent?.trim() || '';
  const text = c.querySelector('.content, .note-text, .comment-text')?.textContent?.trim() || '';
  if (text && text.length > 5) comments.push({user, text: text.slice(0, 250)});
});
const seen = new Set();
const uC = comments.filter(c => { const k = c.user+c.text.slice(0,40); if(seen.has(k)) return false; seen.add(k); return true; });
JSON.stringify({t, a, dt, d: d.slice(0,800), comments: uC.slice(0,10)});
```

### 登录状态检查

```javascript
const avatar = document.querySelector('.user-avatar, .reds-avatar, [class*="avatar"]');
const loginBtn = document.querySelector('[class*="login"], .login-btn');
JSON.stringify({
  loggedIn: !!avatar && !loginBtn,
  hasAvatar: !!avatar,
  hasLoginBtn: !!loginBtn
});
```

---

## 排除规则

以下帖子应被自动排除：
- 标题含 "求职" / "秋招" / "春招" / "面经" / "offer" / "实习招聘"
- 标题含 "课程" / "培训" / "报名" / "优惠"
- 内容以推广/广告为主（通过关键词检测）
- 纯转发（无原创内容）

---

## 已知限制

1. **图片帖**：核心内容在图片中无法自动提取，只能记录标题+评论
2. **评论分页**：只能获取首屏评论（通常10-15条），热门帖评论可能有100+条
3. **反爬限制**：频繁操作可能触发验证码，需要控制节奏
4. **DOM 变更风险**：小红书可能更新页面结构，需要定期验证 selector
5. **登录会话过期**：长时间不操作可能需要重新登录
