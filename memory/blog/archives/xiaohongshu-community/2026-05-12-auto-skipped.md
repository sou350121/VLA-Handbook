# 小红书 VLA 社区采集 — 2026-05-12 自动运行（已跳过）

> **结果**：SKIPPED（未采集任何新帖）
> **原因**：小红书会话未登录（/explore 和 /search_result 均渲染登录浮层，不返回笔记数据）
> **采集器**：xiaohongshu-vla-collector (Scheduled Task)
> **浏览器**：Browser 1 (Windows) — deviceId `347dcd68-ecf4-4e09-b3cd-73643fedd766`（沿用 2026-05-04 的同一台）
> **上一次成功运行**：2026-05-04（[2026-05-04-auto.md](./2026-05-04-auto.md)，采 12 篇详细 + 7 篇标题）

---

## 发生了什么

Phase 1（初始化）阶段表面通过：导航到 `https://www.xiaohongshu.com` 后，`[class*="avatar"]` 选择器命中元素、未发现 `.login-btn`，初步判定登录态有效。

但 Phase 2 真正打开 `/search_result?keyword=OpenVLA 部署` 之后，DOM 实际渲染的是登录浮层（"登录后查看搜索结果"），且 `.note-item` / `[class*="card"]` 均为 0 个。回退验证 `/explore`，同样返回"登录后推荐更懂你的笔记"——确认会话已掉线，初次的 avatar 命中是登录浮层内部元素造成的误判。

Phase 2 抓到的现象（用于将来调试登录态检测）：
- 已登录态：homepage `.side-bar-component` + `.user-name` + 真正的 `.feeds-page` 笔记卡片。
- 未登录态：homepage 显示登录浮层，浮层内含 `[class*="avatar"]` 占位元素（**导致 SKILL.md 当前的登录检查脚本误判**），且 `/search_result` 直接渲染"登录后查看搜索结果"占位 + 10 个公开预览卡（无 `/explore/{id}` 链接）。

按照 scheduled-task 协议（"when in doubt, producing a report of what you found is the correct output"）和 SKILL.md Phase 1（"未登录 → 提示用户登录后重试"）本次不做任何采集写操作：
- `collected_urls.json` **仅追加一条 skipped 运行记录**（不新增任何 URL）
- `deployment/community_field_notes_xiaohongshu.md` **未追加新内容**
- `2026-05-12-auto-skipped.md` 即本文件

## 用户需要做什么

1. **在 Browser 1 重新登录小红书**：打开 Chrome → `xiaohongshu.com` → 用手机扫码或微信登录 → 刷新首页确认 `.feeds-page` 出现真实笔记卡。
2. **手动重触发**：登录后对 Claude 说 "跑一下小红书采集"，会立刻再跑一轮；或等到下一次调度。
3. **加固登录检查（建议）**：当前 SKILL.md 用 `[class*="avatar"]` 单一选择器，登录浮层会让它误判。下一版可改为多信号 AND：`.side-bar-component` 存在 + `/explore` 页面非登录浮层 + `document.body.innerText` 不含 "登录后" 关键词。

## 下一次运行建议

距上次成功采集 8 天（2026-05-04 → 今天 05-12），节奏正常。建议登录后立刻手动触发，使用本轮**未消耗的关键词组**：

- 模型复现类：**OpenVLA 部署**（本轮已尝试，但未抓到）
- 工程实践类：**sim2real 失败**
- 硬件经验类：**ALOHA 组装**
- 方向判断类：**具身智能 技术路线**
- 新兴话题类：**world model 机器人**

---

*自动生成 — 本文件为空运行日志，不含社区数据，不影响 belief graph。*
