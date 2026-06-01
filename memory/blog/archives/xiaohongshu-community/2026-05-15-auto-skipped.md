# 小红书 VLA 社区采集 — 2026-05-15（已被取代）

> **状态**：SUPERSEDED —— 本文件最初记录的是当日两次自动调度因未登录而跳过。
> 之后用户在 Browser「小紅書」登录小红书，手动重新触发，采集**成功完成**。
> **请查看实际采集结果** → [2026-05-15-auto.md](./2026-05-15-auto.md)（13 篇详细 + 2 篇标题，共 15 篇新帖）

---

## 经过

- 当日 scheduled task 运行时，小红书会话未登录（`__INITIAL_STATE__.user.loggedIn=false`，搜索页返回 0 笔记），与 2026-05-12 同一故障 → 跳过。
- 用户登录后说「登录完成」，手动重跑：登录态复核通过（`loggedIn=true`、有 nickname、无登录浮层）。
- 5 组关键词（OpenVLA 部署 / sim2real 失败 / ALOHA 组装 / 具身智能 技术路线 / world model 机器人）共命中 91 条新帖，详情提取 15 篇。
- 结果写入 `2026-05-15-auto.md`，`collected_urls.json` 追加 15 条 URL（count 92→107）+ 一条 manual-trigger 运行记录。

*本文件保留作运行日志；实际社区数据见 2026-05-15-auto.md。*
