# 📡 订阅 Pulsar 照見

> Pulsar 每天从 arXiv / GitHub / 社区源提纯 VLA + AI 信号。
> RSS 订阅让新内容**主动推送到你的阅读器** — 绕开算法推荐，零流失。

---

## 为什么订阅 RSS 而不是刷网站？

| 维度 | 刷网站 | RSS 订阅 |
|------|--------|----------|
| **控制权** | 你记得打开才看 | 新内容主动到达 |
| **算法** | 无 | 无（你自己的阅读器决定顺序） |
| **留存率** | 浏览器书签 > 90% 会遗忘 | 订阅者 1 年留存 > 80% |
| **整合** | 只能看 | 可连 Slack / Discord / Telegram / 邮件 |
| **成本** | 免费 | 免费 |

---

## 4 个主题频道（按需订阅）

一个 feed 一个主题，可以只订你关心的。

### 🧠 VLA 新文章

新加入 VLA-Handbook `theory/` 目录的深度解读文章，每天 3-5 篇。

- **URL**：`https://sou350121.github.io/pulsar-web/rss/vla-theory.xml`
- **适合谁**：关注 VLA 技术进展的研究者、工程师
- **更新节奏**：每天 10:00-17:00 多批次

### ⚡ VLA 每日信号

当日 ⚡🔧 级筛选论文 + SOTA 榜变动。**主动过滤 ❌📖**，只留 Pulsar 评级认为值得读的。

- **URL**：`https://sou350121.github.io/pulsar-web/rss/vla-daily.xml`
- **适合谁**：不想被低质量论文淹没、只要看值得花时间的
- **更新节奏**：每天 09:05-10:00

### 📘 AI 每日

AI Agent 生态每日精选（3 条）+ AI 深度解读文章。

- **URL**：`https://sou350121.github.io/pulsar-web/rss/ai-daily.xml`
- **适合谁**：做 AI 应用 / Agent 工程 / 工具链选型的
- **更新节奏**：每天 08:00-17:00

### 📚 周/双周深度报告

- **周报** = 前瞻侦察（意外发现 / 可证伪命题 / 观察清单）
- **双周报** = 回顾分析（趋势识别 / 交叉洞察 / 预测验证）

- **URL**：`https://sou350121.github.io/pulsar-web/rss/weekly.xml`
- **适合谁**：想要战略视角、长期思考而不是每天信息流的
- **更新节奏**：每周五 / 每两周

---

## 🎁 一键订阅（最快）

到 **[订阅页](https://sou350121.github.io/pulsar-web/subscribe)** — 每个 feed 都有：

- **[Feedly]** 按钮 — 点击即跳转到 Feedly 订阅对话框
- **[Inoreader]** 按钮 — 同上
- **[复制 URL]** 按钮 — 给其他阅读器用

**一键订阅全部**：下载 [OPML 文件](https://sou350121.github.io/pulsar-web/rss/opml.xml) 导入任何支持 OPML 的阅读器（基本都支持），一次性订阅 4 个 feed。

---

## 分阅读器教学

### 桌面浏览器

- **Feedly / Inoreader**：到 [订阅页](https://sou350121.github.io/pulsar-web/subscribe) 点对应按钮即可
- **其他**：复制 feed URL → 阅读器的「Add Subscription」→ 粘贴

**最懒方式**：Feedly / Inoreader 支持直接贴**网站首页** `https://sou350121.github.io/pulsar-web/`，阅读器会自动发现 4 个 feed（我们在 HTML `<head>` 里标注了 `<link rel="alternate">`）。

### macOS 原生

**NetNewsWire**（https://netnewswire.com/）免费开源，原生支持 OPML：
- 下载 → File → Import Feeds → 选 `opml.xml`
- 4 个 feed 瞬间出现

### iOS / Android

- **iOS**：Reeder 5 / NetNewsWire / Feedly app
- **Android**：FeedMe / Readably / Feedly app
- 都支持 OPML 导入

### 命令行 / 自动化

```bash
# 用 curl 测试 feed
curl https://sou350121.github.io/pulsar-web/rss/vla-theory.xml | head -50

# Zapier / n8n 可把 feed 接到 Slack / Discord / Telegram
# rss2email 可转邮件日报
# Buttondown / Feedburner 做广播邮件
```

---

## 内容许可 · License

所有订阅内容以 **[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)** 授权：

- ✅ **自由转载**
- ✅ **AI 训练可用**
- ✅ **商用可用**
- 🔗 **只需署名**：`sou350121 · Pulsar 照見 · https://sou350121.github.io/pulsar-web/`

每个 feed XML 都自带 `<copyright>` 标记，聚合器会自动保留。

---

## FAQ

### Q: 为什么 feed 不放全文？

**A**: 故意的。标题+摘要+链接 → 你点进去看全文 → 我们能知道哪些文章受欢迎，帮助 Pulsar 优化评级。这是给创作者的合理回报，也让 Pulsar 能持续运营。

### Q: 更新频率会变吗？

**A**: Pulsar 管道每日 08:00-17:00 运行，站点每日自动重建。RSS 跟着一起更新。如果某天 pipeline 出问题，feed 会显示「feed temporarily unavailable · being restored」占位信息让你知道。

### Q: 未来会有英文版 feed 吗？

**A**: 如果订阅者里海外研究者比例增高，会加。现在可在 Issue 投票：https://github.com/sou350121/VLA-Handbook/issues

### Q: 可以只订某个子主题吗？（例如只要触觉/世界模型）

**A**: 暂时不行，4 个频道已经按内容类型最佳切分。如果有强烈需求可在 Issue 提出。

### Q: Feedly 免费版够用吗？

**A**: 够。4 个 feed 完全在免费限额内（免费版支持 100 个 feed）。

### Q: RSS 会消失吗？

**A**: 只要 pulsar-web 站点还在就会在。`/rss/*` 路径在 CI smoke test 保护下，pipeline 架构变动不会让 feed 静默失效（会直接 CI fail）。

### Q: 链接里的 `?utm_source=rss&utm_medium=feed&...` 是什么？

**A**: 是 **UTM 追踪参数**。目的是未来如果接入 Umami/Plausible 等轻量分析，可以区分「读者从 RSS 过来」vs「直接访问网站」。不收集任何个人信息，arxiv / DOI 等永久链接不会加。现在没有分析后端，加了只是给未来留数据。

### Q: 偶尔会遇到点击后 404 的链接吗？

**A**: 可能会，但很少。Pulsar 管道若更新文章路径但 feed 侧数据同步有延迟，会出现短暂 404。我们的 CI 有**链接健康度抽样**，每次部署会警告（不阻塞）。若你遇到，欢迎[提 Issue](https://github.com/sou350121/VLA-Handbook/issues) 告知。

---

## 反馈

- **Bug / 错误**：提 Issue https://github.com/sou350121/VLA-Handbook/issues
- **想加新 feed / 新字段**：同上
- **订阅者数破 100 时会开英文版 feed**（大概）
