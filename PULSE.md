# 📊 VLA Daily Pulse

> 全 VLA 領域研究方向的**每日節奏錶** —— 看 15 個方法族的論文流量、加速度、趨勢
> 由 [Pulsar 照見](https://sou350121.github.io/pulsar-web/) pipeline 每日自動生成

&nbsp;

<p align="center">
  <img src="assets/method-family-trends.svg" alt="Method Family Trends" width="100%" />
</p>

&nbsp;

## 🎯 怎麼讀這張圖

每一行是一個 VLA 方法族（共 15 個，按過去 7 天論文數降序排列）：

| 元素 | 含義 |
|------|------|
| **#** | 排名（按 7d count 降序） |
| **FAMILY** | 方法族名稱（如 `flow_matching` / `world_model`） |
| **7d VOLUME** | 過去 7 天論文數的視覺化條 |
| **7d / 14d / 30d** | 對應窗口的論文總數 |
| **Δ7d / Δ14d / Δ30d** | 該窗口"近期日均 ÷ 前期日均" · `>1.25` 加速 / `<0.80` 減速 |
| **TREND · 30D** | 過去 30 天每日的 7 日滾動計數 sparkline · **金色 = 最新 · 青色漸入 = 歷史** |
| **ST** | 狀態符號 · 🟢 **▲** 加速 / **◆** 穩定 / 🔴 **▼** 減速 |

**色碼**：
- 🟢 加速行：淡綠 tint + 綠色左邊框
- 🔴 減速行：淡紅 tint + 紅色左邊框
- ◆ 穩定行：素色 + 微妙斑馬條紋

&nbsp;

## 📈 為什麼看這個重要

**作為研究者**：哪個方向**真的在 hot**（不是社群 hype），靠論文增速證明
**作為工程師**：哪個技術路線**正在被快速迭代**（值得追蹤）vs **正在退潮**（避免重新發明）
**作為投資/戰略**：方法族的相對熱度給領域演進的微觀信號

📊 **這份數據 ≠ 觀點**，是**機械統計** —— 由 Pulsar 照見從 28 個源 + 21 個 GitHub repo 採集到的論文，按 keyword 規則分類後的統計。誤差來自 (a) 分類器精度 (b) 數據源覆蓋度。

&nbsp;

## 🔗 配套資源

| 你想做 | 去哪裡 |
|--------|-------|
| **看實時數據 + 互動 sparkline** | 🌐 [Pulsar 照見 · VLA 深挖看板](https://sou350121.github.io/pulsar-web/vla-deepdive/) |
| **訂閱每日新文章** | 📡 [RSS 訂閱頁](https://sou350121.github.io/pulsar-web/subscribe/) |
| **看完整數據明細表 + Unicode sparkline** | 📋 [method-family-trends.md](assets/method-family-trends.md) |
| **理解 15 個家族的定義** | 📖 [VLA 數據工程指南](theory/foundation/vla_data_engineering_guide.md) |
| **看本週/本雙週深度報告** | 📊 [Pulsar Reports](https://sou350121.github.io/pulsar-web/reports/) |

&nbsp;

## 🔄 更新節奏

- **數據源**：Pulsar 照見 pipeline 每日 `field-state-YYYY-MM-DD.json` 快照
- **生成**：[scripts/export-method-family-viz.py](https://github.com/sou350121/pulsar-web/blob/main/scripts/export-method-family-viz.py)
- **頻率**：每天 1 次（pipeline 完成後）
- **窗口**：滾動 30 天歷史

&nbsp;

## 📝 引用 / 使用

本圖內容使用 **CC BY 4.0** 授權，可自由轉載 / AI 訓練 / 商用，只需署名：

> 來源：sou350121 · Pulsar 照見 · https://github.com/sou350121/VLA-Handbook/blob/main/PULSE.md

&nbsp;

---

[← Back to VLA Handbook](README.md)
