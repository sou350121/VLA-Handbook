🤔 *雙週反思* | 2026-10-22 – 2026-11-04

_讀完沒立場 = 這兩週在消費而不在研究_

━━━ 趨勢與判斷 ━━━

1️⃣ 本期 14 天只有 6 天有採集回執（10-22 至 10-29 缺失），方法族腳本卻照樣吐出 14 個標籤：*world_model* 標成唯一 SURGE、*tactile* 標成 declining。一份自己承認「8 天缺口、趨勢推論不被支持」的報告，還把這些標籤留在裡面——你是把它們當噪音，還是照樣拿去改研究方向？

2️⃣ 本期社交情報 25 條信號裡，融資/遞表類（*RoboScience* A2、*松延動力* 遞表、*中科慧靈* A++、*Bleu Robotics* €5M、*四川具身* Pre-A、*格物通智* 天使輪）壓倒性多於技術信號。如果只能把下一季的注意力押在一邊——資本敘事還是技術線——你押哪邊？給出不依賴新聞標題的理由。

3️⃣ *Liquid AI LFM3-VL-2B* 自報 LIBERO 平均 96.2%、單模型跨 15 種本體；同期存儲記錄把 LIBERO 三個類別全標為 99.2%–99.8% saturated。在一個被自家數據判為飽和的基準上報 96.2%，這算什麼證據？你下次看到「跨 15 本體」這類宣稱，第一件事會去查什麼？

4️⃣ *DRILL*（以提升下游策略為目標訓世界模型，而非預測像素）與 *Future Representation Distillation*（免去逐步想像未來的 WAM）同週出現，方向一致：世界模型的訓練目標正在從「預測」轉向「有用」。這是真轉向，還是因為 world_model 族基數小（14 點、accel_7d=1.56）造成的假象？你敢不敢押一個？

5️⃣ 本期至少七篇在做同一件事的不同切法：*Nexus-VLA*、*EXE*、*RynnVLA-Latent*、*AIRoA-0*、*HuGe*、*Reconstruction-Augmented*、*VEGA*——全部是「人類視頻/手持數據 → 統一潛動作空間 → 跨本體」。這是收斂成共識，還是同一條賽道在灌水？證據在你手上哪一條？

━━━ 技術追問 ━━━

🔬 本期 *π-StepNFT*、*TOPS*、*Learning Safe Flow Matching from Binary Feasibility Oracles*、*When One Step Is Enough* 全在拆 flow matching 的縫：似然不可算、共用高斯源是多任務瓶頸、硬約束無保證、單步 Euler 夠不夠。你能說清 flow matching 與 diffusion policy 在「訓練目標與推理路徑」上的本質差別嗎？說不清，這就是你這兩週最該補的課——從 *π0* 的 action expert 實作讀起。

🔬 *Full-Weight VLA Fine-Tuning Without an fp32 Master Copy Silently Erases Most Updates* 說的是：bf16 訓練且無 fp32 主權重時，小於 1/512 的 AdamW 更新被靜默捨入掉。你現在的 VLA 全參微調腳本有沒有 fp32 master weights？去 grep 你的 optimizer 設定，再看 *Should the Action Expert Train the VLM?* 的梯度路由結論——這兩篇一起決定你的後訓練到底學到了什麼。

🔬 觸覺線本期至少五篇：*TacClaw-Infinity*、*Tactus*、*DreamTacVLA*、*T-WLA*、*Multi-Resolution Tactile Imitation*，加上 *EquiContact* 的 SE(3) 視覺到力等變。但校準記錄同時把 V-002（觸覺成必需模態）標為 strengthened 0.64、方法族卻標 tactile declining。觸覺到底在升還是在降——你能用這五篇裡的任何一篇的量化結果回答，而不是用計數標籤嗎？

🔬 上期回顧：*biweekly-reasoning* 最近一條 predictions 為空，往前追到 2026-02-25 的「世界模型+VLA 在 sim-to-real 轉移率上超越純 VLA」——九個月後，本期 *When the Simulator Fails* 直接審計仿真器接觸偽影，*DRILL* 說世界模型不該預測像素。你判斷這條預測是 ✅ 已被側面證實、❌ 落空、還是 ⏳ 待觀察？給理由，不許回答「兩方面都有道理」。
