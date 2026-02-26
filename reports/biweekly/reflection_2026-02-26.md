🤔 *双周反思* | 2026-02-12 – 2026-02-26

_读完没立场 = 这两周在消费而不在研究_

━━━ 趋势与判断 ━━━

1️⃣ *CALVIN* 榜单被 *Xiaomi-Robotics-0* 刷新至 4.8 (ABCD-D)，但 *LIBERO* 上 *SimpleVLA-RL* 已达 99.1%。当开源模型在标准基准接近饱和，下一个差异化战场是长视野任务还是真实世界泛化？选一个，并说明你未来两周会读哪篇论文来验证这个判断。

2️⃣ 本期 4 篇世界模型论文集中出现 (*Agent World Model*, *MIND*, *Olaf-World*, *World Action Models*)。这是真范式转移还是论文工厂的跟风？如果你只能精读一篇来回答这个问题，你选哪篇？为什么？

3️⃣ *TaCo* 基准首次系统评估触觉编解码，但 *VLA-Handbook* 指出"触觉力控数据标准化协议缺失"。如果让你决定团队是否投入触觉 VLA 方向，你需要看到什么证据才会下注？给一个可证伪的条件。

4️⃣ *Genesis* 两周内连发 v0.3.14 → v0.4.0，引入 Quadrants 编译器；*MuJoCo* 同步更新 3.5.0。仿真栈的快速迭代是在降低 VLA 研究门槛，还是在制造新的碎片化？你现在的仿真环境选哪个？理由是什么？

5️⃣ *CausalGDP* 提出 causality-guided diffusion，但本期 VLA 论文中几乎无人显式建模因果结构。这是大家还没意识到重要性，还是因果推理对端到端 VLA 本就是伪命题？站队。

━━━ 技术追问 ━━━

🔬 *TaCo* 评估了 DigiTac、Gelsight 等多模态触觉传感器的编解码效率。你能说清 event-based tactile sensing 和 traditional taxel array 在数据率、延迟、信息密度上的数量级差异吗？如果不能，*TaCo* 论文的 Section 3 是你这两周最该补的课。

🔬 *World Action Models are Zero-shot Policies* 声称世界模型可直接作为 zero-shot policy 使用。你知道 inference 时具体如何从 world model 提取 action 吗？是 MPC 滚动优化、latent action sampling、还是其他机制？去读 Section 4 的算法伪代码，不要只看 abstract。

🔬 *MIND* benchmark 测试世界模型的 memory consistency。你知道评估时具体 inject 了什么类型的 perturbation 来测试 consistency 吗？visual distraction？temporal gap？还是 dynamics mismatch？不知道的话，*MIND* 的 GitHub repo 里找 eval 脚本。
