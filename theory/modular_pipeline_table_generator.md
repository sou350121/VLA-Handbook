# 弹性模组化架构 Table 生成器（VLA Modular Pipelines）

**A) Stages + Table**

**Stages:** `Inputs → Perception → Backbone → World Model → Planner → Action/Control`

| Inputs | Perception | Backbone | World Model | Planner | Action/Control | Model/Policy | Notes |
|---|---|---|---|---|---|---|---|
| Image<br>Language | EfficientNet + FiLM<br>TokenLearner | Transformer | — | — | Discrete Action Tokens (256 bins) | RT-1 | 经典 token 化控制路径，强调离散动作与高效推理。 |
| Image<br>Language | DINOv2 / SigLIP | Llama 2 (7B) | — | — | Action Head (Linear)<br>Action Detokenization | OpenVLA | VLM 主干 + 专用动作头，非 diffusion/flow。 |
| Image<br>Proprioception<br>Language | Observation Encoder (ResNet-18 / ViT + Linear) | Transformer Decoder<br>CVAE Encoder (train) | — | — | Action Chunking<br>Temporal Ensemble | ACT | CVAE + chunk 方案，低延迟、闭环友好。 |
| Image (RGB)<br>Proprioception | Visual Encoder (ResNet/ViT) | Denoising Network (CNN/U-Net or Transformer/DiT) | — | — | Action Chunking<br>Horizon Control (RHC) | Diffusion Policy | 扩散生成动作，强多峰分布表达。 |
| Images (multi-view)<br>Language<br>Proprioception | SigLIP | PaliGemma 2B VLM | — | — | Action Expert (Flow Matching)<br>ODE Solver (Euler/RK4)<br>Action Chunk | π0 | VLM 条件 + Flow/ODE 动作生成。 |
| RGB Image Sequence<br>Natural Language Cmd | VLM Backbone (3B-5B) | Unified Transformer | — | Latent Thought | FAST Tokenizer (training)<br>Flow Matching (inference)<br>50Hz Action Output | π0.5 | 训练离散、推理连续的混合架构。 |
| — | — | 5B VLM Backbone | — | — | Action Expert<br>Velocity Field v(a_t,t) | π0.6 / π*0.6 | 引入 Action Expert 与 Recap（offline RL）后训练。 |
| Image<br>Language<br>Proprioception<br>Noisy Action x_t<br>Timestep t | Condition Encoder (SigLIP + T5 + Proprio Projection + Fusion) | DiT Backbone (AdaLN-Zero Transformer) | — | — | DDPM/DDIM Denoising Sampling<br>Denoised Action x_{t-1} | RDT-1B | 条件编码 + DiT 扩散基础模型路径。 |
| Visual Input (HD)<br>Language Instruction<br>Real-time RGB<br>Proprioception/Joint States | Large VLM Encoder (System 2) | Diffusion Transformer (System 1, AdaLN-Zero) | — | Latent Task Tokens (Asynchronous Injection) | Denoising Process (3-5 Steps)<br>Action Chunking Output (~50Hz) | GR00T-N1.6 | 异步双系统（慢语义 + 快控制）。 |
| 一帧初始画面<br>文本指令 | — | — | Text-conditioned Video Diffusion World Model | 并行候选视频 + VLM Evaluator 选优 | Inverse Dynamics Model (IDM)<br>Reject/Resample<br>时间平滑 | 1XWM | 先“想象未来视频”，再逆动力学映射动作。 |
| 图像<br>指令 | Qwen2.5 VLMoE Backbone | Qwen2.5 VLMoE | — | Hierarchical CoT<br>(Reasoning → Sub-goal → Action Synthesis) | 离散头 (FAST Token)<br>连续头 (Flow Match)<br>X² Control | WALL-OSS | 分层 CoT + 双 head（离散规划/连续控制）。 |

**B) Mermaid**

```mermaid
flowchart LR

subgraph inputsStage [Inputs]
rt1In["Image + Language"]
openvlaIn["Image + Language"]
actIn["Image + Proprioception + Language"]
dpIn["Image (RGB) + Proprioception"]
pi0In["Images (multi-view) + Language + Proprioception"]
pi05In["RGB Image Sequence + Natural Language Cmd"]
pi06In["—"]
rdtIn["Image + Language + Proprioception + Noisy Action x_t + Timestep t"]
gr00tIn["Visual Input (HD) + Language + Real-time RGB + Proprioception"]
oneXIn["一帧初始画面 + 文本指令"]
wallIn["图像 + 指令"]
end

subgraph perceptionStage [Perception]
rt1Perc["EfficientNet + FiLM + TokenLearner"]
openvlaPerc["DINOv2 / SigLIP"]
actPerc["Observation Encoder (ResNet-18 / ViT + Linear)"]
dpPerc["Visual Encoder (ResNet/ViT)"]
pi0Perc["SigLIP"]
pi05Perc["VLM Backbone (3B-5B)"]
pi06Perc["—"]
rdtPerc["Condition Encoder (SigLIP + T5 + Proprio Projection + Fusion)"]
gr00tPerc["Large VLM Encoder (System 2)"]
oneXPerc["—"]
wallPerc["Qwen2.5 VLMoE Backbone"]
end

subgraph backboneStage [Backbone]
rt1Back["Transformer"]
openvlaBack["Llama 2 (7B)"]
actBack["Transformer Decoder + CVAE Encoder(train)"]
dpBack["Denoising Network (CNN/U-Net or Transformer/DiT)"]
pi0Back["PaliGemma 2B VLM"]
pi05Back["Unified Transformer"]
pi06Back["5B VLM Backbone"]
rdtBack["DiT Backbone (AdaLN-Zero)"]
gr00tBack["Diffusion Transformer (System 1)"]
oneXBack["—"]
wallBack["Qwen2.5 VLMoE"]
end

subgraph worldModelStage [World Model]
rt1WM["—"]
openvlaWM["—"]
actWM["—"]
dpWM["—"]
pi0WM["—"]
pi05WM["—"]
pi06WM["—"]
rdtWM["—"]
gr00tWM["—"]
oneXWM["Text-conditioned Video Diffusion World Model"]
wallWM["—"]
end

subgraph plannerStage [Planner]
rt1Plan["—"]
openvlaPlan["—"]
actPlan["—"]
dpPlan["—"]
pi0Plan["—"]
pi05Plan["Latent Thought"]
pi06Plan["—"]
rdtPlan["—"]
gr00tPlan["Latent Task Tokens (Async Injection)"]
oneXPlan["并行候选视频 + VLM Evaluator"]
wallPlan["Hierarchical CoT"]
end

subgraph actionStage [Action/Control]
rt1Act["Discrete Action Tokens (256 bins)"]
openvlaAct["Action Head (Linear) + Action Detokenization"]
actAct["Action Chunking + Temporal Ensemble"]
dpAct["Action Chunking + RHC"]
pi0Act["Action Expert (Flow Matching) + ODE Solver + Action Chunk"]
pi05Act["FAST(training) + Flow Matching(inference) + 50Hz"]
pi06Act["Action Expert + Velocity Field v(a_t,t)"]
rdtAct["DDPM/DDIM Sampling + Denoised Action x_{t-1}"]
gr00tAct["Denoising(3-5) + Action Chunking(~50Hz)"]
oneXAct["IDM + Reject/Resample + 时间平滑"]
wallAct["离散头(FAST) + 连续头(Flow Match) + X² Control"]
end

subgraph modelStage [Model/Policy]
rt1Model["RT-1"]
openvlaModel["OpenVLA"]
actModel["ACT"]
dpModel["Diffusion Policy"]
pi0Model["π0"]
pi05Model["π0.5"]
pi06Model["π0.6 / π*0.6"]
rdtModel["RDT-1B"]
gr00tModel["GR00T-N1.6"]
oneXModel["1XWM"]
wallModel["WALL-OSS"]
end

rt1In --> rt1Perc --> rt1Back --> rt1WM --> rt1Plan --> rt1Act --> rt1Model
openvlaIn --> openvlaPerc --> openvlaBack --> openvlaWM --> openvlaPlan --> openvlaAct --> openvlaModel
actIn --> actPerc --> actBack --> actWM --> actPlan --> actAct --> actModel
dpIn --> dpPerc --> dpBack --> dpWM --> dpPlan --> dpAct --> dpModel
pi0In --> pi0Perc --> pi0Back --> pi0WM --> pi0Plan --> pi0Act --> pi0Model
pi05In --> pi05Perc --> pi05Back --> pi05WM --> pi05Plan --> pi05Act --> pi05Model
pi06In --> pi06Perc --> pi06Back --> pi06WM --> pi06Plan --> pi06Act --> pi06Model
rdtIn --> rdtPerc --> rdtBack --> rdtWM --> rdtPlan --> rdtAct --> rdtModel
gr00tIn --> gr00tPerc --> gr00tBack --> gr00tWM --> gr00tPlan --> gr00tAct --> gr00tModel
oneXIn --> oneXPerc --> oneXBack --> oneXWM --> oneXPlan --> oneXAct --> oneXModel
wallIn --> wallPerc --> wallBack --> wallWM --> wallPlan --> wallAct --> wallModel
```

**C) Missing Info**

- `RT-1` 缺 `World Model`：需要在 `theory/` 中找到其是否包含显式环境动态建模（关键词：world model / latent dynamics / predictive model）。\n+- `OpenVLA` 缺 `Planner`：需要在 `theory/vla_arch.md` 或 OpenVLA 专文中找到显式任务分解/规划模块描述（关键词：planner / sub-goal / reasoning chain）。\n+- `π0.6 / π*0.6` 缺 `Inputs`：需要在 `theory/pi0_6_dissection.md` 或 `theory/pi0_code_analysis.md` 找到完整输入模态定义（图像/语言/本体感知）。\n+- `π0.6 / π*0.6` 缺 `Perception`：需要补充其具体视觉编码器或条件编码器名称（关键词：vision encoder / SigLIP / tokenizer / condition encoder）。\n+- `RDT-1B` 缺 `Planner`：需要在 `theory/rdt.md` 或 `theory/frontier/rdt2_umi_zero_shot_cross_embodiment_2026.md` 查找是否有显式规划层（关键词：planner / hierarchical / sub-goal）。\n+- `1XWM` 缺 `Perception`：需要在 `theory/frontier/one_x_world_model.md` 补充世界模型前端编码器细节（关键词：vision encoder / text encoder / tokenizer）。\n+- `1XWM` 缺 `Backbone`：需要补充世界模型主干的具体网络命名（关键词：video diffusion backbone / transformer / U-Net）。\n+
