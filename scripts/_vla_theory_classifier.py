# -*- coding: utf-8 -*-
"""
_vla_theory_classifier.py
Classifies VLA theory articles into 10 thematic directories.

Uses two-pass strategy:
  Pass 1: Specific article rules (named models, exact papers)
  Pass 2: Method family keywords (domain-expert taxonomy from 15 families)
  Fallback: frontier/

Usage:
    from _vla_theory_classifier import classify_theory_article
    target_dir = classify_theory_article("pi0_5_dissection.md", "Pi 0.5 模型解剖")
    # Returns "vla-core"
"""

import re

# ── Method Family → Directory mapping (from pipeline's 15 families) ─────────

METHOD_FAMILY_KEYWORDS = {
    "diffusion_policy": {
        "dir": "diffusion-flow",
        "kw": ["diffusion policy", "diffusion policies", "score-based",
               "denoising diffusion", "noise prediction", "DDPM",
               "action diffusion"],
    },
    "flow_matching": {
        "dir": "diffusion-flow",
        "kw": ["flow matching", "flow-matching", "continuous normalizing",
               "rectified flow", "conditional flow"],
    },
    "world_model": {
        "dir": "world-model",
        "kw": ["world model", "world-model", "world action model",
               "video prediction", "latent dynamics", "dreamer",
               "imagination", "simulator"],
    },
    "rl_finetuning": {
        "dir": "rl",
        "kw": ["reinforcement learning", r"\brl\b", "reward",
               "online rl", "offline rl", "policy gradient",
               "ppo", "dpo", "rlhf", "grpo"],
    },
    "tactile": {
        "dir": "tactile",
        "kw": ["tactile", "force sensing", "force feedback",
               "haptic", "contact-rich", "touch sensing",
               "visuo-tactile", "grasp force"],
    },
    "dexterous_hand": {
        "dir": "deployment",
        "kw": ["dexterous hand", "dexterous grasp",
               "finger manipulation", "hand manipulation"],
    },
    "sim_to_real": {
        "dir": "deployment",
        "kw": ["sim-to-real", "sim2real", "sim to real",
               "domain randomization", "reality gap"],
    },
    "mobile_manipulation": {
        "dir": "deployment",
        "kw": ["mobile manipulation", "mobile robot",
               "navigation.*manipulation"],
    },
    "human_robot": {
        "dir": "deployment",
        "kw": ["human-robot", "human robot interaction",
               "teleoperation", "demonstration collection"],
    },
    "3d_representation": {
        "dir": "perception",
        "kw": ["3d representation", "point cloud", "pointcloud",
               "slam", "depth estimation", "stereo", "nerf",
               "3d vision", "spatial representation"],
    },
    "instruction_tuning": {
        "dir": "foundation",
        "kw": ["instruction tuning", "instruction following",
               "fine-tuning", "finetuning", "SFT",
               "supervised fine"],
    },
    "language_grounding": {
        "dir": "vla-core",
        "kw": ["language grounding", "vision-language-action",
               "vision language action", r"\bvla\b"],
    },
    "cross_embodiment": {
        "dir": "vla-core",
        "kw": ["cross-embodiment", "cross embodiment",
               "embodiment transfer", "multi-embodiment"],
    },
    "long_horizon": {
        "dir": "planning",
        "kw": ["long-horizon", "long horizon", "hierarchical",
               "task planning", "chain of thought",
               "reasoning", "skill chaining"],
    },
    "multi_task": {
        "dir": "vla-core",
        "kw": ["multi-task", "multi task", "multitask",
               "generalist policy", "general-purpose"],
    },
}

# ── Specific rules (named models, exact papers) ────────────────────────────

_SPECIFIC_RULES = [
    ("tactile", [
        "tactile", "tac_", "tacmamba", "tacrefine", "supertac", "univtac",
        "force_adaptive", "force feedback", "force sensing", "genforce",
        "touch", "haptic", "contact_rich", "contact-rich",
        "omnivta", "visuo-tactile", "visual-tactile", "visual_tactile",
    ]),
    ("diffusion-flow", [
        "diffusion_policy", "diffusion policy", "flow_matching", "flow matching",
        "contractive_diffusion", "action_representations", r"\baction generation",
        "traditional_action_generation", "action_chunks",
        "neural_implicit_action_fields",
        "pixel_motion_diffusion",
        "compression_gap", "discrete_tokenization",
    ]),
    ("world-model", [
        "world_model", "world model", "world-model", "world_action",
        "dreamzero", "causal_world_modeling",
        "egosim", "wanderland",
        "simulation_distillation.*world_model",
        "learning_physics_from_pretrained.*world",
        "ctrl_world.*worldarena",
        "worldeval.*world_model",
    ]),
    ("rl", [
        "reinforcement_learning", "reinforcement learning",
        r"\brl\b", "rl_finetuning", "online_rl", "offline_rl",
        "reward", "reward_discovery",
        "prioritized_generative_replay",
    ]),
    ("planning", [
        "chain_of_thought", "reasoning", "planning",
        "motion_planning", "safety", "alignment",
        "llm_reasoning",
    ]),
    ("perception", [
        "perception", "slam", "pointcloud", "point_cloud",
        "3d_representation", "multimodal_models",
        "state_estimation", "sensor_fusion",
        "language_shapes_perception",
    ]),
    ("vla-core", [
        r"\bact\b", "act.md", r"\bfast\b", "fast.md",
        "pi0", "gr00t", "galaxea", "openvla", "starvla",
        "vla_arch", "vla_research_mainline",
        "small_vla_models",
        "instructvla",
        "learning_structured_robot_policies",
    ]),
    ("foundation", [
        "knowledge_distillation", "distillation", "knowledge_insulation",
        "transfer_learning", "co_training", "self_supervised",
        "peft_lora", "dora_weight", "quantization",
        "scaling", "pretraining", "pre_training",
        "flash_attention", "transformer_vs_cnn",
        "kv_cache", "evaluation.md",
        "vla_loss_functions",
        "data_flywheel",
    ]),
    ("deployment", [
        "sim2real", "sim_to_real", "sim-to-real",
        "dexterous_hand", "dext", "grasp",
        "robot_control", "robot_dynamics",
        "isaac_lab", "hardware",
    ]),
]


def _classify_by_method_family(text):
    for _fam, spec in METHOD_FAMILY_KEYWORDS.items():
        for kw in spec["kw"]:
            if re.search(kw.lower(), text):
                return spec["dir"]
    return None


def classify_theory_article(filename, title):
    """
    Classify a VLA theory article into one of 10 directories.

    Args:
        filename: e.g. "pi0_5_dissection.md"
        title: e.g. "Pi 0.5 模型解剖 (Dissecting π0.5)"

    Returns:
        Directory name, e.g. "vla-core"
    """
    text = "{} {}".format(filename, title).lower()

    # Pass 1: specific rules
    for target_dir, keywords in _SPECIFIC_RULES:
        for kw in keywords:
            if re.search(kw.lower(), text):
                return target_dir

    # Pass 2: method family keywords
    mf = _classify_by_method_family(text)
    if mf:
        return mf

    return "frontier"


# ── CLI test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        fn, title = sys.argv[1], " ".join(sys.argv[2:])
    elif len(sys.argv) == 2:
        fn, title = sys.argv[1], ""
    else:
        fn, title = "test.md", "test"
    print(classify_theory_article(fn, title))
