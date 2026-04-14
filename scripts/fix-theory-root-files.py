#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix-theory-root-files.py
Automated maintenance for VLA-Handbook theory/ directory.

1. Move any theory/ root files to correct subdirectories (using _vla_theory_classifier)
2. Update article counts in theory/README.md + key articles
3. Optionally re-fetch vla-github-theory.json for pulsar-web

Designed to run daily after deep dive pipeline finishes (e.g. cron at 17:30).
Safe to run multiple times — no-op if nothing needs fixing.

Usage:
    python3 fix-theory-root-files.py                    # dry run
    python3 fix-theory-root-files.py --apply            # actually fix
    python3 fix-theory-root-files.py --apply --verbose   # fix + print details
"""

from __future__ import print_function
import argparse
import base64
import json
import os
import re
import sys
import time

# ── Config ──────────────────────────────────────────────────────────────────

REPO = "sou350121/VLA-Handbook"
API = "https://api.github.com/repos/%s" % REPO
HEADERS = {}  # filled in main()

# Files that contain article counts to keep in sync
COUNT_FILES = [
    "theory/README.md",
    "theory/vla-core/vla_research_mainline.md",
    "theory/vla-core/physical_intelligence_sergey_levine_foundation_model_vision_2026.md",
]

# Valid subdirectories
VALID_DIRS = {
    "vla-core", "diffusion-flow", "world-model", "rl", "tactile",
    "perception", "planning", "foundation", "deployment", "frontier",
}


# ── GitHub API ──────────────────────────────────────────────────────────────

def _api(method, path, data=None):
    try:
        from urllib.request import Request, urlopen
        from urllib.error import HTTPError
    except ImportError:
        print("ERROR: urllib not available", file=sys.stderr)
        sys.exit(1)

    url = path if path.startswith("http") else "%s%s" % (API, path)
    body = json.dumps(data).encode() if data else None
    req = Request(url, data=body, headers=HEADERS, method=method)

    for attempt in range(3):
        try:
            with urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except HTTPError as e:
            if e.code in (429, 502, 503) and attempt < 2:
                time.sleep(5 * (attempt + 1))
                continue
            raise
        except Exception:
            if attempt < 2:
                time.sleep(3)
                continue
            raise


def _get_tree():
    """Get full repo tree."""
    ref = _api("GET", "/git/ref/heads/main")
    sha = ref["object"]["sha"]
    commit = _api("GET", "/git/commits/%s" % sha)
    tree = _api("GET", "/git/trees/%s?recursive=1" % commit["tree"]["sha"])
    return sha, tree


def _read_blob(sha):
    """Read blob content."""
    blob = _api("GET", "/git/blobs/%s" % sha)
    return base64.b64decode(blob["content"]).decode("utf-8", errors="replace")


# ── Classifier ──────────────────────────────────────────────────────────────

def _classify(filename, title):
    """Classify a theory article into a subdirectory."""
    # Try to import the shared classifier
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)
    try:
        from _vla_theory_classifier import classify_theory_article
        return classify_theory_article(filename, title)
    except ImportError:
        pass

    # Fallback: simple keyword rules
    text = ("%s %s" % (filename, title)).lower()
    if any(k in text for k in ("tactile", "touch", "haptic", "force_sens")):
        return "tactile"
    if any(k in text for k in ("world_model", "world model", "dreamer", "simulator")):
        return "world-model"
    if any(k in text for k in ("reinforcement", "\\brl\\b", "reward")):
        return "rl"
    if any(k in text for k in ("diffusion", "flow_matching", "action_represent")):
        return "diffusion-flow"
    return "frontier"


# ── Step 1: Find and move root files ────────────────────────────────────────

def find_root_files(tree):
    """Find .md files directly in theory/ (not in subdirectories)."""
    root_files = []
    for e in tree["tree"]:
        if (e["path"].startswith("theory/")
                and e["path"].endswith(".md")
                and e["type"] == "blob"
                and e["path"].count("/") == 1
                and e["path"] != "theory/README.md"):
            root_files.append(e)
    return root_files


def classify_root_files(root_files):
    """Classify each root file and return moves dict."""
    moves = {}
    for entry in root_files:
        filename = entry["path"].split("/")[-1]
        # Read blob to get title
        content = _read_blob(entry["sha"])
        title = ""
        for line in content.split("\n")[:20]:
            if line.startswith("# "):
                title = line[2:].strip()
                break
        subdir = _classify(filename, title)
        new_path = "theory/%s/%s" % (subdir, filename)
        moves[entry["path"]] = {"new_path": new_path, "sha": entry["sha"], "subdir": subdir}
    return moves


# ── Step 2: Count articles per directory ────────────────────────────────────

def count_articles(tree, exclude_root=True):
    """Count .md files per theory/ subdirectory."""
    from collections import Counter
    counts = Counter()
    total = 0
    for e in tree["tree"]:
        if (e["path"].startswith("theory/")
                and e["path"].endswith(".md")
                and e["type"] == "blob"
                and e["path"] != "theory/README.md"):
            parts = e["path"].split("/")
            if len(parts) == 3:  # theory/<subdir>/<file>.md
                if parts[-1] != "README.md":
                    counts[parts[1]] += 1
                    total += 1
            elif len(parts) == 2 and not exclude_root:
                counts["(root)"] += 1
                total += 1
    return total, dict(counts)


# ── Step 3: Update article counts in files ──────────────────────────────────

# Mermaid diagram count patterns
MERMAID_PATTERNS = {
    "vla-core": r"VLA 核心架构<br/><b>(\d+) 篇</b>",
    "diffusion-flow": r"扩散 · Flow Matching<br/>(\d+) 篇",
    "world-model": r"世界模型 · 仿真<br/>(\d+) 篇",
    "rl": r"强化学习 · 奖励<br/>(\d+) 篇",
    "tactile": r"触觉感知<br/>(\d+) 篇",
    "perception": r"3D · SLAM<br/>(\d+) 篇",
    "planning": r"推理 · 安全 · 规划<br/>(\d+) 篇",
    "foundation": r"基础理论 · 工具箱<br/>(\d+) 篇",
    "deployment": r"部署 · 硬件<br/>(\d+) 篇",
    "frontier": r"跨域 · 神经科学<br/>(\d+) 篇",
}

HEADER_PATTERNS = {
    "vla-core": r"<code>(\d+) 篇</code></h3></summary>\n\n\*\*一句话\*\*：所有 VLA",
    "diffusion-flow": r"<code>(\d+) 篇</code></h3></summary>\n\n\*\*一句话\*\*：机器人的动作",
    "world-model": r"<code>(\d+) 篇</code></h3></summary>\n\n\*\*一句话\*\*：让机器人在",
    "rl": r"<code>(\d+) 篇</code></h3></summary>\n\n\*\*一句话\*\*：模仿学习有天花板",
    "tactile": r"<code>(\d+) 篇</code></h3></summary>\n\n\*\*一句话\*\*：闭上眼睛",
    "perception": r"<code>(\d+) 篇</code></h3></summary>\n\n\*\*一句话\*\*：机器人的",
    "planning": r"<code>(\d+) 篇</code></h3></summary>\n\n\*\*一句话\*\*：让机器人",
    "foundation": r"<code>(\d+) 篇</code></h3></summary>\n\n\*\*一句话\*\*：不按顺序",
    "deployment": r"<code>(\d+) 篇</code></h3></summary>\n\n\*\*一句话\*\*：仿真里",
    "frontier": r"<code>(\d+) 篇</code></h3></summary>\n\n\*\*一句话\*\*：VLA 的下",
}


def update_counts_in_content(content, total, counts):
    """Replace stale article counts with current ones."""
    changes = 0

    # Total count patterns
    for pattern in [r"(\d+) 篇深度解析", r"\*\*(\d+)\*\* articles",
                    r"基于 (\d+) 篇", r"(\d+) 篇语料"]:
        m = re.search(pattern, content)
        if m and int(m.group(1)) != total:
            content = content.replace(m.group(0), m.group(0).replace(m.group(1), str(total)))
            changes += 1

    # Mermaid diagram counts
    for subdir, pattern in MERMAID_PATTERNS.items():
        if subdir in counts:
            m = re.search(pattern, content)
            if m and int(m.group(1)) != counts[subdir]:
                old = m.group(0)
                new = old.replace(m.group(1), str(counts[subdir]))
                content = content.replace(old, new)
                changes += 1

    return content, changes


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Fix VLA-Handbook theory/ root files + counts")
    ap.add_argument("--apply", action="store_true", help="Actually make changes (default: dry run)")
    ap.add_argument("--verbose", action="store_true", help="Print detailed info")
    args = ap.parse_args()

    # Get GitHub token
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        for p in ["/home/admin/.clawdbot/.env", "/home/admin/.moltbot/.env"]:
            try:
                with open(p) as f:
                    for line in f:
                        if line.strip().startswith("GITHUB_TOKEN="):
                            token = line.strip().split("=", 1)[1].strip().strip("'\"")
                            break
            except Exception:
                pass
            if token:
                break

    if not token:
        print("ERROR: no GITHUB_TOKEN", file=sys.stderr)
        sys.exit(1)

    HEADERS.update({
        "Authorization": "token %s" % token,
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json",
    })

    # Step 0: Get tree
    print("[1/4] Reading repository tree...")
    current_sha, tree = _get_tree()

    # Step 1: Find root files
    root_files = find_root_files(tree)
    print("[2/4] Root files to move: %d" % len(root_files))

    moves = {}
    if root_files:
        moves = classify_root_files(root_files)
        for old_path, info in sorted(moves.items()):
            print("  %s → %s" % (old_path.split("/")[-1][:50], info["subdir"]))

    # Step 2: Count articles
    total, counts = count_articles(tree)
    print("[3/4] Article counts (total %d):" % total)
    if args.verbose:
        for d in sorted(counts, key=lambda x: -counts[x]):
            print("  %s: %d" % (d, counts[d]))

    # Step 3: Check if counts need updating in files
    files_to_update = {}
    for fpath in COUNT_FILES:
        for e in tree["tree"]:
            if e["path"] == fpath:
                content = _read_blob(e["sha"])
                new_content, change_count = update_counts_in_content(content, total, counts)
                if change_count > 0:
                    files_to_update[fpath] = new_content
                    print("  %s: %d count updates needed" % (fpath.split("/")[-1], change_count))
                break

    # Summary
    total_changes = len(moves) + len(files_to_update)
    if total_changes == 0:
        print("[4/4] Everything is clean. No changes needed.")
        print(json.dumps({"ok": True, "moves": 0, "count_updates": 0, "total": total}))
        return

    print("[4/4] Total changes: %d moves + %d file updates" % (len(moves), len(files_to_update)))

    if not args.apply:
        print("\n[DRY RUN] Use --apply to execute changes.")
        print(json.dumps({
            "ok": True, "dry_run": True,
            "moves": len(moves), "count_updates": len(files_to_update),
            "total": total,
        }))
        return

    # Apply changes via atomic commit
    print("\nApplying changes...")

    # Build new tree entries
    new_entries = []
    moved_old_paths = set(moves.keys())

    for e in tree["tree"]:
        if e["type"] == "tree":
            continue
        if e["path"] in moved_old_paths:
            continue  # Skip old location of moved files
        if e["path"] in files_to_update:
            # Create new blob for updated content
            blob = _api("POST", "/git/blobs", {
                "content": files_to_update[e["path"]],
                "encoding": "utf-8",
            })
            new_entries.append({
                "path": e["path"],
                "mode": e["mode"],
                "type": "blob",
                "sha": blob["sha"],
            })
        else:
            new_entries.append({
                "path": e["path"],
                "mode": e["mode"],
                "type": e["type"],
                "sha": e["sha"],
            })

    # Add moved files at new locations
    for old_path, info in moves.items():
        new_entries.append({
            "path": info["new_path"],
            "mode": "100644",
            "type": "blob",
            "sha": info["sha"],
        })

    # Create tree + commit
    new_tree = _api("POST", "/git/trees", {"tree": new_entries})

    parts = []
    if moves:
        parts.append("move %d root files to subdirs" % len(moves))
    if files_to_update:
        parts.append("update counts to %d" % total)
    msg = "fix(auto): %s" % " + ".join(parts)

    new_commit = _api("POST", "/git/commits", {
        "message": msg,
        "tree": new_tree["sha"],
        "parents": [current_sha],
    })
    _api("PATCH", "/git/refs/heads/main", {"sha": new_commit["sha"]})

    print("Done! %s/commit/%s" % ("https://github.com/" + REPO, new_commit["sha"]))
    print(json.dumps({
        "ok": True,
        "moves": len(moves),
        "count_updates": len(files_to_update),
        "total": total,
        "commit": new_commit["sha"][:12],
    }))


if __name__ == "__main__":
    main()
