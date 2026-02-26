#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vla-trend-snapshot.py - Pre-compute weekly VLA topic trend data.

Reads vla-daily-hotspots.json, groups rated papers by ISO week,
classifies into the 4-layer VLA framework, writes vla-trend-snapshot.json.

4-Layer Framework:
  Layer 1: 基礎能力  - perception, grounding, language, action generation
  Layer 2: 泛化能力  - generalization, sim-to-real, few-shot, open-vocab
  Layer 3: 長時序能力 - memory, planning, long-horizon, multi-step
  Layer 4: 商業部署  - latency, cost, deployment, hardware, efficiency

Python 3.6+ compatible.
"""

from __future__ import print_function

import json, os, sys
import datetime as _dt
from collections import defaultdict, Counter

MEM_DIR   = "/home/admin/clawd/memory"
HOTSPOTS  = os.path.join(MEM_DIR, "vla-daily-hotspots.json")
OUTPUT    = os.path.join(MEM_DIR, "vla-trend-snapshot.json")
TMP_DIR   = os.path.join(MEM_DIR, "tmp")

LAYER_KW = {
    "Layer 1: 基礎能力": [
        "perception","grounding","action generation","policy learning","pretraining",
        "pretrain","foundation model","visual encoder","tokeniz","embedding",
        "representation","sensorimotor","end-to-end","language understanding",
    ],
    "Layer 2: 泛化能力": [
        "generali","sim-to-real","sim2real","few-shot","few shot","open-vocab",
        "open vocab","zero-shot","zero shot","transfer","cross-task","cross task",
        "domain adaptation","domain randomiz","out-of-distribution","unseen",
        "novel object","novel task","bimanual","dexterous","manipulation",
    ],
    "Layer 3: 長時序能力": [
        "memory","planning","long-horizon","long horizon","multi-step","multi step",
        "reasoning","chain-of-thought","chain of thought","hierarchical",
        "task planning","decompos","subgoal","temporal","sequential","world model",
    ],
    "Layer 4: 商業部署": [
        "latency","real-time","real time","efficiency","efficient","lightweight",
        "distill","quantiz","compress","deploy","hardware","production","cost",
        "fast inference","mobile","humanoid","commercial",
    ],
}

def _classify(title, reason):
    text = ((title or "") + " " + (reason or "")).lower()
    scores = {l: sum(1 for kw in kws if kw in text) for l, kws in LAYER_KW.items()}
    best = max(scores, key=lambda l: scores[l])
    return best if scores[best] > 0 else "Unclassified"

def _iso_week(date_str):
    try:
        dt = _dt.datetime.strptime(date_str, "%Y-%m-%d").date()
        iso = dt.isocalendar()
        return (iso[0], iso[1])
    except Exception:
        return (0, 0)

def _week_label(y, w):
    return "%d-W%02d" % (y, w)

def _load(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default if default is not None else {}

def _save(path, obj):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")

def main():
    papers = (_load(HOTSPOTS, {}).get("reported_papers") or [])

    # Supplement today's unrated papers from rating-out tmp
    today = (_dt.datetime.utcnow() + _dt.timedelta(hours=8)).strftime("%Y-%m-%d")
    ro_path = os.path.join(TMP_DIR, "vla-daily-rating-out-%s.json" % today)
    extra = {}
    if os.path.exists(ro_path):
        for rp in (_load(ro_path, {}).get("papers") or []):
            if isinstance(rp, dict) and rp.get("rating"):
                extra[rp.get("title", "")] = rp
    for p in papers:
        if p.get("rating", "?") in ("?", "", None):
            rp = extra.get(p.get("title", ""))
            if rp:
                p["rating"] = rp.get("rating", "?")
                p["reason"] = rp.get("reason", "")
                p["affiliation"] = rp.get("affiliation", "")

    # Group by ISO week
    weekly = defaultdict(list)
    for p in papers:
        yw = _iso_week(p.get("date", ""))
        if yw != (0, 0):
            weekly[yw].append(p)

    weeks_out = []
    for yw in sorted(weekly.keys()):
        wp = weekly[yw]
        rated    = [p for p in wp if p.get("rating") in ("\u26a1", "\U0001f527", "\U0001f4d6", "\u274c")]
        highlight = [p for p in wp if p.get("rating") in ("\u26a1", "\U0001f527")]
        rc = Counter(p.get("rating") for p in rated)
        lc = Counter(_classify(p.get("title",""), p.get("reason","")) for p in highlight)
        weeks_out.append({
            "week":           _week_label(yw[0], yw[1]),
            "year":           yw[0],
            "iso_week":       yw[1],
            "total_papers":   len(wp),
            "rated_papers":   len(rated),
            "ratings": {
                "\u26a1":         rc.get("\u26a1", 0),
                "\U0001f527":     rc.get("\U0001f527", 0),
                "\U0001f4d6":     rc.get("\U0001f4d6", 0),
                "\u274c":         rc.get("\u274c", 0),
            },
            "layer_distribution": dict(lc),
            "dominant_layer": lc.most_common(1)[0][0] if lc else "insufficient data",
            "top_affiliations": [p.get("affiliation","") for p in highlight if p.get("affiliation")][:5],
            "highlight_titles": [p.get("title","")[:80] for p in highlight],
        })

    rated_weeks = [w for w in weeks_out if w["rated_papers"] > 0]
    trend_delta = None
    if len(rated_weeks) >= 2:
        prev, curr = rated_weeks[-2], rated_weeks[-1]
        trend_delta = {
            "from_week":      prev["week"],
            "to_week":        curr["week"],
            "layer_shift":    curr["dominant_layer"] != prev["dominant_layer"],
            "prev_dominant":  prev["dominant_layer"],
            "curr_dominant":  curr["dominant_layer"],
            "highlight_delta": (curr["ratings"]["\u26a1"] + curr["ratings"]["\U0001f527"]
                              - prev["ratings"]["\u26a1"] - prev["ratings"]["\U0001f527"]),
        }

    out = {
        "generated_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_weeks":  len(weeks_out),
        "rated_weeks":  len(rated_weeks),
        "weeks":        weeks_out,
        "trend_delta":  trend_delta,
        "note": "Meaningful ratings accumulate from 2026-02-26 onward (timeout fix). "
                "Weeks with rated_papers=0 show only volume data.",
    }
    _save(OUTPUT, out)
    print(json.dumps({"ok": True, "output": OUTPUT,
                      "total_weeks": len(weeks_out), "rated_weeks": len(rated_weeks),
                      "trend_delta": trend_delta}, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    sys.exit(main())
