#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backfill-vla-history.py

Backfills wind_direction (and optional synthetic quality-review entries) for
past weeks using LOCAL written articles as source material:
  - vla-daily-hotspots.json  (all paper titles, including unrated '?')
  - biweekly-reasoning.json  (key signals per period)
  - vla-theory-articles.json (theory articles)

Outputs:
  - Adds wind_direction + wind_trend to existing quality-review.json entries
  - Creates new historical quality-review entries for weeks not yet covered
  - Updates vla-trend-snapshot.json to include all-paper layer distributions

Python 3.6+ compatible.
"""

from __future__ import print_function

import json, os, sys, datetime as _dt
from collections import defaultdict, Counter

MEM_DIR      = "/home/admin/clawd/memory"
HOTSPOTS     = os.path.join(MEM_DIR, "vla-daily-hotspots.json")
BIWEEKLY_J   = os.path.join(MEM_DIR, "biweekly-reasoning.json")
THEORY       = os.path.join(MEM_DIR, "vla-theory-articles.json")
SOCIAL       = os.path.join(MEM_DIR, "vla-social-intel.json")
QR_PATH      = os.path.join(MEM_DIR, "quality-review.json")
SNAPSHOT     = os.path.join(MEM_DIR, "vla-trend-snapshot.json")
TMP_DIR      = os.path.join(MEM_DIR, "tmp")

LAYER_KW = {
    "Layer 1: 基礎能力": [
        "perception","grounding","action generation","policy learning","pretraining",
        "pretrain","foundation model","visual encoder","tokeniz","embedding",
        "representation","sensorimotor","end-to-end","language understanding",
        "instruction follow","tactile","touch","force","haptic",
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
        "imagination","latent action","causal","reflection",
    ],
    "Layer 4: 商業部署": [
        "latency","real-time","real time","efficiency","efficient","lightweight",
        "distill","quantiz","compress","deploy","hardware","production","cost",
        "fast inference","mobile","humanoid","commercial","robot arm","service robot",
        "streaming","on-device","edge",
    ],
}


def _load(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default if default is not None else {}


def _save(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print("  Saved:", path)


def _classify(title, reason=""):
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


def _week_to_date_range(year, week):
    """Return (mon_str, sun_str) for given ISO week."""
    jan4 = _dt.date(year, 1, 4)
    start = jan4 - _dt.timedelta(days=jan4.isocalendar()[2] - 1)
    mon = start + _dt.timedelta(weeks=week - 1)
    sun = mon + _dt.timedelta(days=6)
    return mon.strftime("%Y-%m-%d"), sun.strftime("%Y-%m-%d")


# ------------------------------------------------------------------
# Step 1: Load hotspots, classify ALL papers by title
# ------------------------------------------------------------------
def build_weekly_layers():
    """Group ALL papers (rated+unrated) by ISO week and classify layers."""
    data = _load(HOTSPOTS, {})
    papers = data.get("reported_papers") or []

    # Supplement with today's rating-out file
    today = (_dt.datetime.utcnow() + _dt.timedelta(hours=8)).strftime("%Y-%m-%d")
    ro_path = os.path.join(TMP_DIR, "vla-daily-rating-out-%s.json" % today)
    extra = {}
    if os.path.exists(ro_path):
        for rp in (_load(ro_path, {}).get("papers") or []):
            if isinstance(rp, dict):
                extra[rp.get("title", "")] = rp

    # Apply rating supplements
    for p in papers:
        if p.get("rating", "?") in ("?", "", None):
            rp = extra.get(p.get("title", ""))
            if rp and rp.get("rating"):
                p["rating"] = rp["rating"]
                p["reason"] = rp.get("reason", "")

    # Group by week
    weekly = defaultdict(list)
    for p in papers:
        yw = _iso_week(p.get("date", ""))
        if yw != (0, 0):
            weekly[yw].append(p)

    # Compute layer distribution for each week
    result = {}
    for yw, wp in sorted(weekly.items()):
        rated  = [p for p in wp if p.get("rating") in ("\u26a1", "\U0001f527", "\U0001f4d6", "\u274c")]
        hot    = [p for p in wp if p.get("rating") in ("\u26a1", "\U0001f527")]
        # Use hot > rated > all (all = title-based fallback)
        source = hot if hot else (rated if rated else wp)
        basis  = "hot" if hot else ("rated" if rated else "all-title-classified")

        layer_cnt = Counter(_classify(p.get("title",""), p.get("reason","")) for p in source)
        total = sum(layer_cnt.values())

        # Dominant layer (need at least 2 papers in top)
        if total == 0:
            dominant = "insufficient data"
        else:
            top = max(layer_cnt, key=lambda l: layer_cnt[l])
            dominant = top if layer_cnt[top] >= 2 else "insufficient data"

        # Top affiliations (from hot or rated papers)
        affils = []
        for p in (hot or rated):
            a = p.get("affiliation", "")
            if a and a not in affils:
                affils.append(a)

        # Key titles (up to 3, from hot or all)
        highlights = [p.get("title","")[:80] for p in (hot or wp)[:3]]

        result[yw] = {
            "week":               _week_label(yw[0], yw[1]),
            "year":               yw[0],
            "iso_week":           yw[1],
            "total_papers":       len(wp),
            "rated_papers":       len(rated),
            "hot_papers":         len(hot),
            "layer_distribution": dict(layer_cnt),
            "dominant_layer":     dominant,
            "source_basis":       basis,
            "top_affiliations":   affils[:4],
            "highlight_titles":   highlights,
        }

    return result


# ------------------------------------------------------------------
# Step 2: Load biweekly-reasoning signals, map to weeks
# ------------------------------------------------------------------
def load_biweekly_signals():
    data = _load(BIWEEKLY_J, {})
    entries = data.get("biweekly_reasoning") or []
    out = []
    for e in entries:
        period = e.get("period", "")
        try:
            parts = [p.strip() for p in period.split(" to ")]
            ps = parts[0]
            pe = parts[1] if len(parts) > 1 else parts[0]
        except Exception:
            continue
        out.append({
            "date":         e.get("date",""),
            "period_start": ps,
            "period_end":   pe,
            "key_signals":  e.get("key_signals", []),
            "predictions":  e.get("predictions", []),
        })
    return out


def signals_for_week(bw_entries, year, week):
    mon_str, sun_str = _week_to_date_range(year, week)
    matched_signals, matched_preds = [], []
    for e in bw_entries:
        if sun_str >= e["period_start"] and mon_str <= e["period_end"]:
            matched_signals.extend(e["key_signals"])
            matched_preds.extend(e["predictions"])
    seen = set()
    deduped = []
    for s in matched_signals:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    return deduped, matched_preds


# ------------------------------------------------------------------
# Step 3: Build wind_direction for a week
# ------------------------------------------------------------------
def build_wind_direction(yw, week_data, bw_signals, bw_preds, prev_dominant=None):
    dominant   = week_data["dominant_layer"]
    layer_dist = week_data["layer_distribution"]
    total      = sum(layer_dist.values())
    top_count  = max(layer_dist.values()) if layer_dist else 0

    is_converging  = (top_count / total >= 0.5) if total > 0 else False
    shift_detected = (prev_dominant is not None
                      and dominant != prev_dominant
                      and dominant not in ("insufficient data",)
                      and prev_dominant not in ("insufficient data",))

    signal_summary = "; ".join(bw_signals[:3]) if bw_signals else "（本週信號不足，依標題分類推估）"

    layer_pct = {}
    if total > 0:
        for l, c in sorted(layer_dist.items(), key=lambda x: -x[1]):
            layer_pct[l] = round(100.0 * c / total)

    return {
        "dominant_layer":    dominant,
        "layer_distribution": layer_pct,
        "is_converging":     is_converging,
        "shift_from_prev":   shift_detected,
        "signal_basis":      week_data["source_basis"],
        "key_signals":       bw_signals[:4],
        "predictions":       bw_preds[:2],
        "commentary":        signal_summary,
    }


# ------------------------------------------------------------------
# Step 4: Build wind_trend (cross-week history)
# ------------------------------------------------------------------
def build_wind_trend(history_layers, current_yw):
    current_dominant = None
    for yw, dl in history_layers:
        if yw == current_yw:
            current_dominant = dl
            break

    past = [(yw, dl) for yw, dl in history_layers
            if yw < current_yw and dl != "insufficient data"]
    past = past[-4:]

    streak = 0
    if current_dominant and current_dominant != "insufficient data":
        for yw, dl in reversed(past):
            if dl == current_dominant:
                streak += 1
            else:
                break

    return {
        "recent_dominants": [
            {"week": _week_label(yw[0], yw[1]), "dominant_layer": dl}
            for yw, dl in past
        ],
        "current_streak_weeks":
            streak + 1 if current_dominant and current_dominant != "insufficient data" else 0,
        "current_dominant": current_dominant or "insufficient data",
        "shift_detected":   (bool(past) and past[-1][1] != current_dominant
                             and current_dominant not in (None, "insufficient data")
                             and past[-1][1] != "insufficient data"),
    }


# ------------------------------------------------------------------
# Step 5: Update existing quality-review entries
# ------------------------------------------------------------------
def update_quality_reviews(weekly_layers, bw_entries):
    qr_data = _load(QR_PATH, {"quality_reviews": []})
    reviews  = qr_data.get("quality_reviews", [])
    reviews_sorted = sorted(reviews, key=lambda r: r.get("date",""))

    sorted_weeks   = sorted(weekly_layers.keys())
    history_layers = [(yw, weekly_layers[yw]["dominant_layer"]) for yw in sorted_weeks]

    updated = 0
    for r in reviews_sorted:
        if r.get("wind_direction"):
            print("  Skipping %s (wind_direction already present)" % r["date"])
            continue

        period = r.get("period", "")
        try:
            end_date = period.split(" to ")[-1].strip()
        except Exception:
            end_date = r["date"]

        yw = _iso_week(end_date) if end_date else _iso_week(r["date"])
        if yw == (0, 0):
            yw = _iso_week(r["date"])

        week_data = weekly_layers.get(yw)
        if not week_data:
            for delta in [(-1, 0), (0, -1), (0, 1), (1, 0)]:
                c = (yw[0] + delta[0], yw[1] + delta[1])
                if c in weekly_layers:
                    week_data = weekly_layers[c]
                    yw = c
                    break

        if not week_data:
            print("  No week data for %s" % r["date"])
            continue

        bw_sigs, bw_preds = signals_for_week(bw_entries, yw[0], yw[1])
        prev_yw  = (yw[0], yw[1] - 1)
        prev_dom = weekly_layers.get(prev_yw, {}).get("dominant_layer")

        r["wind_direction"] = build_wind_direction(yw, week_data, bw_sigs, bw_preds, prev_dom)
        r["wind_trend"]     = build_wind_trend(history_layers, yw)
        updated += 1
        print("  Updated %s (ISO %s, dominant: %s)" % (
            r["date"], _week_label(yw[0], yw[1]), r["wind_direction"]["dominant_layer"]))

    qr_data["quality_reviews"] = sorted(reviews_sorted, key=lambda r: r.get("date",""), reverse=True)
    _save(QR_PATH, qr_data)
    print("  %d existing entries updated" % updated)
    return updated


# ------------------------------------------------------------------
# Step 6: Add synthetic historical entries for uncovered weeks
# ------------------------------------------------------------------
def add_historical_entries(weekly_layers, bw_entries):
    qr_data = _load(QR_PATH, {"quality_reviews": []})
    reviews  = qr_data.get("quality_reviews", [])

    sorted_weeks   = sorted(weekly_layers.keys())
    history_layers = [(yw, weekly_layers[yw]["dominant_layer"]) for yw in sorted_weeks]

    today      = _dt.date.today()
    current_yw = today.isocalendar()[:2]

    added = 0
    for yw in sorted_weeks:
        if yw >= current_yw:
            continue

        mon_str, sun_str = _week_to_date_range(yw[0], yw[1])

        # Skip if already covered
        covered = False
        for r in reviews:
            ps = r.get("period","").split(" to ")[0].strip()
            pe = r.get("period","").split(" to ")[-1].strip()
            if ps <= sun_str and pe >= mon_str:
                covered = True
                break
        if covered:
            print("  %s already covered" % _week_label(yw[0], yw[1]))
            continue

        week_data = weekly_layers[yw]
        if week_data["total_papers"] < 3:
            print("  Skipping %s (only %d papers)" % (_week_label(yw[0], yw[1]), week_data["total_papers"]))
            continue

        bw_sigs, bw_preds = signals_for_week(bw_entries, yw[0], yw[1])
        prev_yw  = (yw[0], yw[1] - 1)
        prev_dom = weekly_layers.get(prev_yw, {}).get("dominant_layer")

        wind_dir   = build_wind_direction(yw, week_data, bw_sigs, bw_preds, prev_dom)
        wind_trend = build_wind_trend(history_layers, yw)

        entry = {
            "date":           sun_str,
            "period":         "%s to %s" % (mon_str, sun_str),
            "note":           "backfill: synthesized from local hotspot titles + biweekly signals",
            "overall":        None,
            "wind_direction": wind_dir,
            "wind_trend":     wind_trend,
        }
        reviews.append(entry)
        added += 1
        print("  Added synthetic entry for %s (dominant: %s, papers: %d)" % (
            _week_label(yw[0], yw[1]), wind_dir["dominant_layer"], week_data["total_papers"]))

    qr_data["quality_reviews"] = sorted(reviews, key=lambda r: r.get("date",""), reverse=True)
    _save(QR_PATH, qr_data)
    print("  %d synthetic entries added" % added)
    return added


# ------------------------------------------------------------------
# Step 7: Update vla-trend-snapshot.json with all-paper classification
# ------------------------------------------------------------------
def update_snapshot(weekly_layers):
    existing = _load(SNAPSHOT, {})
    existing_weeks = {w.get("week"): w for w in (existing.get("weeks") or [])}

    weeks_out = []
    for yw in sorted(weekly_layers.keys()):
        wl  = weekly_layers[yw]
        key = wl["week"]
        ex  = existing_weeks.get(key, {})

        # Prefer existing if it has real hot-rated data
        if ex.get("rated_papers", 0) > 0:
            if "all_paper_layer_distribution" not in ex:
                ex["all_paper_layer_distribution"] = wl["layer_distribution"]
            weeks_out.append(ex)
        else:
            weeks_out.append({
                "week":               wl["week"],
                "year":               yw[0],
                "iso_week":           yw[1],
                "total_papers":       wl["total_papers"],
                "rated_papers":       wl["rated_papers"],
                "hot_papers":         wl["hot_papers"],
                "layer_distribution": wl["layer_distribution"],
                "dominant_layer":     wl["dominant_layer"],
                "source_basis":       wl["source_basis"],
                "top_affiliations":   wl["top_affiliations"],
                "highlight_titles":   wl["highlight_titles"],
            })

    snapshot = {
        "generated_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_weeks":  len(weeks_out),
        "rated_weeks":  sum(1 for w in weeks_out if w.get("hot_papers", w.get("rated_papers", 0)) > 0),
        "backfilled":   True,
        "weeks":        weeks_out,
    }
    _save(SNAPSHOT, snapshot)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    print("=== VLA History Backfill ===")
    print()

    print("[1] Classifying all papers by week (including unrated)...")
    weekly_layers = build_weekly_layers()
    for yw, data in sorted(weekly_layers.items()):
        print("  %s: %d papers total, %d rated, %d hot | dominant=%s (%s)" % (
            data["week"], data["total_papers"], data["rated_papers"], data["hot_papers"],
            data["dominant_layer"], data["source_basis"]))
    print()

    print("[2] Loading biweekly signals...")
    bw_entries = load_biweekly_signals()
    for e in bw_entries:
        print("  period ending %s: %d signals" % (e["period_end"], len(e["key_signals"])))
    print()

    print("[3] Updating existing quality-review entries with wind_direction...")
    update_quality_reviews(weekly_layers, bw_entries)
    print()

    print("[4] Adding synthetic historical entries for uncovered weeks...")
    add_historical_entries(weekly_layers, bw_entries)
    print()

    print("[5] Updating vla-trend-snapshot.json with all-paper classification...")
    update_snapshot(weekly_layers)
    print()

    print("=== Done ===")


if __name__ == "__main__":
    main()
