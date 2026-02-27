# Scripts Reference

> Last updated: 2026-02-27

## Naming Conventions

| Prefix/Pattern | Role |
|---|---|
| `vla-*` | VLA pipeline (robotics/embodied AI) |
| `ai-app-*` or `ai-*` | AI agent / app pipeline |
| `prep-<pipeline>.py` | Phase 1: collect + filter candidates |
| `run-<pipeline>-two-phase.py` | Phase 2: LLM agent runner (two-phase) |
| `post-<pipeline>.py` | Phase 3: deterministic post-processor / TG push |
| `_*.py` | Private shared module (imported by other scripts) |
| `gh-*.py` | GitHub Contents API operations |

---

## VLA Pipeline

```
vla-rss-collect.py          → vla-rss-YYYY-MM-DD.json
rate-vla-daily.py           → vla-daily-rating-in/out-YYYY-MM-DD.json (Phase 1.5)
run-vla-daily-two-phase.py  → vla-daily-hotspots.json + TG push

prep-vla-theory.py  → agent → post-vla-theory.py   (theory dissection)
prep-vla-social.py  → run-vla-social-two-phase.py → post-vla-social.py
prep-vla-sota.py    → run-vla-sota-two-phase.py   → post-vla-sota.py
prep-vla-release.py → run-vla-release-two-phase.py → post-vla-release.py
prep-vla-weekly.py  → run-vla-weekly-two-phase.py → post-vla-weekly.py

vla-daily-rerate-push.py    (re-push with ratings after keyword-fallback day)
vla-trend-snapshot.py       (weekly trend snapshot for quality review)
backfill-vla-history.py     (one-off historical backfill utility)
_vla_expert.py              (shared: get_api_key, call_qwen, fetch_handbook_context)
```

## AI Agent Pipeline

```
ai-app-rss-collect.py            → ai-app-rss-YYYY-MM-DD.json
ai-daily-pick-collect.py         → ai-daily-pick-sources-YYYY-MM-DD.json

prep-ai-app-rss-filtered.py → write-ai-app-daily.py   (tool news / 日报)
prep-ai-app-social.py       → run-ai-app-social-two-phase.py  (社交情报)
prep-ai-app-workflow.py     → run-ai-app-workflow-two-phase.py  (工作流灵感)
prep-ai-deep-dive.py        → agent → post-ai-deep-dive.py  (深度解析)
prep-ai-weekly.py           → run-ai-weekly-two-phase.py → post-ai-weekly.py

post-ai-app-daily.py        (deterministic daily stats → ai-app-daily-stats.json)
prep-ai-app-dedup.py        (shared dedup helper)
```

## Calibration and Quality

```
prep-calibration-check.py   (daily 11:00 assumption scan)
monthly-calibration-agg.py  (monthly aggregation + confidence update)
```

## Watchdog and System Health

```
daily-watchdog.py            (v6: 15 checks, self-healing, lockfile)
emit-system-health.py        (emit system-health.json)
evaluate-shadow-config.py    (A/B shadow config evaluator)
gateway-preflight.py         (gateway connectivity check)
server-health-check.sh       (shell-level health check)
system-moltbot-monitor.sh    (moltbot process monitor)
```

## Memory Management

```
memory-janitor.py    (age-out old records from JSON stores)
memory-snapshot.py   (daily tar.gz snapshot → snapshots/)
memory-upsert.py     (atomic upsert helper)
```

## GitHub Operations

```
gh-contents-upload.py                   (generic GitHub Contents API PUT)
gh-app-index-update.py                  (update AI app index on GitHub)
gh-paper-index-update.py                (update VLA paper index on GitHub)
gh-handbook-changes-collect.py          (collect VLA-Handbook git changes)
gh-agent-library-blog-readme-patch.py   (patch Agent-Playbook README)
```

## Shared Helpers

```
_heartbeat_run.py                    (heartbeat keepalive for long-running agents)
_paper_index_input_from_hotspots.py  (extract paper index input from hotspots)
```

---

## Known Naming Exceptions

- `write-ai-app-daily.py` — writes to ai-app-daily.json; named "write" not "post" because
  it is a pure atomic writer with no agent dependency. Referenced in jobs.json by this name.
- `rate-vla-daily.py` — Phase 1.5 between RSS collect and two-phase runner; no prep/run/post
  framing because it runs inline before the LLM agent step.
- `vla-daily-rerate-push.py` — special re-push called by watchdog on keyword-fallback days.
