# gameplayer

Black-box macOS agent runtime and multi-agent Swarm setup for playing **Grand Theft Auto V** (via Game Porting Toolkit / Wine) using clean `16:9` Quartz window capture (`CGWindowListCreateImage`), native macOS Vision OCR (`VNRecognizeTextRequest`), minimap/reticle perception, and fail-safe synthetic `CGEvent` HID inputs.

## Key Documents

- [LESSONS_LEARNED.md](LESSONS_LEARNED.md) — Engineering retrospective across Session 1 (`2026-11-17`) and Session 2 (`2026-11-20`–`21`, **Prologue Complete -> *"Franklin and Lamar"***): quantitative scorecard (`23.2%` overall / `56.4%` peak `red_reticle`, `77%` fewer mission resets, `132` offline tests), Checkpoints 1–12, root-cause findings (macOS `Fn+Q` flag leak, Wine keyboard-loop wedge, silhouette noise, stall-clock reset), and next-iteration architecture.
- [AGENTS.md](AGENTS.md) — Strict Swarm role ownership (`player`, `toolsmith`, `verifier`, `watcher`, `monitor`), black-box boundaries, input safety guarantees, and gameplay heuristics.
- [TOOLS.md](TOOLS.md) — CLI and Python API documentation for `gameplayer.{window,input,telemetry,capture_daemon,hud,minimap,perception,playbooks,replay_bench,runner,status_store}`.
- [STATUS.md](STATUS.md) — Chronological 56-cycle self-improvement log (35 Session 1 + 21 Session 2 cycles) and stopped checkpoint status.

## Quick Verification (Offline)

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/pytest -v
```
