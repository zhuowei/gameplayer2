# gameplayer

Black-box macOS agent runtime and multi-agent Swarm setup for playing **Grand Theft Auto V** (via Game Porting Toolkit / Wine) using clean `16:9` Quartz window capture (`CGWindowListCreateImage`), native macOS Vision OCR (`VNRecognizeTextRequest`), minimap/reticle perception, and fail-safe synthetic `CGEvent` HID inputs.

## Key Documents

- [LESSONS_LEARNED.md](LESSONS_LEARNED.md) — Engineering retrospective from the ~9.5-hour live Swarm run: quantitative scorecard, what worked (`16:9` capture, zero-leak inputs, 8–12 Hz closed loops, `Carbon` root-cause debugging), what failed (2D minimap aiming without 3D sightline awareness, ADS/hipfire sensitivity jumps, offline mock limits), and next-iteration architecture.
- [AGENTS.md](AGENTS.md) — Strict Swarm role ownership (`player`, `toolsmith`, `verifier`, `watcher`, `monitor`), black-box boundaries, input safety guarantees, and gameplay heuristics.
- [TOOLS.md](TOOLS.md) — CLI and Python API documentation for `gameplayer.{window,input,telemetry,capture_daemon,hud,minimap,runner}`.
- [STATUS.md](STATUS.md) — Chronological 35-cycle self-improvement log and final stopped checkpoint status.

## Quick Verification (Offline)

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/pytest -v
```
