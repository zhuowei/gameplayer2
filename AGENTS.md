# AGENTS.md — Swarm & Gameplay Rules

This file captures the non-negotiable operational rules for playing Grand Theft Auto V (macOS / Game Porting Toolkit) legitimately through Story Mode to end credits. Keep it accurate and concise so all rules survive context compactions.

## 1. Strict Role Ownership & Workspace Isolation

- **`player` (Live Gameplay)**:
  - **ONLY `player`** may focus or send inputs to the live GTA 5 game window.
  - `player` uses tools and **never edits code**.
  - `player` posts immediately on death/mission fail, mission milestones, `>2` stuck turns, or unresponsive inputs/camera.
  - Maximize progress per LLM turn by using closed-loop runtime tools rather than single-step micro-turns.
- **`toolsmith` (Engineering)**:
  - Develops exclusively in an isolated `workspace: "share"` copy of `/Users/zhuowei/Documents/proj/gameplayer`.
  - **NEVER** focuses or sends inputs to the live GTA 5 game window.
  - Merges into `/Users/zhuowei/Documents/proj/gameplayer` (`main`) **ONLY after `verifier` posts `PASS`** on offline tests.
- **`verifier` (Offline QA)**:
  - Tests **offline only** against `toolsmith`'s share workspace/branch using unit tests, mock/replay harnesses, and saved failure fixtures.
  - **NEVER** focuses or sends inputs to the live GTA 5 game window, and **NEVER** edits code itself.
- **`watcher` (Telemetry & Video Review)**:
  - Runs the rolling downscaled 16:9 window capture daemon (`gameplayer.capture_daemon`) and a recurring 3-minute `schedule` (restarting either if stopped, with the self-improvement check embedded directly in the wakeup prompt).
  - On each check, delegates reading the latest 3-minute raw clip + `logs/telemetry.jsonl` to a fresh disposable `invoke_subagent` (`gemini-4-hydrogen-medium`) to keep raw video out of Swarm contexts.
  - Extracts short 10–15s slices only when debugging specific failures.
  - Updates `STATUS.md` and alerts `@coordinator` / `@toolsmith` on stalls, consecutive failures, or high turn friction.
  - **NEVER** focuses or sends inputs to the live GTA 5 game window.
- **`monitor`**:
  - Swarm stall monitor (`swarm_monitor` skill); posts only to break a stall.

## 2. Black-Box Only Boundaries

- Interact strictly via clean, unobstructed `16:9` window capture (`CGWindowListCopyWindowInfo` / `CGWindowListCreateImage`) and synthetic OS inputs (`CGEvent`).
- **NEVER** read or modify game files, Wine state, Wine prefix/registry, save files, or process memory.
- Enable in-game subtitles in GTA 5 settings early on if operating visually without audio.

## 3. Legitimate Play

- Play Story Mode legitimately until end credits (or user stop).
- **NO** mission or checkpoint/section skipping.
- **NO** cheats, mods, trainers, or external save files.
- Normal in-game saves/autosaves and restarting/refocusing the game window on crash or focus-loss are allowed.

## 4. Input Safety Guarantees

- Every input tool **MUST** automatically release **ALL** held keys and mouse buttons on:
  - Normal completion / context exit (`try/finally`)
  - Any exception or error
  - Process exit (`atexit`)
  - Termination/interrupt signals (`SIGINT`, `SIGTERM`)
  - Hard timeout watchdog expiration
- Any code or CLI path capable of sending live OS inputs must require explicit opt-in (`--live` or live `CGEventInputBackend`) and never execute accidentally during offline tests.

## 5. Pause Policy

- Play primarily in continuous real time via closed-loop tools rather than pause-stepping individual moves.
- Pause only strategically when waiting on tool engineering or diagnosing a novel hazard.

## 6. Model Tiering & Quota Efficiency

- Default workers (`player`, `watcher`, `verifier`, routine `toolsmith`, and child subagents) to **Gemini 4 Hydrogen (Medium)** (or **Hydrogen (High)** for harder coding).
- Escalate to **Gemini 4 Carbon (Medium)** strictly for tough architectural, perception, or root-cause problems that fail on Hydrogen—then switch back to **Hydrogen (Medium)** (or kill a one-shot specialist) once solved.
- Switch models if quota runs low, and rotate `player` or `watcher` cleanly if context grows long.

## 7. Continuous Self-Improvement & Mandatory Upgrade Trigger

- Continuously improve both:
  1. Closed-loop runtime tools (movement, aiming/combat, driving, navigation, OCR/UI reading, turn-gap survival)
  2. Persistent docs/skills (`AGENTS.md`, `TOOLS.md`) capturing tactical heuristics and failure lessons.
- **Mandatory Upgrade Trigger**: Whenever any of these occur:
  - `consecutive_failures >= 2`
  - Checkpoint stall `>= 6 minutes`
  - High manual turn friction
  `toolsmith` + `verifier` must land a verified upgrade into `/Users/zhuowei/Documents/proj/gameplayer`, `git commit` (and `git push` if remote exists), and notify `@player`.
- Scope changes incrementally; verify every change offline with unit tests, saved regression fixtures, and mock/replay harnesses before merging.

## 8. Swarm Board Discipline

- **NEVER** use `@all` unless genuinely necessary for the whole swarm.
- Wake only the next acting worker(s) by `@name`, referencing board sequence numbers (`#seq`).
- Keep posts concise and actionable (results, blockers, exact paths/SHAs/commands). Do not post greetings or empty acknowledgements.
