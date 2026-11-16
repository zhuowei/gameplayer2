Please control the running Grand Theft Auto 5 window on macOS and play through the game until you reach the end credits (or until I stop you).

Setup & Persistence (Do This First):
- Initialize (or use) the local git repo with a `.gitignore` excluding videos, screenshots, and large model/cache files.
- Write our core operating rules (black-box boundary, input safety, stall thresholds, and continuous self-improvement) into `AGENTS.md` at the repo root so they persist across context compactions, and maintain clear skill/`TOOLS.md` docs for subagents.
- Maintain a structured `STATUS.md` scoreboard tracking: current mission/objective, consecutive failures, minutes on current checkpoint, minutes since last commit, and a `Problem -> Code/Skill Change -> Verified Outcome` table.

Boundaries & Requirements:
- Treat the game strictly as a black box: interact only via window/screen capture and synthetic OS inputs (do not read or modify game files, Wine state, or process memory).
- Play legitimately from start to finish: do not skip missions or parts/checkpoints of missions, and do not use cheats or external saves.
- Ensure all input tools automatically release held keys/buttons on exit, error, or timeout so inputs never stick at the OS level.

Subagent & Monitoring:
- Use a gameplay subagent (with command execution enabled) to play using fast local tools and single screenshots—keep video recording out of the gameplay subagent's loop, and rotate in a fresh gameplay subagent if its context grows too long.
- Schedule a recurring 3-minute check whose wakeup prompt explicitly includes both monitoring AND the self-improvement check below. Capture downscaled window video for each 3-minute period, delegate full-clip + telemetry triage to a fresh subagent per check (so raw video doesn't fill your main context), and inspect short 10–15s trimmed clips yourself when debugging failures.
- Have the gameplay subagent notify you immediately on death/fail, mission milestones, >2 stuck turns, or unresponsive inputs/camera.

Continuous Self-Improvement:
- Actively improve BOTH your runtime/tools (closed-loop code for fast perception, aiming, combat, driving, navigation, UI/prompt reading, and surviving LLM thinking gaps so the subagent specifies high-level targets/goals rather than raw input timings) AND your skills/subagent instructions/docs (tactical heuristics, tool-usage rules, and failure lessons)—never relying on prompt advice alone when code should handle it.
- Use established third-party libraries, native macOS frameworks, or local pretrained models where appropriate rather than reinventing brittle pixel heuristics from scratch.
- Build and unit-test improvements in parallel while the subagent plays, targeting both failures and slow/repetitive manual actions. Whenever `consecutive_failures >= 2`, a checkpoint stalls for >= 6 minutes, or > 10 minutes pass with unaddressed manual friction, ship and test an upgrade, update the running tools and subagent, `git commit`, and `git push` to origin.
