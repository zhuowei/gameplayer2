# Lessons Learned — GTA V Story Mode Swarm Retrospective

Engineering retrospective from running an autonomous multi-agent Swarm to play **Grand Theft Auto V (Story Mode)** on macOS (Game Porting Toolkit / Wine) strictly as a black box.

---

## 1. Run Overview & Quantitative Scorecard

- **Control Paradigm**: Black-box only — unobstructed `16:9` Quartz window capture (`CGWindowListCopyWindowInfo` / `CGWindowListCreateImage`) and synthetic OS HID inputs (`CGEvent`), with zero access to game files, Wine state/registry, save files, or process memory.
- **Session Window**: `2026-11-17T06:21:18Z` – `2026-11-17T15:59:42Z` (~9h 38m wall-clock, including a manual pause while the user switched GTA V's in-game mouse setting from `"Raw Input"` to `"Windows"`).
- **Codebase & Verification**:
  - `40` commits before this retrospective (`06ad9e3`..`a216bdc`) across `35` logged self-improvement cycles in [STATUS.md](STATUS.md).
  - `7,940` total Python LOC: `4,848` runtime LOC across [gameplayer/](gameplayer/) (`window.py`: 609, `input.py`: 918, `runner.py`: 2,045, `capture_daemon.py`: 491, `minimap.py`: 327, `hud.py`: 317, `telemetry.py`: 130, `__init__.py`: 11) + `3,092` test LOC across `94` offline unit tests in [tests/](tests/).
- **Live Telemetry ([logs/telemetry.jsonl](logs/telemetry.jsonl))**:
  - `316,602` logged events total: `133,205` `window.capture`, `73,808` `input.mouse_delta`, `66,532` `minimap.inspect`, `32,554` `hud.inspect_frame`, `5,683` `runner.fire_burst` (`363` with `red_reticle`), `1,106` `input.release_all`, `478` `window.focus`, `288` `runner.on_foot_stuck_recovery`, `219` `runner.cover_peek_combat`, `122` `runner.advance_under_fire`, `114` `runner.advance_recover`, `183` completed 3-minute video segments, and `47` `banner:MISSION_FAILED` events.
  - `0` unhandled key/button leaks or stuck OS inputs across the entire run.
- **Gameplay Progression**:
  - **Cleared**: Prologue bank interior (hostage room control, vault detonation, character switch via `lalt`/`f7` to Trevor, precision hostage headshot via `hostage_shot`, and rolling shutter breach).
  - **Reached Repeatedly**: Outdoor 5-star North Yankton depot yard -> security gate -> snowy barn road shootout (cleared the yard wave `red=0` on multiple runs, rejoined Michael & Brad, and increased single-attempt survival from `<15s` early on to `285s`–`335s`).
  - **Stall Point**: Did not reliably finish pushing down the outside barn road to enter the getaway car before user stop.

---

## 2. What Went Right

### A. Non-Intrusive `16:9` Quartz Capture & Reliable Wine Focus Recovery
- `CGWindowListCreateImage` (`kCGWindowImageBoundsIgnoreFraming` + deterministic `16:9` titlebar/center crop in [gameplayer/window.py](gameplayer/window.py)) allowed `player` and `watcher` (`capture_daemon`) to capture clean `1280x720` and `640x360` frames concurrently at ~8–12 Hz without disrupting focus.
- Upgrading `focus_game_window` (`359eb68`) to combine Accessibility (`AXRaise` + `AXFrontmost`), Carbon `SetFrontProcess`, `NSRunningApplication.activateWithOptions_`, `System Events` fallback, `NSRunLoop` pumping, and an `obstructed_by == ()` check resolved cases where `GTA5.exe` reported focused metadata while hidden behind `OBS Studio`.
- Anchoring `CGEvent` cursor warps to `WindowBounds.center` (`28c7eaf`), passing `kCGEventSourceStateHIDSystemState` (`359eb68`), and disabling macOS's `0.25s` post-warp event suppression (`LocalEventsSuppressionInterval=0.0` in `9998ca2`) stopped cursor coordinates from drifting off-display (`x=22032` in `1d5973f`) and fixed stalled `10–12 Hz` camera deltas.

### B. Zero-Leak Synthetic Input Safety
- Every input primitive in [gameplayer/input.py](gameplayer/input.py) enforces fail-safe cleanup via `try/finally`, `with InputController(...)` context exit, per-key fault-tolerant `release_all()`, `atexit`, `SIGINT`/`SIGTERM` chaining, and `10.0s` `threading.Timer` watchdogs.
- Strict explicit opt-ins (`--live`, `allow_focus=True`) prevented `94` offline unit tests and `verifier` QA runs from ever focusing or sending inputs to the desktop or live game window.

### C. Moving Control From Multi-Second LLM Turns to ~8–12 Hz Local Loops
- Early open-loop turns died in `5–15s` LLM round-trip gaps. Moving tactical state machines (`cover_peek_combat`, `advance_under_fire`, `follow_route_on_foot`, `follow_blue_ally`, `follow_gps_drive`) into [gameplayer/runner.py](gameplayer/runner.py) brought survival up to `4.5–5.5 minutes` per attempt (`285s` in `#226`, `335s` in `#239`).
- Native macOS Vision OCR (`VNRecognizeTextRequest`, fast path in [gameplayer/hud.py](gameplayer/hud.py)) reliably extracted `WASTED`/`BUSTED`/`MISSION_FAILED`/`MISSION_PASSED` banners, tutorial prompts, and crew-tether warnings (`"The crew were left behind."` in `3ae89fe`) locally in milliseconds.

### D. Swarm Isolation, Worktree Gate, and Atomic Video Slicing
- Strict ownership prevented collisions: `player` was the sole writer to the live GTA window, `toolsmith` worked in an isolated `workspace: "share"` worktree, `verifier` gated every merge into `main` on offline `pytest`, and `watcher` kept rolling capture independent (`capture_daemon` wrote `.in_progress_seg_*.mp4` and atomically renamed to `seg_*.mp4` after `ffmpeg +faststart`).
- Extracting short `10–15s` slices (`gameplayer.capture_daemon slice`) around telemetry anomalies (`low_health`, `MISSION_FAILED`, `advance_flank_unblock`) gave engineering workers direct frame-by-frame evidence without flooding swarm context windows with 3-minute recordings.

### E. High-Signal Root-Cause Escalation (`Carbon` Specialists)
Escalating deep perception and control failures to `Carbon` specialists produced the biggest measurable jumps in performance:
1. **Interior-Row Minimap Calibration & Siren Rejection (`545f985`, `f2059e2`)**: Fixed empty gray health tracks reading `1.000` HP down to ~26% HP and filtered 5-star magenta-red/blue flashing siren washes (`rgb=(191, 51, 71)`) that generated ghost enemy blips (`bearing=+0.76`).
2. **`step=1` Blip Segmentation (`ee53574`)**: Replaced `5x5` minimap dilation with `step=1` connected-component labeling for `red`/`blue` blips so clusters of 3–4 cops stopped merging into a single empty gap centroid (`2.6%` `red_reticle`).
3. **ADS vs. Hipfire `_recenter_sweep` Ordering (`c93b2b3`)**: Fixed `_run_loop` `finally` ordering via `on_before_release`, because releasing `right` mouse button *before* `_recenter_sweep()` caused downward sweep reset deltas to run at `~3x` hipfire sensitivity, slamming Trevor's camera `45°` into the ground/car roof on every timeout/low-HP exit.
4. **Horizon Sweep + Phantom-Peek Detection + Empirical Recoil (`bc0de07`, `a216bdc`)**:
   - Replaced a `92%` sky-biased sweep (`cum_y = -34..-72`) with a 14-step horizon-centered bidirectional sweep.
   - Added `ReticleStatus.is_visible` (`RGB(224,224,224)` Simple white dot detection) in [gameplayer/runner.py](gameplayer/runner.py) to catch "phantom peeks" behind wide concrete pillars where cover held `right` for `35s` with `0` reticle drawn and `0` bullets fired (`cover_edge_shift` / `"cover_blocked"`).
   - Empirically calibrated `0.75 px/round` downward recoil compensation across both `red_reticle` (`+2px` per 3-round burst) and `suppressive` bursts (`+1px`/`+2px` on ticks `6`/`12`), raising `cover_peek_combat` `red_reticle` lock rate from **`0%–3.9%` (`0/125`, `2/51`) to `27.0%` (`17/63`)** in retry 33 (`#263`).

---

## 3. What Did Not Work (Root Causes & Failure Modes)

### A. 2D Minimap Bearing Steering Without 3D Screen-Space Vision (Primary Gameplay Blocker)
- Minimap blips only tell us 2D horizontal map direction—not **elevation/pitch**, **3D line-of-sight occlusion** (concrete pillars, Bobcat truck, police cruiser doors, wooden crates), or **exact ADS yaw-to-pixel gain**.
- Searching blindly around minimap bearing via `DEFAULT_RETICLE_SWEEP` + a `3x3` center `detect_red_reticle` check failed whenever cops crouched behind vehicles or terrain differed from cover horizon. Even in the final run (`a216bdc`, `clips/yard_flank_stall_1551.mp4`), while `cover_peek_combat` hit `27.0%`, open `advance_under_fire` got only **`2.0%` (`1/50`) `red_reticle`** because Trevor stood inside the depot bay aiming down the empty snowy gap between police cruisers.
- **Unresolved Timing Bug in `advance_flank_unblock` ([gameplayer/runner.py](gameplayer/runner.py#L1031-L1048))**: Inspecting the post-mortem code shows `flank_unblock_until_t = now_t + 0.75 + flank_move_s` computes the deadline from `now_t` captured *before* `self._sleep_chunked(0.20)` (`"s"`) and `self._sleep_chunked(0.55)` (`flank_key`). Because `0.75s` elapses during those two sleeps, `flank_unblock_until_t` expires almost immediately in the subsequent `move` phase, letting `hold_engage_threat` re-trigger `engage` before Trevor finishes walking around the pillar/truck.

### B. Wine Input Modes & Sensitivity Asymmetry
- GTA V's default `"Raw Input"` mouse setting ignored synthetic `CGEvent` mouse deltas completely until the user manually switched GTA V in-game to `"Windows"` mouse mode.
- Furthermore, GTA V's `MouseMoved` (hipfire) sensitivity (`~2.5–3x` higher) differs sharply from `RightMouseDragged` (ADS) sensitivity, and Wine drops/reorders `mouse_delta` if `mouse_up("right")` occurs in the same frame (`0.02s` settle added in `bc0de07`). Mixing uncalibrated deltas across button states repeatedly broke camera pitch.

### C. Reactive Parameter Whack-a-Mole in a Monolithic `runner.py`
- [gameplayer/runner.py](gameplayer/runner.py) grew to `2,045` lines with dozens of interacting scalar heuristics (`recoil_comp_dy`, `peek_burst_s`, `max_engage_hold_s`, `recover_health`, `recover_target_health`, `exposed_duck_drop_threshold`). Fixing one edge case frequently shifted another:
  - Recoil compensation swung from `0` (`+76px` sky climb) -> `1` on `red_reticle` only -> `2` (`2.0 px/round`, `+404px` downward snow dive in `seg_0032`) -> empirical `0.75 px/round` (`a216bdc`).
  - `recover_target_health=0.50` (`545f985`) looped 21 times because GTA V's Prologue passive health regeneration caps around `~0.45–0.48` HP (`ed610d5` lowered it to `0.45`).

### D. Blind Offline Unit Mocks Missed Closed-Loop Physics & Clock Semantics
- All `94` tests passed in pytest using static/scripted `FrameState` sequences and no-op `sleep_fn` stubs (`FakeClock` did not advance during `_sleep_chunked()`).
- Consequently, multi-second timing bugs (like the `now_t` vs. `self._clock()` bug in `advance_flank_unblock`, camera pitch ratcheting, recoil-vs-sensitivity mismatches, and cover occlusion) only surfaced during `5–8 minute` live game runs.

### E. Stateful Toggle Blindness (`q` Cover) & Fragile Inline Player Scripts
- GTA V's `q` cover key is a toggle with no minimap bit. Despite `enter_cover=False` flags and `"low_health"` warnings, multi-step inline scripts run by `player` occasionally toggled *out* of cover at `0.09–0.24` HP (`#230`, `#254`, `#262`) or chained into `advance_under_fire` / `follow_blue_ally` while critically wounded (`7s` later `MISSION_FAILED`).

### F. Swarm Operational Friction
- **Git Worktree Dirty-File Drift**: `watcher` updated `STATUS.md` directly in `main` while `toolsmith` committed in a separate `share` worktree, leaving `STATUS.md` unstaged/modified on `main` across merges.
- **Role Tool Mismatch & Model Capacity Churn**: `watcher` lacked `invoke_subagent` in its declared toolset (`#241`), falling back to telemetry + direct `15s` slices. Repeated `503 No capacity available` errors on `gemini-4-hydrogen-medium`, `gemini-4-hydrogen-high`, and `gemini-4-carbon-medium` also interrupted long worker turns until switched (`argon-medium` / `carbon-high`).

---

## 4. Architecture Recommendations for Next Iteration

1. **Screen-Space Visual Target & Horizon Estimation**:
   - Add a lightweight local detector (e.g., CoreML/ONNX person/head/muzzle-flash detector or cropped temporal motion/silhouette tracker seeded by minimap `bearing`) plus visual horizon estimation on `1280x720` frames so `(dx, dy)` locks onto visible body/head targets directly rather than sweeping blindly at `3x3` reticle pixels.
2. **Closed-Loop Kinematic & Advancing-Clock Simulator (`tests/sim_harness.py`)**:
   - Build a lightweight offline 3D-projected / 2D+pitch simulator where `sleep_fn` advances `clock()`, ADS (`RightMouseDragged`) vs. hipfire (`MouseMoved`) have realistic `1:2.8` sensitivity ratios, gunfire applies `+0.8 px/round` upward recoil, health caps at `0.48` regen, and rectangular occluders block LoS + hide the `is_visible` reticle dot. Run A/B parameter benchmarks offline in `<1s` before touching the live game.
3. **Visual Posture / Cover-State Classifier & Re-usable Mission Playbooks**:
   - Classify crouch-cover vs. standing vs. behind-vehicle posture directly from pixels so `enter_cover()` / `exit_cover()` are idempotent, and promote repeatable mission sequences out of ad-hoc inline `player` scripts into versioned, simulator-tested playbooks in [gameplayer/](gameplayer/).
4. **Decouple Live Telemetry/Scoreboard From Git-Tracked Files**:
   - Keep `STATUS.md` updates committed or write machine-updated scoreboard snapshots to `logs/status.json` (`.gitignore`d) so `toolsmith` worktree merges never collide with `watcher` monitoring updates.

---

## 5. Checkpoint 1 (2026-11-20 23:32 PST)

- **Progression & Perception (`410ee3f`, `f87c9a2`, Cycle 3)**:
  - Preserving native `0x20000000` `CGEventGetFlags` bits + `kCGEventFlagMaskNumericPad | kCGEventFlagMaskSecondaryFn` on arrow/fn keys (`gameplayer/input.py`) and activating via `click("left")` instead of `right` (which aims/cancels the phone) unblocked the Prologue bank vault phone detonation prompt (`player-4` reached Vault / Hallway Escort).
  - Offline `gameplayer.replay_bench` across saved failure clips (`clips/yard_zero_red_1411.mp4`, `clips/yard_flank_stall_1551.mp4`, `clips/yard_low_red_1502.mp4`) verified `VisionHumanDetectorBackend` runs in `~11.5–23.3 ms/frame` post-warmup and locks **`57%–60%` (`17–18/30`, `6/10`)** of frames onto visible enemies where blind `red_reticle` had `0%`.
  - **Cycle 1 (`410ee3f`)**: Fixed `advance_flank_unblock` post-sleep timing (`_elapsed_now()`), added advancing-clock 2.5D simulator (`tests/sim_harness.py` with `0.75 px/rd` recoil, `1:2.8` ADS/hipfire sensitivity, `0.48` health regen cap, and box occluders), `Vision` + silhouette visual target steering, and FFT phase-correlation `CameraMotionTracker`.
  - **Cycle 2 (`f87c9a2`)**: Added `classify_posture` + idempotent `enter_cover()`/`exit_cover(min_health=0.42)`, `<=5s` `calibrate_controls()`, `gameplayer/replay_bench.py`, simulator-tested `gameplayer/playbooks.py`, and gitignored `logs/status.json` (`gameplayer/status_store.py`).
  - **Cycle 3**: Fixed `follow_route_on_foot` indoor wall-grinding (`abs(last_bearing) > 0.35` now releases `w`/`lshift`, backs off `"s"` `0.25s`, skips off-axis wall-vaulting, and steers aggressively toward `last_bearing`).
