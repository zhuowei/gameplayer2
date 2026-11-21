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
  - **Cycle 3 (`176774f`, `6982ad6`)**: Fixed `follow_route_on_foot` indoor wall-grinding (`abs(last_bearing) > 0.35` now releases `w`/`lshift`, backs off `"s"` `0.25s`, skips off-axis wall-vaulting, and steers aggressively toward `last_bearing`; lowered default `stuck_window_s=0.9`).

---

## 6. Checkpoint 2 (2026-11-21 00:02 PST)

- **Progression & Telemetry (`faed7a6`, Cycle 5)**:
  - `player-4` collected the `$177,000` vault cash (`#292`) and cleared the Prologue security-guard hostage ambush via `reticle_shot` headshot (`#298`) after two timing retries on the Trevor wheel switch (`#296`–`#297`).
  - **Cycle 4 (`faed7a6`)**: Added Bug-algorithm `bypass_bearing_until_t` (`0.75s` post-sidestep) + immediate `"s"` back-off on all `follow_route_on_foot` stuck recoveries so indoor walls are bypassed instead of immediately turning back into the same wall.
  - **Cycle 5 (`retro2_check.png`, `#299`)**: After the hostage shot (`08:00:07–08:01:17Z`), `11` `on_foot_stuck_recovery` calls oscillated `d, a, d, a` + `vault=True` against the narrow left steel vault doorway jamb because `0.55s` lateral hops overshot the doorway and alternating every step bounced jamb-to-door. Updated `_perform_on_foot_sidestep` / `follow_route_on_foot` to commit `2` gentle steps per side (`0.28s`, `0.42s`) before flipping (`((attempt - 1) // 2) % 2`), skip `vault` on doorway jamb retries (`attempt >= 2`), and extend `bypass_bearing_until_t` to `1.1s`.

---

## 7. Checkpoint 3 (2026-11-21 00:32 PST)

- **Progression & Telemetry (`1e6fc22`, Cycle 7)**:
  - **Cycle 6 (`1e6fc22`, `#303`)**: Exiting the vault doorway left Trevor wedged against a Shutter Room concrete pillar while `follow_route_on_foot` panned `dx=101..20` (`_road_motion_score >= 2.5`), masking the stall for `10s`. Added minimap `min_target_dist` / `target_stall_elapsed_s` stall detection (`dist_stall_window_s = max(stuck_window_s * 2.2, 2.2)`) plus boosted off-axis steer gain (`max(steer_gain, 240.0)`); Trevor cleared the doorway into the Shutter Room (`clips/check_0827.mp4`).
  - **Root Causes in `23m` Shutter Room Stall (`#308`, `#310`, `#311`)**:
    1. **Unverified vertical mouse sign (`captures/retro3_check.png`)**: `8` `+dy` deltas (`+1,270` total) pitched the camera straight UP into the ceiling because `calibrate_controls()` never probed vertical `dy` (`pitch_down_sign` was `-1`).
    2. **`enter_cover()` untoggled chest-high cover**: `08:28:03Z` produced `shift_score=19.1` (entering chest-high counter cover), but `classify_posture()` only scanned full-height rows `27..153` and returned `"standing"`, causing subsequent `enter_cover()` calls to toggle Trevor back out (`shift_score=6.51, 7.22`).
    3. **Close-range orbiting & missing cover-objective loop**: `follow_route_on_foot` kept `"w"` held while turning `dist < 0.20` (`orbiting`) and had no closed-loop `approach_and_enter_cover()` / `"get in cover"` stop.
  - **Cycle 7 Fixes**: Added lower chest-high cover detection (`rows 88..162`) + ambiguous `"unknown"` band (`0.96..1.15`), latched `_in_cover_hint = True` on `shift_score >= 10.0` so `enter_cover()` never untoggles cover, added vertical `pitch_down_sign` calibration to `calibrate_controls()` (`logs/calibration.json`), added sign-agnostic `level_camera_pitch()` with confidence-gated clamp checks, added close-range pivot + `"get in cover"` stop to `follow_route_on_foot`, and added `ActionRunner.approach_and_enter_cover()` / `prologue_shutter_cover()`.

---

## 8. Checkpoint 4 (2026-11-21 01:02 PST)

- **Progression & Telemetry (`4925b9a`, `79b292e`, Cycle 8)**:
  - **Cycle 7b (`79b292e`, `#317`, `117 passed`)**: Live `calibrate_controls()` measured `pitch_down_sign=1`, `pitch_px_per_dy=0.348`, `hipfire_px_per_dx=0.206`, `warning=None`, and `follow_route_on_foot` hit `target_reached` (`distance=0.0456`, `captures/p5_c7b_step3.png`). Adaptive `probe_dy` (`>= 24`, scaled from `_pitch_px_per_dy`) fixed `level_camera_pitch`'s `14 * 0.225 < 4.0px` false `"unresponsive"`, and `approach_and_enter_cover()` added pre-cover bearing alignment + non-untoggling `"a"`/`"d"` cover slides.
  - **Root Causes in `#321` Shutter Room Cover Blocker**:
    1. **`q` (`0x0C`) verified valid vs blocked geometry**: `AppKit.NSEvent` confirmed `KEYCODES["q"] = 0x0C -> 'q'` with identical `0x20800000` flags as `w/a/s/d`. Live captures (`p5_c7b_step3.png`, `q_after.png`, `p5_test_q_pillar.png`, `p5_test_q_wall.png`) showed Trevor wedged behind a slanted grey tarp/chair on the back of the `B-K SHIPPING` crate (`yellow` blip `+79°` right at `bearing=+0.439`) or facing the unscripted back corridor `!` wall (`yellow` at `bearing=-0.923`), while local `w/a/d/s` micro-probes never backed out around the crate to the clear shutter-facing side.
    2. **Background focus preemption (`Notes(pid=6487)`)**: `CGEventCreateKeyboardEvent` on macOS returned `0x20800000` (setting `kCGEventFlagMaskSecondaryFn = 0x800000`), and `CGEventInputBackend._clear_mod_mask` did not strip `SecondaryFn`/`NumericPad` from normal keys — turning every `'q'` press into macOS `Fn+Q` (**Quick Note**, launching/focusing `Notes` and eating `'q'` before it reached `GTA5.exe`!).
  - **Cycle 8 + 8b Fixes**: Stripped `kCGEventFlagMaskSecondaryFn | kCGEventFlagMaskNumericPad` in `_clear_mod_mask` (explicitly re-applying them ONLY via `_key_intrinsic_flags` for arrows/fn keys), added `stop_on_focus_loss: bool = True` (`"focus_lost"` early abort + `release_all`), and upgraded `approach_and_enter_cover()` with crate/pillar orbit probes.

---

## 9. Checkpoint 5 (2026-11-21 01:32 PST)

- **Progression & Telemetry (`645e456`, Cycle 10)**:
  - **Root-Cause Verification (`645e456`, `120 passed`)**: Stripping `kCGEventFlagMaskSecondaryFn` (`0x800000`) immediately stopped `Fn+Q` Quick Note (`Notes(pid=6487)`) from stealing focus, enabled live `q` cover snaps (`shift_score=16.87..27.88`, `clips/check_0928.mp4`), eliminated false red minimap alarm blips (`p5_exit_side2.png`), and `player-5` cleared the Prologue Shutter Room cover objective at the `GoPostal` crate (`#337`, `captures/p5_brad_cover15.png`).
  - **Remaining Friction (`clips/check_0928.mp4`, `#335`)**:
    1. Trevor snapped into cover against a narrow center pillar (Michael's `EXIT` pillar) occupying `x=[0.34..0.58]`, which fell between `classify_posture`'s left (`0.18..0.36`) and right (`0.48..0.66`) bands and returned `"standing"`/`"unknown"`.
    2. Snapping to a side prop while `yellow` remained `> 0.08` away triggered cover slides on the wrong prop instead of exiting cover (`q`) and advancing to `yellow`.
  - **Cycle 10 Fixes**: Added center pillar band `(0.34..0.58)` (full + chest-high `88..162`) and lower-half `"unknown"` fallback (`c_ratio >= 0.96` or `cc_ratio >= 1.05`) in `classify_posture()`, and updated `approach_and_enter_cover()` to detect `wrong_prop_cover` (cover latched while `target.distance > max(arrive_distance, 0.08)`), call `exit_cover(force=True)`, and run `follow_route_on_foot` directly onto `yellow`.

---

## 10. Checkpoint 6 (2026-11-21 02:02 PST)

- **Progression & Telemetry (`4aeca4e`, `8101e43`, Cycle 11–12)**:
  - **Cycle 11 (`8101e43`, `#344`, `123 passed`)**: Replacing `no_lock_ticks in (4, 10)` with immediate visual-alignment suppressive fire (`aligned_ticks == 1 or aligned_ticks % 4 == 0`) + zero-sum micro-dither pushed live `cover_peek_combat` `red_reticle` accuracy to **`22/39` (`56.4%`)** in `09:50–09:53Z` (`#351`).
  - **Root Causes in Yard Shootout Deaths (`#348`, `#349`, `#350`, `clips/check_0954.mp4`)**:
    1. `_silhouette_fallback` inside `_detect_visual_target` triggered `35/54` (`65%`, up to `53/54` with scan steers) false locks on yellow-striped concrete pillars, vertical drainpipes, doorway edges, and floor shadows, hijacking `advance_under_fire` away from `VisionHumanDetectorBackend` + minimap bearing + `DEFAULT_RETICLE_SWEEP` (`0/18` = `0%` `red_reticle`).
    2. `VisionHumanDetectorBackend` also locked onto closer occluding ally **Brad** (`[0.483, 0.544]`) when `red` cops lay behind him on the same bearing (`min_red <= 0.16` bypassed ally filtering), and lower-horizon floor artifacts (`ay > 0.57`).
    3. `advance_under_fire` aborted `"low_health"` (`0.247` HP at `09:57:47Z`) exposed without snapping `q` cover during the turn gap (`cover_on_low_health`).
  - **Cycle 12 Fixes**: Removed `_silhouette_fallback` inside `_detect_visual_target` when using `VisionHumanDetectorBackend` (keeping `SilhouetteHumanDetectorBackend` for explicit injection/offline tests with strict thin-edge/top-pillar filtering), added closer-ally occlusion + `ay <= 0.57` horizon filters in `select_visual_enemy_target()`, yielded persistent visual misses (`aligned_ticks >= 5` with `0` `red_reticle`) back to `DEFAULT_RETICLE_SWEEP` (`visual_suppressed_until_t`), and added `cover_on_low_health=True` to `advance_under_fire`.

---

## 11. Checkpoint 7 (2026-11-21 02:32 PST)

- **Progression & Telemetry (`c48b247`, Cycle 13)**:
  - **Milestone (`#352`, `#355`)**: `player-5` cleared the entire Yard wave (`red_count: 0`), pushed past the North Yankton gate with Michael & Brad, cleared a roadblock wave (`cover_peek_combat`: `enemies_cleared` at `10:29:50Z`), and `cover_on_low_health=True` saved Trevor live at `0.247` HP (`10:29:12Z`).
  - **Root Causes at Snowy Barn Roadblock (`clips/check_1026.mp4`, `#358`, `#359`)**:
    1. Crouched/hood-peeking cops behind white Police SUV roadblocks were missed by the full-height Vision ROI (`0.20..0.76`), while blind minimap bearing fired `2`-burst suppressive rounds into the SUV steel door (`67` bursts, `3/70` = `4.3%` `red_reticle`, depleting rifle reserve to `12 | 21`).
    2. `hold_engage_threat` doubled engage hold (`3.2s -> 6.5s`) even with `0` `red_reticle` hits, and `advance_flank_unblock` (`0.55s` `'d'` + `0.65s` `'w'`) failed to clear wide `~3m` SUVs and reset `blocked_flank_streak` across `advance_under_fire` timeouts.
  - **Cycle 13 Fixes**: Added hood-height `VisionHumanDetectorBackend` fallback ROI `(0.22, 0.25, 0.88, 0.58)` when primary ROI yields `None`; cut blind suppressive bursts when `engage_red_hits == 0` and `visual is None` to a single `1`-burst probe (`no_lock_ticks == 8` in `advance_under_fire`, `bursts=1` in `cover_peek_combat`) and skipped `hold_engage_threat` extension when `engage_red_hits == 0`; widened `advance_flank_unblock` to `0.85s` lateral strafe + `0.85s` forward clearance and persisted `self._last_flank_streak` across `advance_under_fire` calls.

---

## 12. Checkpoint 8 (2026-11-21 03:03 PST / 11:03Z)

- **What went right (`508bfb9`, `01ed49f`, `#372`)**:
  - Cycle 13 (`01ed49f`) lifted `10:43–11:03Z` `red_reticle` bursts to **`46/130` (`35.4%`; `52/151` = `34.4%` across the 30-min window)**, and `advance_under_fire` hit `target_reached` out the gate (`10:38:19`), putting Trevor beside the getaway SUV (`blue=0.099`, `#361`).
  - Cycle 14 (`508bfb9`, `127 passed`) eliminated the blind `cover_peek_combat(enter_cover=True)` `'q'` untoggle trap after `advance_under_fire` `low_health` (`tap_cover=True`) via idempotent `self.enter_cover(force=False)` and exposed-standing `cover_duck` guard.
- **What did not work (`#363`, `#368`, `#372`)**:
  1. **Getaway SUV overshoot (`#363`, `10:40:11`)**: Blind `w 0.8s + f` ran past the getaway SUV, so `follow_gps_drive` ran `35s` on foot into the snowfield -> `"The crew were left behind."`.
  2. **`advance_flank_unblock` 4-streak one-side lock (`runner.py:1922`)**: Locked `4` consecutive attempts (`((streak - 1) // 4) % 2 == 0`) onto `'d'` (`10:57:46–10:58:15`, `11:01:12–11:01:47`) while `enemy_distance` stayed pinned at `0.157–0.180`, wasting full `50s` advance runs.
  3. **Tall-pillar `enter_cover` (`10:43:44`, `10:50:00`, `#368`)**: `shift_score=26.85 / 22.0` latched `_in_cover_hint=True` while `classify_posture` returned `"standing"` near tall pillar silhouettes, and flanking cops on the front pillar still damaged Trevor during `cover_duck` (`10:52:23 -> 10:52:59`).
- **Cycle 15 Fixes**: Flipped `advance_flank_unblock` polarity after **2** blocked attempts (`// 2` instead of `// 4`, picking initial side away from `enemy.bearing` when `abs(enemy.bearing) > 0.12` on `streak == 1`), and added `ActionRunner.enter_nearest_vehicle()` (closed-loop `follow_route_on_foot` approach + `'f'` tap + HUD/route/scene boarding verification before `follow_gps_drive`).

---

## 13. Checkpoint 9 (2026-11-21 03:32 PST / 11:32Z)

- **What went right (`508bfb9`, `90629bd`, `#376`, `#377`)**:
  - **Zero `MISSION_FAILED` banners** and **`cover_exposed: 0`** across `11:13Z–11:32Z` (Cycle 14's idempotent `enter_cover(force=False)` + `cover_duck` standing guard held cleanly).
  - `player-5` cleared the North Yankton roadblock cops and **boarded the getaway SUV** with Michael & Brad (`#376`, `captures/p5_boarded_suv2.png`, `"Shut the fuck up and drive."`).
- **What did not work (`#377`)**:
  1. **Low `red_reticle` vs cruiser-crouched cops (`6/47` = `12.8%`, `11:13–11:31Z`)**: ~20m roadblock grind before clearing.
  2. **`enter_nearest_vehicle` far-distance false confirmation (`pre_dist=0.373` at `11:27:30`) & short `settle_s=2.5` (`11:28:45`)**: Ungated `shift_score >= 12.0` / `cur_has_drive` confirmed boarding far from the SUV, while `2.5s` cut off the door-walk animation at `pre_dist=0.058`.
  3. **Binary full-tick driving steer (`runner.py:2365-2368`)**: Holding `'a'`/`'d'` across full `100ms` ticks on mild `0.08..0.24` curves risks high-speed fishtailing/weaving.
- **Cycle 16 Fixes**: Gated `enter_nearest_vehicle` `shift_score >= 12.0` on `pre_target is None or pre_target.distance <= max(arrive_distance * 2.5, 0.16)` and `cur_has_drive`/route on `pre_target.distance <= 0.14`, raised default `settle_s=3.8`, and added smooth `0.045s` mild-curve steer pulsing (`steer_deadband <= |bearing| < 0.24`) in `follow_gps_drive`.

---

## 14. Checkpoint 10 (2026-11-21 04:04 PST / 12:04Z)

- **What went right (`b92fb64`, `3cd2628`, `#387`)**:
  - Sustained Prologue getaway driving across `14` `follow_gps_drive` runs (~`540s` drive time), reaching within **`0.13–0.14mi`** of the helicopter objective (`p5_drive_5.png`, `clips/check_1145.mp4`).
  - Rapid visual-telemetry-fix loop: Cycle 16 (`b92fb64`, `129 passed`) shipped `0.045s` mild-curve steer pulsing + distance-gated `enter_nearest_vehicle`; `watcher-3` isolated bumper-rocking wheelspin masking `_road_motion_score` (`#382`, `0` recoveries across `2x45s`); and Cycle 17 (`3cd2628`, `130 passed`) landed & pushed `sharp` brake pulsing, `high_bearing_stall`, `target_dist_stall`, and forward steer-around recovery.
- **What did not work (`#378`, `#382`, `#383`, `#387`)**:
  - Two driving `MISSION_FAILED` resets (`11:37:39` `"The car was stuck in the snow."` at `0.14mi` `#378`, `11:54:35` `"The route was abandoned."` `#383`) plus an `18m` getaway-drive grind before `3cd2628` merged (`04:03:31`), because wheelspin/rocking against roadside snowbanks & civilian cars kept `_road_motion_score >= 3.0`, `not sharp` disabled `stuck_recovery` whenever `|bearing| >= 0.45`, and continuous `'s'` during sharp turns caused reverse-rocking.
- **Cycle 17 Fixes (Verified `3cd2628`, `#385`)**: Replaced continuous `'s'` during `sharp` turns with a `0.25s` brake pulse (`1.6s` cooldown), kept `enable_stuck_recovery` active across `sharp` turns, triggered `stuck_recovery` on sustained `high_bearing_stall` (`|bearing| >= 0.30 >= 2.5s`) and `target_dist_stall` (`>= 4.0s`), and committed a `<= 0.45s` forward steer-around pulse after reverse.

---

## 15. Checkpoint 11 (2026-11-21 04:44 PST / 12:44Z)

- **What went right (`3cd2628`, `b1c448e`, `#396`, `#399`, `#401`, `#402`)**:
  - **PROLOGUE COMPLETE (`#396`, `#399`)**: Completed the North Yankton getaway drive (`3cd2628`, `12:21:49Z`), cleared the train-crash ambush, and rolled into the Los Santos opening credits (`captures/p5_ambush_check2.png`, Vespucci Beach -> *"Franklin and Lamar"*).
  - **Cycle 19 Hotfix (`b1c448e`, `130 passed`)**: Fixed `follow_gps_drive` `_perform_stuck_recovery` missing `last_motion_check_t = None` (eliminating the `8x`-in-`19s` rapid reverse loop from `12:08:06–12:08:25Z`) and raised `high_bearing_stall` threshold (`>= 0.58 >= 3.0s` + improvement reset `< prev_best - 0.08`) ahead of Los Santos driving.
  - **Clean Worker Rotation (`#401`)**: Rotated 1,457-step `player-5` cleanly to fresh `player-6` for Los Santos Story Mode.
- **What did not work (`#388`, `#391`, `#395`, `#402`)**:
  - Three driving `MISSION_FAILED` resets (`12:05:33`, `12:09:58` bridge guardrail `#391`, `12:17:10` final `0.12mi` turnoff `#395`) occurred on `3cd2628` before `b1c448e` merged (`04:36:04`), caused by the unreset `last_motion_check_t` `1.3s` recovery-sleep `dt` inflation and `0.30` sweeping-curve trigger.

## 16. Checkpoint 12 (2026-11-21 06:15 PST / 14:15Z)

- **What went right (`2ed735e`, `#403`, `#408`–`#412`)**:
  - **Opening Transition Into *"Franklin and Lamar"***: After Los Santos credits completed (`2ed735e`), `player-6` opened the mission cleanly (`captures/p6_sidewalk.png`, `12:58:10Z`–`12:59:04Z`), moving Franklin along the sidewalk toward the parked sports cars (`w`/`a` inputs worked until `12:59:52Z`).
  - **Root-Cause Isolation (`#409`, `#411`, `#412`)**: Read-only OS & live probe diagnostics conclusively narrowed the `12:59:52Z` freeze to an internal Wine/`GTA5.exe` keyboard message-loop wedge rather than our code: macOS HID/Combined flags were clean (`0x20000002`), `IsSecureEventInputEnabled` was `False`, `GTA5.exe` (`pid=1212`, `wid=135`) was frontmost with `AXFocusedWindow`/`AXMainWindow` valid, and `mouse_delta` + clicks still rotated/reset the camera (`p6_diag2_est.png`), yet **all 4 external macOS keyboard injection paths** (`kCGHIDEventTap`, `Finder` focus cycle + `click_to_activate`, `CGEventPostToPid(1212)`, `kCGSessionEventTap` with `source=None, flags=0`, and `osascript System Events`) produced zero keyboard response in-game. Also observed an auxiliary `66x20` `GTA5.exe` `'Window'` (`wid=155`) alongside `wid=135`.
  - **Cycle 21 Recovery Tooling (`gameplayer/window.py`, `TOOLS.md`)**: Added `cycle_app_focus: bool = False` / `yield_app_focus` support to `focus_game_window()` so `player` can cleanly cycle `Finder` -> `GTA5.exe` + click for a fresh Wine `WM_ACTIVATEAPP`, and documented the read-only diagnostic + escalation steps (`#412`).
- **What did not work (`#403`, `#409`–`#412`)**:
  - **Internal Wine Keyboard Loop Wedge (`12:59:52Z`–`14:15Z`)**: Immediately after `enter_nearest_vehicle` tapped `'f'` at `12:59:52Z` (`captures/p6_car_enter2.png` / `p6_car_enter3.png`), `GTA5.exe` (`-useNativeInputSupport -scEnableOverlayV3`) stopped consuming any macOS keyboard events while still accepting mouse movement/clicks, stalling *"Franklin and Lamar"* for `~75` minutes (`> 6 min` trigger) because no external synthetic macOS API can unstick Wine's wedged keyboard consumer without a physical check or game restart (`#412`).

---

## 17. Session 2 Final Synthesis (2026-11-20 – 2026-11-21): Full Retrospective & Next-Time Playbook

### A. Session 1 vs. Session 2 Quantitative Scorecard

| Metric | Session 1 (`2026-11-17`, `06ad9e3..a216bdc`) | Session 2 (`2026-11-20`–`21`, `410ee3f..a1d305f`) |
| :--- | :--- | :--- |
| **Window** | `06:21Z–15:59Z` (~9h 38m) | `06:59Z–17:25Z` (~10h 26m; ~6h active play to `12:59Z` + ~4.4h Wine wedge diagnosis/idle) |
| **Commits / Cycles** | `40` commits / `35` cycles | `23` commits / `21` cycles across `12` 30-min retrospectives |
| **Codebase Size** | `7,940` LOC (`4,848` runtime + `3,092` tests) | `12,654` LOC (`7,282` runtime + `5,372` tests) |
| **Offline Test Suite** | `94` unit tests (static mocks) | `132` tests (+ advancing-clock 2.5D simulator + offline `replay_bench`) |
| **Telemetry Volume** | `316,602` events (`183` 3-min video segments) | `132,999` events (`192` 3-min video segments) |
| **Combat Efficiency (`red_reticle / fire_burst`)** | `363 / 5,683` (**`6.4%`** overall; peak `27.0%`) | `95 / 410` (**`23.2%`** overall; peak **`56.4%` (`22/39`)** in `cover_peek_combat` `09:50–09:53Z` — **3.6× hit rate with 13.9× fewer rounds**) |
| **`banner:MISSION_FAILED` Loops** | `47` resets | **`11` loop fails** (`4` cover, `1` advance, `5` getaway driving, `1` SUV overshoot — **77% fewer resets**) |
| **Input Safety Leaks** | `0` stuck OS keys (`1,106` `release_all`) | `0` stuck OS keys (`698` `release_all`, `4` clean `focus_lost` aborts) |
| **Story Mode Milestone** | Stalled in Prologue Snowy Barn Road | **PROLOGUE COMPLETE (`#399`)** -> Los Santos opening credits -> entered **"Franklin and Lamar"** (`p6_sidewalk.png`, halted by Wine keyboard wedge) |

### B. What Went Right in Session 2

1. **Screen-Space Visual Targeting (`VisionHumanDetectorBackend`) Transformed Combat**:
   - Running Apple's native `VNDetectHumanRectanglesRequest` (`gameplayer/perception.py`) at `11.5–23.3 ms/frame` plus a hood-height fallback ROI `(0.22, 0.25, 0.88, 0.58)` (`01ed49f`) and immediate visual-aligned fire (`8101e43`) lifted live `cover_peek_combat` `red_reticle` accuracy to **`56.4%` (`22/39`)** and raised overall session hit efficiency from `6.4%` to **`23.2%`** while cutting total bursts fired by `13.9×` (`5,683 -> 410`).
   - Crucially, pruning noisy heuristic fallbacks (`_silhouette_fallback` disabled under Vision in `c48b247`), adding closer-ally (`Brad`) occlusion filtering, and capping blind suppressive ammo (`01ed49f`) let `player-5` clear the entire North Yankton depot yard and roadblock with `0` deaths in `11:13Z–11:32Z`.
2. **Advancing-Clock 2.5D Simulator & Offline Video Replay Benchmark**:
   - `tests/sim_harness.py` (`492` LOC) + `tests/test_sim_rollout.py` (`1,198` LOC) and `gameplayer/replay_bench.py` (`204` LOC) moved multi-second timing, recoil, ADS/hipfire sensitivity, cover-toggle, and driving-stall regressions out of slow live retries into deterministic `<1s` offline rollouts.
3. **macOS HID Flag Hygiene Solved Two "Mystery" Input Failures**:
   - Preserving native `0x20000000` bits + setting `NumericPad | SecondaryFn` on arrow/Fn keys (`410ee3f`, `f87c9a2`) unblocked the Prologue vault phone.
   - Discovering that `CGEventCreateKeyboardEvent` silently attached `kCGEventFlagMaskSecondaryFn (0x800000)`—turning `'q'` into macOS `Fn+Q` (**Quick Note**, launching `Notes(pid=6487)` over `GTA5.exe`)—and stripping it in `_clear_mod_mask` (`645e456`) immediately unlocked cover snapping and cleared the Shutter Room.
4. **Defensive Cover Idempotency & Focus Guards Eliminated Turn-Gap Wipes**:
   - Combining `stop_on_focus_loss=True` (`eb21097`), `cover_on_low_health=True` (`c48b247`, saving Trevor live at `0.247` HP), idempotent `enter_cover(force=False)` in `cover_peek_combat` (`508bfb9`), and `cover_duck` standing guards drove `cover_exposed` to `0` across the final shootout push.
5. **Closed-Loop Driving & Vehicle Boarding Cleared Prologue**:
   - Iterating `enter_nearest_vehicle()` (`90629bd`, `b92fb64`) and `follow_gps_drive()` (`0.045s` mild-curve steer pulsing, `0.25s` sharp brake pulse, `high_bearing_stall` + `target_dist_stall`, and `last_motion_check_t` reset in `3cd2628`/`b1c448e`) cleared the North Yankton getaway drive (`#396`) and completed **Prologue (`#399`)**.
6. **Operational Discipline (30-Minute Retrospectives, `logs/status.json`, Clean Rotation)**:
   - Decoupling machine-written scoreboard snapshots into gitignored `logs/status.json` (`gameplayer/status_store.py`) eliminated Session 1's `STATUS.md` worktree merge collisions.
   - Scheduled 30-minute retrospectives (`Checkpoints 1–12`) kept engineering tightly coupled to telemetry and short video slices, and rotating `player-4 -> player-5 -> player-6` kept live gameplay crisp across mission transitions.

### C. What Did Not Work (Root Causes & Remaining Failure Modes)

1. **Internal Wine / GPTK Keyboard Loop Wedge (`#403`, `#409`, `#411` — Final Hard Blocker)**:
   - Right after `tap("f")` at `12:59:52Z` in *"Franklin and Lamar"*, `GTA5.exe` (`pid=1212`) stopped consuming all keyboard input (`f`, `w`, `s`, `v`, `escape`, `p`, `enter`, `space`, `tab`) while live highway traffic and `mouse_delta`/right-click camera controls kept working.
   - All four external macOS keyboard paths (`kCGHIDEventTap`, `CGEventPostToPid(1212)`, `kCGSessionEventTap` with `source=None, flags=0`, and `osascript System Events`) plus `Finder` app-focus cycling and modifier pulses failed despite `AXIsProcessTrusted=True`, `CGPreflightPostEventAccess=True`, `IsSecureEventInputEnabled=False`, and `Held=[]`. Because there was no fast dead-keyboard abort/escalation rule, *"Franklin and Lamar"* stalled `>75` minutes before escalating to the user.
2. **Heuristic Fallbacks Poisoning Clean Perception**:
   - Leaving `_silhouette_fallback` enabled behind `VisionHumanDetectorBackend` injected `130` false visual steers (`0/18` `red_reticle` on pillars/pipes/shadows) until disabled (`c48b247`).
   - Column-brightness posture heuristics (`classify_posture`) repeatedly misclassified crouch-cover near tall pillars or center props (`shift_score=22..26` yet `"standing"`), requiring multiple band patches (`4925b9a`, `4aeca4e`).
3. **Stall-Detector Blind Spots & Clock Reset Bugs**:
   - `_road_motion_score` alone could not distinguish forward progress from camera panning (`1e6fc22`) or wheelspin/bumper-rocking against a parked car (`#382`, `0` recoveries in `2x45s`).
   - Conversely, forgetting `last_motion_check_t = None` inside `_perform_stuck_recovery()` charged `1.3s` of recovery sleep to the next tick's `dt`, and setting `high_bearing_stall` at `0.30` triggered `8` reverse slams in `19s` (`#391`, `Michael died`).
4. **Partial Occlusion Behind Vehicles**:
   - Even with the hood-height ROI (`01ed49f`), cops crouched tightly behind police cruiser doors/roofs (`p5_road_push_6.png`) rarely triggered full-body `VNDetectHumanRectanglesRequest`, dropping `red_reticle` rate to `12.8%` (`6/47`) during the final `~20m` roadblock grind.
5. **Monolithic `runner.py` & Context Bloat**:
   - [gameplayer/runner.py](gameplayer/runner.py) grew from `2,045` to `3,333` LOC (+63%), increasing edit friction and coupling unrelated subsystems (on-foot nav, combat, cover, vehicle entry, driving).
   - `player-5` (`1,457` steps) and `toolsmith-3` (`1,750+` steps, compacting mid-task) ran too long before rotation, and full `pytest` wall-clock time grew from `~5s` to `93–206s` under live Wine CPU contention.
6. **No Audio Perception Yet**:
   - `ScreenCaptureKit` / `AVFoundation` Python bindings were not installed in `.venv`, leaving gunfire direction, hitmarkers, sirens, vehicle collision/stall audio, and dialogue completion strictly inferred from pixels/OCR.

### D. What We Can Do Better Next Time

1. **Automated Dead-Keyboard Watchdog & Rapid Escalation (Plus Virtual Gamepad Evaluation)**:
   - Add a `verify_keyboard_liveness()` guard triggered on any `2+` consecutive zero-response movement turns: run `focus_game_window(cycle_app_focus=True)`, pulse modifiers, and probe `p`/`escape` for `pause_menu`. If still unresponsive, abort and alert the user within `<2 minutes` (or run a user-approved relaunch script that reloads the latest Story Mode autosave).
   - Evaluate legitimate macOS virtual controller / Wine XInput support for future runs: analog left/right sticks eliminate binary WASD steering wobble, unlock GTA V's native controller lock-on/aim smoothness, and bypass `winemac.drv`'s keyboard message-loop wedge entirely.
2. **Split `gameplayer/runner.py` (`3,333` LOC) Into Focused Submodules**:
   - Refactor into `gameplayer/loop_engine.py` (shared tick loop, focus/banner guards, shared `StallDetector` with enforced `on_recovery_reset()` clock hygiene), `gameplayer/nav_foot.py`, `gameplayer/combat.py`, and `gameplayer/driving.py`. Keep each module `<700` LOC so engineering edits and offline reviews stay fast.
3. **Upper-Body Vision Detection & Flow-Based Cover/Stall Sensing**:
   - Enable `VNDetectHumanRectanglesRequest` with `upperBodyOnly = True` (Apple Vision supports upper-body mode natively) as the secondary pass behind full-body detection so cops peeking over car hoods and crates lock reliably without noisy dark-blob fallbacks.
   - Replace `_road_motion_score` stall checks with **minimap-interior phase correlation / scroll velocity** (measuring map-tile motion inside the minimap crop, completely immune to 3D camera pans, blowing snow, and tire spin) and classify cover via **camera-offset / peek-shift verification** rather than fixed brightness columns.
4. **Proportional PWM Driving & Wanted-Evasion Mode for Los Santos**:
   - Upgrade `follow_gps_drive` from two-tier steer (`0.045s` pulse vs full hold) to continuous bearing-proportional PWM steering (`duty = clamp((|bearing| - deadband) / 0.35, 0.2, 1.0)`) plus a Los Santos `"lose the cops"` evasion loop (detect flashing red/blue minimap siren border + wanted stars, steer away from police cones into side streets/off-road until stars clear).
5. **Add Audio Capture via `ScreenCaptureKit`**:
   - Install `pyobjc-framework-ScreenCaptureKit` and feed lightweight audio energy / stereo direction / dialogue-silence features into telemetry and combat/cutscene detectors (`ScreenCaptureAccess` is already granted).
6. **Proactive Worker Rotation & Fast/Slow Test Split**:
   - Rotate `player` and `toolsmith` automatically around `~400–500` steps (at natural mission checkpoints) to avoid compaction stalls.
   - Tag heavy simulator rollouts (`@pytest.mark.slow`) so routine hotfix verification runs fast unit tests in `<10s` while full simulator rollouts run in parallel or on major cycles.

---

## 18. Deeper Architectural & Operational Blind Spots (What Else We Could Have Done Better)

Beyond the immediate Session 2 fixes in Section 17, stepping back across ~20 hours of live runs exposes eight broader structural improvements that would have accelerated progress dramatically:

### 1. Persistent Stateful Runtime Daemon (Eliminate Cold-Start Subprocess Amnesia)
- **What happened**: Every `player` turn spawned a fresh `.venv/bin/python -c "..."` process, re-importing `Quartz`/`Vision`/`cv2`/`numpy`, re-warming `VNDetectHumanRectanglesRequest`, and discarding in-memory state (`_in_cover_hint`, `_last_flank_streak`, `CameraMotionTracker`, calibration gains) across turns. Multiple live deaths (`#366`, repeated flank resets) stemmed directly from inter-turn state amnesia and `1–3s` startup gaps.
- **Better**: Run one persistent `gameplayer.daemon` process (over a local Unix socket / CLI client) holding warm ML models, live state (`in_cover`, `in_vehicle`, `keyboard_alive`, calibrated gains), and a single shared capture ring buffer.

### 2. Single-Producer Capture Bus (No Competing Screen Captures)
- **What happened**: `player` (`~10 Hz`) and `watcher`'s `capture_daemon` (`2 Hz`) called `CGWindowListCreateImage` independently while Wine ran GTA V, adding WindowServer/GPU contention (offline `pytest` dropped from `93–206s` during live runs to `11.38s` once Wine/capture stopped) and decoupling video timestamps from loop telemetry.
- **Better**: Publish frames from one capture producer into a shared memory ring buffer consumed by both closed-loop controllers and the MP4 segment recorder/slicer, guaranteeing exact frame-to-telemetry alignment at half the capture overhead.

### 3. Typed Action CLI Instead of Ad-Hoc Inline `python -c` Scripts
- **What happened**: Even after adding [gameplayer/playbooks.py](gameplayer/playbooks.py), `player` workers still wrote ad-hoc inline Python scripts every turn, introducing avoidable one-off errors (blind `w 0.8s + f` overshooting the getaway SUV in `#363`, wrong `enter_cover=True` flag in `#366`, bare `sleep` calls without inspection).
- **Better**: Constrain `player` to a narrow, typed CLI (`gameplayer act cover-peek`, `gameplayer act enter-vehicle`, `gameplayer act drive-gps`, `gameplayer status`) so every parameter is validated, every action is uniformly logged, and reusable behaviors are forced into tested modules.

### 4. Automated Real-Frame Visual Regression Bank in CI
- **What happened**: We recorded `375` 3-minute MP4 segments and hundreds of ad-hoc PNGs (`captures/`, `clips/`), yet `replay_bench.py` was only invoked manually on a few clips.
- **Better**: Auto-archive every anomaly (`MISSION_FAILED`, `cover_blocked`, `focus_lost`, `0%` `red_reticle`) as a compact versioned fixture (`frame.png` + `expected.json`) and run real-image assertions inside `pytest` for minimap segmentation, posture classification, HUD OCR, and Vision target locking. That catches alarm-wash false blips, ally occlusion, and pillar misclassifications before merge.

### 5. Pre-Flight Environment & I/O Smoke Test (`gameplayer preflight`)
- **What happened**: Hours were lost across Sessions 1 & 2 to host/game configuration surprises (`Raw Input` ignoring `CGEvent` deltas, macOS `Fn+Q` launching `Notes`, unverified vertical pitch sign, Wine keyboard wedges).
- **Better**: Run an automated 10-second `preflight` gate before every session: verify screen/accessibility permissions, check for conflicting macOS hotkeys/overlays, confirm subtitles + Simple reticle, calibrate `(dx, dy)` signs/gains, and execute a live `Esc -> Pause Menu -> Esc` round-trip keyboard liveness check.

### 6. Progress-Velocity Watchdogs & Deterministic (Zero-Token) Monitoring
- **What happened**: Mandatory upgrade triggers leaned heavily on `consecutive_failures >= 2`, missing low-death grinds (`player-5` survived `~20m` at `0.506` HP behind a roadblock cruiser with `0` deaths) and `>75m` wedged-keyboard stalls. Meanwhile, LLM-based `watcher`/`monitor` workers consumed tokens polling idle states, and `1,400–1,750`-step workers slowed to `2–4 min/turn` and compacted mid-task.
- **Better**: Run stall/liveness checks (`objective_changed`, `minimap_distance_delta`, `enemy_count_delta`, `keyboard_liveness`) in deterministic Python code that only wakes an LLM when a metric trips; enforce strict hard timeouts (`<2 min` escalation on dead input) and rotate `player`/`toolsmith` every `~400` steps with a compact structured handoff.

### 7. Richer Pixel Cues Beyond Human Bounding Boxes
- **What happened**: Shooting and driving relied almost solely on minimap blips + full-body human boxes + a bottom-center `_road_motion_score` patch.
- **Better**:
  - **Damage-vignette direction detector** (screen-edge red flashes) to know *which side* is flanking Trevor during `cover_duck`.
  - **Forward road-corridor / obstacle-depth cue** (e.g., lightweight CoreML monocular depth or drivable-road mask) so `follow_gps_drive` steers *around* parked civilian cars and bridge guardrails instead of ramming them.
  - **Letterbox / HUD-presence classifier** to distinguish unskippable cutscenes cleanly from frozen gameplay.

### 8. Higher-Leverage Research & Hardware Choices for Future Runs
- **Privileged Dev/Eval Oracle (Black-Box Runtime)**: Log read-only ground truth (`x, y, z, yaw, pitch, in_cover, in_vehicle, enemy LoS`) during *development and calibration* while keeping the live agent pure pixels + HID—accelerating vision dataset creation, system-ID fitting, and root-cause attribution by `10–50×`.
- **Save-State + Frame-Step Testbed First**: Iterate the core perception/driving/combat stack on an emulated 3D/open-world classic first (e.g., PS2 *GTA: San Andreas* via PCSX2 or Dolphin/RetroArch) with instant millisecond save states, `4×–10×` fast-forward regression suites, and zero Wine HID quirks—then port the hardened stack back to GTA V.
- **Hardware USB HID Loopback + Gamepad Emulation**: Inject inputs via a wired `$4` Raspberry Pi Pico (TinyUSB composite Xbox/HID Gamepad + Mouse + Keyboard with a `250 ms` firmware release watchdog) instead of macOS `CGEvent`. This bypasses `CGEvent` flag leaks and `winemac.drv` keyboard wedges, enables true `[-1, +1]` analog driving steering/throttle, and unlocks legitimate native GTA V controller aim assist.
