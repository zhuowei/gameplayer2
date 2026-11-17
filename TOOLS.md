# TOOLS.md — Gameplayer Runtime Toolkit & Usage Guide

## Environment Setup

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

Run offline tests:

```bash
.venv/bin/pytest
```

---

## 1. Window Discovery & Clean 16:9 Capture (`gameplayer.window`)

Black-box window metadata discovery (`CGWindowListCopyWindowInfo`) and clean unobstructed window pixel capture (`CGWindowListCreateImage` with `kCGWindowListOptionIncludingWindow` and `kCGWindowImageBoundsIgnoreFraming`). Capturing does **not** focus or alter the game window.

### CLI Usage

```bash
# Read-only window metadata inspection (safe for any role)
.venv/bin/python -m gameplayer.window --inspect

# Capture clean unobstructed 16:9 PNG (safe for any role; does not focus)
.venv/bin/python -m gameplayer.window --capture captures/frame.png
.venv/bin/python -m gameplayer.window --capture captures/frame_640.png --width 640
```

### Python API

- `find_game_window(...) -> WindowInfo`: matches `GTA5.exe` / `Grand Theft Auto V` (`layer == 0`, onscreen).
- `capture_game_window(target_width=None, crop_titlebar=True, require_16_9=True) -> CaptureResult`:
  - Normalizes aspect ratio to 16:9 (`normalize_to_16_9` crops top title bar if taller than 16:9, or horizontal center if wider).
  - `.to_numpy()` returns `(H, W, 3)` `uint8` RGB array; `.save(path)` writes PNG.
- `focus_game_window(..., allow_focus=True, warp_cursor_to_center=True, click_to_activate=False, activate_button="right") -> WindowInfo`:
  - **STRICT OWNERSHIP**: ONLY `player` may call `focus_game_window(..., allow_focus=True)`. Raises `PermissionError` if `allow_focus` is not explicitly `True`.
  - Uses Accessibility (`AXRaise` + `AXFrontmost`), Carbon `SetFrontProcess`, `NSRunningApplication.activateWithOptions_`, and `System Events` fallback, then pumps `NSRunLoop` and verifies both `is_frontmost` and `obstructed_by == ()` (raising `WindowCaptureError` if blocked).
  - Automatically warps the cursor to `window.bounds.center` (`warp_cursor_to_center=True`). Pass `click_to_activate=True` during live gameplay if Wine ignores keyboard/`mouse_delta` despite `is_frontmost=True` (**WARNING**: do NOT use `activate_button="right"` inside the Pause Menu because `right` backs out of menus).

---

## 2. Fail-Safe Synthetic OS Input (`gameplayer.input`)

Sends macOS Quartz `CGEvent` keyboard/mouse events (`CGEventInputBackend`) with deterministic offline testability (`DryRunInputBackend` / custom `InputBackend`).

### Safety Guarantees

- **Context Exit / Exception**: `with InputController(...) as ctrl:` calls `release_all()` in `__exit__`.
- **Per-Action Cleanup**: `hold`, `tap`, `click`, and `run_sequence` wrap execution in `try/finally: release_all()`.
- **Fault-Tolerant Release**: `release_all()` loops over every held key and mouse button individually so one backend error never prevents releasing the remaining inputs.
- **Process Exit & Signals**: `atexit` and `SIGINT`/`SIGTERM` handlers release all held keys/buttons on active controllers before chaining prior handlers.
- **Hard Timeout Watchdog**: Every timed action (`hold`, `tap`, `click`, `mouse_delta`, `run_sequence`) starts a `threading.Timer` watchdog (`default_timeout_s=10.0`, overrideable via `timeout_s` / `--timeout`). On expiration, the watchdog immediately calls `release_all()` and interrupts the action with `InputTimeoutError`.
- **Live Opt-In**: CLI requires `--live` (vs `--dry-run`). **STRICT OWNERSHIP**: ONLY `player` may pass `--live` or instantiate `CGEventInputBackend`.

### CLI Usage

```bash
# Offline dry-run (safe for verification)
.venv/bin/python -m gameplayer.input --dry-run tap w
.venv/bin/python -m gameplayer.input --dry-run hold --keys w lshift --duration 0.5
.venv/bin/python -m gameplayer.input --dry-run mouse-delta 120 -20 --steps 6 --duration 0.1

# Live game inputs (ONLY `player` may run --live)
.venv/bin/python -m gameplayer.input --live --timeout 5.0 hold --keys w --duration 1.0
.venv/bin/python -m gameplayer.input --live --timeout 10.0 sequence path/to/steps.json
```

### Python API (`InputController`)

- `tap(key, duration_s=0.06, timeout_s=None)`
- `click(button="left", duration_s=0.06, timeout_s=None)`
- `click_window_center(window_bounds=None, button="right", hold_s=0.04, timeout_s=None)`: anchors cursor recentering to `window_bounds`, warps to window center, and clicks `button` to restore Wine keyboard/mouse capture (**WARNING**: avoid `button="right"` inside Pause Menu).
- `hold(keys=(), buttons=(), duration_s=0.1, timeout_s=None)`
- `mouse_delta(dx, dy, steps=1, duration_s=0.0, timeout_s=None)` (automatically uses `*MouseDragged` events when a mouse button is held)
- `switch_character(character, method="hotkey", wheel_hold_s=0.30, timeout_s=None)`: switches between `michael` (`f5`), `franklin` (`f6`), `trevor` (`f7`), `online` (`f8`), `method="wheel"` (`lalt` + small 60px `mouse_delta`), or `method="wheel_hold"` (holds `lalt` without mouse nudge when the target character is already pre-highlighted). Note: do NOT mix `key_down("lalt")` with `tap()`/`hold()` outside `run_sequence` because `hold()`/`tap()` enforces `release_all()` on completion; use `switch_character` or `run_sequence`.
- `run_sequence(steps, timeout_s=None)`: executes a list of step dicts (`hold`, `tap`, `click`, `key_down`, `key_up`, `mouse_down`, `mouse_up`, `mouse_delta`, `wait`) in one local call.

---

## 3. Telemetry Logger (`gameplayer.telemetry`)

Appends structured JSONL records to `logs/telemetry.jsonl` (configurable via `GAMEPLAYER_TELEMETRY_PATH`).

- `TelemetryLogger.log(tool, action, status="ok", duration_ms=None, details=None, error=None)`
- `TelemetryLogger.timed(tool, action, details=None)` context manager (records `ok`, `error`, or `timeout`)
- `read_telemetry(path="logs/telemetry.jsonl", tool=None, limit=None)`

---

## 4. Rolling Video Capture Daemon & Slice Extractor (`gameplayer.capture_daemon`)

Designed for `watcher` to continuously record clean, downscaled 16:9 game window segments without focusing the window, and extract 10–15s failure clips on demand.

### CLI Usage

```bash
# Start rolling capture daemon (default: 640x360 @ 2.0 fps, 180s segments, keep newest 10)
.venv/bin/python -m gameplayer.capture_daemon run

# Check daemon status and completed segments
.venv/bin/python -m gameplayer.capture_daemon status

# Extract a 12s debug slice from a completed segment
.venv/bin/python -m gameplayer.capture_daemon slice recordings/seg_XXXX.mp4 clips/debug.mp4 --start 45 --duration 12
```

### Guarantees

- Encodes to `.in_progress_seg_*.mp4` and atomically renames to `seg_*.mp4` only after `ffmpeg` closes cleanly (`+faststart`), so readers never see truncated MP4 files.
- Automatically prunes oldest segments beyond `--max-segments` (default `10`, ~30 minutes rolling window).
- Skips transient `WindowNotFoundError` / `WindowCaptureError` frames cleanly and reports status in `logs/capture_daemon.json`.

---

## 5. HUD Region Cropping & Vision OCR (`gameplayer.hud`)

Offline 16:9 HUD region extractor and native macOS Vision OCR (`VNRecognizeTextRequest`, fast level) for extracting:
- `top_left_help`: tutorial/help prompt box
- `top_right_status`: character, time, wanted/cash/weapon status area
- `center_banner`: classifies `WASTED`, `BUSTED`, `MISSION_FAILED`, and `MISSION_PASSED`
- `bottom_subtitle`: bottom subtitle and mission objective bar

### CLI Usage

```bash
# Inspect any saved 16:9 PNG offline (never touches live game window)
.venv/bin/python -m gameplayer.hud --inspect-frame captures/start.png
.venv/bin/python -m gameplayer.hud --inspect-frame captures/start.png --save-crops tmp/crops
```

### Python API

- `inspect_frame(image_or_path, ocr_backend=None, logger=None) -> HudObservation`
- `crop_hud_regions(image, regions=HUD_REGIONS)`
- `classify_center_banner(lines) -> Optional[str]`
- `VisionOCRBackend()` (macOS Vision) or custom `OCRBackend` for offline tests.

---

## 6. Minimap & Health/Armor Extraction (`gameplayer.minimap`)

Calibrated bottom-left 16:9 minimap parser extracting:
- `health_ratio` (`0.0..1.0`, green or low-health red), `armor_ratio` (`0.0..1.0`, blue), & `special_ratio` (`0.0..1.0`, yellow special ability bar)
- `minimap_visible` (false during cutscenes/menus)
- Colored blips & GPS routes (`yellow`, `purple`, `red`, `blue`) -> `bearing` (`-1.0..+1.0`, `0.0` straight ahead) and relative `distance`

### CLI Usage

```bash
.venv/bin/python -m gameplayer.minimap --inspect-frame captures/prologue_9.png
```

---

## 7. Closed-Loop Action Runner (`gameplayer.runner`)

Fast local ~8–12 Hz perception + control loops (`ActionRunner`) combining `capture_game_window`, `inspect_minimap`, `inspect_frame`, and `InputController`:
- `activate_wine_capture(button="right", hold_s=0.04, timeout_s=None)`: locates the GTA 5 window and calls `click_window_center` when Wine ignores inputs despite `is_frontmost=True` (**WARNING**: do NOT use `button="right"` on the Pause Menu).
- `run_until_event(hold_keys=..., hold_buttons=..., predicate=..., config=...)`
- `follow_route_on_foot(sprint=False, target_color="yellow", stop_on_subtitle_keywords=("return to", "wait for", "stay with", "follow"), enable_stuck_recovery=True, cover_on_health_drop=True, ...)`: automatically aborts forward movement if a crew-tether warning subtitle/help prompt appears (preventing `"The crew were left behind."` fails), performs automatic visual-motion obstacle recovery (`a`/`d` sidestep + `space` vault) if forward motion stalls for `>= 1.5s`, and snaps into cover (`q`) if aborted on `health_drop`.
- `follow_blue_ally(sprint=False, arrive_distance=0.14, ...)`: steers back to the nearest blue ally blip (e.g., Michael & Brad during Prologue yard/gate exit—do NOT sprint far down the road alone!).
- `advance_under_fire(sprint=False, target_color="yellow", arrive_distance=0.08, ally_tether_distance=0.28, move_phase_s=2.2, engage_phase_s=3.2, engage_distance=0.65, close_enemy_hold_distance=0.28, min_engage_health=0.25, recover_health=0.42, recover_target_health=0.45, recover_wait_s=2.5, use_special=True, recoil_comp_dy=1, exposed_recover_drop_threshold=0.04, cover_on_timeout=True, bearing_deadband=0.025, engage_align_bearing=0.06, ...)`: continuous closed-loop shoot-and-move advance (default `30s`, zero LLM turn gaps) alternating Move (`w` + obstacle recovery without combat vaulting + `capslock` special), Engage (fine `0.025` ADS steering, finer `18`-count upward `DEFAULT_RETICLE_SWEEP`, `recoil_comp_dy=1` per burst, red-reticle & aligned `<= 0.06` suppressive bursts every 5th tick, holding aim on close/dense threats), and Recover (`q` cover/pause when `health < 0.42`, or immediate retreat toward `blue` ally / diagonal backpedal-to-engage when exposed with `>= 0.04` HP drop or close fire; snaps `q` cover on `timeout` exit if not already in `recover`). Automatically tethers/falls back toward `blue` allies when `yellow` is empty or on crew-tether subtitles, and stops immediately on banners, `target_reached`, or vehicle-entry prompts (`"get in"`, `"enter"`, `"vehicle"`, `"car"`).
- **Prologue Yard Spawn Sequence (from `p3_retry3_spawn.png`)**: Unpause -> walk forward `~1.1s` (`w`) to the front concrete pillar (or right `1337` electrical box) before pressing `q` so Trevor snaps to real crouch cover -> run `cover_peek_combat(enter_cover=False)` to thin the initial yard cops -> bound with Michael/Brad (`advance_under_fire`) past the police cars and gate to the getaway vehicle (`f` to enter).
- `follow_gps_drive(route_color="yellow", ...)` (includes automatic visual-motion stuck-car recovery: reverses `s` + counter-steers for `0.8s` if road motion stalls for `1.5s`)
- `combat_burst_scan(aim_button="right", fire_button="left", ...)`
- `aim_until_red_reticle(...)` / `aim_and_shoot(...)` / `hostage_shot(...)`: holds aim (`right`), sweeps/steers until `detect_red_reticle()` detects hostile red reticle contrast at `(0.5, 0.5)`, and fires precision `left` bursts (strictly waits for red reticle).
- `shoot_enemies(..., use_special=False, cover_on_health_drop=True)`: multi-target combat loop combining minimap bearing steering, red-reticle precision shots, periodic suppressive bursts when aligned on distant/covered enemies (`fallback_fire_on_aligned=True`), periodic aim re-tap to pop out of cover / snap, and automatic `q` cover snap on `health_drop`.
- `cover_peek_combat(enter_cover=True, use_special=True, min_peek_health=0.45, heal_wait_s=3.5, exit_on_health_drop=False, recoil_comp_dy=1, exposed_duck_drop_threshold=0.12, flank_recovery_min_health=0.55, max_unclamp_attempts=1, stop_when_no_enemies=True, no_enemies_clear_s=1.5, ...)`:
  - **Cover-toggle safety**: `q` is a toggle in GTA 5! Pass `enter_cover=False` when already crouched in cover so it does not stand out of cover.
  - Keeps aim held during in-FOV cover steering (never dropping peek to steer), applies gentle downward `recoil_comp_dy=1` per burst, scans torso/cover heights via `DEFAULT_RETICLE_SWEEP` with `_recenter_sweep()` on ducks and exits (including timeout), attempts one free-look flank-recovery turn + sidestep when clamped (`|bearing| > 0.45`) and `health >= 0.55` (aborting `cover_clamped` if low health or still clamped), aborts immediately as `"exposed_in_cover"` if health keeps dropping `>= 0.12` while ducked, activates `capslock` special when `special_ratio >= 0.20`, ducks/regenerates for `heal_wait_s=3.5s` on `< min_peek_health` or rapid `>= 0.20` health drop and resumes in-loop, and exits early as `"enemies_cleared"` once no red minimap blips appear for `1.5s`.
- `switch_character(character, method="hotkey" | "wheel")`

Automatically releases ALL held inputs in `finally` on `WASTED`/`BUSTED`/`MISSION_FAILED`/`MISSION_PASSED` banners, new `top_left_help` / subtitle changes, rapid `health_drop`, target reached, or `max_duration_s` timeout.
