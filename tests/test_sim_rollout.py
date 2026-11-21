from __future__ import annotations

from PIL import Image, ImageDraw

from gameplayer.hud import OcrLine
from gameplayer.input import InputController
from gameplayer.perception import classify_posture
from gameplayer.playbooks import prologue_shutter_cover
from gameplayer.runner import ActionRunner, LoopConfig
from gameplayer.telemetry import TelemetryLogger, read_telemetry
from tests.sim_harness import (
    BoxOccluder,
    KinematicWorld,
    SimClock,
    SimEnemy,
    SimInputBackend,
    SimWindowBackend,
)
from tests.test_minimap import make_synthetic_minimap_frame
from tests.test_runner import FakeClock, RecordingInputBackend, ReplayWindowBackend, ScriptedHudOCR


def _make_sim_runner(world: KinematicWorld, tmp_path, name: str, initial_time: float = 500.0):
    clock = SimClock(initial_time=initial_time, on_advance=world.step)
    input_backend = SimInputBackend(world, clock)
    log_path = tmp_path / f"{name}.jsonl"
    logger = TelemetryLogger(log_path)
    ctrl = InputController(
        backend=input_backend,
        logger=logger,
        clock=clock,
        sleep_fn=clock.sleep,
        register_signals=False,
    )
    win_backend = SimWindowBackend(world)
    runner = ActionRunner(
        input_controller=ctrl,
        window_backend=win_backend,
        ocr_backend=ScriptedHudOCR(lambda _idx: {}),
        logger=logger,
        clock=clock,
        sleep_fn=clock.sleep,
    )
    return runner, clock, input_backend, log_path


def test_advance_flank_unblock_full_move_window_after_sleeps_nonzero_clock(tmp_path) -> None:
    """Regression for runner.py:1031-1048: flank_unblock_until_t must advance from post-sleep clock."""
    # Place close occluded enemy behind a narrow pillar directly ahead (y=6.0m).
    # Note: non-zero initial_time=500.0 proves flank_unblock_until_t isn't mixing raw clock() vs elapsed_s.
    world = KinematicWorld(
        player_x=0.0,
        player_y=0.0,
        health=0.95,
        occluders=[BoxOccluder(x0=-1.1, y0=2.2, x1=1.1, y1=3.2)],
        enemies=[SimEnemy(x=0.0, y=6.5, hp=90.0)],
    )
    runner, _clock, input_backend, log_path = _make_sim_runner(world, tmp_path, "flank_timing", initial_time=500.0)
    res = runner.advance_under_fire(
        aim_settle_s=0.0,
        move_phase_s=0.25,
        engage_phase_s=0.5,
        scan_pattern=(),
        config=LoopConfig(max_duration_s=10.0, hz=10.0, hud_check_interval_s=0.0, health_drop_threshold=None),
    )

    logs = read_telemetry(log_path, tool="runner")
    flanks = [r for r in logs if r.get("action") == "advance_flank_unblock"]
    assert len(flanks) >= 1
    # After the 0.20s 's' + 0.55s 'd' sleeps, Trevor must actually hold 'w' to walk forward around the box!
    first_flank_idx = next(i for i, ev in enumerate(input_backend.events) if ev == ("key_up", "d"))
    post_flank_events = input_backend.events[first_flank_idx + 1 :]
    assert ("key_down", "w") in post_flank_events
    # Verify Trevor physically moved right (around x1=1.1) and forward past the initial y=0.0 position
    assert world.player_x > 0.8
    assert res.elapsed_s <= 10.2
    assert world.held_keys == set()
    assert world.held_buttons == set()


def test_visual_targeting_steers_directly_onto_off_horizon_enemy(tmp_path) -> None:
    """Visual human/silhouette targeting steers both yaw (dx) and pitch (dy) onto visible enemies without blind sweep."""
    # Enemy is right (+8° yaw) and elevated (+0.9m above eye_z -> requires upward pitch dy < 0).
    # With scan_pattern=(), blind bearing steering alone has dy=0 and cannot hit an elevated enemy.
    world = KinematicWorld(
        player_x=0.0,
        player_y=0.0,
        eye_z=1.45,
        health=0.95,
        enemies=[SimEnemy(x=1.8, y=10.0, z_center=2.55, half_width_m=0.40, half_height_m=0.50, hp=70.0)],
    )
    runner, _clock, input_backend, log_path = _make_sim_runner(world, tmp_path, "visual_steer")
    runner.advance_under_fire(
        aim_settle_s=0.05,
        move_phase_s=0.1,
        engage_phase_s=4.0,
        scan_pattern=(),
        config=LoopConfig(max_duration_s=6.0, hz=10.0, hud_check_interval_s=0.0, health_drop_threshold=None),
    )

    logs = read_telemetry(log_path, tool="runner")
    assert any(r.get("action") == "visual_target_steer" for r in logs)
    assert world.shots_landed >= 1
    assert not world.enemies[0].alive
    # Confirm pitch steered upward (negative dy) to hit elevated enemy even with scan_pattern=()
    assert any(ev[0] == "mouse_delta" and ev[1][1] < 0 for ev in input_backend.events)
    assert world.held_keys == set()
    assert world.held_buttons == set()


def test_cover_peek_combat_detects_cover_yaw_clamp_via_optical_flow_and_respects_048_regen_cap(tmp_path) -> None:
    """Optical flow catches cover yaw clamping (Δyaw ≈ 0) and passive cover regen caps at 0.48 without stalling."""
    world = KinematicWorld(
        player_x=0.0,
        player_y=0.0,
        in_cover=True,
        health=0.30,
        cover_yaw_clamp_deg=12.0,  # Clamps cover aim well short of enemy at +55°
        enemies=[SimEnemy(x=10.0, y=7.0, hp=100.0)],
    )
    runner, _clock, _input_backend, log_path = _make_sim_runner(world, tmp_path, "clamp_and_regen")
    res = runner.cover_peek_combat(
        enter_cover=False,
        use_special=False,
        min_peek_health=0.45,
        heal_wait_s=0.8,
        aim_settle_s=0.05,
        flank_recovery_min_health=0.90,  # Force abort via cover_clamped once clamp is detected
        scan_pattern=(),
        config=LoopConfig(max_duration_s=10.0, hz=10.0, hud_check_interval_s=0.0, health_drop_threshold=None),
    )

    # Health healed passively from 0.30 up to the 0.48 cap (past min_peek_health=0.45), never exceeding 0.48.
    assert 0.45 <= world.health <= 0.4801
    # Optical-flow tracker detected clamped yaw (Δyaw ≈ 0 despite commanded dx >= 18) and aborted cover_clamped.
    assert res.stop_reason == "cover_clamped"
    logs = read_telemetry(log_path, tool="runner")
    assert any(r.get("action") == "camera_yaw_stalled" for r in logs)
    assert world.held_keys == set()
    assert world.held_buttons == set()


def test_optical_flow_compensates_upward_recoil_and_ads_hipfire_sensitivity_stays_level(tmp_path) -> None:
    """0.75 px/round upward recoil triggers optical-flow downward compensation and timeout exit preserves level pitch."""
    world = KinematicWorld(
        player_x=0.0,
        player_y=0.0,
        in_cover=True,
        health=0.95,
        recoil_px_per_round=1.5,  # Exaggerated recoil when recoil_comp_dy=0 tests optical-flow feedback directly
        enemies=[SimEnemy(x=0.0, y=14.0, z_center=1.45, half_width_m=0.55, half_height_m=0.75, hp=1000.0)],
    )
    runner, _clock, _input_backend, log_path = _make_sim_runner(world, tmp_path, "recoil_flow")
    runner.cover_peek_combat(
        enter_cover=False,
        use_special=False,
        aim_settle_s=0.05,
        peek_burst_s=5.0,
        recoil_comp_dy=0,  # Disable open-loop recoil comp so optical-flow detector must compensate
        config=LoopConfig(max_duration_s=4.5, hz=10.0, hud_check_interval_s=0.0, health_drop_threshold=None),
    )

    logs = read_telemetry(log_path, tool="runner")
    assert world.rounds_fired >= 6
    assert any(r.get("action") == "camera_recoil_comp" for r in logs)
    # _recenter_sweep runs BEFORE mouse_up("right") so 2.8x hipfire sensitivity never slams camera pitch into the ground.
    assert abs(world.pitch_deg) <= 8.0
    assert world.held_keys == set()
    assert world.held_buttons == set()


def test_follow_route_on_foot_wall_corner_unstick_backs_up_turns_without_vault(tmp_path) -> None:
    """Cycle 3 (#283): when stuck facing a wall with |bearing| > 0.35, back up 's' and turn without wall-vaulting."""
    world = KinematicWorld(
        player_x=0.0,
        player_y=0.42,
        yaw_deg=0.0,
        health=0.95,
        hipfire_deg_per_px=0.12,
        occluders=[BoxOccluder(x0=-4.0, y0=0.8, x1=4.0, y1=1.4)],
        yellow_target=(-1.5, -5.0),  # Behind-left (bearing ~ -0.91 -> still |bearing| > 0.35 at stuck trigger)
    )
    runner, _clock, input_backend, log_path = _make_sim_runner(world, tmp_path, "wall_unstick", initial_time=600.0)
    res = runner.follow_route_on_foot(
        stuck_window_s=0.4,
        unstuck_duration_s=0.25,
        config=LoopConfig(max_duration_s=4.5, hz=10.0, hud_check_interval_s=0.0, health_drop_threshold=None),
    )

    logs = read_telemetry(log_path, tool="runner")
    recs = [r for r in logs if r.get("action") == "on_foot_stuck_recovery"]
    assert len(recs) >= 1
    assert recs[0]["details"]["vault"] is False
    assert recs[0]["details"]["back_up_s"] == 0.25
    assert recs[0]["details"]["steer_dx"] < 0
    assert ("key_down", "space") not in input_backend.events
    assert ("key_down", "s") in input_backend.events
    # Player turned around from yaw=0 and moved away from the wall toward (-1.5, -5.0)
    assert world.yaw_deg < -100.0
    assert world.player_y < 0.0 or res.stop_reason == "target_reached"
    assert world.held_keys == set()
    assert world.held_buttons == set()


def test_follow_route_on_foot_indoor_wall_bypass_clears_doorway_to_target(tmp_path) -> None:
    """Bug-algorithm bypass: when yellow blip is straight behind an indoor wall, sidestep + brief bypass clears the doorway."""
    world = KinematicWorld(
        player_x=0.0,
        player_y=0.0,
        yaw_deg=0.0,
        health=0.95,
        # Wall blocks straight path to (0.0, 5.5) with doorway open on the right (x > 1.35)
        occluders=[BoxOccluder(x0=-3.0, y0=1.2, x1=1.35, y1=1.8)],
        yellow_target=(0.0, 5.5),
    )
    runner, _clock, _input_backend, log_path = _make_sim_runner(world, tmp_path, "indoor_bypass", initial_time=700.0)
    res = runner.follow_route_on_foot(
        stuck_window_s=0.5,
        unstuck_duration_s=0.65,
        arrive_distance=0.10,
        config=LoopConfig(max_duration_s=7.0, hz=10.0, hud_check_interval_s=0.0, health_drop_threshold=None),
    )

    assert res.stop_reason == "target_reached"
    assert world.player_y > 2.2
    logs = read_telemetry(log_path, tool="runner")
    recs = [r for r in logs if r.get("action") == "on_foot_stuck_recovery"]
    assert len(recs) >= 1
    assert recs[0]["details"]["back_up_s"] == 0.25
    assert world.held_keys == set()
    assert world.held_buttons == set()


def test_follow_route_on_foot_narrow_doorway_jamb_commits_gentle_steps_without_vault(tmp_path) -> None:
    """Cycle 5 (#299): narrow doorway jamb recovery commits 2 gentle steps (0.28s/0.42s), skips jamb vaulting, and clears opening."""
    world = KinematicWorld(
        player_x=0.0,
        player_y=0.0,
        yaw_deg=0.0,
        health=0.95,
        # Doorway opening x in [0.60, 2.05] (walkable slot [0.95, 1.70] with 0.35m collision pad)
        occluders=[
            BoxOccluder(x0=-3.0, y0=1.2, x1=0.60, y1=1.8),
            BoxOccluder(x0=2.05, y0=1.2, x1=4.0, y1=1.8),
        ],
        yellow_target=(1.30, 5.2),
    )
    runner, _clock, _input_backend, log_path = _make_sim_runner(world, tmp_path, "narrow_jamb", initial_time=800.0)
    res = runner.follow_route_on_foot(
        stuck_window_s=0.5,
        arrive_distance=0.10,
        config=LoopConfig(max_duration_s=6.0, hz=10.0, hud_check_interval_s=0.0, health_drop_threshold=None),
    )

    assert res.stop_reason == "target_reached"
    assert world.player_y > 2.2
    logs = read_telemetry(log_path, tool="runner")
    recs = [r for r in logs if r.get("action") == "on_foot_stuck_recovery"]
    assert len(recs) >= 1
    assert recs[0]["details"]["duration_s"] == 0.28
    assert recs[0]["details"]["sidestep_key"] == "d"
    if len(recs) >= 2:
        assert recs[1]["details"]["sidestep_key"] == "d"
        assert recs[1]["details"]["duration_s"] == 0.42
        assert recs[1]["details"]["vault"] is False
    assert world.held_keys == set()
    assert world.held_buttons == set()


def test_follow_route_on_foot_detects_distance_stall_during_camera_pan(tmp_path) -> None:
    """#303: even when camera panning keeps road_motion >= threshold, zero minimap distance progress triggers unstick."""
    world = KinematicWorld(
        player_x=0.0,
        player_y=0.0,
        yaw_deg=110.0,  # Large off-axis initial camera angle so early ticks pan continuously
        health=0.95,
        occluders=[BoxOccluder(x0=-3.0, y0=0.45, x1=0.9, y1=1.2)],
        yellow_target=(0.0, 5.5),
    )
    runner, _clock, _input_backend, log_path = _make_sim_runner(world, tmp_path, "pan_stall", initial_time=900.0)
    res = runner.follow_route_on_foot(
        stuck_window_s=0.9,
        arrive_distance=0.10,
        config=LoopConfig(max_duration_s=9.5, hz=10.0, hud_check_interval_s=0.0, health_drop_threshold=None),
    )

    assert res.stop_reason == "target_reached"
    logs = read_telemetry(log_path, tool="runner")
    assert any(r.get("action") == "on_foot_stuck_recovery" for r in logs)
    assert world.held_keys == set()
    assert world.held_buttons == set()


def test_enter_cover_idempotent_on_chest_high_shift_score_and_posture(tmp_path) -> None:
    """Cycle 7 (#311): chest-high cover band classifies `in_cover`, and shift_score >= 10 latches idempotency."""
    chest_img = Image.new("RGB", (640, 360), (90, 96, 108))
    draw = ImageDraw.Draw(chest_img)
    # Uniform open upper half, vertical chest-high counter seam in lower rows [0.49..0.90]
    for x in range(int(0.18 * 640), int(0.36 * 640), 10):
        draw.rectangle((x, 176, x + 5, 324), fill=(52, 54, 58))
        draw.rectangle((x + 5, 176, x + 10, 324), fill=(78, 80, 84))
    assert classify_posture(chest_img).state == "in_cover"

    # Low-contrast standing frame before cover, then high-shift chest-high frame after pressing `q`.
    f0 = make_synthetic_minimap_frame(health_ratio=0.90)
    d0 = ImageDraw.Draw(f0)
    d0.rectangle((220, 140, 420, 260), fill=(35, 35, 42))
    f1 = make_synthetic_minimap_frame(health_ratio=0.90)
    d1 = ImageDraw.Draw(f1)
    d1.rectangle((220, 140, 420, 260), fill=(215, 215, 220))
    # Add slightly textured non-blank scene so before/after classify standing (<0.85 conf) but shift_score >= 10.
    for x in range(120, 500, 40):
        d0.rectangle((x, 40, x + 16, 120), fill=(48, 52, 60))
        d1.rectangle((x, 40, x + 16, 120), fill=(48, 52, 60))

    clock = FakeClock()
    backend = RecordingInputBackend()
    log_path = tmp_path / "cover_idemp.jsonl"
    logger = TelemetryLogger(log_path)
    ctrl = InputController(backend=backend, logger=logger, clock=clock, sleep_fn=clock.sleep, register_signals=False)
    frames = [f0, f1, f1, f1]
    idx_box = {"i": 0}

    def _next_frame(idx: int) -> Image.Image:
        i = idx_box["i"]
        idx_box["i"] += 1
        return frames[min(i, len(frames) - 1)]

    runner = ActionRunner(ctrl, ReplayWindowBackend(_next_frame), ScriptedHudOCR(lambda _i: {}), logger, clock, clock.sleep)
    runner.enter_cover()
    assert runner._in_cover_hint is True
    q_count_first = sum(1 for ev in backend.events if ev == ("key_down", "q"))
    assert q_count_first == 1

    # Second call must NOT toggle `q` back out of cover
    runner.enter_cover()
    q_count_second = sum(1 for ev in backend.events if ev == ("key_down", "q"))
    assert q_count_second == 1
    assert any(r.get("action") == "enter_cover_noop" for r in read_telemetry(log_path, tool="runner"))


def test_calibrate_controls_vertical_pitch_down_sign_normal_and_inverted(tmp_path) -> None:
    """Cycle 7 (#311): calibrate_controls probes vertical dy and recovers pitch_down_sign (+1 vs -1)."""
    world_norm = KinematicWorld(player_x=0.0, player_y=0.0, health=0.95, invert_pitch=False)
    runner_norm, _, _, _ = _make_sim_runner(world_norm, tmp_path, "cal_norm", initial_time=1000.0)
    rep_norm = runner_norm.calibrate_controls(probe_dx=20, save_path=None)
    assert rep_norm["pitch_down_sign"] == 1
    assert float(rep_norm["pitch_px_per_dy"]) > 0.0
    assert world_norm.held_keys == set()

    world_inv = KinematicWorld(player_x=0.0, player_y=0.0, health=0.95, invert_pitch=True)
    runner_inv, _, _, _ = _make_sim_runner(world_inv, tmp_path, "cal_inv", initial_time=1050.0)
    rep_inv = runner_inv.calibrate_controls(probe_dx=20, save_path=None)
    assert rep_inv["pitch_down_sign"] == -1
    assert float(rep_inv["pitch_px_per_dy"]) > 0.0
    assert world_inv.held_keys == set()


def test_level_camera_pitch_rescues_ceiling_lock_both_signs_and_preserves_horizon(tmp_path) -> None:
    """Cycle 7: level_camera_pitch rescues -60 deg ceiling lock for normal & inverted Y, keeps horizon net 0."""
    for inv in (False, True):
        w_ceil = KinematicWorld(player_x=0.0, player_y=0.0, pitch_deg=-60.0, invert_pitch=inv)
        r_ceil, _, _, _ = _make_sim_runner(w_ceil, tmp_path, f"ceil_{inv}", initial_time=1100.0)
        rep = r_ceil.level_camera_pitch(probe_dy=14, recovery_dy=150, max_passes=2)
        assert rep["status"] == "recovered_ceiling_clamp"
        assert w_ceil.pitch_deg > -35.0
        assert rep["pitch_down_sign"] == (-1 if inv else 1)

    w_hor = KinematicWorld(player_x=0.0, player_y=0.0, pitch_deg=0.0, invert_pitch=True)
    r_hor, _, _, _ = _make_sim_runner(w_hor, tmp_path, "hor_ok", initial_time=1150.0)
    rep_hor = r_hor.level_camera_pitch()
    assert rep_hor["status"] == "horizon_ok"
    assert abs(w_hor.pitch_deg) <= 1.0

    w_drop = KinematicWorld(player_x=0.0, player_y=0.0, pitch_deg=-60.0, drop_mouse_deltas=True)
    r_drop, _, _, _ = _make_sim_runner(w_drop, tmp_path, "drop_pitch", initial_time=1200.0)
    rep_drop = r_drop.level_camera_pitch()
    assert rep_drop["status"] == "unresponsive"


def test_follow_route_on_foot_pivots_close_target_and_stops_on_get_in_cover(tmp_path) -> None:
    """Cycle 7: close-range large-bearing target pivots cleanly instead of orbiting; 'get in cover' stops near blip."""
    world = KinematicWorld(
        player_x=0.0,
        player_y=0.0,
        yaw_deg=135.0,  # Off-axis close target (3.0m -> dist=0.086) that used to orbit endlessly
        health=0.95,
        yellow_target=(0.0, 3.0),
    )
    runner, _, _, _ = _make_sim_runner(world, tmp_path, "close_pivot", initial_time=1250.0)
    res = runner.follow_route_on_foot(
        arrive_distance=0.055,
        config=LoopConfig(max_duration_s=4.5, hz=10.0, hud_check_interval_s=0.0, health_drop_threshold=None),
    )
    assert res.stop_reason == "target_reached"
    assert world.held_keys == set()

    # Subtitle "get in cover" stops automatically when already near the cover blip
    clock2 = FakeClock()
    backend2 = RecordingInputBackend()
    logger2 = TelemetryLogger(tmp_path / "cover_sub.jsonl")
    ctrl2 = InputController(backend=backend2, logger=logger2, clock=clock2, sleep_fn=clock2.sleep, register_signals=False)
    near_frame = make_synthetic_minimap_frame(yellow_points=[(0.55, 0.70)])
    ocr2 = ScriptedHudOCR(
        lambda _idx: {"bottom_subtitle": [OcrLine("T, get in cover. B, set the charge.", 0.95, (0.2, 0.8, 0.6, 0.1))]}
    )
    runner2 = ActionRunner(ctrl2, ReplayWindowBackend(lambda _i: near_frame), ocr2, logger2, clock2, clock2.sleep)
    res2 = runner2.follow_route_on_foot(config=LoopConfig(max_duration_s=3.0, hz=10.0, hud_check_interval_s=0.0))
    assert res2.stop_reason == "subtitle_keyword:get in cover"
    assert backend2.currently_held_keys == set()


def test_approach_and_enter_cover_shutter_room_with_ceiling_lock_and_pillar(tmp_path) -> None:
    """Cycle 7 (#308/#311): full closed-loop Shutter Room recovery levels ceiling pitch, rounds pillar, and snaps cover."""
    world = KinematicWorld(
        player_x=0.0,
        player_y=0.0,
        yaw_deg=85.0,
        pitch_deg=-60.0,
        invert_pitch=True,
        health=0.95,
        occluders=[BoxOccluder(x0=-1.4, y0=1.1, x1=0.8, y1=1.8)],
        yellow_target=(0.3, 4.2),
        cover_snap_radius_m=3.1,
        clear_yellow_on_cover_snap=True,
    )
    runner, _, _, _ = _make_sim_runner(world, tmp_path, "shutter_cover", initial_time=1300.0)
    pb = prologue_shutter_cover(runner, arrive_distance=0.085, max_duration_s=18.0)

    assert pb.outcome == "cover_entered"
    assert world.in_cover is True
    assert world.yellow_target is None
    assert world.pitch_deg > -35.0
    assert world.held_keys == set()
    assert world.held_buttons == set()


def test_level_camera_pitch_adapts_probe_to_low_calibrated_gain_and_cover_slides_without_untoggle(tmp_path) -> None:
    """Cycle 7b (#317): low pitch_px_per_dy auto-scales probe_dy, and latched cover slides 'a'/'d' without untoggling q."""
    # 1) Low pitch sensitivity (~0.06 deg/px ~ 1.3 px per 14 counts, ~4.0+ px per 44 counts)
    w_low = KinematicWorld(player_x=0.0, player_y=0.0, pitch_deg=-60.0, hipfire_deg_per_px=0.06, invert_pitch=False)
    r_low, _, _, _ = _make_sim_runner(w_low, tmp_path, "low_gain_pitch", initial_time=1400.0)
    r_low._pitch_px_per_dy = 0.22  # Triggers adaptive probe_dy = int(round(10 / 0.22)) = 45
    rep = r_low.level_camera_pitch(recovery_dy=300, max_passes=2)
    assert rep["status"] == "recovered_ceiling_clamp"
    assert w_low.pitch_deg > -45.0

    # 2) Cover latched via shift_score >= 10 slides laterally ('a'/'d') instead of spamming q
    f_before = make_synthetic_minimap_frame(health_ratio=0.95, yellow_points=[(0.50, 0.72)])
    ImageDraw.Draw(f_before).rectangle((220, 140, 420, 260), fill=(35, 35, 42))
    f_cover = make_synthetic_minimap_frame(health_ratio=0.95, yellow_points=[(0.50, 0.72)])
    ImageDraw.Draw(f_cover).rectangle((220, 140, 420, 260), fill=(215, 215, 220))
    for x in range(120, 500, 40):
        ImageDraw.Draw(f_before).rectangle((x, 40, x + 16, 120), fill=(48, 52, 60))
        ImageDraw.Draw(f_cover).rectangle((x, 40, x + 16, 120), fill=(48, 52, 60))

    clock2 = FakeClock()
    backend2 = RecordingInputBackend()
    logger2 = TelemetryLogger(tmp_path / "slide_no_untoggle.jsonl")
    ctrl2 = InputController(backend=backend2, logger=logger2, clock=clock2, sleep_fn=clock2.sleep, register_signals=False)
    frames2 = [f_before, f_before, f_cover]
    idx2 = {"i": 0}

    def _next_slide_frame(_idx: int) -> Image.Image:
        cur = frames2[min(idx2["i"], len(frames2) - 1)]
        idx2["i"] += 1
        return cur

    r_slide = ActionRunner(ctrl2, ReplayWindowBackend(_next_slide_frame), ScriptedHudOCR(lambda _i: {}), logger2, clock2, clock2.sleep)
    r_slide.approach_and_enter_cover(level_pitch=False, max_duration_s=1.2, config=LoopConfig(max_duration_s=1.2, hz=10.0, hud_check_interval_s=0.0))
    q_downs = [ev for ev in backend2.events if ev == ("key_down", "q")]
    assert len(q_downs) == 1
    assert ("key_down", "a") in backend2.events
    assert ("key_down", "d") in backend2.events
    assert backend2.currently_held_keys == set()


def test_run_loop_stops_and_releases_on_focus_loss(tmp_path) -> None:
    """Cycle 8 (#322): background app focus theft (e.g. Notes) aborts loops as 'focus_lost' and releases all keys."""
    from gameplayer.window import WindowBounds, WindowInfo

    class _FocusLostWindowBackend:
        def __init__(self) -> None:
            self.calls = 0

        def list_windows(self) -> list[WindowInfo]:
            self.calls += 1
            front = self.calls <= 2
            obs = () if front else ("Notes(pid=6487,wid=44)",)
            return [
                WindowInfo(
                    window_id=135,
                    owner="GTA5.exe",
                    title="Grand Theft Auto V",
                    pid=1212,
                    bounds=WindowBounds(0, 0, 1280, 720),
                    layer=0,
                    is_onscreen=True,
                    is_frontmost=front,
                    obstructed_by=obs,
                )
            ]

        def capture_window_image(self, _wid: int) -> Image.Image:
            return make_synthetic_minimap_frame(yellow_points=[(0.50, 0.35)])

        def focus_window(self, _win: WindowInfo) -> bool:
            return True

    clock = FakeClock()
    backend = RecordingInputBackend()
    log_path = tmp_path / "focus_loss.jsonl"
    logger = TelemetryLogger(log_path)
    ctrl = InputController(backend=backend, logger=logger, clock=clock, sleep_fn=clock.sleep, register_signals=False)
    runner = ActionRunner(ctrl, _FocusLostWindowBackend(), ScriptedHudOCR(lambda _i: {}), logger, clock, clock.sleep)
    res = runner.follow_route_on_foot(config=LoopConfig(max_duration_s=3.0, hz=10.0, hud_check_interval_s=0.0))
    assert res.stop_reason == "focus_lost"
    assert backend.currently_held_keys == set()
    assert backend.currently_held_buttons == set()
    assert any(r.get("action") == "focus_lost" for r in read_telemetry(log_path, tool="runner"))


def test_approach_and_enter_cover_orbits_blocked_face_to_clear_side(tmp_path) -> None:
    """Cycle 8 (#321/#322): when close local probes fail behind a blocked prop face, back-out + orbit reaches clear side."""
    # Player starts directly south of a crate (y in [1.2, 2.2]), yellow cover spot is on the east side (2.0, 2.0)
    # within snap radius 1.4m, so initial south-face q cannot reach (2.0, 2.0) until orbiting right ('d').
    world = KinematicWorld(
        player_x=0.0,
        player_y=0.0,
        yaw_deg=0.0,
        pitch_deg=0.0,
        health=0.95,
        occluders=[BoxOccluder(x0=-1.0, y0=1.2, x1=1.0, y1=2.6)],
        yellow_target=(1.8, 2.0),
        cover_snap_radius_m=1.25,
        clear_yellow_on_cover_snap=True,
    )
    runner, _, in_backend, _ = _make_sim_runner(world, tmp_path, "orbit_cover", initial_time=1500.0)
    res = runner.approach_and_enter_cover(
        arrive_distance=0.09,
        max_duration_s=14.0,
        level_pitch=False,
        config=LoopConfig(max_duration_s=14.0, hz=10.0, hud_check_interval_s=0.0),
    )
    assert res.stop_reason == "cover_entered"
    assert world.in_cover is True
    assert world.yellow_target is None
    assert ("key_down", "s") in in_backend.events
    assert ("key_down", "d") in in_backend.events
    assert world.held_keys == set()
