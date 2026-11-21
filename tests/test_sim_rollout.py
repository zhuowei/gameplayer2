from __future__ import annotations

from gameplayer.input import InputController
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
from tests.test_runner import ScriptedHudOCR


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
