from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from gameplayer.input import InputController
from gameplayer.perception import SilhouetteHumanDetectorBackend, classify_posture
from gameplayer.playbooks import open_phone_menu, prologue_yard_clear_and_push
from gameplayer.replay_bench import benchmark_clip, benchmark_frames
from gameplayer.runner import ActionRunner
from gameplayer.status_store import RunStatusSnapshot, main as status_cli_main, read_status_snapshot, write_status_snapshot
from gameplayer.telemetry import TelemetryLogger
from tests.sim_harness import KinematicWorld, SimClock, SimEnemy, SimInputBackend, SimWindowBackend
from tests.test_minimap import make_synthetic_minimap_frame
from tests.test_runner import FakeClock, RecordingInputBackend, ReplayWindowBackend, ScriptedHudOCR


def _make_cover_vs_standing_scene(*, in_cover: bool) -> Image.Image:
    img = Image.new("RGB", (640, 360), (95, 112, 134))
    draw = ImageDraw.Draw(img)
    # Ground & skyline
    draw.rectangle((0, 180, 640, 360), fill=(195, 206, 218))
    for sx in range(20, 620, 48):
        draw.rectangle((sx, 140, sx + 10, 180), fill=(60, 68, 78))
    if in_cover:
        # Vertical concrete pillar/wall along left-of-Trevor band x=[0.18..0.36] spanning top to bottom
        for x in range(int(0.18 * 640), int(0.36 * 640), 12):
            draw.rectangle((x, 10, x + 6, 350), fill=(70, 72, 76))
            draw.rectangle((x + 6, 10, x + 12, 350), fill=(88, 90, 94))
    return img


def test_classify_posture_and_idempotent_enter_exit_cover(tmp_path) -> None:
    # 1) Blank uniform frame -> unknown; synthetic cover wall -> in_cover; open horizon -> standing
    blank = make_synthetic_minimap_frame(health_ratio=0.90)
    assert classify_posture(blank).state == "unknown"

    cover_img = _make_cover_vs_standing_scene(in_cover=True)
    stand_img = _make_cover_vs_standing_scene(in_cover=False)
    assert classify_posture(cover_img).state == "in_cover"
    assert classify_posture(stand_img).state == "standing"

    # Also validate against real captured GTA 5 screenshots when present
    p_stand = Path("/Users/zhuowei/Documents/proj/gameplayer/captures/p3_after_cover.png")
    p_cover = Path("/Users/zhuowei/Documents/proj/gameplayer/captures/p3_heal_cover.png")
    if p_stand.exists() and p_cover.exists():
        assert classify_posture(Image.open(p_stand)).state == "standing"
        assert classify_posture(Image.open(p_cover)).state == "in_cover"

    # 2) Idempotent enter_cover() skips q when already in_cover; exit_cover() blocks below min_health
    clock = FakeClock()
    backend = RecordingInputBackend()
    logger = TelemetryLogger(tmp_path / "t_posture.jsonl")
    ctrl = InputController(backend=backend, logger=logger, clock=clock, sleep_fn=clock.sleep, register_signals=False)

    runner_already_cover = ActionRunner(
        ctrl, ReplayWindowBackend(lambda _i: cover_img), ScriptedHudOCR(lambda _i: {}), logger, clock, clock.sleep
    )
    post = runner_already_cover.enter_cover()
    assert post.state == "in_cover"
    assert ("key_down", "q") not in backend.events

    # Low-HP exit_cover guard refuses to stand out of cover
    low_hp_cover = make_synthetic_minimap_frame(health_ratio=0.28)
    runner_low = ActionRunner(
        ctrl, ReplayWindowBackend(lambda _i: low_hp_cover), ScriptedHudOCR(lambda _i: {}), logger, clock, clock.sleep
    )
    exited, _ = runner_low.exit_cover(min_health=0.42)
    assert exited is False
    assert ("key_down", "q") not in backend.events


def test_startup_calibrate_controls_measures_ads_hipfire_ratio_and_detects_drop(tmp_path) -> None:
    world = KinematicWorld(player_x=0.0, player_y=0.0, health=0.95)
    clock = SimClock(initial_time=200.0, on_advance=world.step)
    backend = SimInputBackend(world, clock)
    logger = TelemetryLogger(tmp_path / "t_cal.jsonl")
    ctrl = InputController(backend=backend, logger=logger, clock=clock, sleep_fn=clock.sleep, register_signals=False)
    runner = ActionRunner(ctrl, SimWindowBackend(world), ScriptedHudOCR(lambda _i: {}), logger, clock, clock.sleep)

    cal_path = tmp_path / "calibration.json"
    report = runner.calibrate_controls(probe_dx=20, save_path=cal_path)
    assert cal_path.exists()
    assert report["mouse_responsive"] is True
    assert report["warning"] is None
    assert report["elapsed_s"] < 5.0
    # Simulator hipfire=0.28 deg/px, ADS=0.10 deg/px -> ratio ~ 0.357
    assert 0.20 <= float(report["ads_to_hipfire_ratio"]) <= 0.50
    assert world.held_keys == set()
    assert world.held_buttons == set()

    # Broken/Raw-Input dropped mouse deltas must warn immediately
    world_drop = KinematicWorld(player_x=0.0, player_y=0.0, health=0.95, drop_mouse_deltas=True)
    clock2 = SimClock(initial_time=200.0, on_advance=world_drop.step)
    ctrl2 = InputController(
        backend=SimInputBackend(world_drop, clock2), logger=logger, clock=clock2, sleep_fn=clock2.sleep, register_signals=False
    )
    runner2 = ActionRunner(ctrl2, SimWindowBackend(world_drop), ScriptedHudOCR(lambda _i: {}), logger, clock2, clock2.sleep)
    report_drop = runner2.calibrate_controls(probe_dx=20, save_path=None)
    assert report_drop["mouse_responsive"] is False
    assert report_drop["warning"] == "mouse_unresponsive_check_raw_input_or_focus"


def test_status_store_atomic_roundtrip(tmp_path, capsys) -> None:
    status_path = tmp_path / "logs" / "status.json"
    snap = RunStatusSnapshot(
        mission="Prologue",
        checkpoint="Bank Vault",
        consecutive_failures=0,
        minutes_on_checkpoint=2.5,
        llm_turns_on_checkpoint=3,
        last_commit="410ee3f",
        red_reticle_rate=0.27,
        active_issues=["arrow-up phone prompt"],
    )
    written = write_status_snapshot(snap, path=status_path)
    assert written == status_path
    loaded = read_status_snapshot(path=status_path)
    assert loaded is not None
    assert loaded.mission == "Prologue"
    assert loaded.red_reticle_rate == 0.27
    assert status_cli_main(["--path", str(status_path), "--show"]) == 0
    assert "Bank Vault" in capsys.readouterr().out


def test_replay_bench_evaluates_frames_and_clip(tmp_path) -> None:
    f1 = _make_cover_vs_standing_scene(in_cover=True)
    f2 = _make_cover_vs_standing_scene(in_cover=False)
    p1 = tmp_path / "f1.png"
    p2 = tmp_path / "f2.png"
    f1.save(p1)
    f2.save(p2)

    res = benchmark_frames(
        [p1, p2],
        clip_label="synthetic_pair",
        fps=2.0,
        use_vision=False,
        human_detector=SilhouetteHumanDetectorBackend(),
    )
    assert res.frames_evaluated == 2
    assert res.in_cover_frames == 1
    assert res.standing_frames == 1

    clip_path = Path("/Users/zhuowei/Documents/proj/gameplayer/clips/yard_flank_stall_1551.mp4")
    if clip_path.exists():
        clip_res = benchmark_clip(clip_path, fps=2.0, max_frames=4, use_vision=False)
        assert clip_res.frames_evaluated == 4
        assert clip_res.minimap_visible_frames == 4


def test_playbooks_prologue_yard_clear_and_open_phone(tmp_path) -> None:
    # 1) prologue_yard_clear_and_push clears visible enemy from cover, exits cover cleanly, and pushes forward
    world = KinematicWorld(
        player_x=0.0,
        player_y=0.0,
        in_cover=True,
        health=0.90,
        enemies=[SimEnemy(x=0.0, y=10.0, hp=60.0)],
        yellow_target=(0.0, 20.0),
    )
    clock = SimClock(initial_time=300.0, on_advance=world.step)
    backend = SimInputBackend(world, clock)
    logger = TelemetryLogger(tmp_path / "t_pb.jsonl")
    ctrl = InputController(backend=backend, logger=logger, clock=clock, sleep_fn=clock.sleep, register_signals=False)
    runner = ActionRunner(ctrl, SimWindowBackend(world), ScriptedHudOCR(lambda _i: {}), logger, clock, clock.sleep)

    pb_res = prologue_yard_clear_and_push(runner, enter_cover=False, cover_max_s=6.0, advance_max_s=4.0)
    assert not world.enemies[0].alive
    assert pb_res.steps[0]["step"] == "cover_peek_0"
    assert any(s["step"] == "exit_cover" and s["exited"] is True for s in pb_res.steps)
    assert world.player_y > 2.0
    assert world.held_keys == set()

    # 2) open_phone_menu uses left click (never right-click) + up arrow
    clock2 = FakeClock()
    backend2 = RecordingInputBackend()
    ctrl2 = InputController(backend=backend2, logger=logger, clock=clock2, sleep_fn=clock2.sleep, register_signals=False)
    runner2 = ActionRunner(ctrl2, ReplayWindowBackend(lambda _i: make_synthetic_minimap_frame()), ScriptedHudOCR(lambda _i: {}), logger, clock2, clock2.sleep)
    open_phone_menu(runner2, activate_button=None)
    assert ("key_down", "up") in backend2.events
    assert ("mouse_down", "right") not in backend2.events
