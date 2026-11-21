"""Versioned, simulator-tested mission playbooks wrapping `ActionRunner` closed-loop primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from gameplayer.hud import HudObservation
from gameplayer.minimap import MinimapObservation
from gameplayer.runner import ActionRunner, LoopConfig, LoopResult


@dataclass(frozen=True)
class PlaybookResult:
    playbook: str
    outcome: str
    steps: list[dict[str, object]] = field(default_factory=list)
    last_minimap: Optional[MinimapObservation] = None
    last_hud: Optional[HudObservation] = None


def safe_cover_heal_and_peek(
    runner: ActionRunner,
    *,
    enter_cover: bool = False,
    min_health: float = 0.42,
    cover_max_s: float = 45.0,
    max_rounds: int = 2,
) -> tuple[str, list[dict[str, object]], Optional[LoopResult]]:
    """Run cover_peek_combat up to `max_rounds`, staying crouched in cover on `low_health` until healed."""
    steps: list[dict[str, object]] = []
    last_res: Optional[LoopResult] = None

    if enter_cover:
        posture = runner.enter_cover()
        steps.append({"step": "enter_cover", "state": posture.state, "confidence": round(posture.confidence, 3)})

    for idx in range(max(1, max_rounds)):
        res = runner.cover_peek_combat(
            enter_cover=False,
            config=LoopConfig(
                max_duration_s=cover_max_s,
                hz=10.0,
                hud_check_interval_s=0.25,
                health_drop_threshold=None,
            ),
        )
        last_res = res
        mm = res.last_minimap
        hp = round(mm.health_ratio, 3) if mm else None
        red_count = len(mm.targets["red"]) if mm else None
        steps.append(
            {
                "step": f"cover_peek_{idx}",
                "stop_reason": res.stop_reason,
                "elapsed_s": round(res.elapsed_s, 2),
                "health_ratio": hp,
                "red_count": red_count,
            }
        )
        if res.banner is not None or res.stop_reason.startswith("banner:"):
            return "mission_banner", steps, last_res
        if res.stop_reason == "enemies_cleared" or (mm is not None and len(mm.targets["red"]) <= 1):
            if mm is not None and mm.minimap_visible and mm.health_ratio < min_health:
                continue
            return "ready_to_push", steps, last_res
        if res.stop_reason == "low_health" or (mm is not None and mm.minimap_visible and mm.health_ratio < min_health):
            # Stay crouched in cover to heal; never toggle q out of cover while critically low HP.
            continue
        if res.stop_reason in ("cover_blocked", "cover_clamped"):
            return res.stop_reason, steps, last_res

    mm_final = last_res.last_minimap if last_res else None
    if mm_final is not None and mm_final.minimap_visible and mm_final.health_ratio < min_health:
        return "low_health_hold_cover", steps, last_res
    return (last_res.stop_reason if last_res else "timeout"), steps, last_res


def prologue_yard_clear_and_push(
    runner: ActionRunner,
    *,
    enter_cover: bool = False,
    min_exit_health: float = 0.42,
    cover_max_s: float = 45.0,
    advance_max_s: float = 60.0,
    ally_tether_distance: float = 0.28,
) -> PlaybookResult:
    """Clear/thin yard wave from cover, exit cover ONLY when `health >= 0.42`, then advance with allies."""
    outcome, steps, last_res = safe_cover_heal_and_peek(
        runner,
        enter_cover=enter_cover,
        min_health=min_exit_health,
        cover_max_s=cover_max_s,
        max_rounds=2,
    )
    if outcome in ("mission_banner", "low_health_hold_cover"):
        return PlaybookResult(
            playbook="prologue_yard_clear_and_push",
            outcome=outcome,
            steps=steps,
            last_minimap=last_res.last_minimap if last_res else None,
            last_hud=last_res.last_hud if last_res else None,
        )

    exited, posture = runner.exit_cover(min_health=min_exit_health)
    steps.append(
        {
            "step": "exit_cover",
            "exited": exited,
            "posture": posture.state,
        }
    )
    if not exited:
        return PlaybookResult(
            playbook="prologue_yard_clear_and_push",
            outcome="low_health_hold_cover",
            steps=steps,
            last_minimap=last_res.last_minimap if last_res else None,
            last_hud=last_res.last_hud if last_res else None,
        )

    adv = runner.advance_under_fire(
        sprint=False,
        ally_tether_distance=ally_tether_distance,
        config=LoopConfig(
            max_duration_s=advance_max_s,
            hz=10.0,
            hud_check_interval_s=0.25,
            health_drop_threshold=None,
        ),
    )
    steps.append(
        {
            "step": "advance_under_fire",
            "stop_reason": adv.stop_reason,
            "elapsed_s": round(adv.elapsed_s, 2),
            "health_ratio": round(adv.last_minimap.health_ratio, 3) if adv.last_minimap else None,
        }
    )
    return PlaybookResult(
        playbook="prologue_yard_clear_and_push",
        outcome=adv.stop_reason,
        steps=steps,
        last_minimap=adv.last_minimap,
        last_hud=adv.last_hud,
    )


def open_phone_menu(
    runner: ActionRunner,
    *,
    activate_button: Optional[str] = "left",
    up_hold_s: float = 0.25,
    settle_s: float = 0.35,
) -> None:
    """Safely open GTA 5's phone without right-click aim/cancel interference."""
    if activate_button is not None:
        runner.activate_wine_capture(button=activate_button, hold_s=0.04)
        runner._sleep(0.10)
    runner._tap_key_preserving_held("up", hold_s=up_hold_s)
    if settle_s > 0:
        runner._sleep(settle_s)
