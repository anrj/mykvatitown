"""Lead-bot finite state machine: autonomous lane following with traffic-sign
slowing/stopping and a fixed programmed route executed at red-line intersections.

The lead follows no one. It:
  - lane-follows on straightaways,
  - slows in SLOW zones (AprilTag SLOW signs),
  - remembers a STOP sign across the gap until the intersection's red line, halts
    ~1s, then executes the next route maneuver (turn left/right, cross straight,
    or final stop),
  - slows briefly after each maneuver so the follower can reacquire it.

Pure logic over a WorldModel + an optional encoder-derived yaw, so it is unit
testable without the vision stack or hardware.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from tasks.project.packages.world_model import WorldModel

Rgb = Tuple[float, float, float]

GREEN = (0.0, 1.0, 0.0)
YELLOW = (1.0, 0.7, 0.0)
RED = (1.0, 0.0, 0.0)
WHITE = (1.0, 1.0, 1.0)

# Corner LEDs: 0=front-left, 2=front-right, 3=back-left, 4=back-right.
_LED_INDICES = (0, 2, 3, 4)


@dataclass
class Decision:
    state_name: str
    base_speed: float
    steering: float
    leds: Dict[int, Rgb]


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def all_leds(color: Rgb) -> Dict[int, Rgb]:
    return {idx: color for idx in _LED_INDICES}


STATE_LANE       = "LANE_FOLLOW"
STATE_STOP       = "STOP_AT_SIGN"
STATE_SLOW       = "SLOW_ZONE"
STATE_CROSS      = "CROSS_STRAIGHT"
STATE_TURN_L     = "TURN_LEFT"
STATE_TURN_R     = "TURN_RIGHT"
STATE_SLOW_AFTER = "SLOW_AFTER_TURN"
STATE_DONE       = "ROUTE_DONE"

_TURN_STEPS = ("left", "right")
_MANEUVER_STEPS = ("left", "right", "straight")


class LeadFSM:
    def __init__(self, cfg: Optional[dict] = None):
        cfg = cfg or {}
        self.cruise_speed = float(cfg.get("cruise_speed", 0.3))
        self.slow_factor  = float(cfg.get("slow_factor", 0.5))
        self.stop_duration = float(cfg.get("stop_duration", 1.0))
        self.slow_duration = float(cfg.get("slow_duration", 1.5))
        self.slow_cooldown = float(cfg.get("slow_cooldown", 3.0))
        # A STOP tag whose bbox covers >= this fraction of the frame is treated
        # as "close" and halts the bot directly (not only at a red line).
        self.stop_tag_near_area_frac = float(cfg.get("stop_tag_near_area_frac", 0.04))
        self.stop_tag_halt_s = float(cfg.get("stop_tag_halt_s", 1.0))

        self.route = [str(s).lower() for s in (cfg.get("route") or ["stop"])]

        # Maneuver shape / timing
        self.turn_base   = float(cfg.get("turn_base_speed", 0.25))
        self.turn_steer  = float(cfg.get("turn_steer", self.turn_base))
        # Left turns at a grid intersection sweep a wider arc than rights: scale
        # the left steer law down (and optionally its forward speed up).
        self.left_widen       = float(cfg.get("left_widen", 0.7))
        self.left_base_factor = float(cfg.get("left_base_factor", 1.0))
        self.cross_base  = float(cfg.get("cross_base_speed", 0.25))
        self.min_turn_s  = float(cfg.get("min_turn_s", 0.8))
        self.max_turn_s  = float(cfg.get("max_turn_s", 3.0))
        self.min_cross_s = float(cfg.get("min_cross_s", 0.4))
        self.max_cross_s = float(cfg.get("max_cross_s", 2.0))
        self.turn_yaw_target = float(cfg.get("turn_yaw_target_rad", 1.40))  # ~80 deg
        self.slow_after_turn_s = float(cfg.get("slow_after_turn_s", 2.0))
        self.slow_after_factor = float(cfg.get("slow_after_factor", 0.6))

        # Closed-loop odometry maneuver targets (used only when the agent passes
        # encoder distance/yaw; otherwise the timed values above are the fallback).
        self.cross_distance_m = float(cfg.get("cross_distance_m", 0.35))
        self.heading_kp       = float(cfg.get("maneuver_heading_kp", 0.6))
        self.turn_kp          = float(cfg.get("turn_kp", 0.8))
        self.cross_dist_tol_m = float(cfg.get("cross_dist_tol_m", 0.03))
        self.turn_yaw_tol_rad = float(cfg.get("turn_yaw_tol_rad", 0.08))

        # Red-line intersection firing gates + clear latch
        self.fire_dist   = float(cfg.get("stopline_fire_dist", 0.45))
        self.fire_width  = float(cfg.get("stopline_fire_width", 0.40))
        self.clear_frames = int(cfg.get("stopline_clear_frames", 5))

        # mutable state
        self.route_idx = 0
        self._stop_until = -1.0
        self._slow_until = -1.0
        self._slow_cooldown_until = -1.0
        self._stop_tag_cooldown_until = -1.0
        self._sign_pending = False
        self._maneuver: Optional[str] = None
        self._maneuver_start = 0.0
        self._slow_after_until = -1.0
        self._pending_step: Optional[str] = None
        self._consumed = False
        self._red_clear = 0
        self._done = False

        # surfaced for the agent/debug
        self.request_lane_reset = False
        self.last_step: Optional[str] = None

    # --- public ---------------------------------------------------------------
    def step(self, wm: WorldModel, turn_yaw_rad: Optional[float] = None,
             fwd_dist_m: Optional[float] = None) -> Decision:
        self.request_lane_reset = False
        t = wm.t
        self._update_latch(wm)
        self._ingest_signs(wm)

        if self._done:
            return self._decide(STATE_DONE, 0.0, 0.0, RED)

        # 1) an active maneuver takes precedence
        if self._maneuver is not None:
            return self._run_maneuver(wm, t, turn_yaw_rad, fwd_dist_m)

        # 2) halting at the line for a pending STOP
        if t < self._stop_until:
            return self._decide(STATE_STOP, 0.0, 0.0, RED)
        if self._pending_step is not None:
            step = self._pending_step
            self._pending_step = None
            self._begin_step(step, t)
            if self._done:
                return self._decide(STATE_DONE, 0.0, 0.0, RED)
            return self._run_maneuver(wm, t, turn_yaw_rad, fwd_dist_m)

        # 3) slow-after-turn window (give the follower time to reacquire)
        if t < self._slow_after_until:
            return self._decide(STATE_SLOW_AFTER, self.cruise_speed * self.slow_after_factor,
                                wm.lane.steering_suggestion, YELLOW)

        # 4) intersection event -> consume the next route step
        if self._intersection_fires(wm):
            self._consumed = True
            self._red_clear = 0
            step = self.route[self.route_idx] if self.route_idx < len(self.route) else "stop"
            self.route_idx += 1
            self.last_step = step
            if self._sign_pending:
                self._sign_pending = False
                self._stop_until = t + self.stop_duration
                self._pending_step = step  # maneuver begins after the 1s halt
                return self._decide(STATE_STOP, 0.0, 0.0, RED)
            self._begin_step(step, t)
            if self._done:
                return self._decide(STATE_DONE, 0.0, 0.0, RED)
            return self._run_maneuver(wm, t, turn_yaw_rad, fwd_dist_m)

        # 5) slow zone (timed)
        if t < self._slow_until:
            return self._decide(STATE_SLOW, self.cruise_speed * self.slow_factor,
                                wm.lane.steering_suggestion, YELLOW)

        # 6) default: lane following
        return self._decide(STATE_LANE, self.cruise_speed, wm.lane.steering_suggestion, GREEN)

    # --- internals ------------------------------------------------------------
    def _ingest_signs(self, wm: WorldModel) -> None:
        kinds = {s.kind for s in wm.signs}
        if "STOP" in kinds:
            self._sign_pending = True  # remembered until the intersection consumes it
            # A STOP tag that fills enough of the frame is close: halt directly so
            # the bot reacts to the sign itself, not only at the intersection red
            # line (symmetric with SLOW acting immediately). Gated on bbox size so
            # a far / handheld tag doesn't trip it, and cooled down so it doesn't
            # re-arm every frame while the sign stays in view.
            frame_area = max(1, wm.frame_w * wm.frame_h)
            stop_area = max((max(0, s.bbox[2] - s.bbox[0]) * max(0, s.bbox[3] - s.bbox[1])
                             for s in wm.signs if s.kind == "STOP"), default=0)
            if (stop_area / frame_area >= self.stop_tag_near_area_frac
                    and wm.t >= self._stop_tag_cooldown_until):
                self._stop_until = wm.t + self.stop_tag_halt_s
                self._stop_tag_cooldown_until = (
                    wm.t + self.stop_tag_halt_s + self.slow_cooldown)
        if "SLOW" in kinds and wm.t >= self._slow_cooldown_until:
            self._slow_until = wm.t + self.slow_duration
            self._slow_cooldown_until = wm.t + self.slow_duration + self.slow_cooldown

    def _update_latch(self, wm: WorldModel) -> None:
        rl = wm.red_line
        red_present = rl is not None and rl.present and rl.width_frac >= (self.fire_width * 0.5)
        # Only count "line cleared" frames once we've finished reacting to the
        # last intersection. While a maneuver / stop / slow-after is in progress
        # the bot is still on top of the same physical line; counting clear
        # frames here would re-arm _consumed mid-reaction and let that one line
        # re-fire, burning the whole route down to the terminal 'stop'.
        busy = (self._maneuver is not None
                or wm.t < self._slow_after_until
                or wm.t < self._stop_until)
        if red_present:
            self._red_clear = 0
        elif not busy:
            self._red_clear += 1
            if self._consumed and self._red_clear >= self.clear_frames:
                self._consumed = False  # cleared the line -> ready for next intersection

    def _intersection_fires(self, wm: WorldModel) -> bool:
        rl = wm.red_line
        if rl is None or not rl.present or self._consumed:
            return False
        return rl.dist_proxy >= self.fire_dist and rl.width_frac >= self.fire_width

    def _begin_step(self, step: str, t: float) -> None:
        if step == "stop":
            self._done = True
            self._maneuver = None
            return
        self._maneuver = step if step in _MANEUVER_STEPS else "straight"
        self._maneuver_start = t

    def _run_maneuver(self, wm: WorldModel, t: float,
                      turn_yaw_rad: Optional[float],
                      fwd_dist_m: Optional[float]) -> Decision:
        step = self._maneuver
        elapsed = t - self._maneuver_start
        is_turn = step in _TURN_STEPS
        min_s = self.min_turn_s if is_turn else self.min_cross_s
        max_s = self.max_turn_s if is_turn else self.max_cross_s

        # Closed-loop on encoder odometry when both scalars are available;
        # otherwise fall back to the legacy timed / lane-reacquire behaviour.
        have_odo = turn_yaw_rad is not None and fwd_dist_m is not None

        done = False
        if elapsed >= max_s:                                   # hard safety timeout
            done = True
        elif have_odo:
            if is_turn and abs(turn_yaw_rad) >= (self.turn_yaw_target - self.turn_yaw_tol_rad):
                done = True                                    # turned to target heading
            elif (not is_turn) and fwd_dist_m >= (self.cross_distance_m - self.cross_dist_tol_m):
                done = True                                    # crossed the target distance
        elif is_turn and turn_yaw_rad is not None and abs(turn_yaw_rad) >= self.turn_yaw_target:
            done = True                                        # encoder yaw target reached
        elif elapsed >= min_s and wm.lane.healthy:
            done = True                                        # lane reacquired -> self-correct

        if done:
            self._maneuver = None
            self._slow_after_until = t + self.slow_after_turn_s
            self.request_lane_reset = True  # agent clears stale lane PID on re-entry
            return self._decide(STATE_SLOW_AFTER, self.cruise_speed * self.slow_after_factor,
                                wm.lane.steering_suggestion, YELLOW)

        if is_turn:
            # Lane convention: +steer turns LEFT, -steer turns RIGHT.
            sign = 1.0 if step == "left" else -1.0
            # Left turns sweep a wider arc than rights at a grid intersection:
            # scale the whole steer law (ceiling + floor) down for lefts so the
            # radius opens out, and optionally raise the forward speed.
            widen       = self.left_widen if step == "left" else 1.0
            steer_cap   = self.turn_steer * widen
            steer_floor = 0.10 * widen
            base        = self.turn_base * (self.left_base_factor if step == "left" else 1.0)
            if have_odo and self.turn_yaw_target > 0:
                # Taper steer as the remaining yaw error shrinks, with a floor so
                # the bot keeps rotating until it reaches the target heading.
                yaw_err = self.turn_yaw_target - abs(turn_yaw_rad)
                mag = clamp(steer_cap * self.turn_kp * yaw_err / self.turn_yaw_target,
                            steer_floor, steer_cap)
                steer = sign * mag
            else:
                steer = sign * steer_cap                       # legacy fixed arc
            name = STATE_TURN_L if step == "left" else STATE_TURN_R
            return self._decide(name, base, steer, WHITE)

        # Straight cross: hold heading with a small P term on yaw drift.
        # yaw > 0 means drifting left -> negative steer corrects back right.
        steer = 0.0
        if have_odo:
            steer = clamp(-self.heading_kp * turn_yaw_rad, -self.turn_steer, self.turn_steer)
        return self._decide(STATE_CROSS, self.cross_base, steer, WHITE)

    @staticmethod
    def _decide(name: str, speed: float, steering: float, color) -> Decision:
        return Decision(state_name=name, base_speed=speed,
                        steering=clamp(steering, -1.0, 1.0), leds=all_leds(color))
