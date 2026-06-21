"""Basic tests for convoy leader packages."""

import time

import numpy as np

from tasks.project_leader.packages.sign_behavior import SignBehavior, detect_red_line
from tasks.project_leader.packages.leader_lane import LeaderLaneAgent, centerline_xs
from tasks.project_leader.packages.leader_agent import load_config
from tasks.project_leader.packages.intersection_turn import IntersectionPlanner


def test_detect_red_line_on_synthetic_red_strip():
    cfg = load_config()
    sb = SignBehavior(cfg)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Full-width red band in bottom near ROI
    img[468:, 180:460] = [200, 0, 0]
    hit, near, far = detect_red_line(sb, img, at_stop_line=True)
    assert hit is True
    assert near > 0.3


def test_far_only_red_does_not_trigger():
    cfg = load_config()
    sb = SignBehavior(cfg)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[442:460, 200:440] = [200, 0, 0]  # far band only
    hit, near, far = detect_red_line(sb, img)
    assert hit is False


def test_intersection_turn_order():
    cfg = load_config()
    planner = IntersectionPlanner(cfg)
    assert planner.pending_direction() == 'left'
    planner.advance_maneuver()
    assert planner.pending_direction() == 'right'
    planner.advance_maneuver()
    assert planner.pending_direction() == 'straight'
    assert planner.is_spin_turn() is False
    planner.advance_maneuver()
    assert planner.pending_direction() == 'none'


def test_centerline_offsets_single_edge_slices():
    """One edge per slice must offset by half-width, not use raw edge x."""
    mids = centerline_xs([100, -1], [-1, 300], half_width=200)
    assert mids == [300, 100]


def test_pwm_from_velocity_turn_in_place():
    from tasks.project_leader.packages.intersection_turn import pwm_from_velocity
    left, right = pwm_from_velocity(0.0, 1.0, radius=0.0318, baseline=0.1)
    assert left < 0 and right > 0


def test_cross_distance_odometry():
    from tasks.project_leader.packages.intersection_turn import IntersectionPlanner

    class FakeWheels:
        left_pwm = 0.25
        right_pwm = 0.25
        encoders = None

    cfg = {'maneuver': {'cross_distance_m': 0.10}, 'intersection': {'turns': ['left']}}
    planner = IntersectionPlanner(cfg)
    planner.begin_cross_segment()
    w = FakeWheels()
    for _ in range(80):
        planner.record_motion(w, 0.05)
    assert planner.segment_distance() >= 0.08
    assert planner.cross_segment_complete(time.monotonic())


def test_modcon_pid_turn_converges():
    from tasks.project_leader.packages.intersection_turn import IntersectionPlanner

    class FakeWheels:
        left_pwm = 0.0
        right_pwm = 0.0
        encoders = None

    cfg = {
        'intersection': {
            'turns': ['left'],
            'turn_angle_deg': 45.0,
            'turn_timeout_s': 5.0,
            'converge_samples': 5,
            'tolerance_deg': 8.0,
        },
    }
    planner = IntersectionPlanner(cfg)
    planner.begin_turn(time.monotonic())
    w = FakeWheels()
    done = False
    now = time.monotonic()
    for i in range(200):
        done, left, right = planner.pid_step(w, 0.05, now + i * 0.05)
        w.left_pwm = left
        w.right_pwm = right
        planner.record_motion(w, 0.05)
        if done:
            break
    assert done is True
    assert planner.pending_direction() == 'none'


def test_bottom_line_visible():
    lane = LeaderLaneAgent()
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[470:, 100:200] = 255
    info = {'yellow_mask': mask, 'white_mask': np.zeros_like(mask)}
    assert lane.bottom_line_visible(info, 'left', 25) is True
