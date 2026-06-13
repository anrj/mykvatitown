"""State- and branch-coverage tests for tasks/project/packages/agent.py
(the convoy *follower*).

Strategy
--------
* Pure units (LeaderDetector, Controller, set_leds, _annotate) are exercised
  directly to hit every conditional branch.
* The SEARCH / FOLLOW / HOLD state machine in main() is driven by a scripted
  fake detector + fake hardware so every state transition is taken.
"""

import os
import sys
import unittest
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from _fakes import FakeCamera, FakeWheels, FakeLeds, FakeStop, blank_frame  # noqa: E402
import tasks.project.packages.agent as agent  # noqa: E402


def make_cfg(**overrides):
    cfg = {
        'leader': {
            'grid_cols': 7, 'grid_rows': 3, 'dot_spacing_m': 0.0125,
            'target_span': 0.45, 'stop_span': 0.60, 'span_deadband': 0.03,
            'blob_min_area': 6.0,
        },
        'control': {
            'max_speed': 0.45, 'chase_speed': 0.30,
            'steer_kp': 0.55, 'steer_kd': 0.30,
            'dist_kp': 2.0, 'accel_rate': 0.05, 'decel_rate': 0.08,
            'search_turn': 0.10, 'search_after_frames': 3, 'loop_hz': 1000,
        },
        'camera': {'matrix': None, 'dist_coeffs': None},
    }
    for sec, vals in overrides.items():
        cfg.setdefault(sec, {}).update(vals)
    return cfg


class ScriptedDetector:
    """Drop-in for agent.LeaderDetector. Pops (found, e, span, centers)."""

    script = []

    def __init__(self, cfg):
        self._it = iter(type(self).script)

    def detect(self, frame):
        try:
            return next(self._it)
        except StopIteration:
            return False, 0.0, 0.0, None


# ---------------------------------------------------------------------------
# Config loading + live-tuning guard
# ---------------------------------------------------------------------------

class TestConfig(unittest.TestCase):
    def test_load_config_missing_file(self):
        with mock.patch.object(agent, 'CONFIG_FILE', 'nope_xyz.yaml'):
            cfg = agent.load_config()                # FileNotFoundError -> defaults
        self.assertEqual(cfg['leader']['grid_cols'], 7)

    def test_load_config_dict_and_scalar_sections(self):
        import yaml
        tmp = os.path.join(ROOT, 'config', '_tmp_test_project.yaml')
        with open(tmp, 'w') as f:
            yaml.dump({'control': {'max_speed': 0.9}, 'scalar_section': 5}, f)
        try:
            with mock.patch.object(agent, 'CONFIG_FILE', '_tmp_test_project.yaml'):
                cfg = agent.load_config()
            self.assertEqual(cfg['control']['max_speed'], 0.9)
            self.assertEqual(cfg['scalar_section'], 5)   # non-dict -> else branch
        finally:
            os.remove(tmp)

    def test_sync_cfg_guard_returns_when_uninitialised(self):
        with mock.patch.object(agent, '_ctrl', None), \
             mock.patch.object(agent, 'CFG', None):
            agent._sync_cfg()                        # early-return guard branch


# ---------------------------------------------------------------------------
# LeaderDetector (real detection maths, cv2.findCirclesGrid mocked)
# ---------------------------------------------------------------------------

class TestLeaderDetector(unittest.TestCase):
    def setUp(self):
        self.det = agent.LeaderDetector(make_cfg())

    def test_grid_not_found(self):
        with mock.patch.object(agent.cv2, 'findCirclesGrid',
                               return_value=(False, None)):
            found, e, span, centers = self.det.detect(blank_frame())
        self.assertFalse(found)
        self.assertEqual((e, span), (0.0, 0.0))
        self.assertIsNone(centers)

    def test_grid_found_centre(self):
        # 21 dots clustered at frame centre -> error ~0.
        w = 64
        xs = np.linspace(w / 2 - 5, w / 2 + 5, 21)
        pts = np.stack([xs, np.full(21, 24.0)], axis=1).reshape(-1, 1, 2).astype(np.float32)
        with mock.patch.object(agent.cv2, 'findCirclesGrid', return_value=(True, pts)):
            found, e, span, centers = self.det.detect(blank_frame())
        self.assertTrue(found)
        self.assertAlmostEqual(e, 0.0, delta=0.05)
        self.assertGreater(span, 0.0)
        self.assertEqual(centers.shape, (21, 2))

    def test_grid_found_offset_clamped(self):
        # Dots pushed to far right -> positive error, span large.
        w = 64
        xs = np.linspace(w - 12, w - 1, 21)
        pts = np.stack([xs, np.full(21, 24.0)], axis=1).reshape(-1, 1, 2).astype(np.float32)
        with mock.patch.object(agent.cv2, 'findCirclesGrid', return_value=(True, pts)):
            found, e, span, _ = self.det.detect(blank_frame())
        self.assertTrue(found)
        self.assertGreater(e, 0.0)
        self.assertLessEqual(e, 1.0)

    def test_build_grid_object_points(self):
        obj = self.det.build_grid_object_points(0.0125)
        self.assertEqual(obj.shape, (21, 3))
        self.assertTrue(np.all(obj[:, 2] == 0.0))


# ---------------------------------------------------------------------------
# Controller (steering / distance / ramp branches)
# ---------------------------------------------------------------------------

class TestController(unittest.TestCase):
    def setUp(self):
        self.c = agent.Controller(make_cfg())

    def test_steering_uses_derivative(self):
        self.c.steering(0.0)
        out = self.c.steering(0.4)            # P + D on rising error
        self.assertAlmostEqual(out, 0.55 * 0.4 + 0.30 * 0.4, places=6)

    def test_distance_speed_too_close_holds(self):
        self.assertEqual(self.c.distance_speed(0.65, 0.45), 0.0)   # span >= stop_span

    def test_distance_speed_within_deadband(self):
        self.assertEqual(self.c.distance_speed(0.45, 0.45), 0.0)   # |err| < deadband

    def test_distance_speed_forward(self):
        v = self.c.distance_speed(0.30, 0.45)                      # leader too far
        self.assertGreater(v, 0.0)

    def test_distance_speed_capped(self):
        v = self.c.distance_speed(0.0, 0.10)                       # huge err -> cap
        self.assertEqual(v, 0.10)

    def test_ramp_accel_then_decel(self):
        up = self.c.ramp(0.40)                                     # target > cur
        self.assertAlmostEqual(up, 0.05)
        down = self.c.ramp(0.0)                                    # target < cur
        self.assertAlmostEqual(down, 0.05 - 0.08 if 0.05 - 0.08 > 0 else 0.0)

    def test_reset_steer_and_current_speed(self):
        self.c.steering(0.5)
        self.c.reset_steer()
        self.assertEqual(self.c._prev_e, 0.0)
        self.c.ramp(0.2)
        self.assertEqual(self.c.current_speed, self.c._cur_v)


# ---------------------------------------------------------------------------
# set_leds / _annotate (pure rendering branches)
# ---------------------------------------------------------------------------

class TestLedsAndAnnotate(unittest.TestCase):
    def test_set_leds_none_returns(self):
        agent.set_leds(None, 'FOLLOW', 0.5)        # no exception, early return

    def test_set_leds_turn_right(self):
        leds = FakeLeds()
        agent.set_leds(leds, 'FOLLOW', 0.5)        # turn > 0.12 -> right indicator
        self.assertTrue(leds.rgb)

    def test_set_leds_turn_left(self):
        leds = FakeLeds()
        agent.set_leds(leds, 'HOLD', -0.5)         # turn < -0.12 -> left indicator
        self.assertTrue(leds.rgb)

    def test_set_leds_straight(self):
        leds = FakeLeds()
        agent.set_leds(leds, 'SEARCH', 0.0)        # |turn| <= 0.12 -> else branch
        self.assertTrue(leds.rgb)

    def test_annotate_with_and_without_centers(self):
        pts = np.array([[10, 10], [20, 20]], dtype=float)
        out1 = agent._annotate(blank_frame(), 'FOLLOW', 0.4, 0.1, pts)
        out2 = agent._annotate(blank_frame(), 'SEARCH', 0.0, 0.0, None)
        self.assertEqual(out1.shape, out2.shape)


# ---------------------------------------------------------------------------
# main() state machine
# ---------------------------------------------------------------------------

class TestFollowerStateMachine(unittest.TestCase):
    def _run(self, script, frames=None, leds=None):
        ScriptedDetector.script = script
        n = len(frames) if frames is not None else len(script)
        cam = FakeCamera(frames if frames is not None else
                         [(True, blank_frame())] * n)
        wheels = FakeWheels()
        stop = FakeStop(iterations=n)
        with mock.patch.object(agent, 'LeaderDetector', ScriptedDetector), \
             mock.patch.object(agent, 'load_config', return_value=make_cfg()):
            agent.main(cam, wheels, leds, stop)
        return wheels

    def test_follow_hold_and_brief_loss(self):
        f = blank_frame()
        pts = np.array([[1.0, 1.0]])
        script = [
            (True,  0.5, 0.30, pts),   # FOLLOW, forward, turn>0
            (True,  0.5, 0.45, pts),   # FOLLOW, deadband -> v=0
            (True,  0.5, 0.65, pts),   # HOLD (span >= stop_span)
            (False, 0.0, 0.0,  None),  # brief loss, was close -> HOLD
            (True, -0.5, 0.30, pts),   # FOLLOW, turn<0, last_e<0
            (False, 0.0, 0.0,  None),  # brief loss, was far -> chase FOLLOW
            (False, 0.0, 0.0,  None),  # still chasing
            (False, 0.0, 0.0,  None),  # lost long enough -> SEARCH (last_e<0)
        ]
        frames = [(True, f)] * len(script)
        wheels = self._run(script, frames, leds=FakeLeds())
        # One command per loop + the guaranteed stop issued in main()'s finally.
        self.assertEqual(len(wheels.commands), len(script) + 1)
        self.assertEqual(wheels.commands[-1], (0.0, 0.0))

    def test_camera_not_ok_branch(self):
        script = [(True, 0.0, 0.30, np.array([[1.0, 1.0]]))]
        frames = [(False, None), (True, blank_frame())]
        # 2 iterations: first read fails (continue), second runs detect.
        ScriptedDetector.script = script
        cam = FakeCamera(frames)
        wheels = FakeWheels()
        stop = FakeStop(iterations=2)
        with mock.patch.object(agent, 'LeaderDetector', ScriptedDetector), \
             mock.patch.object(agent, 'load_config', return_value=make_cfg()):
            agent.main(cam, wheels, FakeLeds(), stop)
        # failed read issues nothing; ok frame -> 1 command; finally -> 1 stop.
        self.assertEqual(len(wheels.commands), 2)
        self.assertEqual(wheels.commands[-1], (0.0, 0.0))

    def test_search_positive_turn(self):
        pts = np.array([[1.0, 1.0]])
        script = [
            (True, 0.6, 0.30, pts),    # last_e > 0
            (False, 0.0, 0.0, None),
            (False, 0.0, 0.0, None),
            (False, 0.0, 0.0, None),   # SEARCH with last_e >= 0 -> positive turn
        ]
        wheels = self._run(script, leds=None)       # leds=None path in set_leds
        self.assertEqual(len(wheels.commands), len(script) + 1)


if __name__ == '__main__':
    unittest.main()
