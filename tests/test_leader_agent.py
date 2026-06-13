"""State- and branch-coverage tests for the convoy *leader*:
  tasks/project_leader/packages/leader_agent.py   (CRUISE/AT_LINE/STOP machine)
  tasks/project_leader/packages/leader_lane.py    (lane helpers)

The lane agent is replaced by a scripted fake so every mode transition in
main() is deterministic; leader_lane helpers are tested directly with mocked
student CV so each detection branch is taken.
"""

import os
import sys
import unittest
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from _fakes import FakeCamera, FakeWheels, FakeLeds, FakeStop, Clock, blank_frame  # noqa: E402
import tasks.project_leader.packages.leader_agent as la  # noqa: E402
import tasks.project_leader.packages.leader_lane as ll   # noqa: E402


def small_masks(h=48, w=64, paint=False):
    m = np.zeros((h, w), dtype=np.uint8)
    if paint:
        m[h - 10:, :] = 255
    return m


# ---------------------------------------------------------------------------
# leader_agent pure helpers
# ---------------------------------------------------------------------------

class TestLeaderHelpers(unittest.TestCase):
    def test_norm_side(self):
        self.assertEqual(ll._norm_side('right'), 'right')
        self.assertEqual(ll._norm_side('white'), 'right')
        self.assertEqual(ll._norm_side('LEFT'), 'left')
        self.assertEqual(ll._norm_side(None), 'left')

    def test_migrate_legacy_keys(self):
        ls = {'min_yellow_frames': 7, 'yellow_lost_frames': 3, 'bottom_yellow_px': 30}
        la._migrate_lane_stop_keys(ls)
        self.assertEqual(ls['min_line_frames'], 7)
        self.assertEqual(ls['line_lost_frames'], 3)
        self.assertEqual(ls['bottom_line_px'], 30)

    def test_lane_stop_cfg_defaults(self):
        out = la._lane_stop_cfg({})
        self.assertEqual(out['trigger'], 'left')
        self.assertEqual(out['follow'], 'right')
        self.assertEqual(out['min_frames'], 10)

    def test_load_config_success(self):
        cfg = la.load_config()                       # reads real leader_config.yaml
        self.assertIn('lane_stop', cfg)
        self.assertIn('control', cfg)

    def test_load_config_missing_file(self):
        with mock.patch.object(la, 'CONFIG_FILE', 'does_not_exist_xyz.yaml'):
            cfg = la.load_config()                   # FileNotFoundError -> defaults
        self.assertEqual(cfg['lane_stop']['stop_trigger_line'], 'left')

    def test_load_config_dict_and_scalar_sections(self):
        import yaml
        tmp = os.path.join(ROOT, 'config', '_tmp_test_leader.yaml')
        with open(tmp, 'w') as f:
            yaml.dump({'control': {'loop_hz': 25}, 'scalar_section': 1}, f)
        try:
            with mock.patch.object(la, 'CONFIG_FILE', '_tmp_test_leader.yaml'):
                cfg = la.load_config()
            self.assertEqual(cfg['control']['loop_hz'], 25)
            self.assertEqual(cfg['scalar_section'], 1)   # non-dict -> else branch
        finally:
            os.remove(tmp)

    def test_visualize_with_masks(self):
        info = {
            'yellow_mask': small_masks(paint=True),
            'white_mask': small_masks(paint=True),
            'slice_ys': [40],
            'follow_line': 'right',
            'lane_detected': True,
        }
        img = la._visualize(blank_frame(), info, la.MODE_CRUISE, True, 5, 0, 'left')
        self.assertEqual(img.shape, (48, 64, 3))

    def test_visualize_no_masks(self):
        img = la._visualize(blank_frame(), {}, la.MODE_STOP, False, 0, 3, 'left')
        self.assertEqual(img.shape, (48, 64, 3))


# ---------------------------------------------------------------------------
# LeaderLaneAgent helpers
# ---------------------------------------------------------------------------

class TestLeaderLaneAgent(unittest.TestCase):
    def setUp(self):
        self.agent = ll.LeaderLaneAgent(config_path='no_such_config.yaml')

    def test_half_width_from_config(self):
        tmp = os.path.join(ROOT, 'config', '_tmp_test_lane.yaml')
        with open(tmp, 'w') as f:
            f.write('lane_half_width_px: 240\nmin_lane_half_width_px: 200\n')
        try:
            agent = ll.LeaderLaneAgent(config_path=tmp)
            self.assertEqual(agent._lane_half_width, 240.0)
            self.assertEqual(agent._min_half_width, 200.0)
        finally:
            os.remove(tmp)

    def test_half_width_default_when_no_config_path(self):
        agent = ll.LeaderLaneAgent(config_path=None)     # skips the file read
        self.assertEqual(agent._lane_half_width, agent._min_half_width)

    def test_calculate_error_clamps_half_width_floor(self):
        self.agent._min_half_width = 180.0
        self.agent._lane_half_width = 100.0          # below floor
        # Both edges absent -> base returns prev_error and leaves width as-is;
        # the override then clamps it back up to the floor.
        self.agent._calculate_error([], [], False, False, 640)
        self.assertGreaterEqual(self.agent._lane_half_width, 180.0)

    def test_calculate_error_no_clamp_when_above_floor(self):
        self.agent._min_half_width = 100.0
        self.agent._lane_half_width = 220.0          # already above floor -> no clamp
        self.agent._calculate_error([], [], False, False, 640)
        self.assertEqual(self.agent._lane_half_width, 220.0)

    def test_reset_steering_state(self):
        self.agent._prev_error = 0.9
        self.agent._filtered_error = 0.5
        self.agent._left_history.append(1.0)
        self.agent._right_history.append(1.0)
        self.agent.reset_steering_state()
        self.assertEqual(self.agent._prev_error, 0.0)
        self.assertEqual(len(self.agent._left_history), 0)

    def test_bottom_line_visible_none_mask(self):
        self.assertFalse(self.agent.bottom_line_visible({}, 'left', 25))

    def test_bottom_line_visible_true(self):
        info = {'yellow_mask': small_masks(paint=True)}
        self.assertTrue(self.agent.bottom_line_visible(info, 'left', 25))

    def test_bottom_line_visible_false_when_empty(self):
        info = {'white_mask': small_masks(paint=False)}
        self.assertFalse(self.agent.bottom_line_visible(info, 'right', 25))

    def _patch_student(self, left_paint, right_paint, yellow_xs, white_xs):
        ml = (small_masks(paint=left_paint) > 0).astype(float)
        mr = (small_masks(paint=right_paint) > 0).astype(float)
        return (mock.patch.object(ll.student, 'detect_lane_markings',
                                  return_value=(ml, mr)),
                mock.patch.object(ll, 'detect_lines_in_slices',
                                  return_value=(yellow_xs, white_xs)))

    def test_single_line_follow_right(self):
        p1, p2 = self._patch_student(False, True, [], [30])
        with p1, p2:
            left, right = self.agent.compute_single_line_commands(blank_frame(), 'right')
        self.assertTrue(0.0 <= left <= 1.0 and 0.0 <= right <= 1.0)
        self.assertEqual(self.agent.last_debug_info['follow_line'], 'right')

    def test_single_line_follow_left(self):
        p1, p2 = self._patch_student(True, False, [10], [])
        with p1, p2:
            left, right = self.agent.compute_single_line_commands(blank_frame(), 'left')
        self.assertEqual(self.agent.last_debug_info['follow_line'], 'left')

    def test_single_line_follow_neither(self):
        self.agent._prev_error = 0.2
        p1, p2 = self._patch_student(False, False, [], [])
        with p1, p2:
            self.agent.compute_single_line_commands(blank_frame(), 'right')  # falls back to prev_error

    def test_single_line_detect_error_returns_zero(self):
        with mock.patch.object(ll.student, 'detect_lane_markings',
                               side_effect=RuntimeError('boom')):
            out = self.agent.compute_single_line_commands(blank_frame(), 'right')
        self.assertEqual(out, (0.0, 0.0))


# ---------------------------------------------------------------------------
# leader main() state machine
# ---------------------------------------------------------------------------

def leader_cfg(stop_hold_s=2.0):
    return {
        'lane_stop': {
            'stop_trigger_line': 'left', 'at_line_follow_line': 'right',
            'min_line_frames': 2, 'line_lost_frames': 2, 'bottom_line_px': 25,
            'stop_hold_s': stop_hold_s, 'line_speed_mult': 0.35,
        },
        'control': {'loop_hz': 1000, 'accel_rate': 0.5},
    }


class FakeLane:
    """Scripted stand-in for LeaderLaneAgent.

    Each non-STOP loop iteration consumes one bool from `triggers`, which is
    stored in last_debug_info['_trigger'] and reported by bottom_line_visible.
    """

    def __init__(self, triggers):
        self._triggers = list(triggers)
        self.resets = 0
        self.last_debug_info = self._info(False)

    def _info(self, trig):
        return {
            '_trigger': trig,
            'yellow_mask': small_masks(paint=trig),
            'white_mask': small_masks(paint=trig),
            'slice_ys': [40],
            'follow_line': 'right',
            'lane_detected': True,
            'lateral_error': 0.1,
        }

    def _next(self):
        trig = self._triggers.pop(0) if self._triggers else False
        self.last_debug_info = self._info(trig)
        return trig

    def bottom_line_visible(self, info, side, px):
        return bool((info or {}).get('_trigger', False))

    def compute_commands(self, rgb):
        self._next()
        return 0.3, 0.3

    def compute_single_line_commands(self, rgb, side):
        self._next()
        return 0.2, 0.2

    def reset_steering_state(self):
        self.resets += 1


class TestLeaderStateMachine(unittest.TestCase):
    def _run(self, triggers, n, cfg, leds=None):
        clock = Clock(start=1000.0, step=1.0)
        cam = FakeCamera([(True, blank_frame())] * n)
        wheels = FakeWheels()
        stop = FakeStop(iterations=n, on_wait=clock.tick)
        fake = FakeLane(triggers)
        with mock.patch.object(la, 'LeaderLaneAgent', lambda *a, **k: fake), \
             mock.patch.object(la, 'load_config', return_value=cfg), \
             mock.patch.object(la.time, 'time', clock.time):
            la.main(cam, wheels, leds, stop)
        return wheels, fake

    def test_full_cruise_atline_stop_cycle(self):
        # Drives CRUISE -> AT_LINE -> STOP(hold) -> AT_LINE -> CRUISE.
        triggers = [True, True, False, False, True, True]
        wheels, fake = self._run(triggers, n=7, cfg=leader_cfg(stop_hold_s=2.0),
                                  leds=FakeLeds())
        self.assertEqual(len(wheels.commands), 7 + 1)  # 7 loops + finally stop
        self.assertGreaterEqual(fake.resets, 1)        # reset on STOP and on re-acquire
        self.assertIn((0.0, 0.0), wheels.commands)     # a full-stop command was issued

    def test_stop_releases_immediately_when_hold_zero(self):
        # stop_hold_s=0 -> STOP transitions straight to AT_LINE next iteration.
        triggers = [True, True, False, False, True]
        wheels, _ = self._run(triggers, n=6, cfg=leader_cfg(stop_hold_s=0.0),
                              leds=FakeLeds())
        self.assertEqual(len(wheels.commands), 6 + 1)

    def test_camera_not_ok_and_no_leds(self):
        clock = Clock()
        cam = FakeCamera([(False, None), (True, blank_frame())])
        wheels = FakeWheels()
        stop = FakeStop(iterations=2, on_wait=clock.tick)
        fake = FakeLane([True])
        with mock.patch.object(la, 'LeaderLaneAgent', lambda *a, **k: fake), \
             mock.patch.object(la, 'load_config', return_value=leader_cfg()), \
             mock.patch.object(la.time, 'time', clock.time):
            la.main(cam, wheels, None, stop)           # leds=None -> `if leds` false branch
        self.assertEqual(len(wheels.commands), 2)      # 1 loop + finally stop


if __name__ == '__main__':
    unittest.main()
