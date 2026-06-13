"""Fake hardware / harness used by the coverage tests.

The agents take (camera, wheels, leds, stop_event) and run an infinite control
loop. These fakes let a test drive a deterministic, finite number of loop
iterations and inspect every command the agent issued.
"""

import numpy as np


def blank_frame(h=48, w=64):
    """Small BGR frame so the CV calls in _visualize/_annotate are cheap."""
    return np.zeros((h, w, 3), dtype=np.uint8)


class FakeCamera:
    """Yields a scripted list of (ok, frame) tuples, then (False, None)."""

    def __init__(self, frames):
        self._frames = list(frames)
        self.reads = 0

    def read(self):
        self.reads += 1
        if self._frames:
            return self._frames.pop(0)
        return False, None


class FakeWheels:
    def __init__(self):
        self.commands = []

    def set_wheels_speed(self, left, right):
        self.commands.append((left, right))

    def __getattr__(self, name):
        # Tolerate change_scene / etc. that some servers call.
        return lambda *a, **k: None


class FakeLeds:
    def __init__(self):
        self.rgb = []
        self.off_count = 0

    def set_rgb(self, idx, color):
        self.rgb.append((idx, list(color)))

    def all_off(self):
        self.off_count += 1


class FakeStop:
    """Stops the agent loop after exactly `iterations` passes of `while`.

    is_set() is evaluated once at the top of each loop iteration. Returning
    False for the first `iterations` checks runs the body that many times.
    `wait()` optionally advances a controllable clock so time-based branches
    (e.g. STOP hold) are deterministic.
    """

    def __init__(self, iterations, on_wait=None):
        self.iterations = iterations
        self.checks = 0
        self.waits = 0
        self._on_wait = on_wait

    def is_set(self):
        self.checks += 1
        return self.checks > self.iterations

    def wait(self, dt):
        self.waits += 1
        if self._on_wait is not None:
            self._on_wait(dt)


class Clock:
    """Monotonic-ish fake clock; `tick` advances it by a fixed step per wait."""

    def __init__(self, start=1000.0, step=1.0):
        self.now = float(start)
        self.step = float(step)

    def time(self):
        return self.now

    def tick(self, _dt=None):
        self.now += self.step
