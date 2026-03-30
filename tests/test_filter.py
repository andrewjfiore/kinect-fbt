"""Tests for the 1-Euro filter."""
import sys
import os
import time
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'kinect_server'))

from filter import LowPassFilter, OneEuroFilter, OneEuroFilter3D


class TestLowPassFilter:
    def test_first_value_passthrough(self):
        f = LowPassFilter(alpha=0.5)
        assert f.filter(3.0) == 3.0

    def test_smoothing_towards_target(self):
        f = LowPassFilter(alpha=0.5)
        f.filter(0.0)
        v1 = f.filter(10.0)
        assert 0.0 < v1 < 10.0, "Should be smoothed between 0 and 10"

    def test_alpha_clamp(self):
        f = LowPassFilter(alpha=0.5)
        f.set_alpha(2.0)
        assert f._alpha <= 1.0
        f.set_alpha(-1.0)
        assert f._alpha >= 0.0


class TestOneEuroFilter:
    def test_first_call_returns_value(self):
        f = OneEuroFilter()
        t = time.monotonic()
        result = f.filter(5.0, t)
        assert result == 5.0

    def test_reduces_noise(self):
        f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
        t = 0.0
        # Noisy signal alternating around 1.0
        values = [1.0, 1.5, 0.5, 1.2, 0.8, 1.1, 0.9]
        outputs = []
        for v in values:
            outputs.append(f.filter(v, t))
            t += 0.033
        # Variance of outputs should be less than variance of inputs
        import statistics
        assert statistics.variance(outputs[1:]) < statistics.variance(values[1:])

    def test_same_timestamp_returns_last_value(self):
        f = OneEuroFilter()
        t = 1.0
        f.filter(5.0, t)
        result = f.filter(10.0, t)  # same timestamp → dt=0
        assert result == 5.0  # returns last value unchanged

    def test_tracks_slow_signal(self):
        """Filter should follow a slowly-changing signal without too much lag."""
        f = OneEuroFilter(min_cutoff=1.0, beta=0.1)
        t = 0.0
        # Ramp from 0 to 3 over 3 seconds
        for i in range(30):
            target = i * 0.1
            out = f.filter(target, t)
            t += 0.1
        # After 3 seconds of tracking, output should be close to 3.0
        assert abs(out - 3.0) < 0.5


class TestOneEuroFilter3D:
    def test_returns_tuple(self):
        f = OneEuroFilter3D()
        result = f.filter(1.0, 2.0, 3.0, timestamp=0.0)
        assert len(result) == 3

    def test_first_call_returns_input(self):
        f = OneEuroFilter3D()
        x, y, z = f.filter(1.0, 2.0, 3.0, timestamp=0.0)
        assert x == 1.0 and y == 2.0 and z == 3.0

    def test_filters_each_axis_independently(self):
        """3D filter smooths each axis; stable axes remain close to their input."""
        f = OneEuroFilter3D()
        t = 0.0
        outputs_y = []
        outputs_z = []
        for i in range(15):
            noise = 0.5 * (1 if i % 2 == 0 else -1)
            ox, oy, oz = f.filter(1.0 + noise, 2.0, 3.0, timestamp=t)
            outputs_y.append(oy)
            outputs_z.append(oz)
            t += 0.033
        # Stable axes (y=2.0, z=3.0) should converge quickly
        assert abs(sum(outputs_y[5:]) / len(outputs_y[5:]) - 2.0) < 0.1
        assert abs(sum(outputs_z[5:]) / len(outputs_z[5:]) - 3.0) < 0.1
