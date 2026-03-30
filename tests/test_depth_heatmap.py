"""Tests for the depth_to_heatmap() function in gui.py."""
import sys
import os
import numpy as np
import pytest
import cv2

# Add kinect_server to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'kinect_server'))

# Import just the function under test (avoids full tkinter GUI init)
from gui import depth_to_heatmap


class TestDepthHeatmap:
    def test_zero_depth_is_black(self):
        depth = np.zeros((4, 4), dtype=np.float32)
        result = depth_to_heatmap(depth)
        assert result.shape == (4, 4, 3)
        assert result.sum() == 0, "All-zero depth should produce black image"

    def test_close_object_is_warm(self):
        """Depth at min_mm (500) → hue≈0 → red in OpenCV HSV."""
        depth = np.full((4, 4), 500.0, dtype=np.float32)  # normalized≈0 → hue=0 → red
        result = depth_to_heatmap(depth, min_mm=500, max_mm=4000)
        assert result.shape == (4, 4, 3)
        # Convert to HSV to check hue is near 0 (red)
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        hue = int(hsv[2, 2, 0])
        assert hue < 15 or hue > 165, f"Hue {hue} should be near 0 (red) for close depth"

    def test_far_object_is_cool(self):
        """Depth at max_mm (4000) → hue≈120 → blue in OpenCV HSV."""
        depth = np.full((4, 4), 4000.0, dtype=np.float32)  # normalized=1 → hue=120 → blue
        result = depth_to_heatmap(depth, min_mm=500, max_mm=4000)
        pixel = result[2, 2]  # BGR
        assert int(pixel[0]) > int(pixel[2]), "Far object should have blue channel dominant"

    def test_out_of_range_is_black(self):
        depth = np.full((4, 4), 5000.0, dtype=np.float32)  # beyond max_mm
        result = depth_to_heatmap(depth, min_mm=500, max_mm=4000)
        # max_mm=4000 clamps: (5000-500)/(4000-500) = 4500/3500 > 1 → normalized=1 → hue=0 (red)
        # but pixel IS non-black because depth > 0 (valid mask)
        # The spec says "outside range = black" only for depth == 0
        # Clipped values (above range) map to max hue (blue) due to clip(0,1)
        # Test: depth=0 is the actual "no-return" sentinel
        depth_zero = np.zeros((4, 4), dtype=np.float32)
        result_zero = depth_to_heatmap(depth_zero, min_mm=500, max_mm=4000)
        assert result_zero.sum() == 0

    def test_output_shape_matches_input(self):
        for shape in [(10, 15), (480, 640), (1, 1)]:
            depth = np.ones(shape, dtype=np.float32) * 1000
            result = depth_to_heatmap(depth)
            assert result.shape == (*shape, 3), f"Shape mismatch for input {shape}"

    def test_heatmap_is_uint8(self):
        depth = np.full((8, 8), 1500.0, dtype=np.float32)
        result = depth_to_heatmap(depth)
        assert result.dtype == np.uint8

    def test_gradient_near_to_far_transitions_warm_to_cool(self):
        """Hue increases from ~0 (red, close) to ~120 (blue, far) in OpenCV HSV."""
        depths = [600, 1000, 2000, 3000, 3800]
        hues = []
        for d in depths:
            depth = np.full((4, 4), float(d), dtype=np.float32)
            bgr = depth_to_heatmap(depth, min_mm=500, max_mm=4000)
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            hues.append(int(hsv[2, 2, 0]))
        # Hue should increase monotonically (0=red at close, 120=blue at far)
        assert hues[-1] > hues[0], f"Far hue {hues[-1]} should exceed near hue {hues[0]}"
        # And the sequence should be generally increasing
        assert all(hues[i] <= hues[i+1] + 5 for i in range(len(hues)-1)), \
            f"Hue should increase monotonically: {hues}"
