"""
End-to-end chain test: mock camera → filter → fuse → OSC (dry-run).
No hardware or network access required.
"""
import sys
import os
import time
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'kinect_server'))

from camera import CameraFrame, Landmark3D, JOINT_INDICES, NUM_LANDMARKS
from fusion import MultiCameraFusion
from filter import OneEuroFilter3D


def _synthetic_frame(cam_id: int, t_offset: float = 0.0) -> CameraFrame:
    """Generate a CameraFrame with a walking-like pose."""
    landmarks = [None] * NUM_LANDMARKS
    pose = {
        "LEFT_HIP":      (0.12, 1.0 + 0.02 * np.sin(t_offset), 2.0),
        "RIGHT_HIP":     (-0.12, 1.0 + 0.02 * np.sin(t_offset + 0.1), 2.0),
        "LEFT_SHOULDER": (0.25, 1.45, 2.0),
        "RIGHT_SHOULDER":(-0.25, 1.45, 2.0),
        "LEFT_KNEE":     (0.15, 0.55, 2.05),
        "RIGHT_KNEE":    (-0.15, 0.55, 2.05),
        "LEFT_ANKLE":    (0.15, 0.05, 2.0),
        "RIGHT_ANKLE":   (-0.15, 0.05, 2.0),
    }
    for name, (x, y, z) in pose.items():
        idx = JOINT_INDICES[name]
        landmarks[idx] = Landmark3D(
            x=x, y=y, z=z,
            visibility=0.95, depth_confidence=0.9,
            index=idx,
        )
    return CameraFrame(
        camera_id=cam_id,
        landmarks=landmarks,
        timestamp=time.monotonic(),
        depth_frame=np.full((424, 512), 2000.0, dtype=np.float32),
    )


class TestE2EChain:
    def test_full_pipeline_produces_trackers(self):
        """mock camera → fusion → trackers non-empty."""
        cal = {0: np.eye(4)}
        fusion = MultiCameraFusion(cal, user_height=1.75)

        frame = _synthetic_frame(0)
        joints = fusion.update([frame])
        trackers = fusion.get_trackers()

        assert len(trackers) > 0
        tracker_ids = [t.tracker_id for t in trackers]
        assert 1 in tracker_ids  # hip

    def test_pipeline_runs_multiple_frames(self):
        """Run 30 frames and verify FPS tracking doesn't crash."""
        cal = {0: np.eye(4)}
        fusion = MultiCameraFusion(cal)

        for i in range(30):
            frame = _synthetic_frame(0, t_offset=i * 0.033)
            fusion.update([frame])

        assert fusion.joints_tracked_count() > 0

    def test_dual_camera_fusion_smoke(self):
        """Two cameras with identity calibration."""
        cal = {0: np.eye(4), 1: np.eye(4)}
        fusion = MultiCameraFusion(cal)

        f0 = _synthetic_frame(0)
        f1 = _synthetic_frame(1)
        joints = fusion.update([f0, f1])

        assert "LEFT_HIP" in joints
        assert joints["LEFT_HIP"].confidence > 0

    def test_osc_dry_run_does_not_crash(self):
        """OSC dry-run path does not raise exceptions."""
        # pythonosc is mocked in conftest.py, so just verify the logic path
        from unittest.mock import patch, MagicMock
        mock_osc = MagicMock()
        sys.modules['pythonosc.udp_client'] = mock_osc
        sys.modules['pythonosc.osc_bundle_builder'] = MagicMock()
        sys.modules['pythonosc.osc_message_builder'] = MagicMock()

        from osc_output import OSCOutput
        from fusion import TrackerData

        osc = OSCOutput("127.0.0.1", 9000, dry_run=True)
        trackers = [TrackerData(1, (0.0, 1.0, 0.0), (0.0, 90.0, 0.0), 0.85)]
        # Should not raise
        osc.send(trackers, cameras_active=1, joints_tracked=8, fps=30.0)

    def test_filter_integrated_in_fusion(self):
        """Fusion-internal 1-euro filter reduces jitter over multiple frames."""
        cal = {0: np.eye(4)}
        fusion = MultiCameraFusion(cal)
        t = 0.0
        raw_xs = []
        fused_xs = []
        for i in range(20):
            noise = 0.1 * (1 if i % 2 == 0 else -1)
            frame = _synthetic_frame(0, t)
            # Inject noise into LEFT_HIP x
            idx = JOINT_INDICES["LEFT_HIP"]
            lm = frame.landmarks[idx]
            frame.landmarks[idx] = Landmark3D(
                x=lm.x + noise, y=lm.y, z=lm.z,
                visibility=lm.visibility, depth_confidence=lm.depth_confidence,
                index=lm.index,
            )
            raw_xs.append(lm.x + noise)
            joints = fusion.update([frame])
            fused_xs.append(joints["LEFT_HIP"].x)
            t += 0.033

        import statistics
        raw_var = statistics.variance(raw_xs[5:])
        fused_var = statistics.variance(fused_xs[5:])
        assert fused_var <= raw_var * 1.1, (
            f"Fused variance {fused_var:.4f} should be <= raw variance {raw_var:.4f}"
        )
