"""
Test configuration: mock all hardware + heavy dependencies so tests run without
Kinect hardware, MediaPipe models, or a display server.
"""
import sys
from unittest.mock import MagicMock

# Block hardware / GUI-requiring imports before any test module loads them
for mod in [
    'freenect', 'freenect2', 'pykinect2',
    'cv2.cuda',
    'pythonosc', 'pythonosc.udp_client', 'pythonosc.dispatcher',
    'pythonosc.osc_server',
    'mediapipe', 'mediapipe.tasks', 'mediapipe.tasks.python',
    'mediapipe.tasks.python.vision', 'mediapipe.tasks.python.core',
    'mediapipe.tasks.python.core.base_options',
    'flask',
    'tkinter', 'PIL', 'PIL.Image', 'PIL.ImageTk',
]:
    sys.modules.setdefault(mod, MagicMock())

import cv2  # allow regular cv2 (CPU only); CUDA already mocked above
import numpy as np
