"""
Real-Time Facial Emotion Detection
=================================
Modular webcam-based emotion recognition using PyTorch models.

Usage
-----
    python realtime_emotion_app.py --architecture resnet
    python realtime_emotion_app.py --architecture vgg
    python realtime_emotion_app.py --architecture densenet
    python realtime_emotion_app.py --architecture vit
    python realtime_emotion_app.py --architecture resnet --camera-index 1
    python realtime_emotion_app.py --architecture resnet --confidence-threshold 0.35
    python realtime_emotion_app.py --architecture resnet --detection-interval 3

Keyboard shortcuts (while the window is focused)
------------------------------------------------
    Q / Esc   Quit
    M         Cycle through model architectures
    S         Save a screenshot to the project folder
    H         Toggle the HUD (heads-up display)
    D         Toggle debug mode (show raw probabilities)
"""

from __future__ import annotations

from emotion_app.app import main


if __name__ == "__main__":
    raise SystemExit(main())
