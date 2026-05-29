from __future__ import annotations

import time
from collections import deque

import cv2
import numpy as np
from torch import Tensor

from .config import (
    COLOR_ACCENT,
    COLOR_ACCENT2,
    COLOR_HUD_BG,
    COLOR_TEXT_DIM,
    COLOR_TEXT_WHITE,
    EMOTION_COLORS,
)


def draw_rounded_rect(
    frame: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    radius: int = 10,
    thickness: int = -1,
    alpha: float = 1.0,
) -> None:
    x1, y1 = pt1
    x2, y2 = pt2
    if x1 >= x2 or y1 >= y2:
        return

    if alpha < 1.0:
        overlay = frame.copy()
        _draw_solid_rounded_rect(overlay, pt1, pt2, color, radius, thickness)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    else:
        _draw_solid_rounded_rect(frame, pt1, pt2, color, radius, thickness)


def _draw_solid_rounded_rect(
    frame: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    radius: int,
    thickness: int,
) -> None:
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    filled = thickness == -1
    lw = thickness if not filled else -1

    if filled:
        cv2.rectangle(frame, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(frame, (x1, y1 + r), (x2, y2 - r), color, -1)
    else:
        cv2.line(frame, (x1 + r, y1), (x2 - r, y1), color, lw)
        cv2.line(frame, (x1 + r, y2), (x2 - r, y2), color, lw)
        cv2.line(frame, (x1, y1 + r), (x1, y2 - r), color, lw)
        cv2.line(frame, (x2, y1 + r), (x2, y2 - r), color, lw)

    for cx, cy, start, end in [
        (x1 + r, y1 + r, 180, 270),
        (x2 - r, y1 + r, 270, 360),
        (x1 + r, y2 - r, 90, 180),
        (x2 - r, y2 - r, 0, 90),
    ]:
        cv2.ellipse(frame, (cx, cy), (r, r), 0, start, end, color, lw)


def put_text_shadow(
    frame: np.ndarray,
    text: str,
    org: tuple[int, int],
    font: int,
    scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    shadow_color = (0, 0, 0)
    cv2.putText(frame, text, (org[0] + 1, org[1] + 1), font, scale, shadow_color, thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, org, font, scale, color, thickness, cv2.LINE_AA)


class HUDRenderer:
    FONT = cv2.FONT_HERSHEY_DUPLEX
    FONT_MONO = cv2.FONT_HERSHEY_PLAIN

    def __init__(self, frame_w: int, frame_h: int) -> None:
        self.W = frame_w
        self.H = frame_h

    def draw_status_bar(
        self,
        frame: np.ndarray,
        model_name: str,
        fps: float,
        device_name: str,
        num_faces: int,
    ) -> None:
        bar_h = 52
        draw_rounded_rect(frame, (0, 0), (self.W, bar_h), COLOR_HUD_BG, radius=0, alpha=0.82)

        chip_text = f"  {model_name}  "
        tw, th = cv2.getTextSize(chip_text, self.FONT, 0.52, 1)[0]
        chip_x1, chip_y1 = 12, 9
        chip_x2, chip_y2 = chip_x1 + tw + 4, chip_y1 + th + 8
        draw_rounded_rect(frame, (chip_x1, chip_y1), (chip_x2, chip_y2), COLOR_ACCENT, radius=6)
        cv2.putText(frame, chip_text, (chip_x1 + 2, chip_y2 - 7), self.FONT, 0.52, (10, 10, 10), 1, cv2.LINE_AA)

        fps_str = f"FPS  {fps:5.1f}"
        fps_col = (60, 230, 80) if fps >= 20 else (60, 180, 230) if fps >= 10 else (60, 60, 230)
        put_text_shadow(frame, fps_str, (chip_x2 + 18, 35), self.FONT, 0.62, fps_col, 1)

        face_str = f"Faces  {num_faces}"
        put_text_shadow(frame, face_str, (chip_x2 + 140, 35), self.FONT, 0.62, COLOR_TEXT_WHITE, 1)

        dev_str = device_name.upper()
        tw2 = cv2.getTextSize(dev_str, self.FONT, 0.48, 1)[0][0]
        dev_x = self.W - tw2 - 20
        put_text_shadow(frame, dev_str, (dev_x, 34), self.FONT, 0.48, COLOR_ACCENT2, 1)

        hint = "[Q] Quit   [M] Model   [S] Screenshot   [H] HUD   [D] Debug"
        put_text_shadow(frame, hint, (12, self.H - 12), self.FONT_MONO, 0.9, (120, 120, 120), 1)

    def draw_probability_panel(
        self,
        frame: np.ndarray,
        probs: Tensor,
        idx_to_class: list[str],
        top_label: str,
    ) -> None:
        panel_w = 220
        panel_h = 8 + len(idx_to_class) * 30 + 10
        panel_x = self.W - panel_w - 14
        panel_y = 62

        draw_rounded_rect(
            frame,
            (panel_x - 6, panel_y - 4),
            (panel_x + panel_w + 6, panel_y + panel_h),
            COLOR_HUD_BG,
            radius=10,
            alpha=0.78,
        )

        put_text_shadow(frame, "EMOTION SCORES", (panel_x, panel_y + 12), self.FONT, 0.42, COLOR_ACCENT2, 1)

        bar_max_w = panel_w - 70
        for i, label in enumerate(idx_to_class):
            prob = float(probs[i]) if i < len(probs) else 0.0
            y = panel_y + 24 + i * 30
            is_top = label == top_label

            label_col = COLOR_ACCENT if is_top else COLOR_TEXT_DIM
            cv2.putText(frame, label[:9], (panel_x, y + 14), self.FONT, 0.42, label_col, 1, cv2.LINE_AA)

            bar_x = panel_x + 80
            bar_y_top, bar_y_bot = y + 6, y + 18
            draw_rounded_rect(frame, (bar_x, bar_y_top), (bar_x + bar_max_w, bar_y_bot), (50, 50, 60), radius=4)

            fill_w = int(prob * bar_max_w)
            if fill_w > 2:
                ec = EMOTION_COLORS.get(label, COLOR_ACCENT)
                bar_col = ec if not is_top else COLOR_ACCENT
                draw_rounded_rect(frame, (bar_x, bar_y_top), (bar_x + fill_w, bar_y_bot), bar_col, radius=4)

            pct_str = f"{prob * 100:4.1f}%"
            tw = cv2.getTextSize(pct_str, self.FONT, 0.38, 1)[0][0]
            cv2.putText(
                frame,
                pct_str,
                (bar_x + bar_max_w - tw - 2, y + 14),
                self.FONT,
                0.38,
                COLOR_TEXT_WHITE if is_top else COLOR_TEXT_DIM,
                1,
                cv2.LINE_AA,
            )

    def draw_face_annotation(
        self,
        frame: np.ndarray,
        box: tuple[int, int, int, int],
        label: str,
        confidence: float,
        probs: Tensor,
        idx_to_class: list[str],
        debug: bool,
    ) -> None:
        x, y, w, h = box
        ec = EMOTION_COLORS.get(label, COLOR_ACCENT)

        draw_rounded_rect(frame, (x - 3, y - 3), (x + w + 3, y + h + 3), ec, radius=10, thickness=4, alpha=0.5)
        draw_rounded_rect(frame, (x, y), (x + w, y + h), ec, radius=8, thickness=2)

        badge_text = f"{label}  {confidence * 100:.1f}%"
        (tw, bh), bl = cv2.getTextSize(badge_text, self.FONT, 0.6, 1)
        pad = 8
        bx1 = x
        by1 = max(0, y - bh - bl - pad * 2)
        bx2 = bx1 + tw + pad * 2
        by2 = max(0, y)
        draw_rounded_rect(frame, (bx1, by1), (bx2, by2), ec, radius=6, alpha=0.92)
        cv2.putText(frame, badge_text, (bx1 + pad, by2 - pad // 2 - 2), self.FONT, 0.6, (10, 10, 10), 1, cv2.LINE_AA)

        if debug and probs is not None:
            bar_h_mini = 4
            bar_w_full = w
            bar_y0 = y + h + 4
            for i, lbl in enumerate(idx_to_class):
                if i >= len(probs):
                    break
                prob = float(probs[i])
                bw = int(prob * bar_w_full)
                bx = x
                by = bar_y0 + i * (bar_h_mini + 2)
                col = EMOTION_COLORS.get(lbl, (100, 100, 100))
                cv2.rectangle(frame, (bx, by), (bx + bar_w_full, by + bar_h_mini), (40, 40, 50), -1)
                if bw > 0:
                    cv2.rectangle(frame, (bx, by), (bx + bw, by + bar_h_mini), col, -1)


class FPSCounter:
    def __init__(self, window: int = 30) -> None:
        self._times: deque[float] = deque(maxlen=window)

    def tick(self) -> float:
        now = time.perf_counter()
        self._times.append(now)
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0


def loading_screen(cap: cv2.VideoCapture, model_name: str) -> None:
    ok, frame = cap.read()
    if not ok:
        return

    overlay = np.zeros_like(frame)
    h, w = frame.shape[:2]

    blurred = cv2.GaussianBlur(frame, (31, 31), 0)
    cv2.addWeighted(blurred, 0.4, overlay, 0.6, 0, blurred)

    msg = f"Loading  {model_name} ..."
    tw, th = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)[0]
    cx = (w - tw) // 2
    cy = (h + th) // 2

    draw_rounded_rect(blurred, (cx - 30, cy - th - 20), (cx + tw + 30, cy + 20), COLOR_HUD_BG, radius=12, alpha=0.9)
    put_text_shadow(blurred, msg, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 1.0, COLOR_ACCENT, 2)
    cv2.imshow("Emotion Detection", blurred)
    cv2.waitKey(1)
