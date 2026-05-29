from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor, nn

from .config import ARCH_ORDER, MODEL_SPECS, ROOT
from .inference import load_class_index_map, load_model, predict_face, resolve_paths
from .ui import FPSCounter, HUDRenderer, loading_screen, put_text_shadow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time facial emotion detection - PyTorch + OpenCV")
    parser.add_argument("--architecture", choices=sorted(MODEL_SPECS), default="resnet", help="Model architecture to load.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Override checkpoint .pt path.")
    parser.add_argument("--class-indices", type=Path, default=None, help="Override class_indices.json path.")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam device index.")
    parser.add_argument("--confidence-threshold", type=float, default=0.0, help="Minimum confidence to show emotion label.")
    parser.add_argument("--width", type=int, default=1280, help="Capture width.")
    parser.add_argument("--height", type=int, default=720, help="Capture height.")
    parser.add_argument(
        "--detection-interval",
        type=float,
        default=3.0,
        help="Seconds between emotion detections. Frames in-between reuse the last result.",
    )
    return parser.parse_args()


def run_app(args: argparse.Namespace) -> int:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    device_name = {
        "cuda": "GPU (CUDA)",
        "mps": "GPU (MPS)",
        "cpu": "CPU",
    }.get(device.type, device.type)
    print(f"[INFO] Using device: {device_name}")

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {args.camera_index}")
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera resolution: {actual_w}x{actual_h}")

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_alt2.xml"
    face_cascade = cv2.CascadeClassifier(str(cascade_path))
    if face_cascade.empty():
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(str(cascade_path))
    if face_cascade.empty():
        print(f"[ERROR] Could not load Haar cascade from {cascade_path}")
        return 1
    print(f"[INFO] Face detector: {cascade_path.name}")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    arch_keys = ARCH_ORDER
    arch_idx = arch_keys.index(args.architecture) if args.architecture in arch_keys else 0
    current_arch = arch_keys[arch_idx]

    models: dict[str, nn.Module] = {}
    idx_to_class_map: dict[str, list[str]] = {}

    def ensure_model_loaded(arch: str) -> tuple[nn.Module, list[str]]:
        if arch not in models:
            spec = MODEL_SPECS[arch]
            checkpoint_path, class_indices_path = resolve_paths(arch, args.checkpoint, args.class_indices)
            c2i = load_class_index_map(class_indices_path)
            print(f"[INFO] Loading {spec.name} from {checkpoint_path} ...")
            loading_screen(cap, spec.name)
            model_value, idx_to_class_value = load_model(arch, checkpoint_path, c2i, device)
            models[arch] = model_value
            idx_to_class_map[arch] = idx_to_class_value
            print(f"[INFO] {spec.name} loaded successfully.")
        return models[arch], idx_to_class_map[arch]

    try:
        model, idx_to_class = ensure_model_loaded(current_arch)
    except Exception as exc:
        print(f"[ERROR] Failed to load model: {exc}")
        cap.release()
        return 1

    spec = MODEL_SPECS[current_arch]

    hud = HUDRenderer(actual_w, actual_h)
    fps_counter = FPSCounter(window=30)
    show_hud = True
    debug_mode = False

    smooth_probs: Tensor | None = None
    smooth_alpha = 0.6

    # Run detection once every N seconds and reuse previous results between runs.
    detection_interval = max(0.0, args.detection_interval)
    last_detection_ts = -1e9
    last_face_results: list[tuple[tuple[int, int, int, int], str, float, Tensor]] = []

    window_name = "Emotion Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, actual_w, actual_h)

    screenshot_dir = ROOT / "screenshots"
    screenshot_dir.mkdir(exist_ok=True)

    print(
        "[INFO] Running. Press Q/Esc to quit, M to cycle models, S to screenshot. "
        f"Detection interval: {detection_interval:.1f}s"
    )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Failed to grab frame.")
                break

            frame = cv2.flip(frame, 1)
            fps = fps_counter.tick()

            now = time.perf_counter()
            should_detect = (now - last_detection_ts) >= detection_interval

            if should_detect:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = clahe.apply(gray)

                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=3,
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE,
                )

                if not isinstance(faces, np.ndarray) or len(faces) == 0:
                    faces = face_cascade.detectMultiScale(gray)

                face_results: list[tuple[tuple[int, int, int, int], str, float, Tensor]] = []
                num_faces = len(faces) if isinstance(faces, np.ndarray) else 0

                for i in range(num_faces):
                    fx, fy, fw, fh = int(faces[i][0]), int(faces[i][1]), int(faces[i][2]), int(faces[i][3])

                    pad_x = int(fw * 0.1)
                    pad_y = int(fh * 0.1)
                    x1 = max(0, fx - pad_x)
                    y1 = max(0, fy - pad_y)
                    x2 = min(frame.shape[1], fx + fw + pad_x)
                    y2 = min(frame.shape[0], fy + fh + pad_y)
                    roi = frame[y1:y2, x1:x2]

                    if roi.size == 0 or np.sum(roi) == 0:
                        continue

                    raw_probs = predict_face(
                        model,
                        roi,
                        image_size=spec.image_size,
                        normalize_input=spec.normalize_input,
                        device=device,
                    )

                    if smooth_probs is None or smooth_probs.shape != raw_probs.shape:
                        smooth_probs = raw_probs
                    else:
                        smooth_probs = smooth_alpha * raw_probs + (1 - smooth_alpha) * smooth_probs

                    top_idx = int(smooth_probs.argmax())
                    confidence = float(smooth_probs[top_idx])
                    label = idx_to_class[top_idx] if top_idx < len(idx_to_class) else "Unknown"
                    if confidence < args.confidence_threshold:
                        label = f"Uncertain ({label})"

                    face_results.append(((fx, fy, fw, fh), label, confidence, smooth_probs))

                last_face_results = face_results
                last_detection_ts = now

            if show_hud and last_face_results:
                _, top_label, _, top_probs = last_face_results[0]
                hud.draw_probability_panel(frame, top_probs, idx_to_class, top_label)

            for box, label, conf, probs in last_face_results:
                hud.draw_face_annotation(frame, box, label, conf, probs, idx_to_class, debug_mode)

            if show_hud:
                hud.draw_status_bar(
                    frame,
                    model_name=MODEL_SPECS[current_arch].name,
                    fps=fps,
                    device_name=device_name,
                    num_faces=len(last_face_results),
                )
                if not last_face_results:
                    msg = "No faces detected - look at the camera"
                    tw = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)[0][0]
                    cx = (actual_w - tw) // 2
                    put_text_shadow(frame, msg, (cx, actual_h // 2), cv2.FONT_HERSHEY_DUPLEX, 0.7, (100, 140, 255), 1)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("m"):
                arch_idx = (arch_idx + 1) % len(arch_keys)
                current_arch = arch_keys[arch_idx]
                spec = MODEL_SPECS[current_arch]
                smooth_probs = None
                last_face_results = []
                last_detection_ts = -1e9
                try:
                    model, idx_to_class = ensure_model_loaded(current_arch)
                    print(f"[INFO] Switched to {spec.name}")
                except Exception as exc:
                    print(f"[ERROR] Could not load {spec.name}: {exc}")
            elif key == ord("s"):
                ts = time.strftime("%Y%m%d_%H%M%S")
                fname = screenshot_dir / f"emotion_{ts}.png"
                cv2.imwrite(str(fname), frame)
                print(f"[INFO] Screenshot saved -> {fname}")
            elif key == ord("h"):
                show_hud = not show_hud
            elif key == ord("d"):
                debug_mode = not debug_mode
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


def main() -> int:
    try:
        args = parse_args()
        args.architecture = args.architecture.lower()
        if args.architecture not in MODEL_SPECS:
            print(f"[ERROR] Unknown architecture '{args.architecture}'. Choose from: {sorted(MODEL_SPECS)}")
            return 1
        return run_app(args)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
        return 0
