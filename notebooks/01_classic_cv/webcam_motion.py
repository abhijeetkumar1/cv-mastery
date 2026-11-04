"""Live motion detector on your webcam (module 01d project).

Run from the project root:
    uv run python notebooks/01_classic_cv/webcam_motion.py            # webcam 0
    uv run python notebooks/01_classic_cv/webcam_motion.py --source data/opencv_samples/vtest.avi

Keys: q = quit, m = show/hide the mask window, s = save a snapshot to data/outputs/.
On macOS, allow camera access for your terminal app when asked.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "outputs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="camera index (0, 1, ...) or a video file path")
    ap.add_argument("--min-area", type=int, default=800, help="ignore blobs smaller than this (pixels)")
    ap.add_argument("--var-threshold", type=float, default=25, help="MOG2 sensitivity: lower = more sensitive")
    args = ap.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {source!r}. Check camera permissions or the file path.")

    sub = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=args.var_threshold, detectShadows=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    show_mask, t_prev = True, time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if isinstance(source, int):
            frame = cv2.flip(frame, 1)  # mirror view feels natural for a webcam

        mask = sub.apply(frame)
        mask = np.where(mask == 255, 255, 0).astype(np.uint8)  # drop shadows
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        n = 0
        for c in contours:
            if cv2.contourArea(c) < args.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            n += 1

        now = time.perf_counter()
        fps, t_prev = 1.0 / max(now - t_prev, 1e-6), now
        cv2.putText(frame, f"moving: {n}  |  {fps:4.1f} FPS  |  q quit, m mask, s snapshot",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("motion detector", frame)
        if show_mask:
            cv2.imshow("foreground mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("m"):
            show_mask = not show_mask
            if not show_mask:
                cv2.destroyWindow("foreground mask")
        if key == ord("s"):
            path = OUT_DIR / f"snapshot_{int(time.time())}.png"
            cv2.imwrite(str(path), frame)
            print("saved", path)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
