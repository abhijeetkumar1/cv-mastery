"""Live object detection on your webcam with any YOLO weights (module 06b).

Run from the project root:
    uv run python notebooks/06_detection/webcam_detect.py                       # COCO-pretrained YOLO11n
    uv run python notebooks/06_detection/webcam_detect.py --weights path/to/best.pt --conf 0.4
    uv run python notebooks/06_detection/webcam_detect.py --source data/opencv_samples/vtest.avi

Keys: q = quit, s = save a snapshot to data/outputs/.
On macOS, allow camera access for your terminal app when asked.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None, help="YOLO .pt file (default: COCO-pretrained yolo11n)")
    ap.add_argument("--source", default="0", help="camera index or video file")
    ap.add_argument("--conf", type=float, default=0.35, help="score threshold")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    from ultralytics import YOLO

    from cvlib.data import model_path
    from cvlib.env import get_device

    weights = args.weights or str(model_path("yolo11n.pt"))
    model = YOLO(weights)
    device = get_device()
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {source!r}. Check camera permissions or the file path.")
    out_dir = ROOT / "data" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    t_prev = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if isinstance(source, int):
            frame = cv2.flip(frame, 1)
        result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, device=device, verbose=False)[0]
        vis = result.plot()  # boxes + labels drawn by Ultralytics
        now = time.perf_counter()
        fps, t_prev = 1.0 / max(now - t_prev, 1e-6), now
        cv2.putText(vis, f"{len(result.boxes)} objects | {fps:4.1f} FPS | {device} | q quit, s snapshot",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("YOLO detector", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            path = out_dir / f"detect_{int(time.time())}.png"
            cv2.imwrite(str(path), vis)
            print("saved", path)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
