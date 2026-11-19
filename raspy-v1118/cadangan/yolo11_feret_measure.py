"""Run an Ultralytics YOLO11 segmentation model and measure maximum Feret diameter.

For each predicted instance mask, the script computes its convex hull and the
longest Euclidean distance between hull points. Annotated images and a JSON
summary use the same format as ``hailo_feret_measure.py``.

Install the runtime dependency if needed:
    pip install ultralytics

Example:
    python yolo11_feret_measure.py \
        --weights yolo11_udang.pt \
        --source shrimp_dataset/test/images \
        --classes 3
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from hailo_feret_measure import annotate, expand_source, longest_mask_points


ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "yolov11_udang.pt"
DEFAULT_SOURCE = ROOT / "shrimp_dataset" / "test" / "images"


def default_project():
    return ROOT / "runs" / "yolo11_feret" / datetime.now().strftime("%Y-%m-%d_%H%M%S")


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO11 segmentation inference and Feret measurement.")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Ultralytics YOLO11 segmentation weights")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Image, directory, or glob")
    parser.add_argument("--imgsz", type=int, default=640, help="Square inference size")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--mask-threshold", type=float, default=0.5, help="Mask threshold")
    parser.add_argument("--classes", type=int, nargs="+", default=[3], help="Class IDs to measure")
    parser.add_argument("--device", default=None, help="Inference device, for example cpu, 0, or mps")
    parser.add_argument("--project", default=str(default_project()), help="Output directory")
    parser.add_argument("--no-save", action="store_true", help="Do not save images or JSON")
    return parser.parse_args()


def to_numpy(value):
    """Convert an Ultralytics NumPy or PyTorch value to a NumPy array."""
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def source_box(box, image_shape):
    """Clamp and round an xyxy box to source-image coordinates."""
    height, width = image_shape[:2]
    x1, y1, x2, y2 = map(float, box[:4])
    return [
        max(0, min(width - 1, round(x1))),
        max(0, min(height - 1, round(y1))),
        max(0, min(width - 1, round(x2))),
        max(0, min(height - 1, round(y2))),
    ]


def main():
    args = parse_args()
    weights = Path(args.weights)
    if not weights.is_absolute() and not weights.exists():
        weights = ROOT / weights
    if not weights.exists():
        raise SystemExit(
            f"Weights not found: {weights}\n"
            "Add your YOLO11 segmentation weights there or pass --weights PATH."
        )

    images = expand_source(args.source)
    if not images:
        raise SystemExit(f"No images found for --source {args.source!r}")

    try:
        from ultralytics import YOLO
        from ultralytics.engine.results import Results
    except ImportError as exc:
        raise SystemExit("ultralytics is unavailable; install it with: pip install ultralytics") from exc

    out_dir = Path(args.project).resolve()
    if not args.no_save:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Weights: {weights}\nImages : {len(images)}\nClasses: {args.classes}")
    model = YOLO(str(weights))
    predict_args = {
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "classes": args.classes,
        "retina_masks": True,
        "verbose": False,
    }
    if args.device is not None:
        predict_args["device"] = args.device

    all_results = []
    for image_index, path in enumerate(images, start=1):
        original = cv2.imread(str(path))
        if original is None:
            continue

        prediction = next(iter(model.predict(source=original, stream=True, **predict_args)))
        if not isinstance(prediction, Results):
            raise RuntimeError(f"Unexpected Ultralytics result type: {type(prediction).__name__}")
        result = prediction
        measurements = []
        if result.boxes is not None and result.masks is not None:
            boxes = to_numpy(result.boxes.xyxy)
            confidences = to_numpy(result.boxes.conf)
            class_ids = to_numpy(result.boxes.cls).astype(np.int64)
            masks = to_numpy(result.masks.data)

            for index, (box, confidence, class_id, mask_values) in enumerate(
                zip(boxes, confidences, class_ids, masks), start=1
            ):
                if mask_values.shape != original.shape[:2]:
                    mask_values = cv2.resize(
                        mask_values.astype(np.float32),
                        (original.shape[1], original.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                mask = mask_values >= args.mask_threshold
                length, endpoint_a, endpoint_b = longest_mask_points(mask)
                measurements.append({
                    "instance": index,
                    "class_id": int(class_id),
                    "confidence": round(float(confidence), 4),
                    "length_px": round(length, 2),
                    "endpoint_a_xy": endpoint_a,
                    "endpoint_b_xy": endpoint_b,
                    "mask_area_px": int(mask.sum()),
                    "box_xyxy": source_box(box, original.shape),
                })

        all_results.append({"image": str(path), "instances": measurements})
        for item in measurements:
            print(f"{image_index:>4}  {path.name[:42]:<42} {item['length_px']:>10.2f} px")
        if not args.no_save:
            cv2.imwrite(str(out_dir / f"{path.stem}_length.jpg"), annotate(original, measurements))

    lengths = [item["length_px"] for row in all_results for item in row["instances"]]
    print(f"Detections: {len(lengths)}")
    if lengths:
        print(f"Length px : min={min(lengths):.2f}, mean={np.mean(lengths):.2f}, max={max(lengths):.2f}")

    if not args.no_save:
        summary_path = out_dir / "measurements.json"
        summary_path.write_text(json.dumps({
            "weights": str(weights),
            "source": str(args.source),
            "classes": args.classes,
            "measurement": "maximum Euclidean distance between foreground mask pixels",
            "results": all_results,
        }, indent=2))
        print(f"Saved: {out_dir}\nSummary: {summary_path}")


if __name__ == "__main__":
    started_at = perf_counter()
    try:
        main()
    finally:
        print(f"Total runtime: {perf_counter() - started_at:.2f} s")
