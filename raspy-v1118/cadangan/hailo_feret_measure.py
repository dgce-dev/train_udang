"""Run a Hailo segmentation HEF and measure shrimp with maximum Feret diameter.

This is the HailoRT version of ``measure_test/feret/measure_test.py``. The
measurement is intentionally the same: rasterize each predicted mask, compute
its convex hull, and use the longest Euclidean distance between hull points.

The script supports either postprocessed detection/mask streams or raw
Ultralytics YOLO11-seg heads. Raw heads are identified from their shapes and
are decoded with DFL, class-aware NMS, mask coefficients, and prototypes.

A postprocessed detection stream must contain rows in this form::

    [x1, y1, x2, y2, confidence, class_id]

Coordinates may be normalized to 0..1 or expressed in network-input pixels.
Its mask stream must contain one mask per detection, as ``[N,H,W]`` or
``[1,N,H,W]``. Use ``--list-outputs`` to inspect the stream names and shapes.

Example:
    .hailo-venv/bin/python hailo_feret_measure.py \
        --hef path/to/shrimp_segmentation.hef \
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


ROOT = Path(__file__).resolve().parent
DEFAULT_HEF = ROOT / "yolov11_udang.hef"
DEFAULT_SOURCE = ROOT / "shrimp_dataset" / "test" / "images"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def default_project():
    return ROOT / "runs" / "hailo_feret" / datetime.now().strftime("%Y-%m-%d_%H%M%S")


def parse_args():
    parser = argparse.ArgumentParser(description="Hailo segmentation inference and Feret measurement.")
    parser.add_argument("--hef", default=str(DEFAULT_HEF), help="Segmentation HEF file")
    parser.add_argument(
        "--interface",
        choices=("pcie", "ethernet"),
        default="pcie",
        help="Hailo device interface (default: pcie; Raspberry Pi HAT/M.2 uses PCIe)",
    )
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Image, directory, or glob")
    parser.add_argument("--imgsz", type=int, default=640, help="Square network input size")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold for raw YOLO outputs")
    parser.add_argument("--mask-threshold", type=float, default=0.5, help="Mask threshold")
    parser.add_argument("--classes", type=int, nargs="+", default=[3], help="Class IDs to measure")
    parser.add_argument("--boxes-output", default=None, help="Detection output stream name")
    parser.add_argument("--masks-output", default=None, help="Mask output stream name")
    parser.add_argument("--list-outputs", action="store_true", help="Print HEF streams and exit")
    parser.add_argument("--project", default=str(default_project()), help="Output directory")
    parser.add_argument("--no-save", action="store_true", help="Do not save images or JSON")
    return parser.parse_args()


def expand_source(source):
    path = Path(source)
    if any(char in str(path) for char in "*?["):
        files = sorted(path.parent.glob(path.name))
    elif path.is_dir():
        files = sorted(path.iterdir())
    elif path.is_file():
        files = [path]
    elif not path.is_absolute():
        return expand_source(ROOT / path)
    else:
        files = []
    return [p for p in files if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def longest_mask_points(mask):
    """Return maximum-Feret length and endpoints for a binary mask."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0.0, None, None
    points = np.column_stack((xs, ys)).astype(np.int32)
    hull = cv2.convexHull(points.reshape(-1, 1, 2)).reshape(-1, 2)
    if len(hull) == 1:
        point = hull[0].tolist()
        return 0.0, point, point
    deltas = hull[:, None, :].astype(np.float64) - hull[None, :, :].astype(np.float64)
    distances_sq = np.einsum("ijk,ijk->ij", deltas, deltas)
    first, second = np.unravel_index(np.argmax(distances_sq), distances_sq.shape)
    return float(np.sqrt(distances_sq[first, second])), hull[first].tolist(), hull[second].tolist()


def letterbox(image, size):
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    resized = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_LINEAR)
    pad_x = (size - resized.shape[1]) // 2
    pad_y = (size - resized.shape[0]) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + resized.shape[0], pad_x:pad_x + resized.shape[1]] = resized
    return canvas, scale, pad_x, pad_y


def restore_mask(mask, original_shape, size, scale, pad_x, pad_y):
    """Undo letterboxing and resize a network mask to the source image."""
    mask = cv2.resize(mask.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)
    height, width = original_shape[:2]
    resized_w, resized_h = round(width * scale), round(height * scale)
    cropped = mask[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w]
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)


def restore_box(box, original_shape, size, scale, pad_x, pad_y):
    """Undo letterboxing for xyxy coordinates."""
    x1, y1, x2, y2 = map(float, box[:4])
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
        x1, x2, y1, y2 = x1 * size, x2 * size, y1 * size, y2 * size
    width, height = original_shape[1], original_shape[0]
    values = [(x1 - pad_x) / scale, (y1 - pad_y) / scale, (x2 - pad_x) / scale, (y2 - pad_y) / scale]
    return [max(0, min(width - 1, round(values[0]))), max(0, min(height - 1, round(values[1]))),
            max(0, min(width - 1, round(values[2]))), max(0, min(height - 1, round(values[3])))]


def annotate(image, measurements):
    output = image.copy()
    for index, item in enumerate(measurements, start=1):
        a, b = item["endpoint_a_xy"], item["endpoint_b_xy"]
        if a is None or b is None:
            continue
        cv2.line(output, tuple(a), tuple(b), (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(output, tuple(a), 5, (255, 255, 0), -1)
        cv2.circle(output, tuple(b), 5, (255, 255, 0), -1)
        label = f"#{index}: {item['length_px']:.1f} px"
        cv2.putText(output, label, (min(a[0], b[0]), max(20, min(a[1], b[1]) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
    return output


class HailoSegmenter:
    def __init__(self, hef_path, imgsz, boxes_name=None, masks_name=None, interface="pcie"):
        try:
            from hailo_platform import (
                ConfigureParams, FormatType, HEF, InferVStreams,
                HailoStreamInterface, InputVStreamParams, OutputVStreamParams, VDevice,
            )
        except ImportError as exc:
            raise SystemExit("hailo_platform is unavailable; run this with HailoRT/Python on the Hailo device.") from exc

        self.imgsz = imgsz
        self.HEF = HEF
        self.VDevice = VDevice
        self.InferVStreams = InferVStreams
        self.hef = HEF(str(hef_path))
        self.input_infos = self.hef.get_input_vstream_infos()
        self.output_infos = self.hef.get_output_vstream_infos()
        self.input_name = self.input_infos[0].name
        self.boxes_name = boxes_name
        self.masks_name = masks_name
        self.FormatType = FormatType
        self.ConfigureParams = ConfigureParams
        self.InputVStreamParams = InputVStreamParams
        self.OutputVStreamParams = OutputVStreamParams
        self.HailoStreamInterface = HailoStreamInterface
        self.interface = getattr(HailoStreamInterface, "PCIe" if interface == "pcie" else "Ethernet")

    def describe(self):
        print("Inputs:")
        for info in self.input_infos:
            print(f"  {info.name}: shape={info.shape}, format={info.format.type}")
        print("Outputs:")
        for info in self.output_infos:
            print(f"  {info.name}: shape={info.shape}, format={info.format.type}")

    def __enter__(self):
        self.device = self.VDevice()
        config = self.ConfigureParams.create_from_hef(
            self.hef, self.interface
        )
        self.network_group = self.device.configure(self.hef, config)[0]
        input_params = self.InputVStreamParams.make_from_network_group(
            self.network_group, quantized=False, format_type=self.FormatType.FLOAT32
        )
        output_params = self.OutputVStreamParams.make_from_network_group(
            self.network_group, quantized=False, format_type=self.FormatType.FLOAT32
        )
        self.pipeline = self.InferVStreams(self.network_group, input_params, output_params)
        self.pipeline.__enter__()
        self.activation = self.network_group.activate(self.network_group.create_params())
        self.activation.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.activation.__exit__(exc_type, exc, tb)
        self.pipeline.__exit__(exc_type, exc, tb)
        self.device.__exit__(exc_type, exc, tb)

    def infer(self, image):
        boxed, scale, pad_x, pad_y = letterbox(image, self.imgsz)
        # The HEF input is UINT8. A non-quantized FLOAT32 vstream still expects
        # values in the tensor's native 0..255 range and quantizes them for us.
        rgb = cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB).astype(np.float32)
        outputs = self.pipeline.infer({self.input_name: np.expand_dims(rgb, axis=0)})
        return outputs, (scale, pad_x, pad_y)


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def nms(boxes, scores, class_ids, iou_threshold):
    """Return indices kept by class-aware non-maximum suppression."""
    kept = []
    for class_id in np.unique(class_ids):
        indices = np.flatnonzero(class_ids == class_id)
        order = indices[np.argsort(scores[indices])[::-1]]
        while len(order):
            current = order[0]
            kept.append(current)
            if len(order) == 1:
                break
            others = order[1:]
            left_top = np.maximum(boxes[current, :2], boxes[others, :2])
            right_bottom = np.minimum(boxes[current, 2:], boxes[others, 2:])
            wh = np.maximum(right_bottom - left_top, 0.0)
            intersection = wh[:, 0] * wh[:, 1]
            area_current = np.prod(np.maximum(boxes[current, 2:] - boxes[current, :2], 0.0))
            area_others = np.prod(np.maximum(boxes[others, 2:] - boxes[others, :2], 0.0), axis=1)
            union = area_current + area_others - intersection
            iou = intersection / np.maximum(union, 1e-7)
            order = others[iou <= iou_threshold]
    return np.asarray(kept, dtype=np.int64)


def decode_raw_yolo_seg(outputs, imgsz, conf_threshold, iou_threshold, wanted_classes):
    """Decode Ultralytics YOLO11-seg DFL/class/mask-coefficient heads."""
    arrays = {}
    for name, value in outputs.items():
        array = np.asarray(value)
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]
        if array.ndim == 3:
            arrays[name] = array

    # The prototype and coefficient tensors share nm channels. The prototype is
    # the tensor with that channel count at the largest spatial resolution.
    channel_counts = {}
    for array in arrays.values():
        channel_counts[array.shape[-1]] = channel_counts.get(array.shape[-1], 0) + 1
    repeated_channels = {channels for channels, count in channel_counts.items() if count >= 4}
    proto_candidates = [
        (name, array) for name, array in arrays.items()
        if array.shape[-1] in repeated_channels
    ]
    if not proto_candidates:
        return None
    proto_name, proto = max(proto_candidates, key=lambda item: item[1].shape[0] * item[1].shape[1])
    mask_channels = proto.shape[-1]

    grouped = {}
    for name, array in arrays.items():
        if name != proto_name:
            grouped.setdefault(array.shape[:2], []).append((name, array))

    heads = []
    for (height, width), tensors in grouped.items():
        coeffs = [(name, array) for name, array in tensors if array.shape[-1] == mask_channels]
        regressions = [
            (name, array) for name, array in tensors
            if array.shape[-1] % 4 == 0 and array.shape[-1] >= 16 and array.shape[-1] > mask_channels
        ]
        if len(coeffs) != 1 or len(regressions) != 1:
            continue
        used = {coeffs[0][0], regressions[0][0]}
        classes = [(name, array) for name, array in tensors if name not in used]
        if len(classes) == 1:
            heads.append((height, width, regressions[0][1], classes[0][1], coeffs[0][1]))
    if not heads:
        return None

    all_boxes, all_scores, all_classes, all_coefficients = [], [], [], []
    wanted_classes = np.asarray(wanted_classes, dtype=np.int64)
    for height, width, regression, class_output, coefficients in heads:
        bins = regression.shape[-1] // 4
        logits = regression.reshape(-1, 4, bins).astype(np.float32)
        logits -= logits.max(axis=2, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=2, keepdims=True)
        distances = probabilities @ np.arange(bins, dtype=np.float32)

        class_logits = class_output.reshape(-1, class_output.shape[-1]).astype(np.float32)
        class_scores = sigmoid(class_logits)
        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(len(class_ids)), class_ids]
        selected = (scores >= conf_threshold) & np.isin(class_ids, wanted_classes)
        if not np.any(selected):
            continue

        grid_y, grid_x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        centers = np.column_stack((grid_x.ravel() + 0.5, grid_y.ravel() + 0.5)).astype(np.float32)
        stride_x, stride_y = imgsz / width, imgsz / height
        boxes = np.column_stack((
            (centers[:, 0] - distances[:, 0]) * stride_x,
            (centers[:, 1] - distances[:, 1]) * stride_y,
            (centers[:, 0] + distances[:, 2]) * stride_x,
            (centers[:, 1] + distances[:, 3]) * stride_y,
        ))
        all_boxes.append(boxes[selected])
        all_scores.append(scores[selected])
        all_classes.append(class_ids[selected])
        all_coefficients.append(coefficients.reshape(-1, mask_channels)[selected])

    if not all_boxes:
        return np.empty((0, 6), dtype=np.float32), np.empty((0, proto.shape[0], proto.shape[1]), dtype=np.float32)

    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    class_ids = np.concatenate(all_classes)
    coefficients = np.concatenate(all_coefficients)
    keep = nms(boxes, scores, class_ids, iou_threshold)
    boxes, scores, class_ids, coefficients = boxes[keep], scores[keep], class_ids[keep], coefficients[keep]

    masks = sigmoid(coefficients @ proto.reshape(-1, mask_channels).T)
    masks = masks.reshape(-1, proto.shape[0], proto.shape[1])
    scale_x, scale_y = proto.shape[1] / imgsz, proto.shape[0] / imgsz
    for index, box in enumerate(boxes):
        x1 = max(0, min(proto.shape[1], int(np.floor(box[0] * scale_x))))
        y1 = max(0, min(proto.shape[0], int(np.floor(box[1] * scale_y))))
        x2 = max(0, min(proto.shape[1], int(np.ceil(box[2] * scale_x))))
        y2 = max(0, min(proto.shape[0], int(np.ceil(box[3] * scale_y))))
        cropped = np.zeros_like(masks[index])
        cropped[y1:y2, x1:x2] = masks[index, y1:y2, x1:x2]
        masks[index] = cropped

    decoded_boxes = np.column_stack((boxes, scores, class_ids.astype(np.float32)))
    return decoded_boxes, masks


def choose_outputs(outputs, boxes_name, masks_name, imgsz, conf_threshold, iou_threshold, wanted_classes):
    if boxes_name and boxes_name not in outputs:
        raise SystemExit(f"Detection output {boxes_name!r} not found. Available: {list(outputs)}")
    if masks_name and masks_name not in outputs:
        raise SystemExit(f"Mask output {masks_name!r} not found. Available: {list(outputs)}")
    arrays = [(name, np.asarray(value).squeeze(0)) for name, value in outputs.items()]
    boxes = outputs[boxes_name] if boxes_name else None
    masks = outputs[masks_name] if masks_name else None
    if boxes is None:
        candidates = [(name, array) for name, array in arrays if array.ndim == 2 and array.shape[-1] >= 6]
        if len(candidates) == 1:
            boxes = outputs[candidates[0][0]]
    if masks is None:
        candidates = [
            (name, array) for name, array in arrays
            if array.ndim == 3 and array.shape[-1] > 8 and array.shape[-2] > 8
        ]
        if len(candidates) == 1:
            masks = outputs[candidates[0][0]]
    if boxes is not None and masks is not None:
        return np.asarray(boxes).squeeze(0), np.asarray(masks).squeeze(0)

    decoded = decode_raw_yolo_seg(outputs, imgsz, conf_threshold, iou_threshold, wanted_classes)
    if decoded is not None:
        return decoded
    raise SystemExit(
        "Could not identify either postprocessed outputs or raw YOLO11-seg heads. "
        "Run --list-outputs to inspect the HEF streams."
    )


def main():
    args = parse_args()
    hef = Path(args.hef)
    if not hef.is_absolute() and not hef.exists():
        hef = ROOT / hef
    if not hef.exists():
        raise SystemExit(f"HEF not found: {hef}")
    images = expand_source(args.source)
    if not images:
        raise SystemExit(f"No images found for --source {args.source!r}")

    segmenter = HailoSegmenter(
        hef, args.imgsz, args.boxes_output, args.masks_output, args.interface
    )
    if args.list_outputs:
        segmenter.describe()
        return
    out_dir = Path(args.project).resolve()
    if not args.no_save:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"HEF   : {hef}\nImages: {len(images)}\nClasses: {args.classes}")
    all_results = []
    with segmenter:
        for image_index, path in enumerate(images, start=1):
            original = cv2.imread(str(path))
            if original is None:
                continue
            outputs, letterbox_state = segmenter.infer(original)
            boxes, masks = choose_outputs(
                outputs, args.boxes_output, args.masks_output, args.imgsz,
                args.conf, args.iou, args.classes,
            )
            scale, pad_x, pad_y = letterbox_state
            measurements = []
            for index, row in enumerate(np.atleast_2d(boxes)):
                if len(row) < 6:
                    continue
                confidence, class_id = float(row[4]), int(row[5])
                if confidence < args.conf or class_id not in args.classes or index >= len(masks):
                    continue
                mask = restore_mask(masks[index], original.shape, args.imgsz, scale, pad_x, pad_y) >= args.mask_threshold
                length, endpoint_a, endpoint_b = longest_mask_points(mask)
                measurements.append({
                    "instance": index + 1, "class_id": class_id, "confidence": round(confidence, 4),
                    "length_px": round(length, 2), "endpoint_a_xy": endpoint_a,
                    "endpoint_b_xy": endpoint_b, "mask_area_px": int(mask.sum()),
                    "box_xyxy": restore_box(row, original.shape, args.imgsz, scale, pad_x, pad_y),
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
            "hef": str(hef), "source": str(args.source), "classes": args.classes,
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
