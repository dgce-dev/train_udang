#!/usr/bin/env python3
"""Combine shrimp and udang2 YOLO datasets, then fine-tune an existing model.

The class order is taken from shrimp_dataset/data.yaml. udang2 classes with an
identical name are mapped automatically. The `uudang` class is ignored (its
annotations are omitted); map other unmatched classes explicitly if needed.

By default this uses udang2_dataset_augmented when present, otherwise
udang2_dataset. The combined dataset is written to a new directory.
"""

import argparse
import ast
import random
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "runs/combined_finetune/udang2_20261117_200808/weights/best.pt"
DEFAULT_UDANG2 = (
    ROOT / "udang2_dataset_augmented"
    if (ROOT / "udang2_dataset_augmented").is_dir()
    else ROOT / "udang2_dataset"
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_names(yaml_path: Path) -> list[str]:
    """Read the names list/dict from a YOLO data.yaml."""
    for line in yaml_path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition(":")
        if not separator or key.strip() != "names":
            continue
        try:
            names = ast.literal_eval(value.strip())
            if isinstance(names, dict):
                return [str(names[index]) for index in sorted(names, key=int)]
            if isinstance(names, list):
                return [str(name) for name in names]
        except Exception:
            pass
    import yaml
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = data.get("names", [])
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names, key=int)]
    return [str(name) for name in names]


def parse_class_map(
    specifications: list[str],
    source_names: list[str],
    target_names: list[str],
    ignored_names: set[str] | None = None,
) -> dict[int, int | None]:
    """Map source classes by exact name, explicitly ignoring selected classes."""
    requested: dict[str, str | None] = {
        name: name for name in source_names if name in target_names
    }
    for name in (ignored_names or set()):
        if name in source_names and name not in requested:
            requested[name] = None
    for spec in specifications:
        if "=" not in spec:
            raise ValueError(f"Invalid class mapping {spec!r}; expected SOURCE=TARGET")
        source_name, target_name = (part.strip() for part in spec.split("=", 1))
        if source_name not in source_names:
            raise ValueError(f"Unknown source class {source_name!r}; available: {source_names}")
        if target_name not in target_names:
            raise ValueError(f"Unknown target class {target_name!r}; available: {target_names}")
        requested[source_name] = target_name

    missing = [name for name in source_names if name not in requested]
    if missing:
        raise ValueError(
            "No class mapping for udang2 class(es): " + ", ".join(missing) +
            ". Supply --udang2-map SOURCE=TARGET after confirming the semantic class."
        )
    return {
        i: (target_names.index(requested[name]) if requested[name] is not None else None)
        for i, name in enumerate(source_names)
    }


def merge_split(
    source_root: Path,
    destination_root: Path,
    split: str,
    source_prefix: str,
    class_map: dict[int, int],
) -> int:
    source_split = source_root / split
    if not source_split.is_dir():
        return 0
    image_dir = source_split / "images"
    label_dir = source_split / "labels"
    images = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    for image_path in images:
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing label file for {image_path}: {label_path}")
        out_stem = f"{source_prefix}__{image_path.stem}"
        shutil.copy2(image_path, destination_root / split / "images" / f"{out_stem}{image_path.suffix.lower()}")

        remapped_rows = []
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            fields = line.split()
            if not fields:
                continue
            try:
                source_id = int(fields[0])
                target_id = class_map[source_id]
            except (ValueError, KeyError) as exc:
                raise ValueError(
                    f"Invalid/unmapped class id in {label_path}:{line_number}: {fields[0]!r}"
                ) from exc
            if target_id is None:
                continue
            fields[0] = str(target_id)
            remapped_rows.append(" ".join(fields))
        out_label = destination_root / split / "labels" / f"{out_stem}.txt"
        out_label.write_text("\n".join(remapped_rows) + ("\n" if remapped_rows else ""), encoding="utf-8")
    return len(images)


def resolve_device(requested: str) -> str:
    value = requested.strip().lower()
    if value in {"auto", "cuda", "gpu"}:
        import torch
        if torch.cuda.is_available():
            return "0"
        if value == "auto":
            print("CUDA unavailable; using CPU.")
            return "cpu"
        raise SystemExit("CUDA requested, but no usable GPU was found.")
    return requested


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shrimp", type=Path, default=ROOT / "shrimp_dataset", help="Original dataset")
    parser.add_argument("--udang2", type=Path, default=DEFAULT_UDANG2, help="Augmented udang2 dataset, or its original")
    parser.add_argument("--output-dataset", type=Path, default=ROOT / "combined_shrimp_udang2", help="New merged dataset directory")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Existing trained Ultralytics .pt checkpoint")
    parser.add_argument(
        "--udang2-map", action="append", default=[], metavar="SOURCE=TARGET",
        help="Map a udang2 class to a shrimp class; repeat for each unmapped class",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto", help="auto, cpu, or GPU index such as 0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial LR; lower default suits fine-tuning")
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "combined_finetune")
    parser.add_argument("--name", default=datetime.now().strftime("udang2_%Y%m%d_%H%M%S"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    shrimp = args.shrimp.expanduser().resolve()
    udang2 = args.udang2.expanduser().resolve()
    output = args.output_dataset.expanduser().resolve()
    weights = args.weights.expanduser().resolve()
    output_ready = (output / "data.yaml").is_file()

    if not output_ready:
        for dataset in (shrimp, udang2):
            if not (dataset / "data.yaml").is_file():
                parser.error(f"Dataset YAML not found: {dataset / 'data.yaml'}")
        if (
            output == shrimp or output == udang2
            or output in shrimp.parents or output in udang2.parents
            or shrimp in output.parents or udang2 in output.parents
        ):
            parser.error("Output dataset must be separate from both input datasets")

    if output.exists() and not output.is_dir():
        parser.error(f"Output path exists and is not a directory: {output}")

    # ponytail: fallback ke yolov8n-seg.pt bila checkpoint lokal belum ada
    if not weights.is_file():
        if args.weights == DEFAULT_WEIGHTS:
            weights = Path("yolov8n-seg.pt")
        else:
            parser.error(f"Model checkpoint not found: {weights}; pass --weights /path/to/best.pt")

    if output_ready:
        target_names = read_names(output / "data.yaml")
    else:
        target_names = read_names(shrimp / "data.yaml")
        shrimp_names = target_names
        udang2_names = read_names(udang2 / "data.yaml")
        shrimp_map = parse_class_map([], shrimp_names, target_names)
        try:
            udang2_map = parse_class_map(
                args.udang2_map, udang2_names, target_names, ignored_names={"uudang"}
            )
        except ValueError as exc:
            parser.error(str(exc))

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install Ultralytics first: python -m pip install ultralytics") from exc

    model = YOLO(str(weights))
    model_names = model.names
    if isinstance(model_names, dict):
        model_names = [model_names[index] for index in sorted(model_names)]
    model_names = [str(name) for name in model_names]
    if len(model_names) == len(target_names) and model_names != target_names:
        raise SystemExit(
            f"Checkpoint classes {model_names} do not match target dataset classes {target_names}. "
            "Use a checkpoint trained with shrimp_dataset's class order."
        )
    if model.task != "segment":
        raise SystemExit(
            f"This dataset has segmentation polygons but checkpoint task is {model.task!r}; "
            "use a YOLO segmentation checkpoint."
        )

    data_yaml = output / "data.yaml"
    if output.exists():
        if not data_yaml.is_file():
            parser.error(f"Existing output directory has no data.yaml: {output}")
        if read_names(data_yaml) != target_names:
            parser.error(
                f"Existing dataset classes in {data_yaml} do not match {target_names}; "
                "move/delete the old output or choose a new --output-dataset path."
            )
        for split in ("train", "valid", "test"):
            if not (output / split / "images").is_dir():
                parser.error(f"Existing combined dataset is missing {split}/images: {output}")
        counts = {
            split: sum(
                1 for path in (output / split / "images").iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            for split in ("train", "valid", "test")
        }
        print(f"Reusing existing combined dataset without rebuilding: {output}")
    else:
        output.mkdir(parents=True)
        try:
            for split in ("train", "valid", "test"):
                (output / split / "images").mkdir(parents=True)
                (output / split / "labels").mkdir(parents=True)
            counts = {}
            for split in ("train", "valid", "test"):
                counts[split] = merge_split(shrimp, output, split, "shrimp", shrimp_map)
                counts[split] += merge_split(udang2, output, split, "udang2", udang2_map)

            yaml_lines = [
                f"path: {output.as_posix()}",
                "train: train/images",
                "val: valid/images",
                "test: test/images",
                f"nc: {len(target_names)}",
                f"names: {target_names!r}",
                "",
            ]
            data_yaml.write_text("\n".join(yaml_lines), encoding="utf-8")
        except Exception:
            shutil.rmtree(output, ignore_errors=True)
            raise

    print("Ignoring udang2 class: uudang (its annotations are omitted when building)")
    print(f"Combined dataset: {data_yaml}")
    print("Images by split: " + ", ".join(f"{split}={counts[split]}" for split in counts))
    device = resolve_device(args.device)
    run_project = args.project.expanduser().resolve()
    random.seed(args.seed)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        workers=args.workers,
        patience=args.patience,
        lr0=args.lr0,
        seed=args.seed,
        project=str(run_project),
        name=args.name,
        exist_ok=False,
    )
    print(f"Fine-tuning complete. Checkpoint: {run_project / args.name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
