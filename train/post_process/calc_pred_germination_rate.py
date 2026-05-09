import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from mmengine import Config
from mmdet.apis import inference_detector, init_detector

VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
GERMINATED = "germinated"
UNGERMINATED = "ungerminated"


def safe_rate(germinated_count: int, ungerminated_count: int) -> float:
    total = germinated_count + ungerminated_count
    if total == 0:
        return 0.0
    return germinated_count / total


def collect_images(input_path: Path, recursive: bool) -> List[Path]:
    if input_path.is_file():
        return [input_path]

    pattern = "**/*" if recursive else "*"
    images: List[Path] = []
    for p in sorted(input_path.glob(pattern)):
        if p.is_file() and p.suffix.lower() in VALID_SUFFIXES:
            images.append(p)
    return images


def build_model(config_path: Path, checkpoint_path: Optional[Path], device: str):
    cfg = Config.fromfile(str(config_path))
    ckpt = str(checkpoint_path) if checkpoint_path else cfg.get("load_from", None)
    if not ckpt:
        raise ValueError(
            "No checkpoint provided. Set --checkpoint or add load_from in config."
        )
    return init_detector(cfg, ckpt, device=device)


def get_class_names(model) -> Tuple[str, ...]:
    classes = model.dataset_meta.get("classes", None)
    if classes is None:
        raise ValueError("Cannot find classes from model.dataset_meta['classes']")
    return tuple(str(c) for c in classes)


def count_by_label(model, image_path: Path, score_thr: float) -> Tuple[int, int]:
    result = inference_detector(model, str(image_path))
    pred = result.pred_instances

    labels = pred.labels.cpu().numpy().tolist()
    scores = pred.scores.cpu().numpy().tolist()

    class_names = get_class_names(model)
    g_count = 0
    u_count = 0

    for label_id, score in zip(labels, scores):
        if float(score) < score_thr:
            continue
        if int(label_id) < 0 or int(label_id) >= len(class_names):
            continue
        label_name = class_names[int(label_id)].strip().lower()
        if label_name == GERMINATED:
            g_count += 1
        elif label_name == UNGERMINATED:
            u_count += 1

    return g_count, u_count


def calc_pred_rates(model, images: Iterable[Path], score_thr: float) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for image_path in images:
        g_count, u_count = count_by_label(model, image_path, score_thr=score_thr)
        rows.append(
            {
                "image_path": str(image_path),
                "image_name": image_path.name,
                "pred_germinated": g_count,
                "pred_ungerminated": u_count,
                "pred_total": g_count + u_count,
                "pred_rate": safe_rate(g_count, u_count),
            }
        )
    return rows


def write_csv(rows: List[Dict[str, object]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_path",
        "image_name",
        "pred_germinated",
        "pred_ungerminated",
        "pred_total",
        "pred_rate",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mmdetection model and calculate predicted germination rates.")
    parser.add_argument("--config", type=str, required=True, help="Path to mmdetection config file.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (.pth).")
    parser.add_argument("--input", type=str, required=True, help="Image file or folder.")
    parser.add_argument("--recursive", action="store_true", help="Recursively search images in folder.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Inference device, e.g. cuda:0 or cpu.")
    parser.add_argument("--score_thr", type=float, default=0.5, help="Score threshold for counting detections.")
    parser.add_argument(
        "--out_csv",
        type=str,
        default="train/post_process/outputs/pred_germination_rates.csv",
        help="Output csv path.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    if checkpoint_path and not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    images = collect_images(input_path, recursive=args.recursive)
    if not images:
        raise FileNotFoundError(f"No images found under: {input_path}")

    model = build_model(config_path, checkpoint_path, device=args.device)
    rows = calc_pred_rates(model, images, score_thr=float(args.score_thr))
    write_csv(rows, Path(args.out_csv))

    rates = [float(r["pred_rate"]) for r in rows]
    mean_rate = sum(rates) / len(rates) if rates else 0.0
    print(f"Processed {len(rows)} images.")
    print(f"Mean predicted germination rate: {mean_rate:.4f}")
    print(f"Saved: {args.out_csv}")


if __name__ == "__main__":
    main()
