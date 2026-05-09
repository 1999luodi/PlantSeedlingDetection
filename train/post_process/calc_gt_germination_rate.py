
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
GERMINATED = "germinated"
UNGERMINATED = "ungerminated"


def parse_one_label_json(json_path: Path) -> Tuple[int, int]:
    """Read one X-AnyLabeling json and return (germinated_count, ungerminated_count)."""
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    shapes = data.get("shapes", [])
    g_count = 0
    u_count = 0
    for shape in shapes:
        label = str(shape.get("label", "")).strip().lower()
        if label == GERMINATED:
            g_count += 1
        elif label == UNGERMINATED:
            u_count += 1
    return g_count, u_count


def safe_rate(germinated_count: int, ungerminated_count: int) -> float:
    total = germinated_count + ungerminated_count
    if total == 0:
        return 0.0
    return germinated_count / total


def collect_json_files(input_path: Path, recursive: bool) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    pattern = "**/*.json" if recursive else "*.json"
    return sorted(input_path.glob(pattern))


def infer_image_name(json_path: Path) -> str:
    base = json_path.stem
    for suffix in VALID_SUFFIXES:
        candidate = json_path.with_suffix(suffix)
        if candidate.exists():
            return candidate.name
    # Keep a deterministic placeholder if image file is not found nearby.
    return f"{base}.jpg"


def calc_gt_rates(json_files: Iterable[Path]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for json_file in json_files:
        g_count, u_count = parse_one_label_json(json_file)
        rows.append(
            {
                "json_file": str(json_file),
                "image_name": infer_image_name(json_file),
                "germinated": g_count,
                "ungerminated": u_count,
                "total": g_count + u_count,
                "gt_rate": safe_rate(g_count, u_count),
            }
        )
    return rows


def write_csv(rows: List[Dict[str, object]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["json_file", "image_name", "germinated", "ungerminated", "total", "gt_rate"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate GT germination rate from labeling json(s).")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Json file or folder containing X-AnyLabeling json files.",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively search json files in input folder.")
    parser.add_argument(
        "--out_csv",
        type=str,
        default="train/post_process/outputs/gt_germination_rates.csv",
        help="Output csv path.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    json_files = collect_json_files(input_path, recursive=args.recursive)
    if not json_files:
        raise FileNotFoundError(f"No json files found under: {input_path}")

    rows = calc_gt_rates(json_files)
    write_csv(rows, Path(args.out_csv))

    rates = [float(r["gt_rate"]) for r in rows]
    mean_rate = sum(rates) / len(rates) if rates else 0.0
    print(f"Processed {len(rows)} samples.")
    print(f"Mean GT germination rate: {mean_rate:.4f}")
    print(f"Saved: {args.out_csv}")


if __name__ == "__main__":
    main()
