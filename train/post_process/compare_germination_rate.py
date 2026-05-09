import argparse
import csv
import json
from pathlib import Path
from statistics import mean, variance
from typing import Dict, List, Union

import matplotlib.pyplot as plt


def load_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def compare_rows(gt_rows: List[Dict[str, str]], pred_rows: List[Dict[str, str]]) -> List[Dict[str, Union[float, str]]]:
    gt_map = {row["image_name"]: row for row in gt_rows}
    pred_map = {row["image_name"]: row for row in pred_rows}

    common_names = sorted(set(gt_map.keys()) & set(pred_map.keys()))
    compared: List[Dict[str, Union[float, str]]] = []
    for name in common_names:
        gt_rate = float(gt_map[name]["gt_rate"])
        pred_rate = float(pred_map[name]["pred_rate"])
        err = pred_rate - gt_rate
        compared.append(
            {
                "image_name": name,
                "gt_rate": gt_rate,
                "pred_rate": pred_rate,
                "error": err,
                "abs_error": abs(err),
                "sq_error": err * err,
            }
        )
    return compared


def summarize(compared: List[Dict[str, Union[float, str]]]) -> Dict[str, Union[float, int]]:
    if not compared:
        return {
            "count": 0,
            "mean_error": 0.0,
            "mean_abs_error": 0.0,
            "var_error": 0.0,
            "rmse": 0.0,
        }

    errors = [float(x["error"]) for x in compared]
    abs_errors = [float(x["abs_error"]) for x in compared]
    sq_errors = [float(x["sq_error"]) for x in compared]

    var_err = variance(errors) if len(errors) > 1 else 0.0
    rmse = (sum(sq_errors) / len(sq_errors)) ** 0.5

    return {
        "count": len(compared),
        "mean_error": mean(errors),
        "mean_abs_error": mean(abs_errors),
        "var_error": var_err,
        "rmse": rmse,
    }


def write_compared_csv(compared: List[Dict[str, Union[float, str]]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_name", "gt_rate", "pred_rate", "error", "abs_error", "sq_error"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(compared)


def write_summary(summary: Dict[str, Union[float, int]], out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def draw_plot(compared: List[Dict[str, Union[float, str]]], summary: Dict[str, Union[float, int]], out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    gt_rates = [float(x["gt_rate"]) for x in compared]
    pred_rates = [float(x["pred_rate"]) for x in compared]
    errors = [float(x["error"]) for x in compared]

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(gt_rates, label="GT rate", marker="o", linewidth=1)
    plt.plot(pred_rates, label="Pred rate", marker="x", linewidth=1)
    plt.ylim(0, 1)
    plt.xlabel("Sample index")
    plt.ylabel("Germination rate")
    plt.title("GT vs Pred")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.hist(errors, bins=15)
    plt.xlabel("Error (pred_rate - gt_rate)")
    plt.ylabel("Count")
    plt.title(
        "Error distribution\n"
        f"mean={float(summary['mean_error']):.4f}, "
        f"var={float(summary['var_error']):.6f}, "
        f"MAE={float(summary['mean_abs_error']):.4f}"
    )
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare GT and prediction germination rates, then visualize error.")
    parser.add_argument(
        "--gt_csv",
        type=str,
        required=True,
        help="Output csv from calc_gt_germination_rate.py",
    )
    parser.add_argument(
        "--pred_csv",
        type=str,
        required=True,
        help="Output csv from calc_pred_germination_rate.py",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="train/post_process/outputs",
        help="Directory to save compare outputs.",
    )
    args = parser.parse_args()

    gt_csv = Path(args.gt_csv)
    pred_csv = Path(args.pred_csv)
    if not gt_csv.exists():
        raise FileNotFoundError(f"GT csv not found: {gt_csv}")
    if not pred_csv.exists():
        raise FileNotFoundError(f"Pred csv not found: {pred_csv}")

    out_dir = Path(args.out_dir)
    compared_csv = out_dir / "germination_rate_compare.csv"
    summary_json = out_dir / "germination_rate_summary.json"
    fig_png = out_dir / "germination_rate_error_plot.png"

    gt_rows = load_csv_rows(gt_csv)
    pred_rows = load_csv_rows(pred_csv)
    compared = compare_rows(gt_rows, pred_rows)
    if not compared:
        raise ValueError("No overlapping image_name between gt_csv and pred_csv.")

    summary = summarize(compared)
    write_compared_csv(compared, compared_csv)
    write_summary(summary, summary_json)
    draw_plot(compared, summary, fig_png)

    print(f"Compared {summary['count']} images.")
    print(
        f"mean_error={float(summary['mean_error']):.4f}, "
        f"var_error={float(summary['var_error']):.6f}, "
        f"MAE={float(summary['mean_abs_error']):.4f}, "
        f"RMSE={float(summary['rmse']):.4f}"
    )
    print(f"Saved: {compared_csv}")
    print(f"Saved: {summary_json}")
    print(f"Saved: {fig_png}")


if __name__ == "__main__":
    main()
