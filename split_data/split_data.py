from __future__ import annotations

import argparse
import math
import random
import shutil
from pathlib import Path


def split_dataset(
    data_dir: Path,
    output_dir: Path,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    labeled_ratio: float = 0.1,
    seed: int = 42,
    overwrite: bool = False,
) -> None:
    """
    按规格划分数据集：
    - 每个类别独立执行 train/val/test 切分（ceil 保底，每集合至少 1 张）
    - train 内再按 labeled_ratio 切分 labeled / unlabeled
    - 所有输出目录均平铺存放，无类别子文件夹
    """
    random.seed(seed)

    train_labeled_dir = output_dir / "train" / "labeled"
    train_unlabeled_dir = output_dir / "train" / "unlabeled"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"
    ann_train_dir = output_dir / "annotations" / "train"
    ann_val_dir = output_dir / "annotations" / "val"
    ann_test_dir = output_dir / "annotations" / "test"

    all_dirs = [
        train_labeled_dir, train_unlabeled_dir,
        val_dir, test_dir,
        ann_train_dir, ann_val_dir, ann_test_dir,
    ]
    for d in all_dirs:
        if overwrite and d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        print(f"未在 {data_dir} 找到任何类别文件夹。")
        return

    total = {"train_labeled": 0, "train_unlabeled": 0, "val": 0, "test": 0}

    for class_dir in class_dirs:
        files = [f for f in class_dir.iterdir() if f.is_file()]
        if not files:
            print(f"跳过空文件夹: {class_dir.name}")
            continue

        random.shuffle(files)
        n = len(files)

        # ceil 保底：val/test 各至少 1 张
        n_val = max(1, math.ceil(n * val_ratio))
        n_test = max(1, math.ceil(n * test_ratio))
        n_train = n - n_val - n_test

        # 极小样本保护：若 train <= 0，缩减 val/test 至各 1 张
        if n_train <= 0:
            n_val = min(1, n)
            n_test = min(1, max(0, n - n_val))
            n_train = n - n_val - n_test

        val_files = files[:n_val]
        test_files = files[n_val: n_val + n_test]
        train_files = files[n_val + n_test:]

        # train 内切分 labeled / unlabeled（ceil 保底至少 1 张 labeled）
        if train_files:
            n_labeled = max(1, math.ceil(len(train_files) * labeled_ratio))
        else:
            n_labeled = 0
        labeled_files = train_files[:n_labeled]
        unlabeled_files = train_files[n_labeled:]

        for f in val_files:
            shutil.copy2(f, val_dir / f.name)
        for f in test_files:
            shutil.copy2(f, test_dir / f.name)
        for f in labeled_files:
            shutil.copy2(f, train_labeled_dir / f.name)
        for f in unlabeled_files:
            shutil.copy2(f, train_unlabeled_dir / f.name)

        total["val"] += len(val_files)
        total["test"] += len(test_files)
        total["train_labeled"] += len(labeled_files)
        total["train_unlabeled"] += len(unlabeled_files)

        print(
            f"{class_dir.name}: 总数 {n} | "
            f"train {len(train_files)} "
            f"(labeled {len(labeled_files)}, unlabeled {len(unlabeled_files)}) | "
            f"val {len(val_files)} | test {len(test_files)}"
        )

    print("\n── 汇总 ──")
    print(f"train/labeled  : {total['train_labeled']}")
    print(f"train/unlabeled: {total['train_unlabeled']}")
    print(f"val            : {total['val']}")
    print(f"test           : {total['test']}")
    print(f"\n划分完成，结果已保存到: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="数据集划分脚本（train/val/test + labeled/unlabeled）")
    parser.add_argument("--data_dir", type=Path, default=None, help="原始数据目录（默认 ../data）")
    parser.add_argument("--output_dir", type=Path, default=None, help="输出目录（默认 ../split_data）")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例（默认 0.1）")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="测试集比例（默认 0.1）")
    parser.add_argument("--labeled_ratio", type=float, default=0.1, help="train 中 labeled 比例（默认 0.1）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    parser.add_argument("--overwrite", action="store_true", help="是否清空已有输出目录后重建")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    data_path = args.data_dir or project_root / "data"
    output_path = args.output_dir or project_root / "split_data"

    split_dataset(
        data_dir=data_path,
        output_dir=output_path,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        labeled_ratio=args.labeled_ratio,
        seed=args.seed,
        overwrite=args.overwrite,
    )
