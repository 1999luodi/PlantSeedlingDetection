# post_process

本目录包含 3 个脚本：

1. `calc_gt_germination_rate.py`
   - 从人工标注 json 计算发芽率：
   - 发芽率 = germinated / (germinated + ungerminated)

2. `calc_pred_germination_rate.py`
   - 通过 mmdetection 配置和模型权重，对图片推理并计算发芽率

3. `compare_germination_rate.py`
   - 对比人工标注发芽率与模型发芽率，输出误差统计和可视化

## 运行示例

在项目根目录下运行。

### 1) 计算人工标注发芽率（支持文件夹）

```powershell
# 单行（推荐）
python train/post_process/calc_gt_germination_rate.py --input split_data/val --out_csv train/post_process/outputs/gt_germination_rates.csv

# 多行（PowerShell 续行符是反引号 `）
python train/post_process/calc_gt_germination_rate.py `
  --input split_data/val `
  --out_csv train/post_process/outputs/gt_germination_rates.csv
```

### 2) 模型检测并计算发芽率

```powershell
python train/post_process/calc_pred_germination_rate.py --config train/mmdetection/configs/seedling-detection/faster-rcnn_r50_fpn_2x_coco.py --checkpoint train/mmdetection/work_dirs/faster-rcnn_r50_fpn_2x_coco/epoch_24.pth --input split_data/val --score_thr 0.5 --device cpu --out_csv train/post_process/outputs/pred_germination_rates.csv
```

如果配置文件内已有 `load_from`，可以不传 `--checkpoint`。

### 3) 对比误差并可视化

```powershell
python train/post_process/compare_germination_rate.py --gt_csv train/post_process/outputs/gt_germination_rates.csv --pred_csv train/post_process/outputs/pred_germination_rates.csv --out_dir train/post_process/outputs
```

输出文件：
- `germination_rate_compare.csv`: 每张图 GT/预测/误差
- `germination_rate_summary.json`: 均值、方差、MAE、RMSE
- `germination_rate_error_plot.png`: 误差可视化图
