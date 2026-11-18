import os

import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO

base_path = "python-example"  # 基础路径
datasets_dir = os.path.join(base_path, "datasets")
datasets_name = "9classic"
datasets_yaml_name = "dataset.yaml"
datasets_yaml_path = os.path.join(datasets_dir, datasets_name, datasets_yaml_name)  # 数据yaml

models_dir = os.path.join(base_path, "models")
models_name = "yolo-official"
model_pth_name = "yolo11x.pt"
model_ckpt = os.path.join(models_dir, models_name, model_pth_name)  # yolo权重
model_type = os.path.splitext(os.path.basename(models_name))[0]

epochs = 100  # 训练30轮
patience = 15  # 早停 patience
imgsz = 640
batch = 16  # CPU 环境小一点更稳
project_name = f"{datasets_name}-yolo-detect"  # 输出目录
name = f"{model_type}-{epochs}epochs-{patience}patience-{imgsz}imgsz-{batch}batch"
# device: CPU 用 "cpu" GPU 服务器从0开始
device = "2"


def ensure_test_in_yaml(data_yaml_path: str):
    """
    如果yaml里没有test字段，自动补上 images/test
    这样 model.val(split='test') 一定能跑到 test 集。
    """
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    has_test = any(line.strip().startswith("test:") for line in lines)
    if has_test:
        return data_yaml_path

    # 简单插入 test: images/test 在 val 后面或末尾
    new_lines = []
    inserted = False
    for line in lines:
        new_lines.append(line)
        if line.strip().startswith("val:") and not inserted:
            new_lines.append("test: images/test")
            inserted = True
    if not inserted:
        new_lines.append("test: images/test")

    new_yaml_path = data_yaml_path.replace(".yaml", "_with_test.yaml")
    with open(new_yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines))

    print(f"[INFO] 原yaml无test字段，已生成新yaml: {new_yaml_path}")
    return new_yaml_path


def plot_curves(results_csv: str, out_dir: str):
    """
    从 Ultralytics 训练输出的 results.csv 中读取每轮指标并画图。
    acc 用 mAP50 (metrics/mAP50) 代替。
    loss 绘制 box_loss / cls_loss / dfl_loss 三条曲线。
    """
    df = pd.read_csv(results_csv)

    # 兼容不同版本列名（Ultralytics 会在 results.csv 里保存这些字段）
    # 检测 "acc" 代理：mAP50
    acc_col_candidates = [c for c in df.columns if "metrics/mAP50" in c]
    if not acc_col_candidates:
        raise ValueError("results.csv 中未找到 metrics/mAP50 列，检查 ultralytics 版本输出。")
    acc_col = acc_col_candidates[0]

    # loss 列（训练 loss）
    box_loss_col = [c for c in df.columns if "train/box_loss" in c]
    cls_loss_col = [c for c in df.columns if "train/cls_loss" in c]
    dfl_loss_col = [c for c in df.columns if "train/dfl_loss" in c]

    # epoch 列
    epoch_col_candidates = [c for c in df.columns if c.strip().lower() == "epoch"]
    epoch_col = epoch_col_candidates[0] if epoch_col_candidates else df.columns[0]

    epochs = df[epoch_col].values

    # 1) acc curve
    plt.figure()
    plt.plot(epochs, df[acc_col].values, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("mAP@0.5 (acc proxy)")
    plt.title("Validation mAP50 Curve")
    plt.grid(True)
    acc_png = os.path.join(out_dir, "acc_curve.png")
    plt.savefig(acc_png, dpi=200, bbox_inches="tight")
    plt.close()

    # 2) loss curve
    plt.figure()
    if box_loss_col:
        plt.plot(epochs, df[box_loss_col[0]].values, label="box_loss")
    if cls_loss_col:
        plt.plot(epochs, df[cls_loss_col[0]].values, label="cls_loss")
    if dfl_loss_col:
        plt.plot(epochs, df[dfl_loss_col[0]].values, label="dfl_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.grid(True)
    loss_png = os.path.join(out_dir, "loss_curve.png")
    plt.savefig(loss_png, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[INFO] acc 曲线已保存: {acc_png}")
    print(f"[INFO] loss 曲线已保存: {loss_png}")


# 1) 训练
model = YOLO(model_ckpt)

results = model.train(
    data=datasets_yaml_path,
    epochs=epochs,
    patience=patience,
    imgsz=imgsz,
    batch=batch,
    device=device,
    project=project_name,
    name=name,
    save=True,
    verbose=True,
    plots=True,
)

# 训练输出目录
run_dir = model.trainer.save_dir
print(f"[INFO] 训练输出目录: {run_dir}")

# 9classic-detect-best.pt 路径
best_pt = os.path.join(run_dir, "weights", "9classic-detect-best.pt")
last_pt = os.path.join(run_dir, "weights", "last.pt")
assert os.path.exists(best_pt), f"9classic-detect-best.pt 未找到: {best_pt}"
print(f"[INFO] 9classic-detect-best.pt: {best_pt}")
print(f"[INFO] last.pt: {last_pt}")

# 2) 记录每轮 acc/loss 并画图
results_csv = os.path.join(run_dir, "results.csv")
assert os.path.exists(results_csv), f"results.csv 未找到: {results_csv}"
plot_curves(results_csv, run_dir)

# 3) 加载 9classic-detect-best.pt 做 test 集评估

best_model = YOLO(best_pt)
best_model.eval()

test_metrics = best_model.val(
    data=datasets_yaml_path,
    split="test",
    device=device,
    project=project_name,
    verbose=True,
    name=f"{name}_test",
)

# Ultralytics results 对象里常见指标：
# test_metrics.box.map50, map, mp, mr 等（不同版本字段略有差异）
# 这里尽量兼容输出
mp = getattr(test_metrics.box, "mp", None)  # mean precision
mr = getattr(test_metrics.box, "mr", None)  # mean recall
map50 = getattr(test_metrics.box, "map50", None)
map5095 = getattr(test_metrics.box, "map", None)

print("\n========== Test Results (9classic-detect-best.pt) ==========")
if mp is not None: print(f"Precision (mp): {mp:.4f}")
if mr is not None: print(f"Recall (mr):    {mr:.4f}")
if map50 is not None: print(f"mAP@0.5:        {map50:.4f}  <-- acc proxy")
if map5095 is not None: print(f"mAP@0.5:0.95:   {map5095:.4f}")
print("===========================================\n")

with open(os.path.join(run_dir, "test_summary.txt"), "w", encoding="utf-8") as f:
    f.write("Test Results (9classic-detect-best.pt)\n")
    if mp is not None: f.write(f"Precision (mp): {mp:.4f}\n")
    if mr is not None: f.write(f"Recall (mr): {mr:.4f}\n")
    if map50 is not None: f.write(f"mAP@0.5 (acc proxy): {map50:.4f}\n")
    if map5095 is not None: f.write(f"mAP@0.5:0.95: {map5095:.4f}\n")

print(f"[INFO] Test summary 已保存到: {os.path.join(run_dir, 'test_summary.txt')}")
