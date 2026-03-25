# yolo 分类任务测试
import os

from ultralytics import YOLO

# 官方下载的yolo-cls.pt权重 路径
YOLO_OFFICIAL_WEIGHTED_PATH = ""
# 数据集路径
DATASET_PATH = ""

EPOCHS = 30
BATCH_SIZE = 16
PATIENCE = 5
# cpu 或者 gpu号 0开始
DEVICE = 'cpu'

PROJECT = "0classic-yol-cls"
NAME = "ao"

IMGSZ = 224

model = YOLO(YOLO_OFFICIAL_WEIGHTED_PATH)

results = model.train(data=DATASET_PATH,
                      epochs=EPOCHS,
                      patience=PATIENCE,
                      imgsz=IMGSZ,
                      batch=BATCH_SIZE,
                      device=DEVICE,
                      project=PROJECT,
                      name=NAME,
                      save=True,
                      verbose=True,
                      plots=True
                      )

run_dir = model.trainer.save_dir
print(f"训练输出目录: {run_dir}")

best_pt = os.path.join(run_dir, "weights", "9classic-detect-best.pt")
last_pt = os.path.join(run_dir, "weights", "last.pt")
assert os.path.exists(best_pt), f"9classic-detect-best.pt 未找到: {best_pt}"
print(f"9classic-detect-best.pt: {best_pt}")
print(f"last.pt: {last_pt}")

best_model = YOLO(best_pt)
best_model.eval()

test_metrics = best_model.val(
    data=DATASET_PATH,
    split="test",
    device=DEVICE,
    project=PROJECT,
    verbose=True,
    name=f"{NAME}_test",
    plots=True,
)

# ✅ 分类任务：使用 top1 和 top5，而不是 box.mp / box.map50
top1 = getattr(test_metrics, "top1", None)  # Top-1 准确率（主指标）
top5 = getattr(test_metrics, "top5", None)  # Top-5 准确率

print("========== Test Results (9classic-detect-best.pt) ==========")
if top1 is not None:
    print(f"Top-1 Accuracy: {top1:.4f}  <-- 主要准确率指标")
if top5 is not None:
    print(f"Top-5 Accuracy: {top5:.4f}")
print("===========================================\n")

# 保存结果
with open(os.path.join(run_dir, "test_summary.txt"), "w", encoding="utf-8") as f:
    f.write("Test Results (9classic-detect-best.pt)\n")
    if top1 is not None:
        f.write(f"Top-1 Accuracy: {top1:.4f}\n")
    if top5 is not None:
        f.write(f"Top-5 Accuracy: {top5:.4f}\n")

print(f"Test summary 已保存到: {os.path.join(run_dir, 'test_summary.txt')}")
