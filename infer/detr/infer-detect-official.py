# detr-detect-infer-official.py 使用detr进行推理
import sys
from pathlib import Path

import requests
import supervision as sv
from PIL import Image
from rfdetr import RFDETRLarge
from rfdetr.util.coco_classes import COCO_CLASSES


def get_project_root(project_name="your_project_root", depth=2):
    """
    1. 优先匹配目录名
    2. 其次寻找特征文件 (.git, requirements.txt)
    3. 最后按给定的层级(depth)强制回退作为兜底
    """
    current_path = Path(__file__).resolve()

    # 逻辑 1 & 2：向上搜索匹配名称或特征文件
    for parent in current_path.parents:
        if parent.name == project_name or (parent / ".git").exists() or (parent / "requirements.txt").exists():
            return parent

    # 逻辑 3：兜底逻辑，按原代码逻辑回退固定层级 (yolo_crop.py 在 project/A/B/ 下，回退2级到project)
    # parents[0]是父目录, parents[1]是爷爷目录...
    return current_path.parents[depth] if len(current_path.parents) > depth else current_path.parent


_PROJECT_NAME = "python-example"
try:
    # 执行获取并添加路径
    PROJECT_ROOT = get_project_root(_PROJECT_NAME, depth=2)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from utilities import get_logger
except StopIteration as e:
    print(f"Project root not found: {_PROJECT_NAME}")

logger = get_logger(name="infer-detr")

logger.info("info")

model = RFDETRLarge()

image = Image.open(requests.get("https://media.roboflow.com/dog.jpg", stream=True).raw)
detections = model.predict(image, threshold=0.5)

# labels = [f"{COCO_CLASSES[class_id]}" for class_id in detections.class_id]
labels = [
    f"{COCO_CLASSES[class_id]} {confidence:.4f}"
    for class_id, confidence
    in zip(detections.class_id, detections.confidence)
]

# BoxAnnotator
box_image = sv.BoxAnnotator().annotate(image.copy(), detections)
box_image.save(f"dog-rfdetr-boxes.jpg")

# LabelAnnotator
annotated_image = sv.LabelAnnotator().annotate(box_image.copy(), detections, labels)
annotated_image.save(f"dog-rfdetr-labels.jpg")

# BackgroundOverlayAnnotator
background_overlay_annotator = sv.BackgroundOverlayAnnotator()
annotated_image = background_overlay_annotator.annotate(
    scene=image.copy(),
    detections=detections
)
annotated_image.save(f"dog-rfdetr-background.jpg")

# # CropAnnotator
crop_annotator = sv.CropAnnotator()
annotated_image = crop_annotator.annotate(
    scene=image.copy(),
    detections=detections
)
annotated_image.save(f"dog-rfdetr-cropped.jpg")
