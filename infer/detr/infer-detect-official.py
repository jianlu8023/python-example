# detr-detect-infer-official.py 使用detr进行推理

import requests
import supervision as sv
from PIL import Image
from rfdetr import RFDETRLarge
from rfdetr.util.coco_classes import COCO_CLASSES

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
annotated_image = sv.LabelAnnotator().annotate(image.copy(), detections, labels)
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
