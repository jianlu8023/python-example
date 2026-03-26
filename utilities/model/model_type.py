from enum import Enum, unique

@unique
class ModelType(Enum):
    YOLO_DETECT = "YOLO-DETECT"
    YOLO_CLASSIFY = "YOLO-CLASSIFY"
    RESNET18 = "ResNet18"
    RFDETR_DETECT = "RFDETR-DETECT"