from rfdetr import RFDETRBase
from ultralytics import YOLO


def rfdetr2onnx(rfdetr_model: RFDETRBase, **kwargs):
    """
    rfdetr2onnx 将rfdetr训练的模型导出onnx格式
    :param rfdetr_model:
    :param kwargs:
    :return:
    """
    # output_dir      Directory where the exported ONNX model will be saved
    # infer_dir       Path to an image file to use for tracing. If not provided, a random dummy image is generated.
    # simplify        Whether to simplify the ONNX model using onnxsim for better compatibility and performance.
    # backbone_only   Export only the backbone feature extractor instead of the full model.
    # opset_version   ONNX opset version to use for export. Higher versions support more operations.
    # verbose         Whether to print verbose export information.
    # force           Force re-export even if simplified model already exists.
    # shape           Input shape as tuple (height, width). Must be divisible by 14. If not provided, uses the model's default resolution.
    # batch_size      Batch size for the exported model.

    rfdetr_model.export(**kwargs)


def yolo2onnx(yolo_model: YOLO, **kwargs):
    """
    yolo2onnx 将yolo训练的模型导出onnx格式
    :param yolo_model:
    :param kwargs:
    :return:
    """

    # format="onnx",      # 导出格式为 ONNX
    # imgsz=(640, 640),   # 设置输入图像的尺寸
    # keras=False,        # 不导出为 Keras 格式
    # optimize=False,     # 不进行优化 False, 移动设备优化的参数，用于在导出为TorchScript 格式时进行模型优化
    # half=False,         # 不启用 FP16 量化
    # int8=False,         # 不启用 INT8 量化
    # dynamic=False,      # 不启用动态输入尺寸
    # simplify=True,      # 简化 ONNX 模型
    # opset=None,         # 使用最新的 opset 版本
    # workspace=4.0,      # 为 TensorRT 优化设置最大工作区大小（GiB）
    # nms=False,          # 不添加 NMS（非极大值抑制）
    # batch=1,            # 指定批处理大小
    # device="cpu"        # 指定导出设备为CPU或GPU，对应参数为"cpu" , "0"

    yolo_model.export(**kwargs)
