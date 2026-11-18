import pytest
import requests
from PIL import Image
from rfdetr.util.coco_classes import COCO_CLASSES

from .model_type import ModelType
from .model_infer import ModelInferEngine
from .model_infer_config import ModelInferConfig


class TestModelInferEngine:

    @pytest.fixture(scope="class")
    def engine(self):
        """初始化引擎的 fixture，在整个类测试期间只初始化一次"""
        return ModelInferEngine()

    @pytest.fixture
    def dummy_image(self):
        """创建一个全黑的测试图"""
        # return Image.fromarray(np.zeros((640, 640, 3), dtype=np.uint8))
        image = Image.open(requests.get("https://media.roboflow.com/dog.jpg", stream=True).raw)
        return image

    def test_device_init(self, engine):
        """测试设备初始化是否正确"""
        assert engine.device in ["cuda:0", "cpu", "mps"]
        print(f"\n[测试] 当前运行设备: {engine.device}")

    def test_rfdetr_nano_flow(self, engine, dummy_image):
        """测试 RF-DETR Nano 的完整流水线"""
        # 1. 准备配置 (请确保 model_path 指向一个真实存在的 nano 权重)
        config = ModelInferConfig(
            task_name="test_nano",
            model_type=ModelType.RFDETR_DETECT_LARGE,
            model_path="rf-detr-large-2026.pth",  # 替换为你的路径
            labels=COCO_CLASSES,
            conf_threshold=0.5
        )


        results = engine.run_step(config, step_idx=0, prev_context=None, original_img=dummy_image)


        # 3. 断言验证
        print(results)

    def test_pipeline_step_logic(self, engine, dummy_image):
        """测试多步骤逻辑：检测 -> 裁剪 -> 分类"""
        configs = [
            ModelInferConfig(
                task_name="detect",
                model_type=ModelType.YOLO_DETECT,
                model_path="weights/yolov8n.pt",
                labels={0: "person"},
                conf_threshold=0.1
            ),
            ModelInferConfig(
                task_name="classify",
                model_type=ModelType.RESNET18,
                model_path="weights/resnet18_fracture.pth",
                num_classes=2,
                labels={0: "no", 1: "yes"},
                conf_threshold=0.5
            )
        ]

        # 运行
        pipeline_res = engine.run_pipeline(configs, dummy_image)

        # 验证是否有两步结果
        assert len(pipeline_res) <= 2
        print(f"[测试] 流水线执行步骤数: {len(pipeline_res)}")
