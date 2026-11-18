from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .model_type import ModelType


@dataclass
class ModelInferConfig:
    model_path: str
    model_type: ModelType  # 使用枚举类型
    labels: Dict[int, str] = field(default_factory=dict)
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    num_classes: Optional[int] = None
    task_name: str = "task"
    extra_params: Dict[str, Any] = field(default_factory=dict)
    verbose: bool = False

    def __post_init__(self):
        # 强制类型检查（可选，用于防止非 dataclass 初始化时的传参错误）
        if isinstance(self.model_type, str):
            try:
                self.model_type = ModelType(self.model_type)
            except ValueError:
                valid_types = [t.value for t in ModelType]
                raise ValueError(f"无效的模型类型: {self.model_type}。可选范围: {valid_types}")

        if self.num_classes is None and self.labels:
            self.num_classes = len(self.labels)
