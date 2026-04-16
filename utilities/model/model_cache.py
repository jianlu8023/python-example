import hashlib
import io
import os
import tempfile
import threading
from collections import OrderedDict

import torch
from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall
from torch import nn
from torchvision.models import resnet18
from ultralytics import YOLO

from ..loggers import get_logger

logger = get_logger(__name__)


def _get_cache_key(prefix, model_id, device, **kwargs):
    """生成唯一的缓存键，必须包含 device"""
    key = f"{prefix}_{model_id}_{device}"
    for k, v in sorted(kwargs.items()):
        key += f"_{k}:{v}"
    return key


def _prepare_device_str(device):
    """统一 RF-DETR 需要的设备字符串格式"""
    d_str = str(device).lower()
    if "cuda" in d_str: return "cuda"
    if "mps" in d_str: return "mps"
    return "cpu"


class ModelCache:
    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelCache, cls).__new__(cls)
        return cls._instance

    def __init__(self, max_cache_size=5):
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self.max_cache_size = max_cache_size
                    self.cache = OrderedDict()
                    self._initialized = True

    def _add_to_cache(self, key, model):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_cache_size:
                # 弹出最久未使用的模型
                removed_key, removed_model = self.cache.popitem(last=False)
                logger.debug(f"LRU 淘汰模型: {removed_key}")
                # 显式清理显存（如果是GPU模型）
                del removed_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            self.cache[key] = model

    def _get_model_from_cache(self, key):
        """统一的缓存获取逻辑，包含线程安全的顺序更新"""
        if key in self.cache:
            with self._lock:
                if key in self.cache:
                    self.cache.move_to_end(key)
                    return self.cache[key]
        return None

    def get_yolo(self, model_path, device):
        # 键值包含 device，防止多线程设备抢占
        key = _get_cache_key("yolo", model_path, device)

        # 1. 尝试从缓存获取
        model = self._get_model_from_cache(key)
        if model: return model

        # 2. 缓存未命中，加锁加载
        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]

            logger.info(f"加载 YOLO 模型: {model_path} -> {device}")
            # Ultralytics YOLO 在初始化时即可指定 device
            model = YOLO(model_path).to(device)
            model.eval()
            self._add_to_cache(key, model)
            return model

    def get_resnet18(self, model_path, num_classes, device):
        key = _get_cache_key("resnet18", model_path, device, n_cls=num_classes)

        model = self._get_model_from_cache(key)
        if model: return model

        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]

            logger.info(f"加载 ResNet18 模型: {model_path} -> {device}")
            # weights=None 避免从网上下权重
            model = resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)

            # 加载权重
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()

            self._add_to_cache(key, model)
            return model

    def _get_rfdetr_generic(self, model_class, model_type_name, model_path, device):
        """抽象 RF-DETR 加载逻辑，避免重复代码和 Bug"""
        key = _get_cache_key(model_type_name, model_path, device)

        model = self._get_model_from_cache(key)
        if model: return model

        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]

            logger.info(f"加载 {model_type_name} 模型: {model_path} -> {device}")
            r_device = _prepare_device_str(device)

            # 动态实例化对应的类
            model = model_class(pretrain_weights=model_path, device=r_device)
            model.optimize_for_inference()
            self._add_to_cache(key, model)
            return model

    def get_rfdetr_nano(self, model_path, device):
        return self._get_rfdetr_generic(RFDETRNano, "rfdetr-nano", model_path, device)

    def get_rfdetr_small(self, model_path, device):
        return self._get_rfdetr_generic(RFDETRSmall, "rfdetr-small", model_path, device)

    def get_rfdetr_medium(self, model_path, device):
        return self._get_rfdetr_generic(RFDETRMedium, "rfdetr-medium", model_path, device)

    def get_rfdetr_large(self, model_path, device):
        return self._get_rfdetr_generic(RFDETRLarge, "rfdetr-large", model_path, device)

    def get_resnet18_from_content(self, model_content, num_classes, device, model_name="unnamed"):
        content_hash = hashlib.md5(model_content).hexdigest()
        key = _get_cache_key("resnet18_content", f"{model_name}_{content_hash}", device, n_cls=num_classes)

        model = self._get_model_from_cache(key)
        if model: return model

        with self._lock:
            if key in self.cache:
                return self.cache[key]

            logger.info(f"从二进制流加载 ResNet18: {model_name} -> {device}")
            model = resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)

            # 直接使用 BytesIO，无需临时文件
            buffer = io.BytesIO(model_content)
            state_dict = torch.load(buffer, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()

            self._add_to_cache(key, model)
            return model

    def get_yolo_from_content(self, model_content, device, model_name="unnamed"):
        """YOLO 必须通过文件路径加载，保留临时文件逻辑"""
        content_hash = hashlib.md5(model_content).hexdigest()
        key = _get_cache_key("yolo_content", f"{model_name}_{content_hash}", device)

        model = self._get_model_from_cache(key)
        if model: return model

        # YOLO 框架限制，必须从磁盘读取
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
            tmp.write(model_content)
            tmp_path = tmp.name

        try:
            with self._lock:
                if key in self.cache:
                    return self.cache[key]

                model = YOLO(tmp_path).to(device)
                model.eval()
                self._add_to_cache(key, model)
                return model
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
