import logging
import threading
from collections import OrderedDict
import hashlib

import torch
from torch import nn
from torchvision.models import resnet18,ResNet18_Weights
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class ModelCache:
    _instance = None
    _lock = threading.Lock()
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelCache, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, max_cache_size=5):
        """
        初始化模型缓存。
        :param max_cache_size: 最大缓存模型数量（LRU 淘汰）
        """
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self.max_cache_size = max_cache_size
                    # 使用 OrderedDict 实现 LRU：最新访问放最后，最旧在最前
                    self.cache = OrderedDict()
                    self._initialized = True
    
    def _add_to_cache(self, key, model):
        """线程安全地添加模型到缓存，自动 LRU 淘汰"""
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_cache_size:
                removed_key, removed_model = self.cache.popitem(last=False)
                logger.debug(f"LRU 淘汰模型: {removed_key}")
                del removed_model
            self.cache[key] = model
        self.cache.move_to_end(key)
    
    def get_yolo(self, model_path, device):
        key = f"yolo_{model_path}"
        # 先尝试读取（无锁，快速路径）
        if key in self.cache:
            with self._lock:
                # 再次确认并更新为最近使用
                if key in self.cache:
                    self.cache.move_to_end(key)
                    return self.cache[key].to(device)
        
        # 不在缓存中，加载模型
        with self._lock:
            # 双重检查：可能其他线程刚加载了
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key].to(device)
            
            logger.info(f"加载 YOLO 模型: {model_path} 到设备 {device}")
            try:
                model = YOLO(model_path)
                model.cpu()
                model.eval()
                self._add_to_cache(key, model)
                return model.to(device)
            except Exception as e:
                logger.error(f"加载 YOLO 模型失败: {e}")
                raise
    
    def get_yolo_from_content(self, model_content, device, model_name=None):
        """
        根据模型内容获取YOLO模型，用于处理上传的模型文件
        :param model_content: 模型文件内容
        :param device: 设备
        :param model_name: 模型名称（可选，用于生成更友好的缓存键）
        :return: YOLO模型实例
        """
        # 使用模型内容的哈希值作为缓存键的一部分
        content_hash = hashlib.md5(model_content).hexdigest()
        if model_name:
            key = f"yolo_content_{model_name}_{content_hash}"
        else:
            key = f"yolo_content_{content_hash}"
        
        # 先尝试读取（无锁，快速路径）
        if key in self.cache:
            with self._lock:
                # 再次确认并更新为最近使用
                if key in self.cache:
                    self.cache.move_to_end(key)
                    return self.cache[key].to(device)
        
        # 不在缓存中，需要临时保存内容并加载模型
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_model:
            tmp_model.write(model_content)
            tmp_model_path = tmp_model.name
        
        try:
            with self._lock:
                # 双重检查：可能其他线程刚加载了
                if key in self.cache:
                    self.cache.move_to_end(key)
                    return self.cache[key].to(device)
                
                logger.info(f"从内容加载 YOLO 模型，缓存键: {key} 到设备 {device}")
                model = YOLO(tmp_model_path)
                model.cpu()
                model.eval()
                self._add_to_cache(key, model)
                return model.to(device)
        except Exception as e:
            logger.error(f"从内容加载 YOLO 模型失败: {e}")
            raise
        finally:
            # 清理临时文件
            if os.path.exists(tmp_model_path):
                os.unlink(tmp_model_path)
    
    def get_resnet18(self, model_path, num_classes, device):
        key = f"resnet18_{model_path}_{num_classes}"
        if key in self.cache:
            with self._lock:
                if key in self.cache:
                    self.cache.move_to_end(key)
                    return self.cache[key].to(device)
        
        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key].to(device)
            
            logger.info(f"加载 ResNet18 模型: {model_path} (num_classes={num_classes}) 到设备 {device}")
            try:
                model = resnet18(weights=ResNet18_Weights.DEFAULT)
                num_features = model.fc.in_features
                model.fc = nn.Linear(num_features, num_classes)
                
                # 尝试加载模型权重
                try:
                    model.load_state_dict(
                        torch.load(model_path, map_location=torch.device(device), weights_only=True))
                except Exception as e:
                    logger.warning(
                        f"ResNet18模型 加载权重出错,尝试使用strict=False: {str(e)}")
                    try:
                        model.load_state_dict(
                            torch.load(model_path, map_location=device), strict=False)
                    except Exception as e2:
                        logger.warning(
                            f"ResNet18模型 以strict=False方式加载权重也出错: {str(e2)}")
                        raise RuntimeError(f"模型文件加载失败") from e2
                
                model.cpu()
                model.eval()
                self._add_to_cache(key, model)
                return model.to(device)
            except Exception as e:
                logger.error(f"加载 ResNet18 模型失败: {e}")
                raise
    
    def get_resnet18_from_content(self, model_content, num_classes, device, model_name=None):
        """
        根据模型内容获取ResNet18模型，用于处理上传的模型文件
        :param model_content: 模型文件内容
        :param num_classes: 分类数
        :param device: 设备
        :param model_name: 模型名称（可选，用于生成更友好的缓存键）
        :return: ResNet18模型实例
        """
        # 使用模型内容的哈希值和分类数作为缓存键的一部分
        content_hash = hashlib.md5(model_content).hexdigest()
        if model_name:
            key = f"resnet18_content_{model_name}_{num_classes}_{content_hash}"
        else:
            key = f"resnet18_content_{num_classes}_{content_hash}"
        
        # 先尝试读取（无锁，快速路径）
        if key in self.cache:
            with self._lock:
                # 再次确认并更新为最近使用
                if key in self.cache:
                    self.cache.move_to_end(key)
                    return self.cache[key].to(device)
        
        # 不在缓存中，需要临时保存内容并加载模型
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_model:
            tmp_model.write(model_content)
            tmp_model_path = tmp_model.name
        
        try:
            with self._lock:
                # 双重检查：可能其他线程刚加载了
                if key in self.cache:
                    self.cache.move_to_end(key)
                    return self.cache[key].to(device)
                
                logger.info(f"从内容加载 ResNet18 模型，缓存键: {key} 到设备 {device}")
                model = resnet18(weights=None)
                num_features = model.fc.in_features
                model.fc = nn.Linear(num_features, num_classes)
                
                # 尝试加载模型权重
                try:
                    model.load_state_dict(
                        torch.load(tmp_model_path, map_location=torch.device(device), weights_only=True))
                except Exception as e:
                    logger.warning(
                        f"ResNet18模型 加载权重出错,尝试使用strict=False: {str(e)}")
                    try:
                        model.load_state_dict(
                            torch.load(tmp_model_path, map_location=device), strict=False)
                    except Exception as e2:
                        logger.warning(
                            f"ResNet18模型 以strict=False方式加载权重也出错: {str(e2)}")
                        raise RuntimeError(f"模型文件加载失败") from e2
                
                model.cpu()
                model.eval()
                self._add_to_cache(key, model)
                return model.to(device)
        except Exception as e:
            logger.error(f"从内容加载 ResNet18 模型失败: {e}")
            raise
        finally:
            # 清理临时文件
            if os.path.exists(tmp_model_path):
                os.unlink(tmp_model_path)