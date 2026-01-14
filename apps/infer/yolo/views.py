import json
import logging
import os
import tempfile

import torch
from PIL import Image
from rest_framework.decorators import api_view
from ultralytics import YOLO

from apps.infer.modelcache import ModelCache
from python_example.common.response.resp import ApiResponse

logger = logging.getLogger(__name__)


# Create your views here.

@api_view(['GET'])
def index(request):
    return ApiResponse.success(data='ok')


@api_view(["POST"])
def detect(request):
    
    image_file = request.FILES.get("file", None)
    if image_file is None:
        return ApiResponse.error(data='缺少推理文件')
    model_file = request.FILES.get("model", None)
    if model_file is None:
        return ApiResponse.error(data='缺少模型权重文件')
    model_label = request.data.get("label", None)
    if model_label is None:
        return ApiResponse.error(data='缺少标签对应关系')
    
    try:
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        model_labels = {}
        if model_label:
            try:
                model_labels = json.loads(str(model_label))
            except json.JSONDecodeError as e:
                logger.warning(f"YOLO模型标签解析失败，使用默认标签映射: {str(e)}")
                model_labels = {}
        
        # 读取模型文件内容
        model_content = model_file.read()
        
        # 使用模型缓存加载YOLO模型
        model_cache = ModelCache()
        yolo_model = model_cache.get_yolo_from_content(model_content, device, model_file.name)
        
        infer_file = Image.open(image_file)
        results = yolo_model(infer_file, device=device, verbose=True)
        
        result_data = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls)
                    class_label = str(class_id)
                    if str(class_id) in model_labels:
                        class_label = model_labels[str(class_id)]
                    
                    result_data.append({
                        'class_id': class_id,
                        'class_label': class_label,
                        'confidence': float(box.conf),
                        'bbox': box.xyxy.tolist()[0] if len(box.xyxy) > 0 else []
                    })
        
        logger.debug(f"YOLO模型 推理结束,汇总推理结果...")
        data = {
            "model": "YOLO",
            "task": 'DETECT',
            "result": result_data,
            'success': True,
        }
        return ApiResponse.success(data=data)
    except Exception as e:
        logger.error(f"YOLO DETECT 发生错误: {str(e)}")
        return ApiResponse.error(data='使用yolo进行目标检测推理过程中出现错误')


@api_view(["POST"])
def classify(request):
    image_file = request.FILES.get("file", None)
    if image_file is None:
        return ApiResponse.error(data='缺少推理文件')
    model_file = request.FILES.get("model", None)
    if model_file is None:
        return ApiResponse.error(data='缺少模型权重文件')
    model_label = request.data.get("label", None)
    if model_label is None:
        return ApiResponse.error(data='缺少标签对应关系')
    try:
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 读取模型文件内容
        model_content = model_file.read()
        
        # 使用模型缓存加载YOLO模型
        model_cache = ModelCache()
        model_yolo = model_cache.get_yolo_from_content(model_content, device, model_file.name)
        
        # 加载模型标签映射
        model_labels = {}
        if model_label:
            try:
                model_labels = json.loads(str(model_label))
            except json.JSONDecodeError as e:
                logger.warning(f"YOLO-CLASSIFY模型标签解析失败，使用默认标签映射: {str(e)}")
                model_labels = {}
        infer_file = Image.open(image_file)
        
        results = model_yolo(infer_file, device=device, verbose=True)
        result_data = []
        for r in results:
            probs = r.probs
            if probs is not None:
                class_id = int(probs.top1)
                confidence = float(probs.top1conf)
                class_label = str(class_id)
                if str(class_id) in model_labels:
                    class_label = model_labels[str(class_id)]
                
                top_5_labels = []
                for idx, top_5_idx in enumerate(probs.top5):
                    label = str(top_5_idx)
                    if str(top_5_idx) in model_labels:
                        label = model_labels[str(top_5_idx)]
                    top_5_labels.append({
                        'class_id': top_5_idx,
                        'class_label': label,
                        'probability': float(probs.top5conf[idx]),
                    })
                
                result_data.append({
                    'class_id': class_id,
                    'class_label': class_label,
                    'confidence': confidence,
                    'top_k_predictions': top_5_labels,
                })
        data = {
            "model": "YOLO",
            "task": 'CLASSIFY',
            "result": result_data,
            'success': True,
        }
        return ApiResponse.success(data=data)
    except Exception as e:
        logger.error(f"YOLO CLASSIFY 发生错误: {str(e)}")
        return ApiResponse.error(data='模型推理失败')
