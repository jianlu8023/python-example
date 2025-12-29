

from python_example.common.response.resp import ApiResponse

import torch
from PIL import Image
from rest_framework.decorators import api_view
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18

# Create your views here.


@api_view(['GET'])
def index(request):
    return ApiResponse.success(data='ok')


@api_view(['POST'])
def resnet18(request):
    logger.debug(f"receive infer/pytorch/resnet18 api")
    
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
        
        infer_file = Image.open(image_file).convert('RGB')
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        model_resnet18 = resnet18(weights=None)
        num_features = model_resnet18.fc.in_features
        
        # 加载模型标签映射
        model_labels = {}
        if model_label:
            try:
                model_labels = json.loads(str(model_label))
            except json.JSONDecodeError as e:
                logger.warning(f"模型标签解析失败，使用默认标签映射: {str(e)}")
                model_labels = {}
        
        num_classes = len(model_labels) if model_labels else 1000
        model_resnet18.fc = nn.Linear(num_features, num_classes)
        
        model_resnet18.to(device)
        
        # 尝试加载模型权重
        try:
            model_resnet18.load_state_dict(
                torch.load(model_file, map_location=torch.device(device), weights_only=True))
        except Exception as e:
            logger.warning(
                f"ResNet18模型 加载权重出错,尝试使用strict=False: {str(e)}")
            try:
                model_resnet18.load_state_dict(
                    torch.load(model_file, map_location=torch.device(device)), strict=False)
            except Exception as e2:
                logger.warning(
                    f"ResNet18模型 以strict=False方式加载权重也出错: {str(e2)}")
                return ApiResponse.error(data='加载模型权重失败')
        
        model_resnet18.eval()
        
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = preprocess(infer_file).unsqueeze(0).to(device)
        with torch.no_grad():
            area_logits = model_resnet18(input_tensor)
            area_pred = torch.argmax(area_logits, dim=1).item()
            area_prob = torch.softmax(area_logits, dim=1)[0].tolist()
            class_label = str(area_pred)
            if str(area_pred) in model_labels:
                class_label = model_labels[str(area_pred)]
            
            # 获取top-k预测结果
            top_k = min(5, len(area_prob))
            top_k_indices = torch.topk(torch.tensor(area_prob), top_k).indices.tolist()
            top_k_labels = []
            for top_k_idx in top_k_indices:
                label = str(top_k_idx)
                if str(top_k_idx) in model_labels:
                    label = model_labels[str(top_k_idx)]
                top_k_labels.append({
                    'class_id': top_k_idx,
                    'class_label': label,
                    'probability': area_prob[top_k_idx]
                })
            
            result_data = [{
                'class_id': area_pred,
                'class_label': class_label,
                'probabilities': area_prob,
                'top_k_predictions': top_k_labels
            }]
        
        data = {
            'model': "ResNet18",
            'success': True,
            'result': result_data,
        }
        return ApiResponse.success(data=data)
    except Exception as e:
        logger.error(f"使用模型resnet18进行推理时发生错误: {str(e)}")
        return ApiResponse.error(data='使用resnet18模型推理发生错误')
