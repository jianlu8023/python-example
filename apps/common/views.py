from datetime import datetime
import logging
import os

from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from rest_framework.decorators import api_view

from python_example.common.response.resp import ApiResponse

logger = logging.getLogger(__name__)


# Create your views here.


@api_view(['GET'])
def index(request):
    return ApiResponse.success(data='index')


@api_view(['POST'])
def file_upload(request):
    """
    file_upload 文件上传
    :param
    :param request: 请求
    :request
    :request file: 文件
    :request save_name: 保存名
    :return: 封装的ApiResponse
    """
    
    file = request.FILES.get('file')
    save_name = request.data.get('save_name')
    
    if not file or not save_name:
        return ApiResponse.error(data='缺少必填项')
    try:
        # 确保目录存在
        upload_dir = 'uploads/models'
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir,
                                 f"{save_name}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}_{file.name}")
        
        # 保存文件
        path = default_storage.save(file_path, ContentFile(file.read()))
        data = {
            "path": path,
        }
        return ApiResponse.success(data=data)
    except Exception as e:
        logger.error(f"上传文件发生错误: {str(e)}")
        return ApiResponse.error(data='上传文件失败')
