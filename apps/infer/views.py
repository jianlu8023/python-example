import json
import logging

import torch
from PIL import Image
from rest_framework.decorators import api_view
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18

from python_example.common.response.resp import ApiResponse

logger = logging.getLogger(__name__)


# Create your views here.


@api_view(['GET'])
def index(request):
    return ApiResponse.success(data='ok')


