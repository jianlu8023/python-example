import logging

from rest_framework.decorators import api_view

from python_example.common.response.resp import ApiResponse

logger = logging.getLogger(__name__)


# Create your views here.


@api_view(['GET'])
def index(request):
    return ApiResponse.success(data='ok')
