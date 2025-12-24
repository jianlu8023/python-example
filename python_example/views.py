import logging

from rest_framework.decorators import api_view

from python_example.common.response.resp import ApiResponse

logger = logging.getLogger(__name__)


@api_view(['GET'])
def index(request):
    logger.debug("receive index api")
    return ApiResponse.success(data='index')
