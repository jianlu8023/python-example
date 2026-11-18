import logging
import traceback

from python_example.common.response.resp import ApiResponse

logger = logging.getLogger(__name__)


class GlobalExceptionMiddleware:
    """
    全局异常处理中间件
    捕获所有未处理的异常并返回统一的错误响应
    """
    
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        try:
            response = self.get_response(request)
        except Exception as e:
            return self.handle_exception(request, e)
        return response
    
    def handle_exception(self, request, exception):
        """
        处理未捕获的异常
        :param request: 请求对象
        :param exception: 异常对象
        :return: 统一错误响应
        """
        # 记录异常日志
        logger.error(f"Unhandled exception occurred: {str(exception)}")
        logger.error(traceback.format_exc())
        
        # 返回统一的错误响应
        return ApiResponse.error(
            message="服务器内部错误，请稍后再试",
            code=500
        )
