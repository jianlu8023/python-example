from django.http import JsonResponse


class ApiResponse:
    """
    统一API响应封装类
    格式: {obj: data, code: int, success: bool, message: str}
    """
    
    @staticmethod
    def success(data=None, message="处理成功", code=200):
        """
        成功响应
        :param data: 返回的数据
        :param message: 提示信息
        :param code: 业务状态码
        :return: JsonResponse
        """
        response_data = {
            'obj': data,
            'code': code,
            'success': True,
            'message': message
        }
        return JsonResponse(response_data, status=200, json_dumps_params={'ensure_ascii': False})
    
    @staticmethod
    def error(message="处理失败", code=500, data=None):
        """
        错误响应
        :param message: 错误信息
        :param code: 业务状态码
        :param data: 返回的数据(可选)
        :return: JsonResponse
        """
        response_data = {
            'obj': data,
            'code': code,
            'success': False,
            'message': message
        }
        return JsonResponse(response_data, status=200, json_dumps_params={'ensure_ascii': False})