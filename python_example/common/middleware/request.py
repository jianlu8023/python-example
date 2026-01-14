import logging

from ipware import get_client_ip

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware:
    """
    RequestLoggingMiddleware 打印请求来源和uri
    """
    
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        client_ip, _ = get_client_ip(request)
        if client_ip is None:
            client_ip = 'unknown'
        
        uri = request.get_full_path()
        
        logger.info(f"receive request from {client_ip} uri {uri}")
        
        response = self.get_response(request)
        return response
