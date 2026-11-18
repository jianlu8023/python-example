import logging
import sys

from django.apps import AppConfig

logger = logging.getLogger(__name__)


class CommonConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.common'
    label = 'common_app'
    
    def ready(self):
        # 只在 runserver 或 gunicorn 启动时执行（避免 migrate 等命令触发）
        if not any(cmd in sys.argv for cmd in ['runserver', 'gunicorn', 'uvicorn']):
            return
        
        # 一些其他操作
        #if 't_basecode' not in connection.introspection.table_names():
        #    return
        
        logger.info(f'Starting {self.label} app...')
