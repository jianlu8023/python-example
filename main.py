import os

if not os.path.exists("logs"):
    os.makedirs("logs", exist_ok=True)
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    
    'formatters': {
        'verbose': {
            'format': '[{levelname}] {asctime} {name} - {message}',
            'style': '{',
        },
        'standard': {
            'format': '%(asctime)s | %(levelname)-7s | %(module)s:%(lineno)d | %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
    },
    
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'level': 'INFO',  # 只处理 info 及以上的日志
        },
        'file': {
            'class': 'logging.handlers.TimedRotatingFileHandler',
            'filename': os.path.join("logs", f"{os.path.splitext(os.path.basename(__file__))[0]}.log"),
            'when': 'midnight',  # 每天零点新建一个日志文件
            'interval': 1,
            'backupCount': 7,  # 保留 7 天
            'formatter': 'standard',
            'encoding': 'utf-8',
            'level': 'DEBUG',  # 处理 debug 及以上的日志
        },
        'file_debug': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join("logs", f"{os.path.splitext(os.path.basename(__file__))[0]}.log"),
            'level': 'DEBUG',
            'backupCount': 5,
            'formatter': 'standard',
            'encoding': 'utf-8',
            'maxBytes': 5 * 1024 * 1024,  # 5 MB
        },
    },
    'root': {
        'handlers': ['console', 'file'],
        'level': 'DEBUG',
    },
    'loggers': {
        '__main__': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',  # Logger 必须能接收 debug 日志
            'propagate': False,  # 防止冒泡到 root logger
        },
    }
}
import logging.config

logging.config.dictConfig(LOGGING)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info(f"hello, this is a info log...")
