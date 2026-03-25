import logging
import logging.config
import os
import threading
from typing import Optional

# 1. 将配置直接硬编码在文件里，确保包的独立性
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s | %(levelname)-6s | %(module)s:%(lineno)d | %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "DEBUG"
        },
        "file": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "filename": "logs/pytorch.log",  # 默认在主程序运行目录创建 logs
            "when": "midnight",
            "interval": 1,
            "backupCount": 7,
            "formatter": "standard",
            "encoding": "utf-8",
            "level": "DEBUG"
        }
    },
    "root": {
        "handlers": ["console", "file"],
        "level": "DEBUG"
    },
    "loggers": {
        "matplotlib": {"level": "INFO", "propagate": False}
    }
}

# 全局锁，确保多线程环境下初始化也是安全的
_lock = threading.Lock()
_initialized = False


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    获取 Logger 实例。
    :param name: logger 的名字。如果为 None，则返回 root logger。
    """
    global _initialized

    if not _initialized:
        with _lock:
            if not _initialized:
                _setup_logging()
                _initialized = True

    return logging.getLogger(name)


def _setup_logging():
    # 自动创建日志目录
    # 遍历配置中的所有 handler，如果是文件类型的，创建其所在的目录
    for handler_conf in LOGGING_CONFIG.get("handlers", {}).values():
        filename = handler_conf.get("filename")
        if filename:
            # 转换为绝对路径，确保日志文件夹创建在执行 main.py 的位置
            log_dir = os.path.dirname(os.path.abspath(filename))
            if not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

    # 直接应用字典配置
    logging.config.dictConfig(LOGGING_CONFIG)
