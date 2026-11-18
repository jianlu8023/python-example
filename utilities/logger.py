import logging
import logging.config
import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class LogConfig:
    """
    日志配置类。
    """
    log_dir: str = "logs"
    file_name: str = "app.log"
    level: str = "DEBUG"  # 全局级别
    console_level: Optional[str] = None  # 如果不设置，默认跟全局级别一致
    file_level: Optional[str] = None  # 如果不设置，默认跟全局级别一致
    backup_count: int = 7
    format: str = "%(asctime)s | %(levelname)-6s | %(module)s:%(lineno)d | %(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"

    def to_dict_config(self) -> Dict[str, Any]:
        """将对象参数转换为 logging.config.dictConfig 所需的字典"""
        log_path = os.path.join(self.log_dir, self.file_name)
        c_level = self.console_level or self.level
        f_level = self.file_level or self.level

        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": self.format,
                    "datefmt": self.datefmt
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                    "level": c_level
                },
                "file": {
                    "class": "logging.handlers.TimedRotatingFileHandler",
                    "filename": log_path,
                    "when": "midnight",
                    "interval": 1,
                    "backupCount": self.backup_count,
                    "formatter": "standard",
                    "encoding": "utf-8",
                    "level": f_level
                }
            },
            "root": {
                "handlers": ["console", "file"],
                "level": self.level
            },
            "loggers": {
                "matplotlib": {"level": "INFO", "propagate": False},
                "PIL": {"level": "INFO", "propagate": False}
            }
        }


class LoggerManager:
    """内部单例管理器"""
    _lock = threading.Lock()
    _initialized = False
    _current_config: Optional[LogConfig] = None

    @classmethod
    def setup(cls, config: Optional[LogConfig] = None, force: bool = False):
        """核心安装方法"""
        with cls._lock:
            if cls._initialized and not force:
                return

            # 如果没有传入 config，使用默认配置
            if config is None:
                config = LogConfig()

            cls._current_config = config
            conf_dict = config.to_dict_config()

            # 自动创建目录
            log_file = conf_dict["handlers"]["file"]["filename"]
            os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)

            # 应用配置
            logging.config.dictConfig(conf_dict)
            cls._initialized = True

    @classmethod
    def get_logger(cls, name: Optional[str] = None) -> logging.Logger:
        """获取 logger，未初始化则按默认配置加载"""
        if not cls._initialized:
            cls.setup()
        return logging.getLogger(name)


# --- 暴露给外部的干净接口 ---

def init_logger(config: Optional[LogConfig] = None, force: bool = False, **kwargs):
    """
    初始化日志。
    方式 1: 直接传 config 对象 -> init_logger(LogConfig(level='INFO'))
    方式 2: 传关键字参数（内部自动转对象） -> init_logger(level='INFO', log_dir='test_logs')
    """
    if config is None:
        # 如果没有传对象，则把 kwargs 里的参数塞进对象里
        config = LogConfig(**kwargs)

    LoggerManager.setup(config=config, force=force)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """获取 Logger 实例"""
    return LoggerManager.get_logger(name)
