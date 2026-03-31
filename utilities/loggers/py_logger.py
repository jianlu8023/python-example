import logging
import os
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from loguru import logger


# 1. 常量定义
class Modules:
    GRPC = "Grpc"
    WEB = "Web"
    JOB = "Job"
    REDIS = "Redis"


# 2. 配置定义

@dataclass
class LoggerConfig:
    default_log_level: str = "DEBUG"
    file_path: str = "logs/app.log"
    max_age: int = 7  # 天
    rotation_time: int = 1  # 小时
    # 针对不同模块的日志级别覆盖
    logger_level: Dict[str, str] = field(default_factory=dict)


def get_default_config() -> LoggerConfig:
    return LoggerConfig()


# 3. 拦截器 (确保原生 logging 模块受控)

class InterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


# 4. 核心控制器

def _logger_formatter(record):
    """自定义格式化器"""
    # 时间 | 级别
    # format_str = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <5}</level> | "
    format_str = "{time:YYYY-MM-DD HH:mm:ss.SSS} | <level>{level: <5}</level> | "
    
    # 智能模块名处理 (防止 [[SDK]] 出现)
    module_name = record["extra"].get("module_name", "").strip()
    if module_name:
        has_brackets = (
                module_name.startswith(('[', '(', '{', '<')) and
                module_name.endswith((']', ')', '}', '>'))
        )
        display_module = module_name if has_brackets else f"[{module_name}]"
        # format_str += f"<magenta>{display_module}</magenta> | "
        format_str += f"{display_module} | "
    
    # 代码位置: 文件:函数:行号
    # format_str += "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>\n"
    format_str += "{name}:{function}:{line} - {message}\n"
    
    if record["exception"]:
        format_str += "{exception}"
    return format_str


class LoggerControl:
    def __init__(self, config: Optional[LoggerConfig] = None):
        self.config = config or get_default_config()
        self._log_map: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._initialized = False
    
    def _dynamic_filter(self, record):
        """动态日志级别过滤逻辑"""
        module_name = record["extra"].get("module_name", "").lower()
        # 获取该模块配置的级别，没配置则用全局默认
        level_limit = self.config.logger_level.get(module_name, self.config.default_log_level)
        return record["level"].no >= logger.level(level_limit.upper()).no
    
    def startup(self):
        """初始化日志服务"""
        with self._lock:
            if self._initialized:
                return
            
            # 1. 清理
            logger.remove()
            
            # 2. 控制台
            logger.add(
                sys.stdout,
                level=0,  # 由 filter 决定具体级别
                format=_logger_formatter,
                # serialize=(self.config.print_format == "json"),
                filter=self._dynamic_filter,
                colorize=True,
                backtrace=True,
                diagnose=True
            )
            
            # 3. 文件
            if self.config.file_path:
                os.makedirs(os.path.dirname(os.path.abspath(self.config.file_path)), exist_ok=True)
                
                rotation = f"{self.config.rotation_time} hours"
                retention = f"{self.config.max_age} days"
                
                logger.add(
                    self.config.file_path,
                    level=0,
                    format=_logger_formatter,
                    # serialize=(self.config.print_format == "json"),
                    filter=self._dynamic_filter,
                    rotation=rotation,
                    retention=retention,
                    enqueue=True,
                    encoding="utf-8"
                )
            
            # 4. 拦截原生 logging
            logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
            # 静音高频三方库
            for noise in ["django.utils.autoreload", "matplotlib", "PIL", "amqp"]:
                logging.getLogger(noise).setLevel(logging.INFO)
            
            self._initialized = True
    
    def gen_logger(self, module_name: str):
        """获取或创建模块 Logger"""
        if not self._initialized:
            self.startup()
        
        key = module_name.lower()
        with self._lock:
            if key not in self._log_map:
                self._log_map[key] = logger.bind(module_name=module_name)
            return self._log_map[key]


# 5. 全局单例接口

_control: Optional[LoggerControl] = None


def init_control(config: Optional[LoggerConfig] = None):
    """项目入口调用此函数进行初始化"""
    global _control
    _control = LoggerControl(config)
    _control.startup()
    return _control


def get_module_logger(module_name: str):
    """
    业务层全局调用。
    例如: log = get_module_logger(Modules.WEB)
    """
    global _control
    if _control is None:
        # 懒加载：如果忘记 init，使用默认配置
        _control = LoggerControl()
        _control.startup()
    return _control.gen_logger(module_name)
