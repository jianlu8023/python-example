import multiprocessing
import os

import gunicorn.glogging

# 绑定地址和端口（监听所有 IP 的 8000 端口）
bind = "0.0.0.0:8000"

# 工作进程数：推荐 (CPU 核心数 * 2) + 1，但至少 2 个
# workers = max(2, multiprocessing.cpu_count() * 2 + 1)
workers = min(4, multiprocessing.cpu_count() * 2 + 1)

# 每个 worker 使用 1 个线程（同步模式）
threads = 1

# 工作模式（默认 sync，适用于大多数 Django 项目）
worker_class = "sync"

# 使用 Uvicorn ASGI Worker
worker_class = "uvicorn.workers.UvicornWorker"
wsgi_app = "python_example.asgi:application"

# 使用 gunicorn wsgi worker
# wsgi_app = "python_example.wsgi:application"

# 最大并发连接数（每个 worker）
worker_connections = 1000

# 限制上传大小（可选）
# limit_request_line = 4096
# limit_request_fields = 100

# Graceful shutdown 超时（秒）
graceful_timeout = 30

# 连接队列大小（默认 2048，高并发可适当调高）
backlog = 2048

# 请求超时时间（秒），防止卡死
timeout = int(os.getenv("GUNICORN_TIMEOUT", "120"))

keepalive = 5

# 每处理 N 个请求后重启 worker（防内存泄漏）
max_requests = 1000
# 随机抖动，避免所有 worker 同时重启
max_requests_jitter = 100

# 预加载应用（节省内存，但需确保无全局状态问题）
preload_app = True

# 访问日志（可设为 '-' 输出到 stdout）
accesslog = "-"

# 错误日志
errorlog = "-"

# 日志级别
loglevel = "debug"

logger_class = gunicorn.glogging.Logger

# 可以设置日志信息 下方为默认
logconfig_dict = {
    "version": 1,
    "disable_existing_loggers": False,
    "root": {
        "level": "INFO",
        "handlers": ["console"]
    },
    "loggers": {
        "gunicorn.error": {
            "level": "DEBUG",
            "handlers": ["error_console"],
            "propagate": False,
            "qualname": "gunicorn.error"
        },
        "gunicorn.access": {
            "level": "DEBUG",
            "handlers": ["console"],
            "propagate": False,
            "qualname": "gunicorn.access"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "generic",
            "stream": "ext://sys.stdout"
        },
        "error_console": {
            "class": "logging.StreamHandler",
            "formatter": "generic",
            "stream": "ext://sys.stderr"
        },
    },
    "formatters": {
        "generic": {
            # "format": "%(asctime)s [%(process)d] [%(levelname)s] %(message)s",
            # "datefmt": "[%Y-%m-%d %H:%M:%S %z]",
            # "class": "logging.Formatter",

            # 自定义
            # "format": "%(asctime)s.%(msecs)03d | %(log_color)s%(levelname)-7s%(reset)s | %(module)16s:%(lineno)-4d | %(message)s",
            "format": "%(asctime)s.%(msecs)03d | %(levelname)-7s | %(module)16s:%(lineno)-4d | %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
            # '()': 'colorlog.ColoredFormatter',
            # 'log_colors': {
            #     'DEBUG': 'cyan',
            #     'INFO': 'blue',
            #     'WARNING': 'yellow',
            #     'ERROR': 'red',
            #     'CRITICAL': 'red,bg_white',
            # },
        }
    }
}

# 是否将 stdout/stderr 重定向到 error log
capture_output = True

pidfile = "app.pid"

# 启动时打印配置（调试用） True 只打印配置文件 不启动服务
print_config = False

# 自定义访问日志格式（兼容 Nginx / Cloudflare / ELK）
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# 用户/组（在容器中通常不需要，宿主机部署时使用）
# user = "www-data"
# group = "www-data"

# 禁用 spew（调试用，生产必须关闭）
spew = False
