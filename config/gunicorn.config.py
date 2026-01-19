import multiprocessing

# 绑定地址和端口（监听所有 IP 的 8000 端口）
bind = "0.0.0.0:8000"

# 工作进程数：推荐 (CPU 核心数 * 2) + 1，但至少 2 个
workers = max(2, multiprocessing.cpu_count() * 2 + 1)

# 每个 worker 使用 1 个线程（同步模式）
threads = 1

# 工作模式（默认 sync，适用于大多数 Django 项目）
worker_class = "sync"

# 最大并发连接数（每个 worker）
worker_connections = 1000

# 限制上传大小（可选）
# limit_request_line = 4096
# limit_request_fields = 100


# 请求超时时间（秒），防止卡死
timeout = 120

keepalive = 5

# 每处理 N 个请求后重启 worker（防内存泄漏）
max_requests = 1000
# 随机抖动，避免所有 worker 同时重启
max_requests_jitter = 1000

# 预加载应用（节省内存，但需确保无全局状态问题）
preload_app = True

# 访问日志（可设为 '-' 输出到 stdout）
accesslog = "-"

# 错误日志
errorlog = "-"

# 日志级别
loglevel = "debug"

# 是否将 stdout/stderr 重定向到 error log
capture_output = True

pidfile = "app.pid"

# 启动时打印配置（调试用）
# print_config = True
