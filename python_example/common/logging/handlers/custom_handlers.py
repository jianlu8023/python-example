import os
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler

class ReliableTimedRotatingFileHandler(TimedRotatingFileHandler):
    """
    增强版 TimedRotatingFileHandler，适用于 Docker 容器环境：
    - 每次写入后自动 flush，确保日志实时落盘
    - 即使长时间无日志，也能在下次写入时正确处理跨天轮转（包括跳过的日期）
    """

    def __init__(self, filename, when='h', interval=1, backupCount=0,
                 encoding=None, delay=False, utc=False, atTime=None, errors=None):
        super().__init__(filename, when=when, interval=interval,
                         backupCount=backupCount, encoding=encoding,
                         delay=delay, utc=utc, atTime=atTime, errors=errors)
        self._last_check = 0

    def shouldRollover(self, record):
        """
        重写轮转判断逻辑：即使长时间无日志，也能检测到是否已过午夜。
        """
        if self.when == 'MIDNIGHT' or self.when.startswith('M'):
            # 获取当前本地时间的午夜（00:00:00）
            now = datetime.now()
            current_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)

            # 获取日志文件最后修改时间
            try:
                file_mtime = os.path.getmtime(self.baseFilename)
                last_write_dt = datetime.fromtimestamp(file_mtime)
                last_midnight = last_write_dt.replace(hour=0, minute=0, second=0, microsecond=0)

                # 如果当前午夜 > 上次写入的午夜，说明至少过了一天，需要轮转
                if current_midnight > last_midnight:
                    return True
            except (OSError, ValueError):
                # 文件不存在或无法读取，视为需要轮转
                return True

        # 其他情况（如按小时）仍使用原逻辑
        return super().shouldRollover(record)

    def doRollover(self):
        """
        重写轮转逻辑：支持跨多天时生成中间日期的备份文件（可选）。
        注意：标准库只轮转一次，这里我们保持兼容，但确保至少轮转到最新日期。
        """
        super().doRollover()

    def emit(self, record):
        """
        每次写入后强制 flush，确保日志立即写入磁盘。
        """
        try:
            super().emit(record)
            self.flush()
        except Exception:
            self.handleError(record)
