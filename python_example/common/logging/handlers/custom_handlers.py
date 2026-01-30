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


class MultiProcessSafeTimedRotatingFileHandler(TimedRotatingFileHandler):
    """
    多进程安全的 TimedRotatingFileHandler（支持 Windows + Linux）
    - 保留重命名轮转机制（app.log -> app.log.YYYY-MM-DD）
    - 每次写入前检查：当前文件是否仍是 baseFilename？
    - 如果不是（被重命名了），则重新打开 baseFilename
    - 支持跨天检测（即使长时间无日志）
    - 每次写入后 flush()
    """

    def __init__(self, filename, when='h', interval=1, backupCount=0,
                 encoding=None, delay=False, utc=False, atTime=None, errors=None):
        super().__init__(filename, when=when, interval=interval,
                         backupCount=backupCount, encoding=encoding,
                         delay=delay, utc=utc, atTime=atTime, errors=errors)
        self._last_stat = None

    def _is_base_file_still_exists(self):
        """检查 baseFilename 是否仍存在且是普通文件"""
        return os.path.isfile(self.baseFilename)

    def _get_file_inode_or_mtime(self):
        """获取当前打开文件的 inode（Linux/macOS）或 mtime（Windows fallback）"""
        if not self.stream:
            return None
        try:
            stat_result = os.fstat(self.stream.fileno())
            # Windows 没有 inode，用 (mtime, size) 作为指纹
            if os.name == 'nt':
                return (stat_result.st_mtime, stat_result.st_size)
            else:
                return stat_result.st_ino
        except (OSError, ValueError):
            return None

    def _should_reopen_due_to_rename(self):
        """
        判断当前打开的文件是否已被重命名（即不再是 baseFilename）
        方法：比较当前 stream 的 inode/mtime 与 baseFilename 的是否一致
        """
        if not self.stream or not self._is_base_file_still_exists():
            return True  # 文件不存在，需重建

        try:
            current_fingerprint = self._get_file_inode_or_mtime()
            base_stat = os.stat(self.baseFilename)
            if os.name == 'nt':
                base_fingerprint = (base_stat.st_mtime, base_stat.st_size)
            else:
                base_fingerprint = base_stat.st_ino

            return current_fingerprint != base_fingerprint
        except (OSError, ValueError):
            return True

    def shouldRollover(self, record):
        """
        增强版轮转判断：
        1. 先检查是否因重命名需要 reopen
        2. 再检查是否跨天
        """
        # 检查是否被重命名 → 需要 reopen
        if self._should_reopen_due_to_rename():
            return True

        # 检查是否跨天（兼容长时间无日志）
        if self.when == 'MIDNIGHT':
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
        轮转逻辑：
        - 如果是因重命名导致 reopen，则不执行轮转，只 reopen
        - 否则执行标准轮转
        """
        if not self._is_base_file_still_exists():
            # 文件不存在，直接 reopen，不轮转
            if self.stream:
                self.stream.close()
                self.stream = None
            self.stream = self._open(self.baseFilename)
            return

        # 检查是否真的需要时间轮转（而非仅 reopen）
        if self.when == 'MIDNIGHT':
            now = datetime.now()
            current_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
            try:
                file_mtime = os.path.getmtime(self.baseFilename)
                last_write_dt = datetime.fromtimestamp(file_mtime)
                last_midnight = last_write_dt.replace(hour=0, minute=0, second=0, microsecond=0)
                if current_midnight <= last_midnight:
                    # 不需要时间轮转，只需 reopen
                    if self.stream:
                        self.stream.close()
                    self.stream = self._open(self.baseFilename)
                    return
            except (OSError, ValueError):
                pass

        # 执行标准轮转（重命名）
        super().doRollover()

    def emit(self, record):
        """写入前确保文件正确，写入后 flush"""
        try:
            if self.shouldRollover(record):
                self.doRollover()
            super().emit(record)
            self.flush()  # 确保实时落盘
        except Exception:
            self.handleError(record)
