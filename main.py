import logging.handlers
import os.path
import sys

from simaple import simple_train

log_dir = "../demo-workspace/logs"
log_name = "app.log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 配置日志记录器
logging.basicConfig(
    level=logging.DEBUG,  # 设置最低日志级别为 DEBUG
    format='%(asctime)s %(name)s::%(filename)s [line:%(lineno)s] %(levelname)s %(message)s',
    # filename=os.path.join(log_dir, log_name),  # 将日志写入文件 (可选)
    # filemode='a',  # 如果要覆盖文件，使用 'w'，如果要追加，使用 'a'
    encoding='utf-8',
    # force=True
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            filename=os.path.join(log_dir, log_name),
            mode='a',
            encoding='utf-8',
            delay=False,

        ),
        logging.handlers.RotatingFileHandler(
            filename=os.path.join(log_dir, log_name),
            mode='a',
            encoding='utf-8',
            delay=False,
            maxBytes=5 * 1024 * 1024,
            backupCount=7
        ),
        logging.handlers.TimedRotatingFileHandler(
            filename=os.path.join(log_dir, log_name),
            when='h',
            interval=1,
            encoding='utf-8',
            backupCount=7,
            delay=False,
            utc=False,
            atTime=None,

        ),
    ],
)

log = logging.getLogger(__name__)

if __name__ == '__main__':
    log.debug("debug...")
    simple_train(epochs=100, counts=10000, batchsize=16)
