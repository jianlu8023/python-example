# 是否开启debug
DEBUG = False

# settings.ALLOWED_HOSTS
ALLOWED_HOSTS = [
    'localhost',
    '127.0.0.1',
]

# 数据库位置 使用sqlite
DB_PATH = "db/db.sqlite3"

# 日志存储位置
LOG_PATH = "logs/app.log"

# 日志保存时间 单位 天
LOG_BACKUP_COUNT = 15

# 时区
TIME_ZONE = 'Asia/Shanghai'

# 语言
LANGUAGE_CODE = 'en-us'

# 是否国际化
USE_I18N = True

# 是否使用时区
USE_TZ = True
