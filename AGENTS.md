# AGENTS.md


## 项目概述

这是一个 Django 5.2.9 项目，采用模块化应用程序结构。主项目位于 `python_example` 目录，自定义应用位于 `apps/common` 目录。

## 架构

- 主项目：`python_example/` - 包含核心 Django 配置、中间件和设置
- 应用结构：`apps/common/` - 包含具有视图、模型和 URL 的通用应用程序
- 自定义中间件：`python_example/common/middleware/except.py` 中的全局异常处理
- 自定义响应处理：`python_example/common/response/resp.py` 用于 API 响应
- 数据库：默认使用 SQLite，在 settings.py 中配置
- 日志：在 `logs/` 目录中带有文件轮转的自定义日志配置
- 文件上传：在 common 应用中的自定义文件上传处理

## 开发命令

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行应用程序
```bash
python manage.py runserver
```

### 数据库管理
```bash
python manage.py migrate
python manage.py makemigrations
```

### 测试
```bash
python manage.py test
python manage.py test apps.common.tests
```

### Django 管理命令
```bash
python manage.py createsuperuser
python manage.py shell
```

## 关键文件和目录

- `manage.py` - Django 管理工具
- `python_example/settings.py` - 项目设置，包括自定义日志、中间件和应用配置
- `python_example/urls.py` - 主 URL 配置
- `apps/common/` - 具有 API 端点的主应用程序
- `python_example/common/middleware/except.py` - 全局异常处理中间件
- `python_example/common/response/resp.py` - 自定义 API 响应包装器