# python-example

Python 示例代码合集：Django 5.2.9 后端项目、PyTorch 训练/推理示例、Python 通用示例。

## 目录结构

```
python-example/
├── apps/                          # Django 应用（common 通用应用 / infer 推理服务）
├── config/                        # Django 配置（config.py / gunicorn.config.py / logging.conf）
├── python_example/                # Django 主项目（settings / urls / 中间件 / 日志 / 响应包装）
├── manage.py                      # Django 管理入口
├── Dockerfile / Dockerfile-prod   # GPU 推理服务镜像（ultralytics 构建）
├── Makefile                       # docker 镜像构建命令
├── AGENTS.md                      # Django 项目说明（AI 协作约定）
│
├── infer/                         # PyTorch 推理示例（detr / yolo）
├── train/                         # PyTorch 训练示例（detr / yolo）
├── models/                        # 自定义模型（pima 等）
├── datasets/                      # 数据集
├── pytorch/                       # PyTorch 基础示例（tensor 等）
├── docs/                          # 文档（tensor 笔记）
├── ttf/                           # 字体资源
│
├── utilities/                     # 共享工具
│   ├── logger.py                  # 日志（get_logger / init_logger）
│   ├── filepath.py                # 文件路径工具
│   ├── image.py                   # 图片处理
│   ├── keras/                     # Keras 导出/可视化
│   ├── images/                    # 图片处理器（Django 用）
│   ├── loggers/                   # 日志工厂（Django 用）
│   └── model/                     # 模型缓存/推理/ONNX（Django 用）
│
├── examples/
│   └── main.py                    # Python 示例：logging 配置（控制台+滚动日志文件）
├── main.py                        # PyTorch 入口示例
├── requirements.txt               # 依赖（django 段 + pytorch 段）
├── .conda / .commands             # 环境创建 / 安装命令备忘
└── other/                         # 零散脚本（csv / matplotlib / func）
```

## Django 项目

Django 5.2.9 + djangorestframework，模块化应用结构，SQLite 默认数据库，gunicorn 部署。

```bash
# 安装依赖
pip install -r requirements.txt

# 运行
python manage.py migrate
python manage.py runserver

# 测试
python manage.py test
```

关键文件：`python_example/settings.py`（设置/日志/中间件）、`python_example/urls.py`、`apps/common/`（主应用）、`python_example/common/middleware/except.py`（全局异常处理）、`python_example/common/response/resp.py`（API 响应包装）。详见 `AGENTS.md`。

## PyTorch 示例

```bash
# 安装依赖
pip install -r requirements.txt
```

| 目录 | 内容 |
|------|------|
| `train/yolo/` | YOLO 分类/检测训练 |
| `train/detr/` | DETR 训练 |
| `infer/yolo/` | YOLO 推理 |
| `infer/detr/` | DETR 推理（官方实现） |
| `pytorch/tensor.py` | Tensor 基础操作示例 |
| `models/` | 自定义模型 |

## Python 示例

- `examples/main.py` — logging 配置示例：控制台 + 按天滚动日志文件（保留 7 天）

## Docker 部署

```bash
make docker      # 构建 GPU 推理镜像（Dockerfile）
make docker-prod # 构建生产镜像（Dockerfile-prod）
```

## 环境

- Python 3.11
- `.conda` — conda 环境创建命令备忘
- `.commands` — 依赖安装命令备忘
