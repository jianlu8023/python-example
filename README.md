# python-example

Python 示例代码合集：Django 5.2.9 后端项目（含 YOLO / PyTorch 推理服务）、模型工具库、零散 Python 示例。

## 目录结构

```
python-example/
├── apps/                          # Django 应用
│   ├── common/                    # 通用应用（首页 / 文件上传 / 自定义 management 命令）
│   └── infer/                     # 推理服务应用
│       ├── yolo/                  #   YOLO 推理（detect / classify，ultralytics）
│       ├── pytorch/               #   PyTorch 推理（resnet18 等 torchvision 模型）
│       └── modelcache.py          #   模型缓存
├── config/                        # 部署配置（config.py / gunicorn.config.py / logging.conf）
├── python_example/                # Django 主项目（settings / urls / views / asgi / wsgi）
│   └── common/                    #   共享组件
│       ├── middleware/            #     全局异常处理（except.py）/ 请求中间件（request.py）
│       ├── response/              #     API 响应包装（resp.py → ApiResponse）
│       └── logging/               #     自定义日志 handler
├── utilities/                     # 共享工具
│   ├── logger.py                  # 日志（get_logger / init_logger）
│   ├── filepath.py                # 文件路径工具
│   ├── image.py                   # 图片处理
│   ├── images/                    # 图片处理器（Django 用）
│   ├── loggers/                   # 日志工厂（Django 用）
│   ├── keras/                     # Keras 导出 / 可视化
│   └── model/                     # 模型缓存 / 推理 / 处理器 / ONNX（yolo、rf-detr 等）
├── other/                         # 零散示例（csv / matplotlib / tensor / pima keras+torch /
│   │                              #   yolo 训练+检测 / detr 训练+推理 等独立脚本）
├── docs/                          # 文档（django.md：Django 使用整理）
├── ttf/                           # 字体资源
├── scripts/                       # 预留脚本目录
│
├── manage.py                      # Django 管理入口
├── main.py                        # 最小示例（logger 使用入口）
├── requirements.txt               # 依赖（django 段 + pytorch 段，清华源 + cu121 扩展源）
├── .conda / .commands             # conda 环境定义 / 依赖安装命令备忘
├── Dockerfile / Dockerfile-prod   # GPU 推理服务镜像（ultralytics 构建）
├── docker-entrypoint.sh           # 容器入口（migrate / runserver / serve，日志落盘 + logrotate）
├── Makefile                       # docker 镜像构建 / 清理命令
└── AGENTS.md                      # Django 项目说明（AI 协作约定）
```

## Django 项目

Django 5.2.9 + djangorestframework，模块化应用结构，SQLite 默认数据库，gunicorn 23 + uvloop 部署。

```bash
# 安装依赖（requirements.txt 已配置清华源 + pytorch cu121 扩展源）
pip install -r requirements.txt

# 运行
python manage.py migrate
python manage.py runserver

# 测试
python manage.py test
```

接口路由：

| 路径 | 说明 |
|------|------|
| `/` | 首页 index |
| `/common/` | 通用接口（`file/upload` 文件上传） |
| `/infer/yolo/detect`、`/infer/yolo/classify` | YOLO 目标检测 / 分类（ultralytics） |
| `/infer/pytorch/resnet18` | PyTorch 模型推理示例 |
| `/admin/` | Django admin |

关键文件：`python_example/settings.py`（设置/日志/中间件）、`python_example/urls.py`、`apps/common/`（主应用，含 `management/commands/example.py`、`init_basemodel.py` 自定义命令）、`python_example/common/middleware/except.py`（全局异常处理）、`python_example/common/response/resp.py`（`ApiResponse` 响应包装）。详见 `AGENTS.md`。

## 零散示例

`other/` 下为独立可运行的 Python 示例脚本：

- `tensor_example.py` / `main.py` — Tensor 基础操作、logger 使用
- `yolo_train_classify.py` / `yolo_train_detect.py` / `yolo_detect_example.py` — YOLO 训练 / 检测
- `detr_train_example.py` / `detr_infer_example.py` — DETR 训练 / 推理
- `pima_keras.py` / `pima_torch.py` / `simple_train.py` — Pima 数据集 Keras / PyTorch 训练
- `csv_example.py` / `matplotlib_example.py` — csv 读写、matplotlib 绑图
- `datasets/plma.csv` — 示例数据集

## Docker 部署

```bash
make docker      # 构建镜像（Dockerfile）
make docker-prod # 构建生产镜像（Dockerfile-prod）
make clean       # 清理 docker buildx 缓存
```

镜像基于 `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime` 自行构建 ultralytics 层，入口 `docker-entrypoint.sh` 支持：

```bash
migrate                                    # 执行 makemigrations + migrate
runserver -address 0.0.0.0:8000 [-migrate] # 开发服务
serve -address 0.0.0.0:8080 -migrate -config config/gunicorn.config.py  # gunicorn 生产服务
```

容器内日志追加写入 `logs/run.log`，后台 logrotate 按天轮转。

## 环境

- Python 3.11
- `.conda` — conda 环境定义（`python-example` / python 3.11）
- `.commands` — 重建环境步骤备忘（uv 安装 torch cu121 + django 段 + pytorch 段依赖）
- 核心版本：Django 5.2.9、torch 2.5.1 (cu121)、ultralytics 8.3.234、rfdetr 1.5.2、onnxruntime 1.23.2、gunicorn 23.0.0
