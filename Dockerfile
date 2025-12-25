# 自己制作 ultralytics/ultralytics:latest 镜像

# FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime AS ultralytics-builder
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel AS ultralytics-builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    MKL_THREADING_LAYER=GNU \
    OMP_NUM_THREADS=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai

ADD https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.ttf \
    https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.Unicode.ttf \
    /root/.config/Ultralytics/

RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list && \
    sed -i s@/security.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc git htop \
    zip unzip \
    wget curl \
    libgl1 libglib2.0-0 gnupg libsm6  \
    tzdata && \
    echo "Asia/Shanghai" > /etc/timezone && \
    ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /ultralytics

RUN git clone -b v8.3.234 https://github.com/ultralytics/ultralytics.git /ultralytics && \
    sed -i '/^\[http "https:\/\/github\.com\/"\]/,+1d' .git/config && \
    sed -i'' -e 's/"opencv-python/"opencv-python-headless/' pyproject.toml
ADD https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt .

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install uv && \
    uv pip install --system -e "." albumentations faster-coco-eval wandb && \
    rm -rf tmp /root/.config/Ultralytics/persistent_cache.json

FROM ubuntu:20.04 AS tools-builder

RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list && \
    sed -i s@/security.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    tini gosu

FROM ultralytics-builder AS pip-builder

WORKDIR /pip

COPY requirements.txt .

RUN pip install -r requirements.txt


FROM pip-builder AS runner

WORKDIR /home/appuser/myapp

COPY . .
COPY --from=tools-builder /usr/bin/tini /usr/bin/tini
COPY --from=tools-builder /usr/sbin/gosu /usr/sbin/gosu

#创建非 root 用户
RUN groupadd -g 1000 -r myusers && \
    useradd -m -r -u 1000 -g myusers appuser && \
    mkdir -p /home/appuser/myapp && \
    mkdir -p /home/appuser/myapp/db && \
    mkdir -p /home/appuser/myapp/logs && \
    mkdir -p /home/appuser/myapp/uploads && \
    mkdir -p /home/appuser/.config/Ultralytics && \
    mv /home/appuser/myapp/docker-entrypoint.sh /docker-entrypoint.sh && \
    chmod +x /docker-entrypoint.sh && \
    chown -R appuser:myusers /home/appuser

USER appuser

VOLUME /home/appuser/myapp/logs \
       /home/appuser/myapp/db

EXPOSE 8000

ENTRYPOINT ["/usr/bin/tini","--","/docker-entrypoint.sh"]

HEALTHCHECK --start-period=60s --retries=3 --timeout=15s --interval=60s \
    CMD curl http://127.0.0.1:8000/ || exit 1

CMD ["runserver", "-migrate", "-address", "0.0.0.0:8000"]
