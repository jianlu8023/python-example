# run-detr.py 用于使用rfdetr库进行detr模型训练

# 设置cuda可见卡的信息 因为 rfdetr无法设置cuda:3
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch.cuda
import sys

# 支持模型列表
# detect
#   RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge
# seg
#   RFDETRSegNano, RFDETRSegSmall, RFDETRSegMedium, RFDETRSegLarge
#   RFDETRSegXLarge, RFDETRSeg2XLarge 这里应该需要安装rfdetr_plus
# 官网链接
#   https://rfdetr.roboflow.com/develop/

from rfdetr import RFDETRNano

# 加载模型到pwd 若无则会联网下载
model = RFDETRNano(resolution=672)
model_type = "nano"  # 模型类型

# 一些配置信息 用于构成存放训练数据的目录
base_path = "python-example"  # 基础路径
base_dataset_path = os.path.join(base_path, "datasets")  # 基础数据集存放路径
dataset_name = "8classic-coco"  # 数据集名称
dataset_dir = os.path.join(base_dataset_path, dataset_name)  # 使用数据集信息
epochs = 30  # 训练轮次
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"  # 运行设备
progress_bar = True  # 程序进度条
lr = 1e-4  # 学习率 默认值
lr_encoder = 1.5e-4  # 默认值
weight_decay = 1e-4  # l2正则 默认值
# GPU	        VRAM	batch_size	grad_accum_steps
# A100	        40-80GB	16	        1
# RTX 4090	    24GB	8	        2
# RTX 3090	    24GB	8	        2
# T4	        16GB	4	        4
# RTX 3070	    8GB	    2	        8
batch_size = 8  # 批大小
grad_acc_steps = 2  #
early_stopping_patience = 10  # 早停等待
early_stopping_min_delta = 0.0001
project_name = f"{dataset_name}"
run_name = f"{epochs}epochs-{model_type}-{lr}lr-{lr_encoder}lr_encoder-{weight_decay}weight_decay-{early_stopping_patience}patience-{early_stopping_min_delta}min_delta"

base_output_path = os.path.join(base_path, "output")
output_name = f"{dataset_name}-{run_name}"
output_dir = os.path.join(base_output_path, output_name)

print("----")
print("rfdetr模型训练相关配置信息如下:")
print(f"模型名称: {model_type}")
print(f"数据集路径: {dataset_dir}")
print(f"训练轮次: {epochs} \n早停patience: {early_stopping_patience} ")
print(f"学习率lr: {lr} \nlr_encoder: {lr_encoder} \nweight_decay: {weight_decay}")
print(f"project: {project_name} \nrun: {run_name}")
print(f"训练存放路径: {output_dir}")
print("----")
print("开始训练...")

try:
    model.train(
        dataset_dir=dataset_dir,
        epochs=epochs,
        batch_size=batch_size,
        grad_acc_steps=grad_acc_steps,
        lr=lr,
        lr_encoder=lr_encoder,
        weight_decay=weight_decay,
        output_dir=output_dir,
        use_ema=True,
        early_stopping=True,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        wandb=False,
        tensorboard=True,
        project=project_name,
        run=run_name,
        progress_bar=progress_bar,
    )
except KeyboardInterrupt as e:
    print("接收到终止按键,停止训练...")
    sys.exit(0)

print("done.")
