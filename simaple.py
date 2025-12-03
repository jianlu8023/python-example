import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging

from func import say

log = logging.getLogger(__name__)


def simple_train(epochs=100, counts=1000, batchsize=32):
    say()
    # 1. 创建一个简单的数据集
    X = torch.randn(counts, 10)
    y = torch.randint(0, 2, (counts,))

    # 2. 创建 DataLoader
    dataset = TensorDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=batchsize, shuffle=True)

    # 3. 定义一个简单的模型
    model = nn.Linear(10, 2)  # 线性模型，10 个输入特征，2 个输出类别

    # 4. 定义优化器
    optimizer = optim.Adam(model.parameters())

    # 5. 训练模型
    # num_epochs = 100  # 设置 Epochs 的数量
    for epoch in range(epochs):  # 遍历 Epochs
        for batch_idx, (data, target) in enumerate(data_loader):  # 遍历每个 Batch
            # 前向传播
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 打印训练信息
            log.debug(f"Epoch: {epoch + 1}/{epochs} Batch: {batch_idx + 1}/{len(data_loader)} Loss: {loss.item():.4f}")
            # print(f"Epoch: {epoch + 1}/{epochs}, Batch: {batch_idx + 1}/{len(data_loader)}, Loss: {loss.item():.4f}")
    say()


if __name__ == "__main__":
    simple_train(10, counts=10000)
