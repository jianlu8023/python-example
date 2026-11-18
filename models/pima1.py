import os

# 设置 keras 后端使用pytorch
os.environ["KERAS_BACKEND"] = "torch"
# 设置 CUDA 可见设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import logging
import math
from typing import AnyStr
import numpy
import torch
import keras
from keras import layers

log = logging.getLogger(__name__)


# plma Pima Indians Diabetes Database（皮马印第安人糖尿病数据集）
# 特征名称	描述	数据类型	单位
# Pregnancies	怀孕次数	整数	次
# Glucose	口服葡萄糖耐量试验中 2 小时后的血浆葡萄糖浓度	整数	mg/dL
# BloodPressure	舒张压	整数	mmHg
# SkinThickness	三头肌皮褶厚度	整数	mm
# Insulin	2 小时血清胰岛素	整数	mu U/mL
# BMI	身体质量指数	浮点数	kg/m^2
# DiabetesPedigreeFunction	糖尿病家族史函数 (根据家族史预测糖尿病风险的函数)	浮点数	无
# Age	年龄	整数	岁
# Outcome (目标变量/类标签)	是否患有糖尿病 (0 表示未患病，1 表示患病)	整数	0 或 1
def plma_model() -> keras.Model:
    return keras.Sequential([
        keras.Input(shape=(8,), name='input'),
        layers.Dense(12, activation='relu', name='dense1'),
        layers.Dense(8, activation='relu', name='dense2'),
        layers.Dense(1, activation='sigmoid', name='sigmoid'),
    ])


def plma_compile(model: keras.Model,
                 loss_func="binary_crossentropy", optimizer_func="adam"):
    model.compile(loss=loss_func, optimizer=optimizer_func, metrics=['accuracy'])


def plma_train(datasets: str, delimiter: AnyStr, epochs=10, batch_size=8):
    try:
        # 1. 确定设备 cpu / cuda
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Using device: {device}")
        
        # 2. 获取模型 并转移到device
        model = plma_model()
        model.to(device)
        plma_compile(model)
        
        # 3. 设置随机种子
        seed = 7
        numpy.random.seed(seed)
        # 设置pytorch 随机种子
        torch.manual_seed(seed)
        
        # 4. 加载数据
        dataset = numpy.loadtxt(fname=datasets, delimiter=delimiter)
        X = dataset[:, 0:8]
        Y = dataset[:, 8]
        # basic(model, X, Y)
        # batch_dataset(model, epochs, batch_size, device, X, Y)
        split_dataset(model, epochs, batch_size, device, X, Y)
    except FileNotFoundError as e:
        log.error(e)
    except KeyboardInterrupt as e:
        log.error(e)


def basic(model: keras.Model, epochs: int, batch_size: int, X: numpy.ndarray, Y: numpy.ndarray):
    for i in range(epochs):
        history = model.fit(X, Y, batch_size=batch_size, epochs=1, verbose=0)
        train_loss, train_acc = history.history["loss"][0], history.history["accuracy"][0]
        scores = model.evaluate(X, Y, batch_size=batch_size, verbose=0)
        # 获取损失值
        test_loss, test_acc = scores[0], scores[1]
        log.debug(
            f'epoch {i + 1 :3d}/{epochs} train_acc {train_acc:.7f} train_loss {train_loss:.7f} test_acc {test_acc:.7f} test_loss {test_loss:.7f}')


def batch_dataset(model: keras.Model, epochs: int, batch_size: int,
                  device: torch.device, X: numpy.ndarray, Y: numpy.ndarray):
    # 将 numpy数据转为pytorch tensor 并移动到device
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    log.info(f"X shape {X.shape}")
    log.info(f"Y shape {Y.shape}")
    
    #
    def divide(a, b):
        result = a / b
        if result == int(result):
            return int(result)
        else:
            return math.ceil(result)
    
    batch_num = divide(X.shape[0], batch_size)
    log.info(f"Batch size: {batch_num}")
    for i in range(epochs):
        train_loss_batch, train_acc_batch = [], []
        test_loss_batch, test_acc_batch = [], []
        for batch in range(batch_num):
            start = batch * batch_size
            end = start + batch_size
            X_batch = X[start:end].to(device)
            Y_batch = Y[start:end].to(device)
            if torch.cuda.is_available():
                with torch.cuda.device(0):
                    # log.info(f"Epoch {i + 1} train on device: {device}")
                    history = model.fit(X_batch, Y_batch, batch_size=batch_size, epochs=1, verbose=0)
            else:
                # log.info(f"Epoch {i + 1} train on cpu")
                history = model.fit(X_batch, Y_batch, batch_size=batch_size, epochs=1, verbose=0)
            train_batch_loss, train_batch_acc = history.history["loss"][0], history.history["accuracy"][0]
            train_loss_batch.append(train_batch_loss)
            train_acc_batch.append(train_batch_acc)
            log.info(f"epoch {i + 1} batch {batch + 1} train_loss {train_batch_loss} train_acc {train_batch_acc}")
        train_loss = float(numpy.mean(numpy.array(train_loss_batch)))  # 计算当前轮训练的损失值
        train_acc = float(numpy.mean(numpy.array(train_acc_batch)))  # 计算当前轮训练集正确率
        
        for batch in range(batch_num):
            start = batch * batch_size
            end = start + batch_size
            X_batch = X[start:end].to(device)
            Y_batch = Y[start:end].to(device)
            if torch.cuda.is_available():
                with torch.cuda.device(0):
                    # log.info(f"Epoch {i + 1} test on device: {device}")
                    scores = model.evaluate(X_batch, Y_batch, batch_size=batch_size, verbose=0)
            else:
                # log.info(f"Epoch {i + 1} test on cpu")
                scores = model.evaluate(X_batch, Y_batch, batch_size=batch_size, verbose=0)
            # 获取损失值
            test_batch_loss, test_batch_acc = scores[0], scores[1]
            test_loss_batch.append(test_batch_loss)
            test_acc_batch.append(test_batch_acc)
            log.info(f"epoch {i + 1} batch {batch + 1} test_loss {test_batch_loss} test_acc {test_batch_acc}")
        test_loss = float(numpy.mean(numpy.array(test_loss_batch)))  # 计算当前轮测试的损失值
        test_acc = float(numpy.mean(numpy.array(test_acc_batch)))  # 计算当前轮测试集正确率
        log.debug(
            f'epoch {i + 1 :3d}/{epochs} train_acc {train_acc:.7f} train_loss {train_loss:.7f} test_acc {test_acc:.7f} test_loss {test_loss:.7f}')


def split_dataset(model: keras.Model, epochs: int, batch_size: int,
                  device: torch.device, X: numpy.ndarray, Y: numpy.ndarray):
    full_dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32),
                                                  torch.tensor(Y, dtype=torch.float32))
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, lengths=[train_size, test_size],
                                                                generator=torch.Generator().manual_seed(0))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size
                                               , shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    for i in range(epochs):
        train_loss_batch, train_acc_batch = [], []
        test_loss_batch, test_acc_batch = [], []
        for batch in train_loader:
            X_batch, Y_batch = batch
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            
            if torch.cuda.is_available():
                with torch.cuda.device(0):
                    # log.info(f"Epoch {i + 1} train on device: {device}")
                    history = model.fit(X_batch, Y_batch, batch_size=batch_size, epochs=1, verbose=0)
            else:
                # log.info(f"Epoch {i + 1} train on cpu")
                history = model.fit(X_batch, Y_batch, batch_size=batch_size, epochs=1, verbose=0)
            train_batch_loss, train_batch_acc = history.history["loss"][0], history.history["accuracy"][0]
            train_loss_batch.append(train_batch_loss)
            train_acc_batch.append(train_batch_acc)
        train_loss = float(numpy.mean(numpy.array(train_loss_batch)))  # 计算当前轮训练的损失值
        train_acc = float(numpy.mean(numpy.array(train_acc_batch)))  # 计算当前轮训练集正确率
        
        for batch in test_loader:
            X_batch, Y_batch = batch
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            if torch.cuda.is_available():
                with torch.cuda.device(0):
                    # log.info(f"Epoch {i + 1} test on device: {device}")
                    scores = model.evaluate(X_batch, Y_batch, batch_size=batch_size, verbose=0)
            else:
                # log.info(f"Epoch {i + 1} test on cpu")
                scores = model.evaluate(X_batch, Y_batch, batch_size=batch_size, verbose=0)
            # 获取损失值
            test_batch_loss, test_batch_acc = scores[0], scores[1]
            test_loss_batch.append(test_batch_loss)
            test_acc_batch.append(test_batch_acc)
        test_loss = float(numpy.mean(numpy.array(test_loss_batch)))  # 计算当前轮测试的损失值
        test_acc = float(numpy.mean(numpy.array(test_acc_batch)))  # 计算当前轮测试集正确率
        
        log.debug(
            f'epoch {i + 1 :3d}/{epochs} train_acc {train_acc:.7f} train_loss {train_loss:.7f} test_acc {test_acc:.7f} test_loss {test_loss:.7f}')
        
        log.info(f'epoch {i + 1:3d}/{epochs} starting predict...')
        for batch in test_loader:
            X_batch, Y_batch = batch
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            if torch.cuda.is_available():
                with torch.cuda.device(0):
                    for j, x in enumerate(X_batch):
                        x_shaped = x.reshape(1, -1)
                        # predictions = (model.predict(x_shaped) > 0.5).astype(int)
                        # print(f'{x.tolist()} predict {predictions[0][0]} correct {Y_batch[i].item()}')
                        pre = (plma_predict(model, x_shaped) > 0.5).astype(int)
                        log.info(f'x_shaped={x_shaped.tolist()} predict {pre} correct {Y_batch[j].item()}')
            else:
                for j, x in enumerate(X_batch):
                    x_shaped = x.reshape(1, -1)
                    # predictions = (model.predict(x_shaped) > 0.5).astype(int)
                    # print(f'{x.tolist()} predict {predictions[0][0]} correct {Y_batch[i].item()}')
                    pre = (plma_predict(model, x_shaped) > 0.5).astype(int)
                    log.info(f'x_shaped={x_shaped.tolist()} predict {pre} correct {Y_batch[j].item()}')
        log.info(f'epoch {i + 1:3d}/{epochs} finish predict...\n')


def plma_predict(model: keras.Model, x: torch.Tensor):
    assert model is not None
    prediction = model.predict(x, verbose=0)
    # print(prediction)
    return prediction
