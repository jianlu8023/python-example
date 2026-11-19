import torch


def tensor_one():
    a = torch.ones(2, 3)
    print(a)


def tensor_zero():
    a = torch.zeros(2, 4)
    print(a)


def tensor_random():
    a = torch.randn(2, 3)
    print(a)


def tensor_from_numpy():
    import numpy as np
    numpy_arr = np.array([[1, 2], [3, 4]])
    a = torch.from_numpy(numpy_arr)
    print(a)
    return a


def tensor_from_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = torch.randn(2, 4, device=device)
    print(a)


def sum_tensor():
    a = tensor_from_numpy()
    b = tensor_from_numpy()
    print(a + b)


def mul_tensor():
    a = tensor_from_numpy()
    b = tensor_from_numpy()
    print(a * b)


if __name__ == "__main__":
    print("tensor_one")
    tensor_one()
    print("tensor_zero")
    tensor_zero()
    print("tensor_random")
    tensor_random()
    print("tensor_from_numpy")
    tensor_from_numpy()
    print("tensor_from_device")
    tensor_from_device()
    print("sum_tensor")
    sum_tensor()
    print("mul_tensor")
    mul_tensor()
