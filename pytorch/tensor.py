import torch


def tensorone():
    a = torch.ones(2, 3)
    print(a)


def tensorzero():
    a = torch.zeros(2, 4)
    print(a)


def tensorrandom():
    a = torch.randn(2, 3)
    print(a)


def tensorfromnumpy():
    import numpy as np
    numpy_arr = np.array([[1, 2], [3, 4]])
    a = torch.from_numpy(numpy_arr)
    print(a)
    return a


def tensorfromdevice():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = torch.randn(2, 4, device=device)
    print(a)


def sumtensor():
    a = tensorfromnumpy()
    b = tensorfromnumpy()
    print(a + b)


def multensor():
    a = tensorfromnumpy()
    b = tensorfromnumpy()
    print(a * b)


if __name__ == "__main__":
    print("tensorone")
    tensorone()
    print("tensorzero")
    tensorzero()
    print("tensorrandom")
    tensorrandom()
    print("tensorfromdevice")
    tensorfromnumpy()
    print("tensorfromdevice")
    tensorfromdevice()
    print("sumtensor")
    sumtensor()
    print("multensor")
    multensor()
