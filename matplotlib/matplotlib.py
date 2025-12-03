import matplotlib.pyplot as plt

from randwalk.randwalk import RandWalk


def demo1():
    spuares = [1, 4, 9, 16, 25]
    fig, ax = plt.subplots()
    ax.plot(spuares)
    plt.show()


def demo2():
    """randwalk"""
    rl=RandWalk()
    rl.walk()
    plt.style.use("classic")
    fig,ax=plt.subplots()
    ax.scatter(rl.x_values, rl.y_values)
    ax.set_aspect("equal")
    plt.show()


if __name__ == '__main__':
    print("demo1")
    # demo1()
    print("demo2")
    demo2()
