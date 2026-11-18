import matplotlib.pyplot as plt

from random import choice


class RandWalk:
    def __init__(self, num_points=5000):
        self.num_points = num_points
        self.x_values = [0]
        self.y_values = [0]
    
    def walk(self):
        while len(self.x_values) < self.num_points:
            x_direction = choice([1, -1])
            x_distance = choice([0, 1, 2, 3, 4])
            x_step = x_direction * x_distance
            
            y_direction = choice([1, -1])
            y_distance = choice([0, 1, 2, 3, 4])
            y_step = y_direction * y_distance
            
            if x_step == 0 and y_step == 0:
                continue
            
            x = self.x_values[-1] + x_step
            y = self.y_values[-1] + y_step
            
            self.x_values.append(x)
            self.y_values.append(y)


def demo1():
    spuares = [1, 4, 9, 16, 25]
    fig, ax = plt.subplots()
    ax.plot(spuares)
    plt.show()


def demo2():
    """randwalk"""
    rl = RandWalk()
    rl.walk()
    plt.style.use("classic")
    fig, ax = plt.subplots()
    ax.scatter(rl.x_values, rl.y_values)
    ax.set_aspect("equal")
    plt.show()


if __name__ == '__main__':
    print("demo1")
    # demo1()
    print("demo2")
    demo2()
