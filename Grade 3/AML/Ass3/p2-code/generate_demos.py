import matplotlib.pyplot as plt

import os

import random
random.seed(0)


markers = [
    "o",
    "s",
    "p",
    "h",
    "v",
    "<",
    "*",
    "+",
    "x",
    "D",
]

makers2int = {marker: i for i, marker in enumerate(markers)}


def random_color():
    return "#" + "".join([
        random.choice("0123456789ABCDEF") for i in range(6)
    ])


def random_size(smin, smax):
    return random.random() * (smax - smin) + smin


def random_position():
    x = 2.0 * random.random() - 1.0
    y = 2.0 * random.random() - 1.0
    return x, y


def plot_demo_markers(t):
    fig = plt.figure(figsize=(10, 4))

    for k, marker in enumerate(markers):
        plt.subplot(2, 5, k + 1)

        s = random_size(200, 400)
        c = random_color()
        x, y = random_position()
        plt.scatter([x], [y], marker=marker, s=s, c=c)
        plt.xticks([])
        plt.yticks([])
        plt.xlim(-1.0, 1.0)
        plt.ylim(-1.0, 1.0)

        plt.title("Class {}".format(k), fontsize=15)

    fig.tight_layout()
    fig.savefig(os.path.join("demos", "demo-{}.jpg".format(t)))
    plt.close()


if __name__ == "__main__":
    # generate demo data
    if not os.path.exists("./demos"):
        os.mkdir("./demos")

    for t in range(5):
        plot_demo_markers(t)
