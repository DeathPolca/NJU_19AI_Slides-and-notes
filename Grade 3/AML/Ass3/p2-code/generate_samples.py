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


def generate_samples(n_samples, split):
    for c, marker in enumerate(markers):
        for i in range(n_samples):
            fig = plt.figure(figsize=(0.64, 0.64))
            size = random_size(100, 200)
            color = random_color()
            px, py = random_position()
            plt.scatter([px], [py], marker=marker, s=size, c=color)
            plt.xticks([])
            plt.yticks([])
            plt.xlim(-1.0, 1.0)
            plt.ylim(-1.0, 1.0)

            fig.tight_layout()
            fig.savefig(os.path.join(
                "figs", split, "class{}-{}.jpg".format(c, i)
            ))
            plt.close()


if __name__ == "__main__":
    # generate train/test data
    if not os.path.exists("./figs"):
        os.mkdir("./figs")

    for split in ["train", "test"]:
        subdir = os.path.join("figs", split)

        if not os.path.exists(subdir):
            os.mkdir(subdir)

        generate_samples(
            n_samples=200,
            split=split
        )
