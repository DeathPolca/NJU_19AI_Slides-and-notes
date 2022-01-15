import os
import numpy as np
from PIL import Image

from collections import Counter


def load_images(split):
    subdir = os.path.join("figs_cut", split)

    xs = []
    ys = []
    for fname in os.listdir(subdir):
        if fname.startswith("class") and fname.endswith(".jpg"):
            label = int(fname.split("-")[0][-1])

            fpath = os.path.join(subdir, fname)

            img = Image.open(fpath).resize((32, 32))
            arr = np.array(img) / 255.0 - 0.5  # [-0.5, 0.5]

            xs.append(arr)
            ys.append(label)

    xs = np.array(xs)
    ys = np.array(ys)
    print(xs.shape, ys.shape)
    print(xs.max(), xs.min())
    print(Counter(ys))
    return xs, ys


if __name__ == "__main__":
    # load data
    train_xs, train_ys = load_images(split="train")
    test_xs, test_ys = load_images(split="test")
