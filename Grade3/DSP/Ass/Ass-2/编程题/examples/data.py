from sklearn import cluster, datasets
from sklearn.utils.validation import _num_samples


def get_moon_data():
    n_samples = 200
    x, y = datasets.make_moons(n_samples=n_samples, noise=.05, 
    random_state=0)
    return x, y


def get_cycle_data():
    n_samples = 200
    x, y = datasets.make_circles(n_samples=n_samples,
                                 factor=.5,
                                 noise=.05, 
                                 random_state=0)
    return x, y


if __name__ == "__main__":
    get_moon_data()
