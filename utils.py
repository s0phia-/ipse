import numpy as np


def random_tiebreak_argmax(x):
    x = np.array(x)
    return np.random.choice(np.flatnonzero(x == x.max()))


def cross_axis_sd(x):
    """
    :returns: for a list of lists, returns the variance of the first items of each list, variance of the second item of
    each list... etc
    """
    return [np.std(i) for i in list(map(list, zip(*x)))]


def cross_axis_mean(x):
    """
    :returns: for a list of lists, returns the mean of the first items of each list, mean of the second item of each
    list... etc
    """
    return [np.mean(i) for i in list(map(list, zip(*x)))]
