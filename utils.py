import numpy as np
from sklearn.linear_model import LinearRegression



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


def fit_stew(x, y, d, lam):
    """
    Fits STEW
    """
    x = np.array(x)
    y = np.array(y)
    a = np.matmul(x.transpose(), x) + lam * np.matmul(d.transpose(), d)
    b = np.matmul(x.transpose(), y)
    return np.matmul(np.linalg.inv(a), b)


def fit_ew(x):
    """
    fits pure equal weights
    """
    x = np.array(x)
    return np.ones(x.shape[1])


def fit_ridge(x, y, lam):
    """
    Fits ridge regression
    """
    x = np.array(x)
    y = np.array(y)
    a = np.matmul(x.transpose(), x) + (lam * np.identity(x.shape[1]))
    b = np.matmul(x.transpose(), y)
    return np.matmul(np.linalg.inv(a), b)


def fit_lin_reg(x, y):
    reg = LinearRegression(fit_intercept=False).fit(x, y)
    return reg.coef_
