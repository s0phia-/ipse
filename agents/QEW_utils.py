from sklearn.linear_model import LinearRegression
import numpy as np


def fit_stew(x, y, d, lam):
    """
    Fits STEW
    """
    a = np.matmul(x.transpose(), x) + lam * np.matmul(d.transpose(), d)
    b = np.matmul(x.transpose(), y)
    return np.matmul(np.linalg.inv(a), b)


def fit_ew(x):
    """
    fits pure equal weights
    """
    return np.ones(x.shape[1]) 


def fit_ridge(x, y, lam):
    """
    Fits ridge regression
    """
    a = np.matmul(x.transpose(), x) + lam * np.identity(x.shape[1])
    b = np.matmul(x.transpose(), y)
    return np.matmul(np.linalg.inv(a), b)


def fit_lin_reg(x, y):
    reg = LinearRegression(fit_intercept=False).fit(x, y)
    return reg.coef_
