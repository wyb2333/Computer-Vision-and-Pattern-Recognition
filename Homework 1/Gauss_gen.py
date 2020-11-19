import numpy as np


def Gauss_fun(sigma, x):
    return np.exp(-(x / sigma) ** 2 / 2) / (sigma * (2 * np.pi) ** 0.5)


def Gauss_gen(sigma, size=3):
    kernel = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            kernel[i][j] = Gauss_fun(sigma, i - size // 2) * Gauss_fun(sigma, j - size // 2)
    kernel /= np.sum(kernel)
    return kernel
