import numpy as np
from Gauss_gen import *


def conv(image, kernel, mode_conv="same", mode_padding="zero"):
    if mode_conv == "full":
        image_padding = padding(image, kernel.shape[0] - 1, kernel.shape[1] - 1, mode_padding=mode_padding)
        image_conv = np.zeros((image.shape[0] + kernel.shape[0] - 1, image.shape[1] + kernel.shape[1] - 1))
    if mode_conv == "same":
        image_padding = padding(image, kernel.shape[0] // 2, kernel.shape[1] // 2, mode_padding=mode_padding)
        image_conv = np.zeros((image.shape[0], image.shape[1]))
    if mode_conv == "valid":
        image_padding = image
        image_conv = np.zeros((image.shape[0] - kernel.shape[0] + 1, image.shape[1] - kernel.shape[1] + 1))
    length0 = kernel.shape[0] // 2
    length1 = kernel.shape[1] // 2
    kernel = kernel[::-1, ::-1]
    for i in range(length0, image_padding.shape[0] - length0):
        for j in range(length1, image_padding.shape[1] - length1):
            image_conv[i - length0, j - length1] = np.sum(image_padding[i - length0:i + length0 + 1, j - length1:j + length1 + 1] * kernel)
    return image_conv


def padding(image, size_edge0, size_edge1, mode_padding):
    img = np.zeros((image.shape[0] + 2 * size_edge0, image.shape[1] + 2 * size_edge1))
    if mode_padding == "zero":
        img[size_edge0:image.shape[0] + size_edge0, size_edge1:image.shape[1] + size_edge1] = image
    if mode_padding == "wrap":
        # center
        img[size_edge0:image.shape[0] + size_edge0, size_edge1:image.shape[1] + size_edge1] = image
        # side
        img[:size_edge0, size_edge1:-size_edge1] = image[image.shape[0] - size_edge0:, :]
        img[image.shape[0] + size_edge0:, size_edge1:-size_edge1] = image[:size_edge0, :]
        img[size_edge0:-size_edge0, :size_edge1] = image[:, image.shape[1] - size_edge0:]
        img[size_edge0:-size_edge0, image.shape[1] + size_edge1:] = image[:, :size_edge1]
        # corner
        img[:size_edge0, :size_edge1] = image[image.shape[0]-size_edge0:, image.shape[1]-size_edge1:]
        img[:size_edge0, image.shape[1] + size_edge1:] = image[image.shape[0] - size_edge0:, :size_edge1]
        img[image.shape[0] + size_edge0:, :size_edge1] = image[:size_edge0, image.shape[1] - size_edge1:]
        img[image.shape[0] + size_edge0:, image.shape[1] + size_edge1:] = image[:size_edge0, :size_edge1]
    if mode_padding == "copy":
        # center
        img[size_edge0:image.shape[0] + size_edge0, size_edge1:image.shape[1] + size_edge1] = image
        # side
        img[:size_edge0, size_edge1:-size_edge1] = image[0, :]
        img[image.shape[0] + size_edge0:, size_edge1:-size_edge1] = image[-1, :]
        img = img.T
        img[:size_edge1, size_edge0:-size_edge0] = image[:, 0]
        img[image.shape[1] + size_edge1:, size_edge0:-size_edge0] = image[:, -1]
        img = img.T
        # corner
        img[:size_edge0, :size_edge1] = image[0, 0]
        img[:size_edge0, image.shape[1] + size_edge1:] = image[0, -1]
        img[image.shape[0] + size_edge0:, :size_edge1] = image[-1, 0]
        img[image.shape[0] + size_edge0:, image.shape[1] + size_edge1:] = image[-1, -1]
    if mode_padding == "reflect":
        # center
        img[size_edge0:image.shape[0] + size_edge0, size_edge1:image.shape[1] + size_edge1] = image
        # side
        img[:size_edge0, size_edge1:-size_edge1] = image[:size_edge0, :][::-1, :]
        img[image.shape[0] + size_edge0:, size_edge1:-size_edge1] = image[image.shape[0] - size_edge0:, :][::-1, :]
        img[size_edge0:-size_edge0, :size_edge1] = image[:, :size_edge1][:, ::-1]
        img[size_edge0:-size_edge0, image.shape[1] + size_edge1:] = image[:, image.shape[1] - size_edge1:][:, ::-1]
        # corner
        img[:size_edge0, :size_edge1] = image[:size_edge0, :size_edge1][::-1, ::-1]
        img[:size_edge0, image.shape[1] + size_edge1:] = image[:size_edge0, image.shape[1] - size_edge1:][::-1, ::-1]
        img[image.shape[0] + size_edge0:, :size_edge1] = image[image.shape[0] - size_edge0:, :size_edge1][::-1, ::-1]
        img[image.shape[0] + size_edge0:, image.shape[1] + size_edge1:] = image[image.shape[0] - size_edge0:, image.shape[1] - size_edge1:][::-1, ::-1]
    return img


def Bilateral_Filter(image, sigma_s, sigma_r, size, mode_conv="same", mode_padding="zero"):
    kernel_s = Gauss_gen(sigma=sigma_s, size=size)
    if mode_conv == "full":
        image_padding = padding(image, kernel_s.shape[0] - 1, kernel_s.shape[1] - 1, mode_padding=mode_padding)
        image_conv = np.zeros((image.shape[0] + kernel_s.shape[0] - 1, image.shape[1] + kernel_s.shape[1] - 1))
    if mode_conv == "same":
        image_padding = padding(image, kernel_s.shape[0] // 2, kernel_s.shape[1] // 2, mode_padding=mode_padding)
        image_conv = np.zeros((image.shape[0], image.shape[1]))
    if mode_conv == "valid":
        image_padding = image
        image_conv = np.zeros((image.shape[0] - kernel_s.shape[0] + 1, image.shape[1] - kernel_s.shape[1] + 1))
    length0 = kernel_s.shape[0] // 2
    length1 = kernel_s.shape[1] // 2
    for i in range(length0, image_padding.shape[0] - length0):
        for j in range(length1, image_padding.shape[1] - length1):
            delta = image_padding[i - length0:i + length0 + 1, j - length1:j + length1 + 1] - image_padding[i, j]
            kernel_r = Gauss_fun(sigma_r, delta)
            kernel = np.multiply(kernel_r, kernel_s)
            kernel /= np.sum(kernel)
            image_conv[i - length0, j - length1] = np.sum(np.multiply(image_padding[i - length0:i + length0 + 1, j - length1:j + length1 + 1], kernel))
    return image_conv
