import numpy as np


def conv1d(image, kernel, mode, padding=None):
    pass


def conv2d(image, kernel, mode_conv="same", mode_padding="zero"):
    if mode_conv == "full":
        image_padding = padding(image, len(kernel) - 1, mode_padding=mode_padding)
        image_conv = np.zeros((image.shape[0] + len(kernel) - 1, image.shape[1] + len(kernel) - 1))
    if mode_conv == "same":
        image_padding = padding(image, len(kernel) // 2, mode_padding=mode_padding)
        image_conv = np.zeros((image.shape[0], image.shape[1]))
    if mode_conv == "valid":
        image_padding = image
        image_conv = np.zeros((image.shape[0] - len(kernel) + 1, image.shape[1] - len(kernel) + 1))
    length = len(kernel) // 2
    for i in range(length, image_padding.shape[0] - length):
        for j in range(length, image_padding.shape[1] - length):
            image_conv[i - length, j - length] = np.sum(image_padding[i - length:i + length + 1, j - length:j + length + 1] * kernel)
    return image_conv


def padding(image, size_edge, mode_padding):
    img = np.zeros((image.shape[0] + 2 * size_edge, image.shape[1] + 2 * size_edge))
    if mode_padding == "zero":
        img[size_edge:image.shape[0] + size_edge, size_edge:image.shape[1] + size_edge] = image
    if mode_padding == "wrap":
        # center
        img[size_edge:image.shape[0] + size_edge, size_edge:image.shape[1] + size_edge] = image
        # side
        img[:size_edge, size_edge:-size_edge] = image[image.shape[0] - size_edge:, :]
        img[image.shape[0] + size_edge:, size_edge:-size_edge] = image[:size_edge, :]
        img[size_edge:-size_edge, :size_edge] = image[:, image.shape[1] - size_edge:]
        img[size_edge:-size_edge, image.shape[1] + size_edge:] = image[:, :size_edge]
        # corner
        img[:size_edge, :size_edge] = image[image.shape[0]-size_edge:, image.shape[1]-size_edge:]
        img[:size_edge, image.shape[1] + size_edge:] = image[image.shape[0] - size_edge:, :size_edge]
        img[image.shape[0] + size_edge:, :size_edge] = image[:size_edge, image.shape[1] - size_edge:]
        img[image.shape[0] + size_edge:, image.shape[1] + size_edge:] = image[:size_edge, :size_edge]
    if mode_padding == "copy":
        # center
        img[size_edge:image.shape[0] + size_edge, size_edge:image.shape[1] + size_edge] = image
        # side
        img[:size_edge, size_edge:-size_edge] = image[0, :]
        img[image.shape[0] + size_edge:, size_edge:-size_edge] = image[-1, :]
        img = img.T
        img[:size_edge, size_edge:-size_edge] = image[:, 0]
        img[image.shape[1] + size_edge:, size_edge:-size_edge] = image[:, -1]
        img = img.T
        # corner
        img[:size_edge, :size_edge] = image[0, 0]
        img[:size_edge, image.shape[1] + size_edge:] = image[0, -1]
        img[image.shape[0] + size_edge:, :size_edge] = image[-1, 0]
        img[image.shape[0] + size_edge:, image.shape[1] + size_edge:] = image[-1, -1]
    if mode_padding == "reflect":
        # center
        img[size_edge:image.shape[0] + size_edge, size_edge:image.shape[1] + size_edge] = image
        # side
        img[:size_edge, size_edge:-size_edge] = image[:size_edge, :][::-1, :]
        img[image.shape[0] + size_edge:, size_edge:-size_edge] = image[image.shape[0] - size_edge:, :][::-1, :]
        img[size_edge:-size_edge, :size_edge] = image[:, :size_edge][:, ::-1]
        img[size_edge:-size_edge, image.shape[1] + size_edge:] = image[:, image.shape[1] - size_edge:][:, ::-1]
        # corner
        img[:size_edge, :size_edge] = image[:size_edge, :size_edge][::-1, ::-1]
        img[:size_edge, image.shape[1] + size_edge:] = image[:size_edge, image.shape[1] - size_edge:][::-1, ::-1]
        img[image.shape[0] + size_edge:, :size_edge] = image[image.shape[0] - size_edge:, :size_edge][::-1, ::-1]
        img[image.shape[0] + size_edge:, image.shape[1] + size_edge:] = image[image.shape[0] - size_edge:, image.shape[1] - size_edge:][::-1, ::-1]
    return img
