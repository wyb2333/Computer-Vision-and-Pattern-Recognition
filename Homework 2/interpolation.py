import numpy as np
import cv2
from math import ceil


def n_amplify(picture, times):
    x, y, z = picture.shape
    x_big = x * times
    y_big = y * times
    picture_big = np.zeros((x_big, y_big, z))
    for i in range(x):
        for j in range(y):
            for k in range(-ceil(times / 2), ceil(times / 2)):
                for l in range(-ceil(times / 2), ceil(times / 2)):
                    picture_big[i * times + k][j * times + l] = picture[i][j]
    return cv2.convertScaleAbs(picture_big)


def b_amplify(picture, times):
    picture = picture.astype("float64")
    x, y, z = picture.shape
    x_big = x * times
    y_big = y * times
    picture_big = np.zeros((x_big, y_big, z))
    pic1 = picture[x - 1, :, :]
    pic2 = picture[:, y - 1, :]
    pic2 = np.insert(pic2, x, values=picture[x-1, y-1, :], axis=0)
    picture = np.insert(picture, x, values=pic1, axis=0)
    picture = np.insert(picture, y, values=pic2, axis=1)
    for i in range(x):
        for j in range(y):
            for k in range(times):
                for l in range(times):
                    line1 = l * ((picture[i][j + 1] - picture[i][j]) / times) + picture[i][j]
                    line2 = l * ((picture[i + 1][j + 1] - picture[i + 1][j]) / times) + picture[i+1][j]
                    picture_big[i * times + k][j * times + l] = k * (line2 - line1) / times + line1
    return cv2.convertScaleAbs(picture_big)


img = cv2.imread("lena.jpg")
cv2.imshow("big1", n_amplify(img, 3))
cv2.imshow("big2", b_amplify(img, 3))
cv2.waitKey(0)
