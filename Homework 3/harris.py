import cv2
import numpy as np
from Gauss_gen import *


def dif(image):
    tmp = np.zeros(image.shape)
    tmp[:, 1:] = image[:, :-1]
    return image - tmp


def gen_M(Ix, Iy, sigma, size):
    dif_matrix = np.zeros((Ix.shape[0], Ix.shape[1], 2, 2))
    kernel = Gauss_gen(sigma, size)
    edge = size // 2
    M = dif_matrix.copy()
    for y in range(Iy.shape[0]):
        for x in range(Ix.shape[1]):
            dif_matrix[y, x, :, :] = np.array([[Ix[y, x] ** 2, Ix[y, x] * Iy[y, x]],
                                               [Ix[y, x] * Iy[y, x], Iy[y, x] ** 2]])
    for y in range(edge, Iy.shape[0] - edge):
        for x in range(edge, Ix.shape[1] - edge):
            tmp = np.zeros((2, 2))
            for i in range(size):
                for j in range(size):
                    tmp += dif_matrix[y - size + i, x - size + j, :, :] * kernel[i, j]
            M[y, x, :, :] = tmp
    return M


def gen_har(M, alpha):
    har = np.zeros((M.shape[0], M.shape[1]))
    for y in range(M.shape[0]):
        for x in range(M.shape[1]):
            har[y, x] = M[y, x, 0, 0] * M[y, x, 1, 1] - M[y, x, 0, 1] ** 2 - alpha * (
                        M[y, x, 0, 0] + M[y, x, 1, 1]) ** 2
            if har[y, x] < 0:
                har[y, x] = 0
    har = har / har.max() * 255
    return har


def NMS(har, size):
    har_NMS = har.copy()
    edge = size // 2
    for y in range(edge, har_NMS.shape[0] - edge, size):
        for x in range(edge, har_NMS.shape[1] - edge, size):
            local_max = har_NMS[y - edge:y + edge + 1, x - edge:x + edge + 1].max()
            mask = har_NMS[y - edge:y + edge + 1, x - edge:x + edge + 1] == local_max
            har_NMS[y - edge:y + edge + 1, x - edge:x + edge + 1] *= mask
    return har_NMS


img = cv2.imread("test.png", 0)
img_Ix, img_Iy = dif(img), dif(img.T).T
img_M = gen_M(img_Ix, img_Iy, 21, 5)
img_har = gen_har(img_M, 0.04)
img_har_NMS_3 = NMS(img_har, 3)
img_har_NMS_5 = NMS(img_har, 5)
img_har_NMS_11 = NMS(img_har, 11)
img_har_NMS_21 = NMS(img_har, 21)
# cv2.imshow("ori", img)
cv2.imshow("size = 3", img_har_NMS_3[7:, 7:])
cv2.imshow("size = 5", img_har_NMS_5[7:, 7:])
cv2.imshow("size = 11", img_har_NMS_11[7:, 7:])
cv2.imshow("size = 21", img_har_NMS_21[7:, 7:])
cv2.waitKey(0)
