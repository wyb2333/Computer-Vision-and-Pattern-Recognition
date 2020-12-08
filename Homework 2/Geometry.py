import numpy as np
import cv2


def trans(img, matrix, shift=False):
    img_trans = np.zeros((img.shape[0], img.shape[1], 3))
    coordinate = np.zeros((3, img.shape[0] * img.shape[1]))
    for i in range(img.shape[1]):
        coordinate[0, i * img.shape[0]:(i + 1) * img.shape[0]] = np.ones(img.shape[0]) * i
        coordinate[1, i * img.shape[0]:(i + 1) * img.shape[0]] = np.array(list(range(img.shape[0])))
        coordinate[2, i * img.shape[0]:(i + 1) * img.shape[0]] = np.ones(img.shape[0])
    if shift:
        bias_x = img.shape[1] // 2
        bias_y = img.shape[0] // 2
        coordinate[0, :] -= bias_x
        coordinate[1, :] = -coordinate[1, :] + bias_y
    coordinate_ori = matrix @ coordinate
    min_x = min(coordinate[0, :])
    max_x = max(coordinate[0, :])
    min_y = min(coordinate[1, :])
    max_y = max(coordinate[1, :])
    coordinate_ori /= coordinate_ori[2, :]
    for i in range(img.shape[1]):
        for j in range(img.shape[0]):
            x_ori, x = coordinate_ori[0, i * img.shape[0] + j], coordinate[0, i * img.shape[0] + j]
            y_ori, y = coordinate_ori[1, i * img.shape[0] + j], coordinate[1, i * img.shape[0] + j]
            if min_x + 1 < x_ori < max_x - 1 and min_y + 1 < y_ori < max_y - 1:
                if shift:
                    x_ori += bias_x
                    x += bias_x
                    y_ori = -y_ori + bias_y
                    y = -y + bias_y
                rate_x = 1 - (x_ori - int(x_ori))
                rate_y = 1 - (y_ori - int(y_ori))
                img_trans[int(y), int(x), :] = rate_x * (rate_y * img[int(y_ori), int(x_ori), :]
                                                         + (1 - rate_y) * img[int(y_ori) + 1, int(x_ori), :]) \
                                               + (1 - rate_x) * (rate_y * img[int(y_ori), int(x_ori) + 1, :]
                                                                 + (1 - rate_y) * img[int(y_ori) + 1, int(x_ori) + 1,
                                                                                  :])
    return img_trans


image = cv2.imread("Lena.jpg")
# image = cv2.imread("cxk.jpeg")
size_edge0 = int(image.shape[0] * 0.5)
size_edge1 = int(image.shape[1] * 0.5)
img_zero = np.zeros((image.shape[0] + 2 * size_edge0, image.shape[1] + 2 * size_edge1, 3))
img_zero[size_edge0:image.shape[0] + size_edge0, size_edge1:image.shape[1] + size_edge1, :] = image
tx, ty = 100, 100
cx, cy = 1.1, 1.3
theta = np.pi / 2
a_11, a_12, a_21, a_22 = 1, 0, 2, 1
coefficient = a_11 * a_22 - a_12 * a_21
h_11, h_12, h_13, h_21, h_22, h_23, h_31, h_32 = 0.97, -0.14, 117, 0.02, 1.02, 160, 0.0005, 0.0005
matrix_trans = np.array([[[1, 0, -tx],
                          [0, 1, -ty],
                          [0, 0, 1]],
                         [[1 / cx, 0, 0],
                          [0, 1 / cy, 0],
                          [0, 0, 1]],
                         [[np.cos(theta), np.sin(theta), 0],
                          [-np.sin(theta), np.cos(theta), 0],
                          [0, 0, 1]],
                         np.multiply([[a_22, -a_12, -a_22 * tx + a_12 * ty],
                                      [-a_21, a_11, a_21 * tx - a_11 * ty],
                                      [0, 0, 1]], 1 / coefficient),
                         np.linalg.inv([[h_11, h_12, h_13],
                                        [h_21, h_22, h_23],
                                        [h_31, h_32, 1]])
                         ])
cv2.imshow("ori", img_zero.astype("uint8"))
# cv2.imshow("trans", trans(img_zero, matrix_trans[4]).astype("uint8"))
# cv2.imshow("trans", trans(img_zero, matrix_trans[4]).astype("uint8"))
# cv2.imshow("trans", trans(img_zero, matrix_trans[4]).astype("uint8"))
# cv2.imshow("trans", trans(img_zero, matrix_trans[4]).astype("uint8"))
# cv2.imshow("trans", trans(img_zero, matrix_trans[4]).astype("uint8"))
cv2.imshow("trans", trans(img_zero, matrix_trans[4], shift=False).astype("uint8"))
cv2.waitKey(0)
