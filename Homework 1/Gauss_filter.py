from Gauss_gen import *
from conv import *
import cv2
import time


img = cv2.imread("Lena.jpg")
# img = cv2.imread("cxk.jpeg")


conv_setting = ["full", "same", "valid"]
padding_setting = ["zero", "wrap", "copy", "reflect"]
mode_conv = conv_setting[1]
mode_padding = padding_setting[0]


kernel = Gauss_gen(sigma=1.5, size=11)
if mode_conv == "full":
    img_gauss = np.zeros((img.shape[0] + len(kernel) - 1, img.shape[1] + len(kernel) - 1, 3))
if mode_conv == "same":
    img_gauss = np.zeros((img.shape[0], img.shape[1], 3))
if mode_conv == "valid":
    img_gauss = np.zeros((img.shape[0] - len(kernel) + 1, img.shape[1] - len(kernel) + 1, 3))

size = 11
sep_filter_0 = np.zeros((size, 1))
for i in range(size):
    sep_filter_0[i, 0] = Gauss_fun(1.5, i - size//2)
sep_filter_0 = sep_filter_0 / np.sum(sep_filter_0)
sep_filter_1 = sep_filter_0.T
img_gauss2d = img_gauss.copy()
t1 = time.time()
for i in range(3):
    img_gauss2d[:, :, i] = conv(image=img[:, :, i], kernel=kernel, mode_conv=mode_conv, mode_padding=mode_padding)
t2 = time.time()
for i in range(3):
    img_gauss[:, :, i] = conv(image=img[:, :, i], kernel=sep_filter_0, mode_conv=mode_conv, mode_padding=mode_padding)
for i in range(3):
    img_gauss[:, :, i] = conv(image=img_gauss[:, :, i], kernel=sep_filter_1, mode_conv=mode_conv, mode_padding=mode_padding)
t3 = time.time()


cv2.imshow("ori", img)
cv2.imshow("gauss", img_gauss.astype("uint8"))
cv2.imshow("gauss2d", img_gauss2d.astype("uint8"))
print(t2-t1, t3-t2)
cv2.waitKey(0)
