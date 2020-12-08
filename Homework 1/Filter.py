from Gauss_gen import *
from conv import *
import cv2
import time


img = cv2.imread("Lena.jpg")
img_grey = cv2.imread("Lena.jpg", 0)
# img = cv2.imread("Lena_noise.jpg")
# img = cv2.imread("cxk.jpeg")


conv_setting = ["full", "same", "valid"]
padding_setting = ["zero", "wrap", "copy", "reflect"]
mode_conv = conv_setting[1]
mode_padding = padding_setting[0]


size = 5
sigma = 3
sigma_r = 50
kernel = Gauss_gen(sigma=sigma, size=size)
if mode_conv == "full":
    img_gauss = np.zeros((img.shape[0] + len(kernel) - 1, img.shape[1] + len(kernel) - 1, 3))
if mode_conv == "same":
    img_gauss = np.zeros((img.shape[0], img.shape[1], 3))
if mode_conv == "valid":
    img_gauss = np.zeros((img.shape[0] - len(kernel) + 1, img.shape[1] - len(kernel) + 1, 3))


sep_filter_0 = np.zeros((size, 1))
for i in range(size):
    sep_filter_0[i, 0] = Gauss_fun(sigma, i - size//2)
sep_filter_0 = sep_filter_0 / np.sum(sep_filter_0)
sep_filter_1 = sep_filter_0.T
img_gauss1d = img_gauss.copy()
img_sharp = img_gauss.copy()
img_bi = img_gauss.copy()
for i in range(3):
    img_gauss[:, :, i] = conv(image=img[:, :, i], kernel=kernel, mode_conv=mode_conv, mode_padding=mode_padding)
for i in range(3):
    img_bi[:, :, i] = Bilateral_Filter(image=img[:, :, i], sigma_s=sigma, sigma_r=sigma_r, size=size, mode_conv=mode_conv, mode_padding=mode_padding)

for i in range(3):
    img_gauss1d[:, :, i] = conv(image=img[:, :, i], kernel=sep_filter_0, mode_conv=mode_conv, mode_padding=mode_padding)
for i in range(3):
    img_gauss1d[:, :, i] = conv(image=img_gauss1d[:, :, i], kernel=sep_filter_1, mode_conv=mode_conv, mode_padding=mode_padding)

alpha = 1
w = np.zeros((size, size))
w[size // 2, size // 2] = 1
kernel_sharp = (1+alpha)*w - alpha*kernel
for i in range(3):
    img_sharp[:, :, i] = conv(image=img[:, :, i], kernel=kernel_sharp, mode_conv=mode_conv, mode_padding=mode_padding)

cv2.imshow("ori", img)
cv2.imshow("gauss", img_gauss.astype("uint8"))
cv2.imshow("gauss1d", img_gauss1d.astype("uint8"))
cv2.imshow("gauss_minus", conv(image=img_grey, kernel=(kernel-Gauss_gen(1, size)), mode_conv=mode_conv, mode_padding=mode_padding).astype("uint8"))
cv2.normalize(img_sharp, img_sharp, 0, 255, cv2.NORM_MINMAX)
cv2.imshow("sharp", img_sharp.astype("uint8"))
cv2.imshow("bi", img_bi.astype("uint8"))
cv2.waitKey(0)
