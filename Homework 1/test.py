from Gauss_gen import *
from conv import *
import cv2


img = cv2.imread("Lena.jpg")
# img = cv2.imread("cxk.jpeg")


conv_setting = ["full", "same", "valid"]
padding_setting = ["zero", "wrap", "copy", "reflect"]
mode_conv = conv_setting[0]
mode_padding = padding_setting[3]


kernel = Gauss_gen(sigma=1.5, size=11)
if mode_conv == "full":
    img_gauss = np.zeros((img.shape[0] + len(kernel) - 1, img.shape[1] + len(kernel) - 1, 3))
if mode_conv == "same":
    img_gauss = np.zeros((img.shape[0], img.shape[1], 3))
if mode_conv == "valid":
    img_gauss = np.zeros((img.shape[0] - len(kernel) + 1, img.shape[1] - len(kernel) + 1, 3))
for i in range(3):
    img_gauss[:, :, i] = conv2d(image=img[:, :, i], kernel=kernel, mode_conv=mode_conv, mode_padding=mode_padding)


cv2.imshow("ori", img)
cv2.imshow("gauss", img_gauss.astype("uint8"))
print(img_gauss.shape)
cv2.waitKey(0)
