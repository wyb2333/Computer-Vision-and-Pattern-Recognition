from Gauss_gen import *
from conv import *
import cv2


img = cv2.imread("Lena.jpg")
edge_size = 50
img_padding = np.zeros((img.shape[0] + 2 * edge_size, img.shape[1] + 2 * edge_size, 3))
# kernel = Gauss_gen(1.5)
# img_gauss = np.zeros((img.shape[0] - len(kernel) + 1, img.shape[1] - len(kernel) + 1, 3))
for i in range(3):
    img_padding[:, :, i] = padding(img[:, :, i], edge_size, "wrap_around")
#     img_gauss[:, :, i] = conv2d(img[:, :, i], kernel, 1)





cv2.imshow("ori", img)
cv2.imshow("zeor", img_padding.astype("uint8"))
# cv2.imshow("gauss", img_gauss.astype("uint8"))
cv2.waitKey(0)
