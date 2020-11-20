import numpy as np
import cv2


img = cv2.imread("lena.jpg", 0)
img_cxk = cv2.imread("cxk_head.png", 0)
cxk_head = img_cxk[img_cxk.shape[0] // 2 - 100:img_cxk.shape[0] // 2 + 156, img_cxk.shape[1] // 2 - 80:img_cxk.shape[1] // 2 + 176]
img_fourier = np.fft.fft2(img)
img_fourier_shift = np.fft.fftshift(img_fourier)
Magnitude = 20*np.log(np.abs(img_fourier))
Magnitude_shift = 20*np.log(np.abs(img_fourier_shift))
Phase = np.angle(img_fourier)
Phase_shift = np.angle(img_fourier_shift)
cv2.imshow("ori", img)
cv2.imshow("Fourier", Magnitude.astype("uint8"))
cv2.imshow("Phase", Phase.astype("uint8"))
cv2.imshow("Fourier_shift", Magnitude_shift.astype("uint8"))
cv2.imshow("Phase_shift", Phase_shift.astype("uint8"))
cv2.imshow("ifft", np.fft.ifft2(img_fourier+cxk_head).astype("uint8"))
Magnitude_lyq = Magnitude + 0.1*cxk_head
cv2.imshow("Magnitude_lyq", Magnitude_lyq.astype("uint8"))
Magnitude_lyq = np.exp(Magnitude_lyq / 20)
lyqnb = np.real(np.fft.ifft2(np.multiply(Magnitude_lyq, np.exp(1j*Phase))))
cv2.imshow("lyqnb", lyqnb.astype("uint8"))
img_fourier_lyq = np.fft.fft2(lyqnb)
Magnitude = 20*np.log(np.abs(img_fourier_lyq))
cv2.imshow("lyqnb_Magnitude", Magnitude.astype("uint8"))
cv2.waitKey(0)
