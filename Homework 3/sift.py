import cv2
import numpy as np

img1 = cv2.imread('image_1.png')
img2 = cv2.imread('image_2.png')
rows, cols = img1.shape[:2]
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])
img_sift = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good[:100], None, flags=2)

cv2.imshow('sift', img_sift)
cv2.waitKey(0)
cv2.destroyAllWindows()
